// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

@testable import MLXNN

/// Tests for `MultiHeadAttention`, the encoder and decoder layers and
/// `Transformer` in `Transformer.swift`.
///
/// The attention and the layers are compared with references that use the
/// layer's own weights and plain ops. Other tests check output shapes, that
/// seeded weights give the same output, and the effect of the causal mask.
class TransformerTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    private func assertClose(
        _ a: MLXArray, _ b: MLXArray, _ message: String = "",
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(a.shape, b.shape, message, file: file, line: line)
        XCTAssertTrue(
            a.allClose(b, rtol: 1e-4, atol: 1e-5).item(Bool.self),
            "\(message)\n\(a)\n\(b)", file: file, line: line)
    }

    private func normal(_ shape: [Int], seed: UInt64) -> MLXArray {
        MLXRandom.normal(shape, key: MLXRandom.key(seed))
    }

    /// Attention with plain ops: project, split into heads, softmax(q k^T * scale + mask) v,
    /// join the heads and project out.
    private func referenceAttention(
        _ mha: MultiHeadAttention, _ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let heads = mha.numHeads
        func toHeads(_ x: MLXArray) -> MLXArray {
            x.reshaped([x.dim(0), x.dim(1), heads, x.dim(2) / heads]).transposed(0, 2, 1, 3)
        }
        let qh = toHeads(mha.queryProjection(q))
        let kh = toHeads(mha.keyProjection(k))
        let vh = toHeads(mha.valueProjection(v))
        var scores = matmul(qh, kh.transposed(0, 1, 3, 2)) * (1 / sqrt(Float(qh.dim(-1))))
        if let mask {
            scores = scores + mask
        }
        let weights = softmax(scores, axis: -1)
        let out = matmul(weights, vh).transposed(0, 2, 1, 3)
        return mha.outProjection(out.reshaped([q.dim(0), q.dim(1), -1]))
    }

    // MARK: - MultiHeadAttention

    func testMultiHeadAttentionMatchesReference() {
        MLXRandom.seed(1)
        let mha = MultiHeadAttention(
            dimensions: 8, numHeads: 2, queryInputDimensions: 6, keyInputDimensions: 4,
            valueInputDimensions: 5, valueDimensions: 6, valueOutputDimensions: 7, bias: true)
        let q = normal([2, 3, 6], seed: 2)
        let k = normal([2, 5, 4], seed: 3)
        let v = normal([2, 5, 5], seed: 4)

        let y = mha(q, keys: k, values: v)
        XCTAssertEqual(y.shape, [2, 3, 7])
        assertClose(y, referenceAttention(mha, q, k, v), "no mask")

        let mask = normal([3, 5], seed: 5)
        assertClose(
            mha(q, keys: k, values: v, mask: mask),
            referenceAttention(mha, q, k, v, mask: mask), "additive mask")
    }

    /// Without `bias: true` the projections have no bias.
    func testMultiHeadAttentionHasNoBiasByDefault() {
        let mha = MultiHeadAttention(dimensions: 8, numHeads: 2)
        let keys = mha.parameters().flattened().map { $0.0 }
        XCTAssertEqual(keys.count, 4)
        XCTAssertTrue(keys.allSatisfy { $0.hasSuffix(".weight") }, "\(keys)")
    }

    func testCausalMaskValues() {
        let mask = MultiHeadAttention.createAdditiveCausalMask(4)
        XCTAssertEqual(mask.dtype, .float32)
        let blocked = Float(DType.float32.finfo!.min)
        let expected = MLXArray(
            [
                0, blocked, blocked, blocked,
                0, 0, blocked, blocked,
                0, 0, 0, blocked,
                0, 0, 0, 0,
            ] as [Float], [4, 4])
        XCTAssertTrue(mask.arrayEqual(expected).item(Bool.self), "\(mask)")
    }

    /// A float16 mask must still be 0 where a position may attend.
    func testCausalMaskFloat16() {
        let mask = MultiHeadAttention.createAdditiveCausalMask(4, dtype: .float16)
        XCTAssertEqual(mask.dtype, .float16)
        let values = mask.asType(.float32)
        let indices = MLXArray(0 ..< 4)
        let blocked = expandedDimensions(indices, axis: 1) .< expandedDimensions(indices, axis: 0)
        XCTAssertFalse(isNaN(values).any().item(Bool.self), "\(values)")
        let ok = which(blocked, values .< -1e4, values .== 0)
        XCTAssertTrue(ok.all().item(Bool.self), "\(values)")
    }

    /// With the causal mask, output at position t does not depend on the input after t.
    func testCausalMaskHidesLaterPositions() {
        MLXRandom.seed(6)
        let mha = MultiHeadAttention(dimensions: 8, numHeads: 2)
        let mask = MultiHeadAttention.createAdditiveCausalMask(5)
        let x = normal([1, 5, 8], seed: 7)
        let changed = concatenated([x[0..., ..<4], x[0..., 4...] + 3], axis: 1)

        let a = mha(x, keys: x, values: x, mask: mask)
        let b = mha(changed, keys: changed, values: changed, mask: mask)
        assertClose(a[0..., ..<4], b[0..., ..<4], "the earlier positions changed")
        XCTAssertFalse(a[0..., 4].allClose(b[0..., 4]).item(Bool.self))

        // Without the mask, the change reaches the first position too.
        let c = mha(x, keys: x, values: x)
        let d = mha(changed, keys: changed, values: changed)
        XCTAssertFalse(c[0..., 0].allClose(d[0..., 0]).item(Bool.self))
    }

    // MARK: - Encoder and decoder layers

    func testEncoderLayerMatchesReference() {
        let x = normal([2, 4, 8], seed: 8)
        let mask = MultiHeadAttention.createAdditiveCausalMask(4)
        for normFirst in [true, false] {
            MLXRandom.seed(9)
            let layer = TransformerEncoderLayer(
                dimensions: 8, numHeads: 2, mlpDimensions: 16, normFirst: normFirst)
            func mlp(_ x: MLXArray) -> MLXArray {
                layer.linear2(maximum(layer.linear1(x), 0))
            }
            func attend(_ x: MLXArray) -> MLXArray {
                referenceAttention(layer.attention, x, x, x, mask: mask)
            }
            let expected: MLXArray
            if normFirst {
                let h = x + attend(layer.ln1(x))
                expected = h + mlp(layer.ln2(h))
            } else {
                let h = layer.ln1(x + attend(x))
                expected = layer.ln2(h + mlp(h))
            }
            assertClose(layer(x, mask: mask), expected, "normFirst: \(normFirst)")
        }
    }

    func testDecoderLayerMatchesReference() {
        let x = normal([2, 4, 8], seed: 10)
        let memory = normal([2, 3, 8], seed: 11)
        let xMask = MultiHeadAttention.createAdditiveCausalMask(4)
        let memoryMask = MLXArray.zeros([4, 3])
        for normFirst in [true, false] {
            MLXRandom.seed(12)
            let layer = TransformerDecoderLayer(
                dimensions: 8, numHeads: 2, mlpDimensions: 16, normFirst: normFirst)
            func mlp(_ x: MLXArray) -> MLXArray {
                layer.linear2(maximum(layer.linear1(x), 0))
            }
            func selfAttend(_ x: MLXArray) -> MLXArray {
                referenceAttention(layer.selfAttention, x, x, x, mask: xMask)
            }
            func crossAttend(_ x: MLXArray) -> MLXArray {
                referenceAttention(layer.crossAttention, x, memory, memory, mask: memoryMask)
            }
            let y = layer(x, memory: memory, xMask: xMask, memoryMask: memoryMask)
            XCTAssertEqual(y.shape, [2, 4, 8])
            if normFirst {
                var h = x + selfAttend(layer.ln1(x))
                h = h + crossAttend(layer.ln2(h))
                assertClose(y, h + mlp(layer.ln3(h)), "normFirst: true")
            } else {
                // The post-norm order from upstream MLX: the cross attention
                // takes its queries from the normalized sum x.
                var h = layer.ln1(x + selfAttend(x))
                h = layer.ln2(h + crossAttend(h))
                let expected = layer.ln3(h + mlp(h))
                assertClose(y, expected, "normFirst: false")
            }
        }
    }

    // MARK: - Transformer

    private func makeTransformer(seed: UInt64, normFirst: Bool) -> Transformer {
        MLXRandom.seed(seed)
        return Transformer(
            dimensions: 8, numHeads: 2, encoderLayerCount: 2, decoderLayerCount: 2,
            mlpDimensions: 16, normFirst: normFirst)
    }

    private struct Inputs {
        let source: MLXArray
        let target: MLXArray
        let sourceMask = MLXArray.zeros([3, 3])
        let targetMask = MultiHeadAttention.createAdditiveCausalMask(4)
        let memoryMask = MLXArray.zeros([4, 3])
    }

    private func run(_ t: Transformer, _ i: Inputs) -> MLXArray {
        t(
            source: i.source, target: i.target, sourceMask: i.sourceMask,
            targetMask: i.targetMask, memoryMask: i.memoryMask)
    }

    func testTransformerShapeAndSeededDeterminism() {
        let inputs = Inputs(
            source: normal([2, 3, 8], seed: 13), target: normal([2, 4, 8], seed: 14))
        for normFirst in [true, false] {
            let a = run(makeTransformer(seed: 15, normFirst: normFirst), inputs)
            let b = run(makeTransformer(seed: 15, normFirst: normFirst), inputs)
            let c = run(makeTransformer(seed: 16, normFirst: normFirst), inputs)
            XCTAssertEqual(a.shape, [2, 4, 8])
            XCTAssertTrue(a.arrayEqual(b).item(Bool.self), "same seed, normFirst: \(normFirst)")
            XCTAssertFalse(a.allClose(c).item(Bool.self), "other seed, normFirst: \(normFirst)")
        }
    }

    /// The encoder and decoder layer counts give the number of layers.
    func testTransformerLayerCounts() {
        let t = Transformer(
            dimensions: 8, numHeads: 2, encoderLayerCount: 1, decoderLayerCount: 3,
            mlpDimensions: 16)
        XCTAssertEqual(t.encoder.layers.count, 1)
        XCTAssertEqual(t.decoder.layers.count, 3)
        XCTAssertEqual(t.encoder.layers[0].linear1.weight.shape, [16, 8])
    }

    /// With a causal target mask, a change to the last target position leaves
    /// the earlier outputs as they were. A change to the source changes all outputs.
    /// The changes are random, not one constant: LayerNorm removes a constant added
    /// to all features, so a constant change would not show in the pre-norm layers.
    func testTransformerTargetMaskIsCausal() {
        let source = normal([1, 3, 8], seed: 17)
        let target = normal([1, 4, 8], seed: 18)
        let changedTarget = concatenated(
            [target[0..., ..<3], target[0..., 3...] + normal([1, 1, 8], seed: 20)], axis: 1)
        let changedSource = source + normal([1, 3, 8], seed: 21)
        for normFirst in [true, false] {
            let t = makeTransformer(seed: 19, normFirst: normFirst)
            let a = run(t, Inputs(source: source, target: target))
            let b = run(t, Inputs(source: source, target: changedTarget))
            assertClose(a[0..., ..<3], b[0..., ..<3], "normFirst: \(normFirst)")
            XCTAssertFalse(
                a[0..., 3].allClose(b[0..., 3]).item(Bool.self), "normFirst: \(normFirst)")

            let c = run(t, Inputs(source: changedSource, target: target))
            XCTAssertFalse(
                a[0..., 0].allClose(c[0..., 0]).item(Bool.self), "normFirst: \(normFirst)")
        }
    }
}
