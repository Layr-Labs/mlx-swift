// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

@testable import MLXNN

/// Tests for `TransformerDecoderLayer` in `Transformer.swift`.
///
/// The layer is compared with a reference that uses the layer's own weights
/// and plain ops.
class TransformerDecoderLayerTests: XCTestCase {

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
                assertClose(y, layer.ln3(h + mlp(h)), "normFirst: false")
            }
        }
    }
}
