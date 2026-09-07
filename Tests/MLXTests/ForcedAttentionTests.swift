import Foundation
import MLX
import XCTest

final class ForcedAttentionTests: XCTestCase {
    private func values(_ shape: [Int], salt: Int) -> MLXArray {
        let count = shape.reduce(1, *)
        return MLXArray((0..<count).map { Float(($0 * salt) % 61 - 30) / 97 }, shape)
    }

    func testFalsePreservesExistingRouting() throws {
        let q = values([1, 4, 17, 256], salt: 7)
        let k = values([1, 1, 97, 256], salt: 11)
        let v = values([1, 1, 97, 256], salt: 13)
        let mask = MLXArray.ones([17, 97], dtype: .bool)
        let old = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 0.0625, mask: .array(mask))
        let explicit = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 0.0625,
            mask: mask, forceFused: false)
        try withError { eval(old, explicit); Stream.gpu.synchronize() }
        XCTAssertTrue(arrayEqual(old, explicit).item(Bool.self))
    }

    func testFusedPitchedViewsAndSinksMatchDenseReference() throws {
        for dimension in [64, 192, 256] {
            let q = values([1, 4, 32, dimension], salt: 7)[0..., 0..., 0..<17, 0...]
            let k = values([1, 1, 113, dimension], salt: 11)[0..., 0..., 0..<97, 0...]
            let v = values([1, 1, 113, dimension], salt: 13)[0..., 0..., 0..<97, 0...]
            let mask = MLXArray((0..<32*113).map { $0 % 113 <= 70 + $0 / 113 }, [32, 113])[0..<17, 0..<97]
            let sinks = MLXArray([Float(-0.8), 0.2, 0.7, 1.3])
            let scale = 1 / sqrt(Float(dimension))
            let scores = matmul(q * scale, k.transposed(0, 1, 3, 2))
            let masked = which(mask, scores, MLXArray(-Float.infinity))
            let sinkScores = broadcast(sinks.reshaped([1, 4, 1, 1]), to: [1, 4, 17, 1])
            let probabilities = softmax(concatenated([masked, sinkScores], axis: -1), axis: -1)
            let allValues = concatenated([v, MLXArray.zeros([1, 1, 1, dimension])], axis: 2)
            let reference = matmul(probabilities, allValues)
            let result = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale,
                mask: .array(mask), sinks: sinks, forceFused: true)
            try withError { eval(result, reference); Stream.gpu.synchronize() }
            XCTAssertEqual(result.shape, [1, 4, 17, dimension])
            XCTAssertTrue(allClose(result, reference, rtol: 1e-4, atol: 2e-5).item(Bool.self))
        }
    }

    func testUnsupportedHeadRefusesInsteadOfFallingBack() {
        let q = MLXArray.zeros([1, 4, 17, 512])
        let k = MLXArray.zeros([1, 1, 33, 512])
        XCTAssertThrowsError(try withError {
            _ = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: k, scale: 0.1,
                mask: .causal, forceFused: true)
        }) { error in
            XCTAssertTrue(String(describing: error).contains("force_fused"))
        }
    }

    func testWideHeadGQA8MatchesDenseReference() throws {
        let q = values([1, 16, 64, 256], salt: 7)[0..., 0..., 0..<33, 0...] * 3
        let k = values([1, 2, 300, 256], salt: 11)[0..., 0..., 0..<273, 0...] * 3
        let v = values([1, 2, 300, 256], salt: 13)[0..., 0..., 0..<273, 0...]
        let mask = MLXArray((0..<64*300).map { $0 % 300 <= 200 + $0 / 300 }, [64, 300])[0..<33, 0..<273]
        let sinks = MLXArray((0..<16).map { Float($0) * 0.2 - 1 })
        let fullK = broadcast(k.expandedDimensions(axis: 2), to: [1, 2, 8, 273, 256])
            .reshaped([1, 16, 273, 256])
        let fullV = broadcast(v.expandedDimensions(axis: 2), to: [1, 2, 8, 273, 256])
            .reshaped([1, 16, 273, 256])
        let scores = matmul(q * 0.0625, fullK.transposed(0, 1, 3, 2))
        let masked = which(mask, scores, MLXArray(-Float.infinity))
        let sinkScores = broadcast(sinks.reshaped([1, 16, 1, 1]), to: [1, 16, 33, 1])
        let probabilities = softmax(concatenated([masked, sinkScores], axis: -1), axis: -1)
        let reference = matmul(probabilities,
            concatenated([fullV, MLXArray.zeros([1, 16, 1, 256])], axis: 2))
        let result = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 0.0625,
            mask: .array(mask), sinks: sinks, forceFused: true)
        try withError { eval(result, reference); Stream.gpu.synchronize() }
        XCTAssertTrue(allClose(result, reference, rtol: 1e-4, atol: 2e-5).item(Bool.self))
    }

    /// Run this suite without concurrent GPU workloads: peakMemory is global
    /// to the process. A composed score tensor alone is 32 MiB in this cell.
    func testFusedDoesNotMaterializeAttentionScores() throws {
        let q = values([1, 16, 128, 256], salt: 7)
        let k = values([1, 2, 4096, 256], salt: 11)
        let v = values([1, 2, 4096, 256], salt: 13)
        let mask = MLXArray.ones([128, 4096], dtype: .bool)
        try withError { eval(q, k, v, mask); Stream.gpu.synchronize() }
        Memory.clearCache()
        let before = Memory.activeMemory
        Memory.peakMemory = 0
        let result = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 0.0625,
            mask: .array(mask), forceFused: true)
        try withError { eval(result); Stream.gpu.synchronize() }
        let extra = Memory.peakMemory - before
        XCTAssertTrue(all(isFinite(result)).item(Bool.self))
        XCTAssertLessThan(extra, 16 << 20,
            "forced full SDPA must not materialize the 32 MiB score tensor")
    }
}
