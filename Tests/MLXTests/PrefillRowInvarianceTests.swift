// Copyright © 2026 Eigen Labs.

import Cmlx
import Foundation
import XCTest

@testable import MLX

private struct ScopeError: Error {}

/// Tests for `GPU.withPrefillRowInvariance(rows:_:)`: the scope sets the calling thread's
/// reference rows only, restores the previous value on return and on throw, ops keep the value
/// of the thread that built them, and the rows of a short quantized product built in the scope
/// have the bits of the same rows in a product of the reference rows.
class PrefillRowInvarianceTests: XCTestCase {

    private static let referenceRows = 4096

    func testNestedScopesRestoreThePreviousRows() {
        XCTAssertEqual(GPU.prefillRowInvarianceRows, 0)
        GPU.withPrefillRowInvariance(rows: 4096) {
            XCTAssertEqual(GPU.prefillRowInvarianceRows, 4096)
            GPU.withPrefillRowInvariance(rows: 512) {
                XCTAssertEqual(GPU.prefillRowInvarianceRows, 512)
            }
            GPU.withPrefillRowInvariance(rows: 0) {
                XCTAssertEqual(GPU.prefillRowInvarianceRows, 0)
            }
            XCTAssertEqual(GPU.prefillRowInvarianceRows, 4096)
        }
        XCTAssertEqual(GPU.prefillRowInvarianceRows, 0)
    }

    func testThrowingBodyRestoresTheRows() {
        XCTAssertThrowsError(
            try GPU.withPrefillRowInvariance(rows: 2048) {
                XCTAssertEqual(GPU.prefillRowInvarianceRows, 2048)
                throw ScopeError()
            })
        XCTAssertEqual(GPU.prefillRowInvarianceRows, 0)
    }

    func testNegativeRowsEndTheScope() {
        GPU.withPrefillRowInvariance(rows: 4096) {
            GPU.withPrefillRowInvariance(rows: -3) {
                XCTAssertEqual(GPU.prefillRowInvarianceRows, 0)
            }
        }
    }

    func testRowsApplyOnlyToTheCallingThread() {
        final class Observation: @unchecked Sendable {
            var rows: Int?
        }
        GPU.withPrefillRowInvariance(rows: 4096) {
            let observation = Observation()
            let done = DispatchSemaphore(value: 0)
            Thread {
                observation.rows = GPU.prefillRowInvarianceRows
                done.signal()
            }.start()
            done.wait()
            XCTAssertEqual(observation.rows, 0)
            XCTAssertEqual(GPU.prefillRowInvarianceRows, 4096)
        }
    }

    /// The Qwen 3.5 9B key/value projection: N = 1024, K = 4096, 4-bit affine weights in groups
    /// of 64, bfloat16 activations.
    private struct Product {
        let x: MLXArray
        let wq: MLXArray
        let scales: MLXArray
        let biases: MLXArray?

        init(rows: Int, n: Int = 1024, k: Int = 4096) {
            x = MLXRandom.normal([rows, k], key: MLXRandom.key(UInt64(k + n))).asType(.bfloat16)
            let w =
                (MLXRandom.normal([n, k], key: MLXRandom.key(UInt64(3 * k + n)))
                / Float(k).squareRoot()).asType(.bfloat16)
            (wq, scales, biases) = quantized(w, groupSize: 64, bits: 4)
            eval(x, wq, scales)
            if let biases { eval(biases) }
        }

        func callAsFunction(_ rows: MLXArray) -> MLXArray {
            quantizedMM(
                rows, wq, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 4)
        }

        func rows(_ count: Int) -> MLXArray {
            let r = contiguous(x[0 ..< count])
            eval(r)
            return r
        }
    }

    private func sameBits(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && a.dtype == b.dtype
            && arrayEqual(a.view(dtype: .uint16), b.view(dtype: .uint16)).item(Bool.self)
    }

    func testShortProductsInTheScopeMatchTheReferenceRows() {
        let product = Product(rows: Self.referenceRows)
        GPU.withPrefillRowInvariance(rows: Self.referenceRows) {
            let reference = product(product.x)
            eval(reference)
            for m in [2, 3, 16, 33, 100, 128, 129, 256, 257, 512, 1000, 2048] {
                let inside = product(product.rows(m))
                eval(inside)
                XCTAssertTrue(sameBits(inside, reference[0 ..< m]), "rows \(m)")
            }
        }
    }

    func testOpsKeepTheRowsOfTheThreadThatBuiltThem() {
        let product = Product(rows: 256)
        let before = product(product.x)
        eval(before)

        // Built outside the scope, evaluated inside it.
        let outsideBuilt = product(product.x)
        GPU.withPrefillRowInvariance(rows: Self.referenceRows) { eval(outsideBuilt) }
        XCTAssertTrue(sameBits(outsideBuilt, before))

        // Built inside the scope, evaluated after it ends.
        let (insideBuilt, insideEvaluated) = GPU.withPrefillRowInvariance(
            rows: Self.referenceRows
        ) { () -> (MLXArray, MLXArray) in
            let built = product(product.x)
            let evaluated = product(product.x)
            eval(evaluated)
            return (built, evaluated)
        }
        eval(insideBuilt)
        XCTAssertTrue(sameBits(insideBuilt, insideEvaluated))

        let after = product(product.x)
        eval(after)
        XCTAssertTrue(sameBits(after, before))
    }
}
