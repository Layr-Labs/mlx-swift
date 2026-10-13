// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

/// Tests for `convolve(_:_:mode:)` with `.same` in `Ops.swift`.
class ConvolveSameModeTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    /// Full 1D convolution, as numpy.convolve(a, b, mode: "full") computes it.
    private func fullConvolve(_ a: [Float], _ b: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: a.count + b.count - 1)
        for i in 0 ..< a.count {
            for j in 0 ..< b.count {
                out[i + j] += a[i] * b[j]
            }
        }
        return out
    }

    /// `.same` gives max(len(a), len(b)) values, from index (len(b) - 1) / 2 of the
    /// full convolution, as numpy and Python MLX do.
    func testConvolveSameEvenKernel() {
        let a: [Float] = [1, 2, 3, 4, 5]
        for k in [[1, 1], [1, 1, 1, 1], [1, 2, 3, 4, 5, 6]] as [[Float]] {
            let full = fullConvolve(a, k)
            let start = (min(a.count, k.count) - 1) / 2
            let expected = Array(full[start ..< start + max(a.count, k.count)])

            let r = convolve(MLXArray(a), MLXArray(k), mode: .same)
            XCTAssertEqual(r.shape, [max(a.count, k.count)], "kernel size \(k.count)")
            if r.shape == [expected.count] {
                XCTAssertEqual(r.asArray(Float.self), expected, "kernel size \(k.count)")
            }
        }
    }
}
