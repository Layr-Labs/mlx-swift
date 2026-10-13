// Copyright © 2026 Eigen Labs.

import MLXLinalg
import XCTest

// Import only the MLX types. Then `triInv` is the deprecated function in the
// MLXLinalg module, not the function in MLX.
import class MLX.MLXArray
import struct MLX.StreamOrDevice

/// The deprecated `MLXLinalg` module function `triInv(_:upper:stream:)` must
/// pass `upper` on.
class LinalgDeprecatedTriInvTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    @available(*, deprecated)
    func testDeprecatedTriInvUpper() {
        // upper [[2, 1], [0, 1]] -> [[0.5, -0.5], [0, 1]]
        let upper = MLXArray([Float(2), 1, 0, 1], [2, 2])
        let upperInv = MLXArray([Float(0.5), -0.5, 0, 1], [2, 2])

        assertEqual(triInv(upper, upper: true, stream: .cpu), upperInv, atol: 1e-6)
    }

    @available(*, deprecated)
    func testDeprecatedTriInvLower() {
        // lower [[2, 0], [1, 1]] -> [[0.5, 0], [-0.5, 1]]
        let lower = MLXArray([Float(2), 0, 1, 1], [2, 2])
        let lowerInv = MLXArray([Float(0.5), 0, -0.5, 1], [2, 2])

        assertEqual(triInv(lower, stream: .cpu), lowerInv, atol: 1e-6)
        assertEqual(triInv(lower, upper: false, stream: .cpu), lowerInv, atol: 1e-6)
    }
}
