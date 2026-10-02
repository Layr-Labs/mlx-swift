// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

/// The free function `triInv(_:upper:stream:)` must pass `upper` on to
/// `MLXLinalg.triInv`.
class LinalgTriInvTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    func testFreeTriInvUpper() {
        // upper [[2, 1], [0, 1]] -> [[0.5, -0.5], [0, 1]]
        let upper = MLXArray([Float(2), 1, 0, 1], [2, 2])
        let upperInv = MLXArray([Float(0.5), -0.5, 0, 1], [2, 2])

        assertEqual(triInv(upper, upper: true, stream: .cpu), upperInv, atol: 1e-6)
        assertEqual(
            triInv(upper, upper: true, stream: .cpu),
            MLXLinalg.triInv(upper, upper: true, stream: .cpu), atol: 1e-6)
    }

    func testFreeTriInvLower() {
        // lower [[2, 0], [1, 1]] -> [[0.5, 0], [-0.5, 1]]
        let lower = MLXArray([Float(2), 0, 1, 1], [2, 2])
        let lowerInv = MLXArray([Float(0.5), 0, -0.5, 1], [2, 2])

        assertEqual(triInv(lower, stream: .cpu), lowerInv, atol: 1e-6)
        assertEqual(triInv(lower, upper: false, stream: .cpu), lowerInv, atol: 1e-6)
    }
}
