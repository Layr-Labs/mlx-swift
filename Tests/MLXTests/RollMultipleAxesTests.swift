// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

/// Tests for `roll(_:shift:axes:)` in `Ops.swift`.
class RollMultipleAxesTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    /// The shift applies to each axis, as Python `mx.roll(a, 1, axis=(0, 1))` does.
    func testRollMultipleAxes() {
        let a = MLXArray(0 ..< 6, [2, 3])
        let r = roll(a, shift: 1, axes: [0, 1])
        XCTAssertEqual(r.shape, [2, 3])
        // roll along axis 0: [[3, 4, 5], [0, 1, 2]]; then along axis 1
        XCTAssertEqual(r.asArray(Int32.self), [5, 3, 4, 2, 0, 1])

        let one = roll(a, shift: -1, axes: [1])
        XCTAssertEqual(one.asArray(Int32.self), [1, 2, 0, 4, 5, 3])
    }
}
