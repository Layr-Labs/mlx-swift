// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

/// Tests for `median(_:keepDims:)` in `Ops.swift`.
class MedianAllAxesTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    /// With no axes, the median is over the full array, as Python `mx.median(a)` does.
    func testMedianAllAxes() {
        let a = MLXArray([3, 1, 2, 9, 7, 8, 4, 6, 5] as [Float], [3, 3])
        let m = median(a)
        XCTAssertEqual(m.shape, [])
        XCTAssertEqual(m.dtype, .float32)
        XCTAssertEqual(m.item(Float.self), 5)

        let k = median(a, keepDims: true)
        XCTAssertEqual(k.shape, [1, 1])
        XCTAssertEqual(k.item(Float.self), 5)

        // an even count gives the mean of the two middle values
        let e = MLXArray([4, 1, 3, 2] as [Float], [2, 2])
        XCTAssertEqual(median(e).item(Float.self), 2.5)
    }
}
