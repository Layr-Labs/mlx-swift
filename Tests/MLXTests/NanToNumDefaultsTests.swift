// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

/// Tests for the default arguments of `nanToNum` in `Ops.swift`.
class NanToNumDefaultsTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    /// Without posInf and negInf, infinities become the largest finite values of the
    /// dtype, as the doc comment and Python `mx.nan_to_num` say.
    func testNanToNumDefaults() {
        let special = MLXArray([Float.nan, Float.infinity, -Float.infinity, 1], [4])
        let r = nanToNum(special)
        XCTAssertEqual(r.dtype, .float32)
        XCTAssertEqual(
            r.asArray(Float.self),
            [0, Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, 1])

        // an explicit value still replaces the infinities
        let z = nanToNum(special, nan: 2, posInf: 0, negInf: 0)
        XCTAssertEqual(z.asArray(Float.self), [2, 0, 0, 1])
    }
}
