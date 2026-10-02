// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import XCTest

/// The deprecated SoftMax layer must normalize over the last axis, as Softmax does.
class SoftMaxAxisTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    @available(*, deprecated)
    func testSoftMaxUsesLastAxis() {
        let rows: [[Float]] = [[1, 2, 3], [-1, 0, 4]]
        let x = MLXArray(rows.flatMap { $0 }, [2, 3])

        // softmax of each row, computed in Swift
        let expected = rows.flatMap { row -> [Float] in
            let e = row.map { exp($0) }
            let s = e.reduce(0, +)
            return e.map { $0 / s }
        }

        let y = SoftMax()(x)
        XCTAssertEqual(y.shape, [2, 3])
        XCTAssertEqual(y.dtype, .float32)
        for (v, e) in zip(y.asArray(Float.self), expected) {
            XCTAssertEqual(v, e, accuracy: 1e-5 + 1e-4 * e)
        }
        assertEqual(y, Softmax()(x))
    }
}
