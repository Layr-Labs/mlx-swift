// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

/// Tests for `Source/MLX/ParameterTypes.swift`: each initializer and literal
/// form of `IntOrPair`, `IntOrTriple`, `IntOrArray` and `FloatOrArray`.
///
/// Note: `T(5)` with a literal is a literal coercion in Swift and calls the
/// literal initializer. The tests pass variables to reach the
/// `init(_ value:)` initializers.
class ParameterTypesCoverageTests: XCTestCase {

    func testIntOrPair() {
        let fromInteger: IntOrPair = 3
        XCTAssertEqual(fromInteger.first, 3)
        XCTAssertEqual(fromInteger.second, 3)

        let fromArray: IntOrPair = [4, 5]
        XCTAssertEqual(fromArray.first, 4)
        XCTAssertEqual(fromArray.second, 5)

        let values = [6, 7]
        let fromCollection = IntOrPair(values)
        XCTAssertEqual(fromCollection.first, 6)
        XCTAssertEqual(fromCollection.second, 7)

        // a slice does not start at index 0
        let slice = [0, 8, 9][1...]
        let fromSlice = IntOrPair(slice)
        XCTAssertEqual(fromSlice.first, 8)
        XCTAssertEqual(fromSlice.second, 9)

        let fromTuple = IntOrPair((10, 11))
        XCTAssertEqual(fromTuple.values.0, 10)
        XCTAssertEqual(fromTuple.values.1, 11)

        let n = 12
        let fromInt = IntOrPair(n)
        XCTAssertEqual(fromInt.first, 12)
        XCTAssertEqual(fromInt.second, 12)
    }

    func testIntOrTriple() {
        let fromInteger: IntOrTriple = 2
        XCTAssertEqual(fromInteger.first, 2)
        XCTAssertEqual(fromInteger.second, 2)
        XCTAssertEqual(fromInteger.third, 2)

        let fromArray: IntOrTriple = [1, 2, 3]
        XCTAssertEqual(fromArray.first, 1)
        XCTAssertEqual(fromArray.second, 2)
        XCTAssertEqual(fromArray.third, 3)

        let values = [4, 5, 6]
        let fromCollection = IntOrTriple(values)
        XCTAssertEqual(fromCollection.first, 4)
        XCTAssertEqual(fromCollection.second, 5)
        XCTAssertEqual(fromCollection.third, 6)

        let slice = [0, 0, 7, 8, 9][2...]
        let fromSlice = IntOrTriple(slice)
        XCTAssertEqual(fromSlice.first, 7)
        XCTAssertEqual(fromSlice.second, 8)
        XCTAssertEqual(fromSlice.third, 9)

        let fromTuple = IntOrTriple((10, 11, 12))
        XCTAssertEqual(fromTuple.values.0, 10)
        XCTAssertEqual(fromTuple.values.1, 11)
        XCTAssertEqual(fromTuple.values.2, 12)

        let n = 13
        let fromInt = IntOrTriple(n)
        XCTAssertEqual(fromInt.first, 13)
        XCTAssertEqual(fromInt.second, 13)
        XCTAssertEqual(fromInt.third, 13)
    }

    func testIntOrArray() {
        let fromInteger: IntOrArray = 4
        XCTAssertEqual(fromInteger.asArray, [4])
        XCTAssertEqual(fromInteger.asInt32Array, [4])
        XCTAssertEqual(fromInteger.count, 1)

        let fromArray: IntOrArray = [1, -2, 3]
        XCTAssertEqual(fromArray.asArray, [1, -2, 3])
        XCTAssertEqual(fromArray.asInt32Array, [1, -2, 3])
        XCTAssertEqual(fromArray.count, 3)

        let values = [5, 6]
        let fromValues = IntOrArray(values)
        XCTAssertEqual(fromValues.asArray, [5, 6])
        XCTAssertEqual(fromValues.asInt32Array, [5, 6])
        XCTAssertEqual(fromValues.count, 2)

        let empty = IntOrArray([Int]())
        XCTAssertEqual(empty.asArray, [])
        XCTAssertEqual(empty.count, 0)

        let n = -7
        let fromInt = IntOrArray(n)
        XCTAssertEqual(fromInt.asArray, [-7])
        XCTAssertEqual(fromInt.asInt32Array, [-7])
        XCTAssertEqual(fromInt.count, 1)

        switch fromInt {
        case .int(let v): XCTAssertEqual(v, -7)
        case .array: XCTFail("expected .int")
        }
        switch fromValues {
        case .int: XCTFail("expected .array")
        case .array(let v): XCTAssertEqual(v, [5, 6])
        }
    }

    func testFloatOrArray() {
        let fromFloat: FloatOrArray = 1.5
        XCTAssertEqual(fromFloat.asArray, [1.5])
        XCTAssertEqual(fromFloat.count, 1)
        XCTAssertEqual(fromFloat.asArray(dimensions: 3), [1.5, 1.5, 1.5])

        let fromArray: FloatOrArray = [0.5, 2.5]
        XCTAssertEqual(fromArray.asArray, [0.5, 2.5])
        XCTAssertEqual(fromArray.count, 2)
        XCTAssertEqual(fromArray.asArray(dimensions: 2), [0.5, 2.5])

        let values: [Float] = [3, 4, 5]
        let fromValues = FloatOrArray(values)
        XCTAssertEqual(fromValues.asArray, [3, 4, 5])
        XCTAssertEqual(fromValues.count, 3)

        let x: Float = 0.25
        let fromScalar = FloatOrArray(x)
        XCTAssertEqual(fromScalar.asArray, [0.25])
        XCTAssertEqual(fromScalar.count, 1)
        XCTAssertEqual(fromScalar.asArray(dimensions: 2), [0.25, 0.25])

        switch fromScalar {
        case .float(let v): XCTAssertEqual(v, 0.25)
        case .array: XCTFail("expected .float")
        }
        switch fromValues {
        case .float: XCTFail("expected .array")
        case .array(let v): XCTAssertEqual(v, [3, 4, 5])
        }
    }
}
