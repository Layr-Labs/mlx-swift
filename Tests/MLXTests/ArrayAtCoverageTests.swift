// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

/// Tests for `Source/MLX/ArrayAt.swift`: each `.at[...]` update operation
/// compared with a plain Swift loop. The indices repeat, so each repeated
/// location must get every update, not only the last one.
class ArrayAtCoverageTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    // index 0 is used three times, index 2 and 3 once, index 1 never
    private let indices: [Int32] = [0, 2, 0, 3, 0]

    /// Apply `op` at `indices` with `updates`, one element at a time.
    private func loop(
        _ base: [Float], _ updates: [Float], _ op: (Float, Float) -> Float
    ) -> [Float] {
        var result = base
        for (i, index) in indices.enumerated() {
            result[Int(index)] = op(result[Int(index)], updates[i])
        }
        return result
    }

    private func check(
        _ result: MLXArray, _ expected: [Float], file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(result.shape, [expected.count], file: file, line: line)
        XCTAssertEqual(result.dtype, .float32, file: file, line: line)
        XCTAssertEqual(result.asArray(Float.self), expected, file: file, line: line)
    }

    func testAdd() {
        let base: [Float] = [10, 20, 30, 40]
        let updates: [Float] = [1, 2, 3, 4, 5]
        let a = MLXArray(base)
        let result = a.at[MLXArray(indices)].add(MLXArray(updates))
        check(result, loop(base, updates, +))

        // a scalar is added once for each use of an index
        check(a.at[MLXArray(indices)].add(Float(1)), loop(base, [1, 1, 1, 1, 1], +))

        // the source array is not changed
        XCTAssertEqual(a.asArray(Float.self), base)
    }

    func testSubtract() {
        let base: [Float] = [10, 20, 30, 40]
        let updates: [Float] = [1, 2, 3, 4, 5]
        let a = MLXArray(base)
        check(a.at[MLXArray(indices)].subtract(MLXArray(updates)), loop(base, updates, -))
        check(a.at[MLXArray(indices)].subtract(Float(2)), loop(base, [2, 2, 2, 2, 2], -))
    }

    func testMultiply() {
        let base: [Float] = [1, 2, 3, 4]
        let updates: [Float] = [2, 3, 4, 5, 6]
        let a = MLXArray(base)
        check(a.at[MLXArray(indices)].multiply(MLXArray(updates)), loop(base, updates, *))
        check(a.at[MLXArray(indices)].multiply(Float(2)), loop(base, [2, 2, 2, 2, 2], *))
    }

    func testDivide() {
        // powers of two keep the float math exact
        let base: [Float] = [8, 16, 32, 64]
        let updates: [Float] = [2, 4, 2, 8, 2]
        let a = MLXArray(base)
        check(a.at[MLXArray(indices)].divide(MLXArray(updates)), loop(base, updates, /))
        check(a.at[MLXArray(indices)].divide(Float(2)), loop(base, [2, 2, 2, 2, 2], /))
    }

    func testMinimum() {
        let base: [Float] = [10, 20, 30, 40]
        let updates: [Float] = [5, 25, 12, 1, 7]
        let a = MLXArray(base)
        check(
            a.at[MLXArray(indices)].minimum(MLXArray(updates)),
            loop(base, updates) { Swift.min($0, $1) })
        check(
            a.at[MLXArray(indices)].minimum(Float(15)),
            loop(base, [15, 15, 15, 15, 15]) { Swift.min($0, $1) })
    }

    func testMaximum() {
        let base: [Float] = [10, 20, 30, 40]
        let updates: [Float] = [5, 35, 12, 50, 7]
        let a = MLXArray(base)
        check(
            a.at[MLXArray(indices)].maximum(MLXArray(updates)),
            loop(base, updates) { Swift.max($0, $1) })
        check(
            a.at[MLXArray(indices)].maximum(Float(25)),
            loop(base, [25, 25, 25, 25, 25]) { Swift.max($0, $1) })
    }

    func testIntegerArray() {
        // an Int scalar takes the dtype of the array
        let a = MLXArray([Int32(1), Int32(2), Int32(3)])
        let idx = MLXArray([Int32(2), Int32(2), Int32(0)])
        let result = a.at[idx].add(10)
        XCTAssertEqual(result.dtype, .int32)
        XCTAssertEqual(result.asArray(Int32.self), [11, 2, 23])

        let product = a.at[idx].multiply(2)
        XCTAssertEqual(product.dtype, .int32)
        XCTAssertEqual(product.asArray(Int32.self), [2, 2, 12])
    }

    func testIndexAndSlice() {
        // a = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        let a = MLXArray(Int32(0) ..< Int32(9), [3, 3])

        // a single integer index updates one row
        let row = a.at[1].add(10)
        XCTAssertEqual(row.asArray(Int32.self), [0, 1, 2, 13, 14, 15, 6, 7, 8])

        // a slice updates a range of rows
        let rows = a.at[1 ..< 3].multiply(2)
        XCTAssertEqual(rows.asArray(Int32.self), [0, 1, 2, 6, 8, 10, 12, 14, 16])

        // a strided slice is turned into an index array
        let strided = a.at[.stride(from: 0, to: 3, by: 2)].subtract(1)
        XCTAssertEqual(strided.asArray(Int32.self), [-1, 0, 1, 3, 4, 5, 5, 6, 7])

        // two indices: rows [0, 0, 2], column 1
        let rowIndex = MLXArray([Int32(0), Int32(0), Int32(2)])
        let cell = a.at[rowIndex, 1].add(100)
        XCTAssertEqual(cell.asArray(Int32.self), [0, 201, 2, 3, 4, 5, 6, 107, 8])

        let cellMax = a.at[rowIndex, 1].maximum(5)
        XCTAssertEqual(cellMax.asArray(Int32.self), [0, 5, 2, 3, 4, 5, 6, 7, 8])
    }

    func testFullSliceUsesBroadcast() {
        // A full slice has no scatter indices. The update then broadcasts
        // over the whole array.
        let base: [Float] = [1, 5, 3, 7]
        let a = MLXArray(base)
        let b = MLXArray([Float(4), Float(4), Float(4), Float(4)])

        check(a.at[0...].add(b), base.map { $0 + 4 })
        check(a.at[0...].subtract(Float(1)), base.map { $0 - 1 })
        check(a.at[0...].multiply(b), base.map { $0 * 4 })
        check(a.at[0...].divide(Float(2)), base.map { $0 / 2 })
        check(a.at[0...].minimum(b), base.map { Swift.min($0, 4) })
        check(a.at[0...].maximum(b), base.map { Swift.max($0, 4) })
    }

    func testSequenceSubscriptAndStream() {
        let base: [Float] = [10, 20, 30, 40]
        let updates: [Float] = [1, 2, 3, 4, 5]
        let a = MLXArray(base)

        // the index list as one array
        let ops: [any MLXArrayIndex] = [MLXArray(indices)]
        check(a.at[ops].add(MLXArray(updates)), loop(base, updates, +))

        // explicit stream
        check(
            a.at[MLXArray(indices), stream: .cpu].add(MLXArray(updates)),
            loop(base, updates, +))
        check(
            a.at[ops, stream: .cpu].maximum(MLXArray(updates)),
            loop(base, updates) { Swift.max($0, $1) })
    }
}
