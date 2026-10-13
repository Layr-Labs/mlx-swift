// Copyright © 2026 Eigen Labs.

import Foundation
import XCTest

@testable import MLX

/// Tests for subscript forms in `MLXArray+Indexing.swift`.
///
/// Every test uses a [2, 3, 4] int32 array that holds 0 ..< 24 and compares the
/// result with values computed by plain Swift loops.
class IndexingFormsTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    /// The source array: value at [i, j, k] is i * 12 + j * 4 + k.
    private func source() -> MLXArray {
        MLXArray(Int32(0) ..< 24, [2, 3, 4])
    }

    /// Values of the source array at the given coordinates, in row-major order.
    private func values(_ iList: [Int], _ jList: [Int], _ kList: [Int]) -> [Int32] {
        var result = [Int32]()
        for i in iList {
            for j in jList {
                for k in kList {
                    result.append(Int32(i * 12 + j * 4 + k))
                }
            }
        }
        return result
    }

    /// Contents of the source array after the given coordinates are set to `value`.
    private func valuesAfterSet(
        _ iList: [Int], _ jList: [Int], _ kList: [Int], _ value: Int32
    ) -> [Int32] {
        var result = (0 ..< 24).map { Int32($0) }
        for i in iList {
            for j in jList {
                for k in kList {
                    result[i * 12 + j * 4 + k] = value
                }
            }
        }
        return result
    }

    private let allI = [0, 1]
    private let allJ = [0, 1, 2]
    private let allK = [0, 1, 2, 3]

    private func check(
        _ result: MLXArray, _ shape: [Int], _ expected: [Int32],
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(result.dtype, .int32, file: file, line: line)
        XCTAssertEqual(result.shape, shape, file: file, line: line)
        XCTAssertEqual(result.asArray(Int32.self), expected, file: file, line: line)
    }

    // MARK: - deprecated range + axis subscript

    @available(*, deprecated)
    func testRangeAxisGetEveryRangeType() {
        let a = source()

        // Range, negative axis
        check(a[1 ..< 3, axis: -1], [2, 3, 2], values(allI, allJ, [1, 2]))

        // Range with negative bounds
        check(a[-3 ..< -1, axis: 2], [2, 3, 2], values(allI, allJ, [1, 2]))

        // ClosedRange
        check(a[0 ... 1, axis: 1], [2, 2, 4], values(allI, [0, 1], allK))

        // PartialRangeUpTo
        check(a[..<1, axis: 0], [1, 3, 4], values([0], allJ, allK))

        // PartialRangeThrough
        check(a[...2, axis: 2], [2, 3, 3], values(allI, allJ, [0, 1, 2]))

        // PartialRangeFrom
        check(a[1..., axis: 1], [2, 2, 4], values(allI, [1, 2], allK))
    }

    @available(*, deprecated)
    func testRangeAxisSet() {
        do {
            let a = source()
            a[1 ..< 3, axis: 2] = MLXArray(Int32(-1))
            check(a, [2, 3, 4], valuesAfterSet(allI, allJ, [1, 2], -1))
        }
        do {
            let a = source()
            a[...0, axis: 0] = MLXArray(Int32(-1))
            check(a, [2, 3, 4], valuesAfterSet([0], allJ, allK, -1))
        }
        do {
            let a = source()
            a[1..., axis: -2] = MLXArray(Int32(-1))
            check(a, [2, 3, 4], valuesAfterSet(allI, [1, 2], allK, -1))
        }
    }

    // MARK: - deprecated index + axis subscript

    @available(*, deprecated)
    func testIndexAxisGetAndSet() {
        let a = source()
        check(a[1, axis: 1], [2, 4], values(allI, [1], allK))
        check(a[-1, axis: 2], [2, 3], values(allI, allJ, [3]))
        check(a[-2, axis: 0], [3, 4], values([0], allJ, allK))

        do {
            let b = source()
            b[1, axis: 1] = MLXArray(Int32(-5))
            check(b, [2, 3, 4], valuesAfterSet(allI, [1], allK, -5))
        }
        do {
            let b = source()
            b[-1, axis: 2] = MLXArray(Int32(-5))
            check(b, [2, 3, 4], valuesAfterSet(allI, allJ, [3], -5))
        }
        do {
            let b = source()
            b[0, axis: 0] = MLXArray(Int32(-5))
            check(b, [2, 3, 4], valuesAfterSet([0], allJ, allK, -5))
        }
    }

    // MARK: - deprecated stride subscript

    @available(*, deprecated)
    func testStrideAxisGetWithBounds() {
        let a = source()
        check(a[from: 1, to: 3, stride: 1, axis: -1], [2, 3, 2], values(allI, allJ, [1, 2]))
        check(a[to: 2, stride: 1, axis: 1], [2, 2, 4], values(allI, [0, 1], allK))
        check(a[from: -3, to: -1, stride: 1, axis: 2], [2, 3, 2], values(allI, allJ, [1, 2]))
    }

    @available(*, deprecated)
    func testStrideAxisSetPositiveStride() {
        do {
            let a = source()
            a[stride: 2, axis: -1] = MLXArray(Int32(-1))
            check(a, [2, 3, 4], valuesAfterSet(allI, allJ, [0, 2], -1))
        }
        do {
            // negative start and end are taken from the end of the axis
            let a = source()
            a[from: -3, to: -1, stride: 1, axis: 1] = MLXArray(Int32(-1))
            check(a, [2, 3, 4], valuesAfterSet(allI, [0, 1], allK, -1))
        }
        do {
            let a = source()
            a[from: 1, stride: 2, axis: 0] = MLXArray(Int32(-1))
            check(a, [2, 3, 4], valuesAfterSet([1], allJ, allK, -1))
        }
    }

    // MARK: - [any MLXArrayIndex] subscript

    func testIndexArraySubscriptGetAndSet() {
        let a = source()
        let indices: [any MLXArrayIndex] = [1, .ellipsis, 2]
        check(a[indices], [3], values([1], allJ, [2]))

        a[indices] = MLXArray(Int32(-7))
        check(a, [2, 3, 4], valuesAfterSet([1], allJ, [2], -7))

        let b = source()
        let ranges: [any MLXArrayIndex] = [0 ..< 1, 1 ... 2]
        check(b[ranges], [1, 2, 4], values([0], [1, 2], allK))
        b[ranges] = MLXArray(Int32(-8))
        check(b, [2, 3, 4], valuesAfterSet([0], [1, 2], allK, -8))
    }

    // MARK: - general subscript, read

    func testNewAxisAndEllipsisReads() {
        let a = source()
        check(a[.newAxis, 1], [1, 3, 4], values([1], allJ, allK))
        check(a[0, .newAxis, .ellipsis], [1, 3, 4], values([0], allJ, allK))
        check(a[.ellipsis, .newAxis], [2, 3, 4, 1], values(allI, allJ, allK))
        check(a[.ellipsis, 1, .newAxis], [2, 3, 1], values(allI, allJ, [1]))
    }

    func testStridedAndNegativeReads() {
        let a = source()

        // single strided slice (reversed first axis)
        check(a[.stride(by: -1)], [2, 3, 4], values([1, 0], allJ, allK))

        // reversed middle axis with an index
        check(a[1, .stride(by: -1)], [3, 4], values([1], [2, 1, 0], allK))

        // strided slice with a start, plus an index on the last axis
        check(a[0..., .stride(from: 2, by: -1), 1], [2, 3], values(allI, [2, 1, 0], [1]))

        // stride with from, to and by
        check(a[.ellipsis, .stride(from: 0, to: 4, by: 3)], [2, 3, 2], values(allI, allJ, [0, 3]))

        // negative index with a range
        check(a[-1, 1...], [2, 4], values([1], [1, 2], allK))
        check(a[-2, -1, -1], [], values([0], [2], [3]))
    }

    func testArrayIndexReads() {
        let a = source()

        // array index then a negative integer index
        let rows = MLXArray([1, 0] as [Int32])
        check(a[rows, -1], [2, 4], values([1], [2], allK) + values([0], [2], allK))

        // integer index then an array index
        let cols = MLXArray([2, 0] as [Int32])
        check(a[0, cols], [2, 4], values([0], [2], allK) + values([0], [0], allK))
    }

    // MARK: - general subscript, write

    func testNegativeIndexWrites() {
        do {
            let a = source()
            a[-1] = MLXArray(Int32(-2))
            check(a, [2, 3, 4], valuesAfterSet([1], allJ, allK, -2))
        }
        do {
            let a = source()
            a[0, -1] = MLXArray(Int32(-2))
            check(a, [2, 3, 4], valuesAfterSet([0], [2], allK, -2))
        }
        do {
            // a row of values broadcast over the last axis
            let a = source()
            a[1, -2] = MLXArray([90, 91, 92, 93] as [Int32])
            var expected = (0 ..< 24).map { Int32($0) }
            for k in 0 ..< 4 {
                expected[12 + 4 + k] = Int32(90 + k)
            }
            check(a, [2, 3, 4], expected)
        }
    }

    func testSliceWritesWithNewAxis() {
        let a = source()
        a[0, .newAxis, 1 ..< 3] = MLXArray(Int32(-4))
        check(a, [2, 3, 4], valuesAfterSet([0], [1, 2], allK, -4))
    }

    func testStridedSliceWritesWithArrayIndex() {
        do {
            // array index first, then a strided slice
            let a = source()
            a[MLXArray([0, 1] as [Int32]), .stride(by: 2)] = MLXArray(Int32(-3))
            check(a, [2, 3, 4], valuesAfterSet(allI, [0, 2], allK, -3))
        }
        do {
            // strided slice first, then an array index
            let a = source()
            a[.stride(by: 2), MLXArray([0, 2] as [Int32])] = MLXArray(Int32(-3))
            check(a, [2, 3, 4], valuesAfterSet([0], [0, 2], allK, -3))
        }
    }

    // MARK: - descriptions and MLXSlice

    func testIndexOperationDescription() {
        XCTAssertEqual(MLXArrayIndexOperation.ellipsis.description, ".ellipsis")
        XCTAssertEqual(MLXArrayIndexOperation.newAxis.description, ".newAxis")
        XCTAssertEqual(MLXArrayIndexOperation.index(3).description, "3")
        XCTAssertEqual(
            MLXArrayIndexOperation.slice(MLXSlice(start: 1, end: 3)).description, "1 ..< 3")
        XCTAssertEqual(
            MLXArrayIndexOperation.slice(MLXSlice(start: 1, end: 7, stride: 2)).description,
            "1 ..< 7 : 2")
        XCTAssertEqual(
            MLXArrayIndexOperation.array(MLXArray([1, 2] as [Int32])).description, "[2](int32)")

        // the operation for each index type
        XCTAssertEqual((1 ... 2).mlxArrayIndexOperation.description, "1 ..< 3")
        XCTAssertEqual((..<2).mlxArrayIndexOperation.description, "0 ..< 2")
        XCTAssertEqual((...2).mlxArrayIndexOperation.description, "0 ..< 3")
        XCTAssertEqual((1...).mlxArrayIndexOperation.description, "1 ..< ")
        XCTAssertEqual((1 ..< 4).mlxArrayIndexOperation.description, "1 ..< 4")
        XCTAssertEqual(5.mlxArrayIndexOperation.description, "5")
        XCTAssertEqual(MLXEllipsisIndex().mlxArrayIndexOperation.description, ".ellipsis")
        XCTAssertEqual(MLXNewAxisIndex().mlxArrayIndexOperation.description, ".newAxis")
    }

    func testIndexOperationKinds() {
        let operations: [MLXArrayIndexOperation] = [
            .ellipsis, .newAxis, .index(1), .slice(MLXSlice()), .array(MLXArray([0] as [Int32])),
        ]
        XCTAssertEqual(operations.map { $0.isEllipsis }, [true, false, false, false, false])
        XCTAssertEqual(operations.map { $0.isNewAxis }, [false, true, false, false, false])
        XCTAssertEqual(operations.map { $0.isIndex }, [false, false, true, false, false])
        XCTAssertEqual(operations.map { $0.isSlice }, [false, false, false, true, false])
        XCTAssertEqual(operations.map { $0.isArray }, [false, false, false, false, true])
        XCTAssertEqual(operations.map { $0.isArrayOrIndex }, [false, false, true, false, true])
        XCTAssertEqual(countNonNewAxisOperations(operations), 4)
    }

    func testSliceBounds() {
        XCTAssertTrue(MLXSlice().isFull)
        XCTAssertTrue(MLXSlice(start: 0, stride: 1).isFull)
        XCTAssertFalse(MLXSlice(end: 3).isFull)
        XCTAssertFalse(MLXSlice(start: 1).isFull)
        XCTAssertFalse(MLXSlice(stride: 2).isFull)

        // positive stride defaults: [0, size)
        let forward = MLXSlice()
        XCTAssertEqual(forward.stride, 1)
        XCTAssertEqual(forward.start(5), 0)
        XCTAssertEqual(forward.end(5), 5)

        // negative stride defaults: numpy style [size - 1, -size - 1)
        let backward = MLXSlice(stride: -1)
        XCTAssertEqual(backward.stride, -1)
        XCTAssertEqual(backward.start(5), 4)
        XCTAssertEqual(backward.end(5), -6)
        XCTAssertEqual(backward.absoluteStart(5), 4)
        XCTAssertEqual(backward.absoluteEnd(5), -1)
        XCTAssertEqual(backward.description, " ..<  : -1")

        // negative bounds resolve against the size
        let fromEnd = MLXSlice(start: -2, end: -1)
        XCTAssertEqual(fromEnd.start(5), -2)
        XCTAssertEqual(fromEnd.absoluteStart(5), 3)
        XCTAssertEqual(fromEnd.absoluteEnd(5), 4)
        XCTAssertEqual(fromEnd.description, "-2 ..< -1")

        XCTAssertEqual(
            MLXSlice.stride(from: 1, to: 4, by: 2), MLXSlice(start: 1, end: 4, stride: 2))
        XCTAssertEqual(MLXSlice.stride(), MLXSlice())
        XCTAssertNotEqual(MLXSlice(start: 1), MLXSlice(start: 2))
    }
}
