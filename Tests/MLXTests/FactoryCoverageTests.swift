// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

/// Tests for `Source/MLX/Factory.swift`: each `MLXArray` static factory and
/// each free factory function, with exact values, shapes and dtypes.
class FactoryCoverageTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    // MARK: - zeros and ones

    func testZeros() {
        let a = MLXArray.zeros([2, 3])
        XCTAssertEqual(a.shape, [2, 3])
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.asArray(Float.self), [Float](repeating: 0, count: 6))

        let b = MLXArray.zeros([4], type: Int32.self)
        XCTAssertEqual(b.dtype, .int32)
        XCTAssertEqual(b.asArray(Int32.self), [0, 0, 0, 0])

        let c = MLXArray.zeros([2, 2], dtype: .int16)
        XCTAssertEqual(c.dtype, .int16)
        XCTAssertEqual(c.asArray(Int16.self), [0, 0, 0, 0])

        let source = MLXArray([UInt8(1), UInt8(2), UInt8(3)], [3, 1])
        let d = MLXArray.zeros(like: source)
        XCTAssertEqual(d.shape, [3, 1])
        XCTAssertEqual(d.dtype, .uint8)
        XCTAssertEqual(d.asArray(UInt8.self), [0, 0, 0])

        // free functions
        let e = MLX.zeros([3], type: Int8.self)
        XCTAssertEqual(e.dtype, .int8)
        XCTAssertEqual(e.asArray(Int8.self), [0, 0, 0])

        let f = MLX.zeros([1, 2], dtype: .float16)
        XCTAssertEqual(f.dtype, .float16)
        XCTAssertEqual(f.shape, [1, 2])
        XCTAssertEqual(f.asArray(Float.self), [0, 0])

        let g = MLX.zeros(like: source)
        XCTAssertEqual(g.dtype, .uint8)
        XCTAssertEqual(g.shape, [3, 1])
    }

    func testOnes() {
        let a = MLXArray.ones([2, 2])
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.asArray(Float.self), [1, 1, 1, 1])

        let b = MLXArray.ones([3], type: Int32.self)
        XCTAssertEqual(b.dtype, .int32)
        XCTAssertEqual(b.asArray(Int32.self), [1, 1, 1])

        let c = MLXArray.ones([2], dtype: .uint16)
        XCTAssertEqual(c.dtype, .uint16)
        XCTAssertEqual(c.asArray(UInt16.self), [1, 1])

        let source = MLXArray([Int32(5), Int32(6)], [1, 2])
        let d = MLXArray.ones(like: source)
        XCTAssertEqual(d.dtype, .int32)
        XCTAssertEqual(d.shape, [1, 2])
        XCTAssertEqual(d.asArray(Int32.self), [1, 1])

        // free functions
        let e = MLX.ones([2, 1], type: Int8.self)
        XCTAssertEqual(e.dtype, .int8)
        XCTAssertEqual(e.shape, [2, 1])
        XCTAssertEqual(e.asArray(Int8.self), [1, 1])

        let f = MLX.ones([3], dtype: .bool)
        XCTAssertEqual(f.dtype, .bool)
        XCTAssertEqual(f.asArray(Bool.self), [true, true, true])

        let g = MLX.ones(like: source)
        XCTAssertEqual(g.dtype, .int32)
        XCTAssertEqual(g.asArray(Int32.self), [1, 1])
    }

    // MARK: - eye and identity

    func testEye() {
        let a = MLXArray.eye(3)
        XCTAssertEqual(a.shape, [3, 3])
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.asArray(Float.self), [1, 0, 0, 0, 1, 0, 0, 0, 1])

        // 2 x 3 with the diagonal moved up by one
        let b = MLXArray.eye(2, m: 3, k: 1, type: Int32.self)
        XCTAssertEqual(b.shape, [2, 3])
        XCTAssertEqual(b.dtype, .int32)
        XCTAssertEqual(b.asArray(Int32.self), [0, 1, 0, 0, 0, 1])

        // diagonal moved down by one
        let c = MLXArray.eye(3, k: -1, dtype: .int32)
        XCTAssertEqual(c.asArray(Int32.self), [0, 0, 0, 1, 0, 0, 0, 1, 0])

        // free functions
        let d = MLX.eye(2, m: 3, type: UInt8.self)
        XCTAssertEqual(d.dtype, .uint8)
        XCTAssertEqual(d.asArray(UInt8.self), [1, 0, 0, 0, 1, 0])

        let e = MLX.eye(3, m: 2, k: -1, dtype: .int16)
        XCTAssertEqual(e.shape, [3, 2])
        XCTAssertEqual(e.dtype, .int16)
        XCTAssertEqual(e.asArray(Int16.self), [0, 0, 1, 0, 0, 1])
    }

    func testIdentity() {
        let a = MLXArray.identity(2)
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.asArray(Float.self), [1, 0, 0, 1])

        let b = MLXArray.identity(3, type: Int32.self)
        XCTAssertEqual(b.dtype, .int32)
        XCTAssertEqual(b.asArray(Int32.self), [1, 0, 0, 0, 1, 0, 0, 0, 1])

        let c = MLXArray.identity(2, dtype: .uint8)
        XCTAssertEqual(c.dtype, .uint8)
        XCTAssertEqual(c.asArray(UInt8.self), [1, 0, 0, 1])

        let d = MLX.identity(2, type: Int16.self)
        XCTAssertEqual(d.dtype, .int16)
        XCTAssertEqual(d.asArray(Int16.self), [1, 0, 0, 1])

        let e = MLX.identity(1, dtype: .bool)
        XCTAssertEqual(e.dtype, .bool)
        XCTAssertEqual(e.asArray(Bool.self), [true])
    }

    // MARK: - full

    func testFull() {
        // values broadcast to the shape; the type argument sets the dtype
        let a = MLXArray.full([2, 3], values: MLXArray(Int32(7)), type: Float.self)
        XCTAssertEqual(a.shape, [2, 3])
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.asArray(Float.self), [7, 7, 7, 7, 7, 7])

        let b = MLXArray.full([2, 2], values: MLXArray(Float(2.5)), dtype: .float16)
        XCTAssertEqual(b.dtype, .float16)
        XCTAssertEqual(b.asArray(Float.self), [2.5, 2.5, 2.5, 2.5])

        // dtype comes from the values; each row gets [1, 2, 3]
        let row = MLXArray([Int32(1), Int32(2), Int32(3)])
        let c = MLXArray.full([2, 3], values: row)
        XCTAssertEqual(c.dtype, .int32)
        XCTAssertEqual(c.asArray(Int32.self), [1, 2, 3, 1, 2, 3])

        // free functions
        let d = MLX.full([3], values: Int32(4), type: Int8.self)
        XCTAssertEqual(d.dtype, .int8)
        XCTAssertEqual(d.asArray(Int8.self), [4, 4, 4])

        let e = MLX.full([2, 1], values: MLXArray(Int32(9)), dtype: .int16)
        XCTAssertEqual(e.dtype, .int16)
        XCTAssertEqual(e.shape, [2, 1])
        XCTAssertEqual(e.asArray(Int16.self), [9, 9])

        let f = MLX.full([2], values: Float(1.5))
        XCTAssertEqual(f.dtype, .float32)
        XCTAssertEqual(f.asArray(Float.self), [1.5, 1.5])

        let g = MLX.full([2], values: true)
        XCTAssertEqual(g.dtype, .bool)
        XCTAssertEqual(g.asArray(Bool.self), [true, true])
    }

    // MARK: - linspace

    func testLinspace() {
        let a = MLXArray.linspace(Int32(0), Int32(10), count: 6)
        XCTAssertEqual(a.dtype, .int32)
        XCTAssertEqual(a.asArray(Int32.self), [0, 2, 4, 6, 8, 10])

        let b = MLXArray.linspace(Float(0), Float(1), count: 5)
        XCTAssertEqual(b.dtype, .float32)
        XCTAssertEqual(b.asArray(Float.self), [0, 0.25, 0.5, 0.75, 1])

        // the default count is 50. The GPU computes the values in float32,
        // so a few of them are off by one or two ulp.
        let c = MLXArray.linspace(Float(0), Float(49))
        XCTAssertEqual(c.shape, [50])
        assertEqual(c, MLXArray((0 ..< 50).map { Float($0) }), atol: 1e-5)

        // free functions
        let d = MLX.linspace(Int16(-4), Int16(4), count: 3)
        XCTAssertEqual(d.dtype, .int16)
        XCTAssertEqual(d.asArray(Int16.self), [-4, 0, 4])

        let e = MLX.linspace(Float(1), Float(2), count: 3)
        assertEqual(e, MLXArray([Float(1), Float(1.5), Float(2)]))

        let f = MLX.linspace(Float16(0), Float16(2), count: 5)
        XCTAssertEqual(f.dtype, .float16)
        XCTAssertEqual(f.asArray(Float.self), [0, 0.5, 1, 1.5, 2])
    }

    // MARK: - arange

    func testArangeInt() {
        let a = MLXArray.arange(5)
        XCTAssertEqual(a.dtype, .int32)
        XCTAssertEqual(a.asArray(Int32.self), [0, 1, 2, 3, 4])

        let b = MLXArray.arange(2, 10, step: 3)
        XCTAssertEqual(b.dtype, .int32)
        XCTAssertEqual(b.asArray(Int32.self), [2, 5, 8])

        let c = MLXArray.arange(4, dtype: .float32)
        XCTAssertEqual(c.dtype, .float32)
        XCTAssertEqual(c.asArray(Float.self), [0, 1, 2, 3])

        let d = MLXArray.arange(1, 7, step: 2, dtype: .int16)
        XCTAssertEqual(d.dtype, .int16)
        XCTAssertEqual(d.asArray(Int16.self), [1, 3, 5])

        // free functions
        let e = MLX.arange(3)
        XCTAssertEqual(e.asArray(Int32.self), [0, 1, 2])

        let f = MLX.arange(10, 0, step: -4)
        XCTAssertEqual(f.asArray(Int32.self), [10, 6, 2])

        let g = MLX.arange(3, dtype: .uint8)
        XCTAssertEqual(g.dtype, .uint8)
        XCTAssertEqual(g.asArray(UInt8.self), [0, 1, 2])

        let h = MLX.arange(-2, 2, dtype: .float32)
        XCTAssertEqual(h.asArray(Float.self), [-2, -1, 0, 1])
    }

    func testArangeDouble() {
        let a = MLXArray.arange(3.0)
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.asArray(Float.self), [0, 1, 2])

        let b = MLXArray.arange(0.0, 1.0, step: 0.25)
        XCTAssertEqual(b.dtype, .float32)
        XCTAssertEqual(b.asArray(Float.self), [0, 0.25, 0.5, 0.75])

        let c = MLXArray.arange(2.5, dtype: .float16)
        XCTAssertEqual(c.dtype, .float16)
        XCTAssertEqual(c.asArray(Float.self), [0, 1, 2])

        // free functions
        let d = MLX.arange(2.0)
        XCTAssertEqual(d.asArray(Float.self), [0, 1])

        let e = MLX.arange(1.0, 2.0, step: 0.5, dtype: .float16)
        XCTAssertEqual(e.dtype, .float16)
        XCTAssertEqual(e.asArray(Float.self), [1, 1.5])

        let f = MLX.arange(-1.0, 1.0)
        XCTAssertEqual(f.asArray(Float.self), [-1, 0])
    }

    // MARK: - repeated

    func testRepeated() {
        // [[0, 1], [2, 3]]
        let a = MLXArray(Int32(0) ..< Int32(4), [2, 2])

        let b = MLXArray.repeated(a, count: 2, axis: 1)
        XCTAssertEqual(b.shape, [2, 4])
        XCTAssertEqual(b.asArray(Int32.self), [0, 0, 1, 1, 2, 2, 3, 3])

        let c = MLXArray.repeated(a, count: 2, axis: 0)
        XCTAssertEqual(c.shape, [4, 2])
        XCTAssertEqual(c.asArray(Int32.self), [0, 1, 0, 1, 2, 3, 2, 3])

        // without an axis the array is flattened first
        let d = MLXArray.repeated(a, count: 2)
        XCTAssertEqual(d.shape, [8])
        XCTAssertEqual(d.asArray(Int32.self), [0, 0, 1, 1, 2, 2, 3, 3])

        // free functions
        let e = MLX.repeated(a, count: 3, axis: 0)
        XCTAssertEqual(e.shape, [6, 2])
        XCTAssertEqual(e.asArray(Int32.self), [0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3])

        let f = MLX.repeated(MLXArray([Int32(7), Int32(8)]), count: 3)
        XCTAssertEqual(f.asArray(Int32.self), [7, 7, 7, 8, 8, 8])
    }

    // MARK: - tri

    func testTri() {
        let a = MLXArray.tri(3)
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.asArray(Float.self), [1, 0, 0, 1, 1, 0, 1, 1, 1])

        let b = MLXArray.tri(2, m: 3, k: 1, type: Int32.self)
        XCTAssertEqual(b.shape, [2, 3])
        XCTAssertEqual(b.dtype, .int32)
        XCTAssertEqual(b.asArray(Int32.self), [1, 1, 0, 1, 1, 1])

        let c = MLXArray.tri(3, k: -1, dtype: .int32)
        XCTAssertEqual(c.asArray(Int32.self), [0, 0, 0, 1, 0, 0, 1, 1, 0])

        // free functions
        let d = MLX.tri(2, type: UInt8.self)
        XCTAssertEqual(d.dtype, .uint8)
        XCTAssertEqual(d.asArray(UInt8.self), [1, 0, 1, 1])

        let e = MLX.tri(3, m: 2, dtype: .int16)
        XCTAssertEqual(e.shape, [3, 2])
        XCTAssertEqual(e.dtype, .int16)
        XCTAssertEqual(e.asArray(Int16.self), [1, 0, 1, 1, 1, 1])
    }
}
