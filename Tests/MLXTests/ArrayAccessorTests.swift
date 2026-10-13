// Copyright © 2026 Eigen Labs.

import Foundation
import Numerics
import XCTest

@testable import MLX

/// Tests for the accessors and conversions in `MLXArray.swift`.
class ArrayAccessorTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    // MARK: - item()

    func testItemReadsEveryIntegerDType() {
        let dtypes: [DType] = [.uint8, .uint16, .uint32, .uint64, .int8, .int16, .int32, .int64]
        for dtype in dtypes {
            let a = MLXArray(Int32(7)).asType(dtype)
            XCTAssertEqual(a.dtype, dtype)

            // signed reads go through itemInt()
            XCTAssertEqual(a.item(Int.self), 7, "\(dtype)")
            XCTAssertEqual(a.item(Int8.self), 7, "\(dtype)")
            XCTAssertEqual(a.item(Int16.self), 7, "\(dtype)")
            XCTAssertEqual(a.item(Int32.self), 7, "\(dtype)")
            XCTAssertEqual(a.item(Int64.self), 7, "\(dtype)")

            // unsigned reads go through itemUInt()
            XCTAssertEqual(a.item(UInt8.self), 7, "\(dtype)")
            XCTAssertEqual(a.item(UInt16.self), 7, "\(dtype)")
            XCTAssertEqual(a.item(UInt32.self), 7, "\(dtype)")
            XCTAssertEqual(a.item(UInt64.self), 7, "\(dtype)")
            XCTAssertEqual(a.item(UInt.self), 7, "\(dtype)")
        }
    }

    func testItemReadsNegativeSignedValues() {
        let dtypes: [DType] = [.int8, .int16, .int32, .int64]
        for dtype in dtypes {
            let a = MLXArray(Int32(-3)).asType(dtype)
            XCTAssertEqual(a.item(Int.self), -3, "\(dtype)")
            XCTAssertEqual(a.item(Int8.self), -3, "\(dtype)")
            XCTAssertEqual(a.item(Int64.self), -3, "\(dtype)")
        }
    }

    func testItemReadsFloatDTypes() {
        // float32 source
        let f32 = MLXArray(Float(2.5))
        XCTAssertEqual(f32.dtype, .float32)
        XCTAssertEqual(f32.item(Float.self), 2.5)
        XCTAssertEqual(f32.item(Float32.self), 2.5)
        XCTAssertEqual(f32.item(Double.self), 2.5)

        // float64 source (made directly, no GPU cast)
        let f64 = MLXArray(float64: 2.5)
        XCTAssertEqual(f64.dtype, .float64)
        XCTAssertEqual(f64.item(Float.self), 2.5)
        XCTAssertEqual(f64.item(Double.self), 2.5)

        #if !arch(x86_64)
            XCTAssertEqual(f32.item(Float16.self), 2.5)
            XCTAssertEqual(f64.item(Float16.self), 2.5)

            // float16 source
            let f16 = MLXArray(Float(2.5)).asType(.float16)
            XCTAssertEqual(f16.dtype, .float16)
            XCTAssertEqual(f16.item(Float.self), 2.5)
            XCTAssertEqual(f16.item(Float32.self), 2.5)
            XCTAssertEqual(f16.item(Float16.self), 2.5)
            XCTAssertEqual(f16.item(Double.self), 2.5)
        #endif
    }

    func testItemConvertsWhenTypeDoesNotMatch() {
        // float array read as an integer: converts (truncates) first
        XCTAssertEqual(MLXArray(Float(2.75)).item(Int.self), 2)
        XCTAssertEqual(MLXArray(Float(2.75)).item(UInt8.self), 2)

        // integer array read as a float: converts first
        XCTAssertEqual(MLXArray(Int32(3)).item(Float.self), 3)

        // integer array read as a bool
        XCTAssertEqual(MLXArray(Int32(1)).item(Bool.self), true)
        XCTAssertEqual(MLXArray(Int32(0)).item(Bool.self), false)

        // bool array read as integers
        XCTAssertEqual(MLXArray(true).item(Int.self), 1)
        XCTAssertEqual(MLXArray(true).item(UInt16.self), 1)
        XCTAssertEqual(MLXArray(false).item(Int32.self), 0)
        XCTAssertEqual(MLXArray(true).item(Bool.self), true)
    }

    func testItemGenericAndComplex() {
        let v: Int32 = MLXArray(Int32(9)).item()
        XCTAssertEqual(v, 9)

        let f: Float = MLXArray([1, 4.5] as [Float])[1].item()
        XCTAssertEqual(f, 4.5)

        let c = MLXArray(real: 1.5, imaginary: -2)
        XCTAssertEqual(c.dtype, .complex64)
        XCTAssertEqual(c.item(Complex<Float>.self), Complex(1.5, -2))
    }

    // MARK: - shape and size

    func testShapeTuplesAndDims() {
        let a = MLXArray(Int32(0) ..< 24, [2, 3, 4])

        let (r, c) = a.reshaped(6, 4).shape2
        XCTAssertEqual(r, 6)
        XCTAssertEqual(c, 4)

        let (d0, d1, d2) = a.shape3
        XCTAssertEqual([d0, d1, d2], [2, 3, 4])

        let (b, h, w, ch) = a.reshaped(1, 2, 3, 4).shape4
        XCTAssertEqual([b, h, w, ch], [1, 2, 3, 4])

        XCTAssertEqual(a.dim(0), 2)
        XCTAssertEqual(a.dim(-1), 4)
        XCTAssertEqual(a.dim(Int32(1)), Int32(3))
        XCTAssertEqual(a.dim(Int32(-1)), Int32(4))

        XCTAssertEqual(a.size, 24)
        XCTAssertEqual(a.count, 2)
        XCTAssertEqual(a.ndim, 3)
        XCTAssertEqual(a.itemSize, 4)
        XCTAssertEqual(a.nbytes, 24 * 4)
    }

    func testScalarShapeAndStrides() {
        let s = MLXArray(Int32(5))
        XCTAssertEqual(s.ndim, 0)
        XCTAssertEqual(s.shape, [])
        XCTAssertEqual(s.size, 1)
        XCTAssertEqual(s.internalStrides, [])
    }

    @available(*, deprecated)
    func testDeprecatedStrides() {
        let a = MLXArray(Int32(0) ..< 24, [2, 3, 4])
        a.eval()
        XCTAssertEqual(a.strides, [12, 4, 1])

        let s = MLXArray(Int32(5))
        XCTAssertEqual(s.strides, [])
    }

    // MARK: - conversion

    func testAsTypeOverloads() {
        let a = MLXArray(Int32(0) ..< 4)
        XCTAssertEqual(a.dtype, .int32)

        // same type returns the same array
        XCTAssertTrue(a.asType(.int32) === a)
        XCTAssertTrue(a.asType(Int32.self) === a)

        let f = a.asType(Float.self)
        XCTAssertEqual(f.dtype, .float32)
        XCTAssertEqual(f.asArray(Float.self), [0, 1, 2, 3])

        let u = a.asType(.uint8)
        XCTAssertEqual(u.dtype, .uint8)
        XCTAssertEqual(u.asArray(UInt8.self), [0, 1, 2, 3])
    }

    func testComplexParts() {
        let re = MLXArray([1, 2] as [Float])
        let im = MLXArray([3, 4] as [Float])

        let i = im.asImaginary()
        XCTAssertEqual(i.dtype, .complex64)
        XCTAssertEqual(i.asArray(Complex<Float>.self), [Complex(0, 3), Complex(0, 4)])

        let c = re + i
        XCTAssertEqual(c.dtype, .complex64)
        XCTAssertEqual(c.asArray(Complex<Float>.self), [Complex(1, 3), Complex(2, 4)])

        let real = c.realPart()
        XCTAssertEqual(real.dtype, .float32)
        assertEqual(real, re)

        let imaginary = c.imaginaryPart()
        XCTAssertEqual(imaginary.dtype, .float32)
        assertEqual(imaginary, im)
    }

    // MARK: - identity and state

    func testUpdateInternalAndCopyContext() {
        let a = MLXArray([1, 2, 3] as [Int32])
        let b = MLXArray([7, 8] as [Int32])

        let copy = a.copyContext()
        XCTAssertFalse(copy === a)
        XCTAssertEqual(copy.asArray(Int32.self), [1, 2, 3])

        a._updateInternal(b)
        XCTAssertEqual(a.shape, [2])
        XCTAssertEqual(a.asArray(Int32.self), [7, 8])

        // the copy keeps the old contents
        XCTAssertEqual(copy.asArray(Int32.self), [1, 2, 3])
    }

    func testInnerStateAndEval() {
        let a = MLXArray([1, 2] as [Int32]) * 3
        let state = a.innerState()
        XCTAssertEqual(state.count, 1)
        XCTAssertTrue(state[0] === a)

        a.eval()
        XCTAssertEqual(a.asArray(Int32.self), [3, 6])
    }

    func testDescription() {
        let a = MLXArray([1, 2, 3] as [Int32])
        let d = a.description
        XCTAssertTrue(d.contains("1, 2, 3"), d)
        XCTAssertTrue(d.contains("int32"), d)

        let s = MLXArray(Float(2.5)).description
        XCTAssertTrue(s.contains("2.5"), s)
        XCTAssertTrue(s.contains("float32"), s)
    }
}
