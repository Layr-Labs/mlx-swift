// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import Numerics
import XCTest

/// Tests for `Source/MLX/DType.swift`: the `DType` properties, `finfo`, the
/// `Codable` form, each `HasDType.asMLXArray` conversion, `toArrays` and the
/// type promotion rules of MLX.
class DTypeCoverageTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    // MARK: - DType properties

    func testDTypeRoundTripThroughArrays() {
        // DType -> mlx_dtype -> DType for every case. The CPU stream is used
        // because MLX does not allow float64 on the GPU.
        for dtype in DType.allCases {
            XCTAssertEqual(
                MLXArray.zeros([1], dtype: dtype, stream: .cpu).dtype, dtype, "\(dtype)")
        }
        XCTAssertEqual(DType.allCases.count, 14)
    }

    func testDTypeKinds() {
        let floating: Set<DType> = [.float16, .float32, .bfloat16, .complex64, .float64]
        let complex: Set<DType> = [.complex64]
        let integer: Set<DType> = [
            .uint8, .uint16, .uint32, .uint64, .int8, .int16, .int32, .int64,
        ]
        let signed: Set<DType> = [.int8, .int16, .int32, .int64]

        for dtype in DType.allCases {
            XCTAssertEqual(dtype.isFloatingPoint, floating.contains(dtype), "\(dtype)")
            XCTAssertEqual(dtype.isComplex, complex.contains(dtype), "\(dtype)")
            XCTAssertEqual(dtype.isInteger, integer.contains(dtype), "\(dtype)")
            XCTAssertEqual(dtype.isSignedInteger, signed.contains(dtype), "\(dtype)")
        }
    }

    func testDTypeSize() {
        let sizes: [DType: Int] = [
            .bool: 1, .uint8: 1, .uint16: 2, .uint32: 4, .uint64: 8,
            .int8: 1, .int16: 2, .int32: 4, .int64: 8,
            .float16: 2, .float32: 4, .bfloat16: 2, .complex64: 8, .float64: 8,
        ]
        for dtype in DType.allCases {
            XCTAssertEqual(dtype.size, sizes[dtype], "\(dtype)")
        }
    }

    // MARK: - finfo

    func testFInfoIsNilForNonFloatingTypes() {
        for dtype in DType.allCases where !dtype.isFloatingPoint {
            XCTAssertNil(dtype.finfo, "\(dtype)")
        }
    }

    func testFInfoFloat16() throws {
        let info = try XCTUnwrap(DType.float16.finfo)
        XCTAssertEqual(info.dtype, .float16)
        XCTAssertEqual(info.eps, pow(2.0, -10))
        XCTAssertEqual(info.max, 65504)
        XCTAssertEqual(info.min, -65504)
        XCTAssertEqual(info.smallestNormal, pow(2.0, -14))
        XCTAssertEqual(info.smallestSubnormal, pow(2.0, -24))
    }

    func testFInfoFloat32() throws {
        let info = try XCTUnwrap(DType.float32.finfo)
        XCTAssertEqual(info.dtype, .float32)
        XCTAssertEqual(info.eps, pow(2.0, -23))
        XCTAssertEqual(info.max, (2 - pow(2.0, -23)) * pow(2.0, 127))
        XCTAssertEqual(info.min, -(2 - pow(2.0, -23)) * pow(2.0, 127))
        XCTAssertEqual(info.smallestNormal, pow(2.0, -126))
        XCTAssertEqual(info.smallestSubnormal, pow(2.0, -149))
    }

    func testFInfoFloat64() throws {
        let info = try XCTUnwrap(DType.float64.finfo)
        XCTAssertEqual(info.dtype, .float64)
        XCTAssertEqual(info.eps, pow(2.0, -52))
        XCTAssertEqual(info.max, 1.7976931348623157e308)
        XCTAssertEqual(info.min, -1.7976931348623157e308)
        XCTAssertEqual(info.smallestNormal, pow(2.0, -1022))
        XCTAssertEqual(info.smallestSubnormal, 5e-324)
    }

    func testFInfoBFloat16() throws {
        // bfloat16 has 8 exponent bits and 7 mantissa bits
        let info = try XCTUnwrap(DType.bfloat16.finfo)
        XCTAssertEqual(info.dtype, .bfloat16)
        XCTAssertEqual(info.eps, pow(2.0, -7))
        XCTAssertEqual(info.max, (2 - pow(2.0, -7)) * pow(2.0, 127))
        XCTAssertEqual(info.min, -(2 - pow(2.0, -7)) * pow(2.0, 127))
        XCTAssertEqual(info.smallestNormal, pow(2.0, -126))

        // Defect: DType.swift:204 gives the smallest normal (2^-126) as the
        // smallest subnormal. The doc comment defines it as the smallest
        // positive value with a leading 0 bit in the mantissa, which for
        // bfloat16 is 2^-126 * 2^-7 = 2^-133.
        XCTExpectFailure("DType.swift:204 bfloat16 smallestSubnormal is the smallest normal") {
            XCTAssertEqual(info.smallestSubnormal, pow(2.0, -133))
        }
    }

    func testFInfoComplex64() throws {
        // MLX (mlx/utils.cpp, finfo::finfo) gives complex64 the limits of its
        // float32 parts.
        let info = try XCTUnwrap(DType.complex64.finfo)
        XCTAssertEqual(info.dtype, .complex64)

        // Defect: DType.swift:137, 154, 171, 188 and 205 give complex64 the
        // float64 limits, not the float32 limits.
        XCTExpectFailure("DType.swift:137-205 complex64 finfo uses float64 limits") {
            XCTAssertEqual(info.eps, pow(2.0, -23))
            XCTAssertEqual(info.max, (2 - pow(2.0, -23)) * pow(2.0, 127))
            XCTAssertEqual(info.min, -(2 - pow(2.0, -23)) * pow(2.0, 127))
            XCTAssertEqual(info.smallestNormal, pow(2.0, -126))
            XCTAssertEqual(info.smallestSubnormal, pow(2.0, -149))
        }
    }

    // MARK: - Codable

    func testCodable() throws {
        // the encoded value is the mlx_dtype raw value
        let encoded = try JSONEncoder().encode([DType.bool, .uint8, .float32, .complex64])
        XCTAssertEqual(String(decoding: encoded, as: UTF8.self), "[0,1,10,13]")

        let decoded = try JSONDecoder().decode([DType].self, from: Data("[7,11,12]".utf8))
        XCTAssertEqual(decoded, [.int32, .float64, .bfloat16])

        let all = try JSONDecoder().decode(
            [DType].self, from: JSONEncoder().encode(DType.allCases))
        XCTAssertEqual(all, DType.allCases)
    }

    // MARK: - HasDType

    func testHasDTypeStaticDType() {
        XCTAssertEqual(Bool.dtype, .bool)
        XCTAssertEqual(Int.dtype, .int64)
        XCTAssertEqual(Int8.dtype, .int8)
        XCTAssertEqual(Int16.dtype, .int16)
        XCTAssertEqual(Int32.dtype, .int32)
        XCTAssertEqual(Int64.dtype, .int64)
        XCTAssertEqual(UInt8.dtype, .uint8)
        XCTAssertEqual(UInt16.dtype, .uint16)
        XCTAssertEqual(UInt32.dtype, .uint32)
        XCTAssertEqual(UInt64.dtype, .uint64)
        XCTAssertEqual(UInt.dtype, .uint64)
        XCTAssertEqual(Float16.dtype, .float16)
        XCTAssertEqual(Float32.dtype, .float32)
        XCTAssertEqual(Float64.dtype, .float64)
        XCTAssertEqual(Complex<Float>.dtype, .complex64)
    }

    func testIntegerScalarAsMLXArray() {
        // with no dtype each type keeps its own dtype
        XCTAssertEqual(Int8(3).asMLXArray(dtype: nil).dtype, .int8)
        XCTAssertEqual(Int16(3).asMLXArray(dtype: nil).dtype, .int16)
        XCTAssertEqual(Int32(3).asMLXArray(dtype: nil).dtype, .int32)
        XCTAssertEqual(Int64(3).asMLXArray(dtype: nil).dtype, .int64)
        XCTAssertEqual(UInt8(3).asMLXArray(dtype: nil).dtype, .uint8)
        XCTAssertEqual(UInt16(3).asMLXArray(dtype: nil).dtype, .uint16)
        XCTAssertEqual(UInt32(3).asMLXArray(dtype: nil).dtype, .uint32)
        XCTAssertEqual(UInt64(3).asMLXArray(dtype: nil).dtype, .uint64)
        XCTAssertEqual(UInt(3).asMLXArray(dtype: nil).dtype, .uint64)

        // a bool dtype is not used for an integer; the own dtype is kept
        XCTAssertEqual(Int8(3).asMLXArray(dtype: .bool).dtype, .int8)
        XCTAssertEqual(Int16(3).asMLXArray(dtype: .bool).dtype, .int16)
        XCTAssertEqual(Int32(3).asMLXArray(dtype: .bool).dtype, .int32)
        XCTAssertEqual(Int64(3).asMLXArray(dtype: .bool).dtype, .int64)
        XCTAssertEqual(UInt8(3).asMLXArray(dtype: .bool).dtype, .uint8)
        XCTAssertEqual(UInt16(3).asMLXArray(dtype: .bool).dtype, .uint16)
        XCTAssertEqual(UInt32(3).asMLXArray(dtype: .bool).dtype, .uint32)
        XCTAssertEqual(UInt64(3).asMLXArray(dtype: .bool).dtype, .uint64)
        XCTAssertEqual(UInt(3).asMLXArray(dtype: .bool).dtype, .uint64)

        // any other dtype is used, and the value is kept
        let a = Int8(-5).asMLXArray(dtype: .float32)
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.item(Float.self), -5)
        let b = UInt16(300).asMLXArray(dtype: .int32)
        XCTAssertEqual(b.dtype, .int32)
        XCTAssertEqual(b.item(Int32.self), 300)
        let c = Int64(7).asMLXArray(dtype: .int16)
        XCTAssertEqual(c.dtype, .int16)
        XCTAssertEqual(c.item(Int16.self), 7)
        let d = UInt(9).asMLXArray(dtype: .uint8)
        XCTAssertEqual(d.dtype, .uint8)
        XCTAssertEqual(d.item(UInt8.self), 9)

        // Bool uses the default HasDType conversion
        let e = true.asMLXArray(dtype: nil)
        XCTAssertEqual(e.dtype, .bool)
        XCTAssertEqual(e.item(Bool.self), true)
    }

    func testFloatScalarAsMLXArray() {
        // Float16 keeps its dtype unless a floating point dtype is given
        XCTAssertEqual(Float16(1.5).asMLXArray(dtype: nil).dtype, .float16)
        XCTAssertEqual(Float16(1.5).asMLXArray(dtype: .int32).dtype, .float16)
        let h = Float16(1.5).asMLXArray(dtype: .bfloat16)
        XCTAssertEqual(h.dtype, .bfloat16)
        XCTAssertEqual(h.item(Float.self), 1.5)

        XCTAssertEqual(Float(1.5).asMLXArray(dtype: nil).dtype, .float32)
        XCTAssertEqual(Float(1.5).asMLXArray(dtype: .bfloat16).dtype, .bfloat16)

        // Double does not promote to float64 unless asked
        let d = Double(2.5).asMLXArray(dtype: nil)
        XCTAssertEqual(d.dtype, .float32)
        XCTAssertEqual(d.item(Float.self), 2.5)
        XCTAssertEqual(Double(2.5).asMLXArray(dtype: .int8).dtype, .float32)
        XCTAssertEqual(Double(2.5).asMLXArray(dtype: .float16).dtype, .float16)
        XCTAssertEqual(Double(2.5).asMLXArray(dtype: .float64).dtype, .float64)

        // complex values stay complex
        let c = Complex<Float>(1, 2).asMLXArray(dtype: .float32)
        XCTAssertEqual(c.dtype, .complex64)
        XCTAssertEqual(c.realPart().item(Float.self), 1)
        XCTAssertEqual(c.imaginaryPart().item(Float.self), 2)
    }

    func testArrayAndMLXArrayAsMLXArray() {
        // MLXArray returns itself and ignores the dtype
        let m = MLXArray([Int32(1), Int32(2)])
        XCTAssertTrue(m.asMLXArray(dtype: .float32) === m)

        // [T] uses the element dtype, or the given dtype
        let a = [Int8(1), Int8(2)].asMLXArray(dtype: nil)
        XCTAssertEqual(a.dtype, .int8)
        XCTAssertEqual(a.asArray(Int8.self), [1, 2])

        let b = [Int32(3), Int32(4)].asMLXArray(dtype: .float32)
        XCTAssertEqual(b.dtype, .float32)
        XCTAssertEqual(b.asArray(Float.self), [3, 4])

        let c = [Float(0.5), Float(1.5)].asMLXArray(dtype: nil)
        XCTAssertEqual(c.dtype, .float32)
        XCTAssertEqual(c.asArray(Float.self), [0.5, 1.5])
    }

    // MARK: - toArrays

    func testToArrays() {
        let i8 = MLXArray([Int8(1), Int8(2)])
        let f16 = MLXArray([Float16(1), Float16(2)])
        let i32 = MLXArray([Int32(1), Int32(2)])

        // two arrays: dtypes are not changed
        let (a1, b1) = toArrays(i8, f16)
        XCTAssertEqual(a1.dtype, .int8)
        XCTAssertEqual(b1.dtype, .float16)

        // array and scalar: the scalar takes the dtype of the array
        let (a2, b2) = toArrays(i8, 10)
        XCTAssertEqual(a2.dtype, .int8)
        XCTAssertEqual(b2.dtype, .int8)
        XCTAssertEqual(b2.item(Int8.self), 10)

        // scalar and array
        let (a3, b3) = toArrays(Float(2.5), f16)
        XCTAssertEqual(a3.dtype, .float16)
        XCTAssertEqual(b3.dtype, .float16)
        XCTAssertEqual(a3.item(Float.self), 2.5)

        // a float scalar with an integer array stays float
        let (_, b4) = toArrays(i32, Float(1.5))
        XCTAssertEqual(b4.dtype, .float32)

        // two scalars: each keeps its own default dtype
        let (a5, b5) = toArrays(3, Float(2.5))
        XCTAssertEqual(a5.dtype, .int32)
        XCTAssertEqual(b5.dtype, .float32)
    }

    // MARK: - promotion

    func testArrayArrayPromotion() {
        // The MLX promotion rules (mlx/dtype.cpp, promote_types), checked
        // through binary operations between two arrays. The CPU stream is
        // used because MLX does not allow float64 on the GPU.
        func promoted(_ a: DType, _ b: DType) -> DType {
            MLX.add(
                MLXArray.zeros([1], dtype: a, stream: .cpu),
                MLXArray.zeros([1], dtype: b, stream: .cpu), stream: .cpu
            ).dtype
        }

        XCTAssertEqual(promoted(.bool, .bool), .bool)
        XCTAssertEqual(promoted(.bool, .int16), .int16)
        XCTAssertEqual(promoted(.int8, .uint8), .int16)
        XCTAssertEqual(promoted(.uint8, .uint16), .uint16)
        XCTAssertEqual(promoted(.int16, .uint16), .int32)
        XCTAssertEqual(promoted(.int32, .uint32), .int64)
        XCTAssertEqual(promoted(.int8, .int64), .int64)
        XCTAssertEqual(promoted(.int8, .float16), .float16)
        XCTAssertEqual(promoted(.int32, .float32), .float32)
        XCTAssertEqual(promoted(.float16, .bfloat16), .float32)
        XCTAssertEqual(promoted(.float16, .float32), .float32)
        XCTAssertEqual(promoted(.float32, .complex64), .complex64)
        XCTAssertEqual(promoted(.float32, .float64), .float64)
    }

    func testScalarPromotion() {
        // a scalar does not change the dtype of the array it is used with
        let i8 = MLXArray([Int8(1), Int8(2)])
        let sum8 = i8 + 10
        XCTAssertEqual(sum8.dtype, .int8)
        XCTAssertEqual(sum8.asArray(Int8.self), [11, 12])
        XCTAssertEqual((10 + i8).dtype, .int8)

        let f16 = MLXArray([Float16(1), Float16(2)])
        let sum16 = f16 + Float(0.5)
        XCTAssertEqual(sum16.dtype, .float16)
        XCTAssertEqual(sum16.asArray(Float.self), [1.5, 2.5])

        // but a float scalar with an integer array gives a float result
        let i32 = MLXArray([Int32(1), Int32(2)])
        let mixed = i32 * Float(0.5)
        XCTAssertEqual(mixed.dtype, .float32)
        XCTAssertEqual(mixed.asArray(Float.self), [0.5, 1])
    }
}
