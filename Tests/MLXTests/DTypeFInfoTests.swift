// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

/// `DType.finfo` values for complex64 and bfloat16.
class DTypeFInfoTests: XCTestCase {

    func testFInfoBFloat16SmallestSubnormal() throws {
        // bfloat16 has 8 exponent bits and 7 mantissa bits. The smallest
        // subnormal is 2^-126 * 2^-7 = 2^-133.
        let info = try XCTUnwrap(DType.bfloat16.finfo)
        XCTAssertEqual(info.smallestNormal, pow(2.0, -126))
        XCTAssertEqual(info.smallestSubnormal, pow(2.0, -133))
    }

    func testFInfoComplex64() throws {
        // MLX (mlx/utils.cpp, finfo::finfo) gives complex64 the limits of its
        // float32 parts.
        let info = try XCTUnwrap(DType.complex64.finfo)
        XCTAssertEqual(info.dtype, .complex64)
        XCTAssertEqual(info.eps, pow(2.0, -23))
        XCTAssertEqual(info.max, (2 - pow(2.0, -23)) * pow(2.0, 127))
        XCTAssertEqual(info.min, -(2 - pow(2.0, -23)) * pow(2.0, 127))
        XCTAssertEqual(info.smallestNormal, pow(2.0, -126))
        XCTAssertEqual(info.smallestSubnormal, pow(2.0, -149))

        // the same values as float32
        let f32 = try XCTUnwrap(DType.float32.finfo)
        XCTAssertEqual(info.eps, f32.eps)
        XCTAssertEqual(info.max, f32.max)
        XCTAssertEqual(info.min, f32.min)
        XCTAssertEqual(info.smallestNormal, f32.smallestNormal)
        XCTAssertEqual(info.smallestSubnormal, f32.smallestSubnormal)
    }
}
