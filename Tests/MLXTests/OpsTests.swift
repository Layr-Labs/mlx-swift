// Copyright © 2024 Apple Inc.

import Foundation
import XCTest

@testable import MLX

class OpsTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    func testAsStridedReshape() {
        // just changing the shape and using the default strides is the same as reshape
        let a = MLXArray(0 ..< 12, [4, 3])

        // this uses [4, 1] as the strides
        let b = asStrided(a, [3, 4])
        assertEqual(b, a.reshaped([3, 4]))

        let c = asStrided(a, [3, 4], strides: [4, 1])
        assertEqual(b, c)
    }

    func testAsStridedTranspose() {
        // strides in the reverse order is a transpose
        let a = MLXArray(0 ..< 12, [4, 3])

        let b = asStrided(a, [3, 4], strides: [1, 3])
        assertEqual(b, a.transposed())
    }

    func testAsStridedOffset() {
        let a = MLXArray(0 ..< 16, [4, 4])

        let b = asStrided(a, [3, 4], offset: 1)
        assertEqual(b, MLXArray(1 ..< 13, [3, 4]))
    }

    func testTensordot() {
        let a = MLXArray(0 ..< 60, [3, 4, 5]).asType(.float32)
        let b = MLXArray(0 ..< 24, [4, 3, 2]).asType(.float32)
        let c = tensordot(a, b, axes: ([1, 0], [0, 1]))

        let expected = MLXArray(
            converting: [
                4400.0, 4730.0,
                4532.0, 4874.0,
                4664.0, 5018.0,
                4796.0, 5162.0,
                4928.0, 5306.0,
            ], [5, 2])
        assertEqual(c, expected)
    }

    /// A [1, K] x [K, 1] product takes the `dot_product` route, whose kernel
    /// ships only in the prebuilt metallib (`dot.metal`).
    func testMatmulVectorDotProduct() {
        let k = 4096
        let a = (MLXArray(0 ..< k) % 4).asType(.float32)
        let b = (MLXArray(0 ..< k) % 3).asType(.float32)

        let product = matmul(a.reshaped([1, k]), b.reshaped([k, 1]))

        assertEqual(product, (a * b).sum().reshaped([1, 1]))
    }

    /// One-row operands with sorted rhs indices take the `gather_mm_rhs`
    /// route, which builds per-expert row offsets with the `gather_mm_offsets`
    /// kernel. That kernel ships only in the prebuilt metallib.
    func testGatherMMSortedRHSIndices() {
        let rows = 6
        let experts = 4
        let k = 64
        let n = 32
        let rhsIndices = MLXArray([0, 0, 1, 3, 3, 3] as [Int32])

        for dtype in [DType.float32, .float16] {
            let a = (MLXArray(0 ..< rows * k) % 4).asType(dtype).reshaped([rows, 1, k])
            let b = (MLXArray(0 ..< experts * k * n) % 3).asType(dtype)
                .reshaped([experts, k, n])

            let sorted = gatherMM(a, b, rhsIndices: rhsIndices, sortedIndices: true)

            assertEqual(sorted, matmul(a, b.take(rhsIndices, axis: 0)))
        }
    }

    func testConvertScalarInt() {
        let a = MLXArray(0 ..< 10)
        let b = a .< (a + 1)
        let c = b * 25
        XCTAssertEqual(b.dtype, .bool)
        XCTAssertEqual(c.dtype, .int32)
    }

    func testConvertScalarFloat16() {
        let a = MLXArray(0 ..< 10)
        let b = a .< (a + 1)
        let c = b * Float16(2.5)
        XCTAssertEqual(b.dtype, .bool)
        XCTAssertEqual(c.dtype, .float16)
    }

    func testConvertScalarFloat() {
        let a = MLXArray(0 ..< 10)
        let b = a .< (a + 1)
        let c = b * Float(2.5)
        XCTAssertEqual(b.dtype, .bool)
        XCTAssertEqual(c.dtype, .float32)
    }

    func testConvertScalarDouble() {
        let a = MLXArray(0 ..< 10)
        let b = a .< (a + 1)
        let c = b * 2.5
        XCTAssertEqual(b.dtype, .bool)
        XCTAssertEqual(c.dtype, .float32)
    }

    func testFlatten() {
        let a = zeros([4, 5, 6, 7])
        let b = flatten(a, startAxis: 1, endAxis: 2)
        let c = unflatten(b, axis: 1, shape: [5, 6])
        assertEqual(a, c)
    }

    func testQuantized() {
        let a = MLXRandom.uniform(low: 0, high: 1, [8, 64])

        let (wq1, s1, b1) = quantized(a, mode: .affine)
        XCTAssertEqual(wq1.dtype, .uint32)
        XCTAssertEqual(wq1.shape, [8, 8])
        XCTAssertEqual(s1.shape, [8, 1])
        if let b1 {
            XCTAssertEqual(b1.shape, [8, 1])
        } else {
            XCTFail("b1 should not be nil")
        }

        let (wq2, s2, b2) = quantized(a, groupSize: 32, mode: .mxfp4)
        XCTAssertEqual(wq2.dtype, .uint32)
        XCTAssertEqual(wq2.shape, [8, 8])
        XCTAssertEqual(s2.shape, [8, 2])
        XCTAssertNil(b2)
    }

}
