// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

// Tests for the free functions in Source/MLX/Ops+Array.swift and for the operators
// and methods in Source/MLX/MLXArray+Ops.swift. Each test calls the free function
// and the method form, and compares the results with values that are computed by
// hand or by a plain Swift loop.

/// Make a float32 array with the given values and shape.
private func arrF(_ values: [Float], _ shape: [Int]? = nil) -> MLXArray {
    MLXArray(values, shape ?? [values.count])
}

/// Make an int32 array with the given values and shape.
private func arrI(_ values: [Int32], _ shape: [Int]? = nil) -> MLXArray {
    MLXArray(values, shape ?? [values.count])
}

/// Apply `f` to each value and make a float32 array with the results.
private func arrMap(_ values: [Float], _ f: (Float) -> Float) -> MLXArray {
    arrF(values.map(f))
}

class MLXArrayOpsCoverageTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    // MARK: - Arithmetic operators

    func testAddSubtractOperators() {
        let a = arrF([1, 2, 3])
        let b = arrF([10, 20, 30])

        XCTAssertEqual((a + b).asArray(Float.self), [11, 22, 33])
        XCTAssertEqual((a + 1).asArray(Float.self), [2, 3, 4])
        XCTAssertEqual((1 + a).asArray(Float.self), [2, 3, 4])
        XCTAssertEqual((b - a).asArray(Float.self), [9, 18, 27])
        XCTAssertEqual((a - 1).asArray(Float.self), [0, 1, 2])
        XCTAssertEqual((10 - a).asArray(Float.self), [9, 8, 7])
        XCTAssertEqual((-a).asArray(Float.self), [-1, -2, -3])

        // the scalar takes the dtype of the array
        let i = arrI([1, 2])
        XCTAssertEqual((i + 1).dtype, .int32)
        XCTAssertEqual((i + 1).asArray(Int32.self), [2, 3])

        var c = arrF([1, 2, 3])
        c += b
        XCTAssertEqual(c.asArray(Float.self), [11, 22, 33])
        c += 1
        XCTAssertEqual(c.asArray(Float.self), [12, 23, 34])
        c -= b
        XCTAssertEqual(c.asArray(Float.self), [2, 3, 4])
        c -= 2
        XCTAssertEqual(c.asArray(Float.self), [0, 1, 2])
    }

    func testMultiplyDivideOperators() {
        let a = arrF([1, 2, 4])
        let b = arrF([2, 4, 8])

        XCTAssertEqual((a * b).asArray(Float.self), [2, 8, 32])
        XCTAssertEqual((a * 3).asArray(Float.self), [3, 6, 12])
        XCTAssertEqual((3 * a).asArray(Float.self), [3, 6, 12])
        XCTAssertEqual((b / a).asArray(Float.self), [2, 2, 2])
        XCTAssertEqual((b / 2).asArray(Float.self), [1, 2, 4])
        XCTAssertEqual((8 / a).asArray(Float.self), [8, 4, 2])

        var c = arrF([1, 2, 4])
        c *= b
        XCTAssertEqual(c.asArray(Float.self), [2, 8, 32])
        c *= 2
        XCTAssertEqual(c.asArray(Float.self), [4, 16, 64])
        c /= b
        XCTAssertEqual(c.asArray(Float.self), [2, 4, 8])
        c /= 2
        XCTAssertEqual(c.asArray(Float.self), [1, 2, 4])
    }

    func testPowerAndRemainderOperators() {
        let a = arrF([1, 2, 3])
        let e = arrF([2, 3, 2])
        assertEqual(a ** e, arrF([1, 8, 9]), rtol: 1e-5)
        assertEqual(a ** 2, arrF([1, 4, 9]), rtol: 1e-5)
        assertEqual(2 ** a, arrF([2, 4, 8]), rtol: 1e-5)

        let i = arrI([7, 8, 9])
        let d = arrI([2, 3, 4])
        XCTAssertEqual((i % d).asArray(Int32.self), [1, 2, 1])
        XCTAssertEqual((i % 5).asArray(Int32.self), [2, 3, 4])
        XCTAssertEqual((20 % d).asArray(Int32.self), [0, 2, 0])
        XCTAssertEqual((i % d).dtype, .int32)
    }

    // MARK: - Comparison and logical operators

    func testComparisonOperators() {
        let a = arrI([1, 2, 3])
        let b = arrI([3, 2, 1])

        XCTAssertEqual((a .== b).asArray(Bool.self), [false, true, false])
        XCTAssertEqual((a .== 3).asArray(Bool.self), [false, false, true])
        XCTAssertEqual((a .!= b).asArray(Bool.self), [true, false, true])
        XCTAssertEqual((a .!= 3).asArray(Bool.self), [true, true, false])
        XCTAssertEqual((a .< b).asArray(Bool.self), [true, false, false])
        XCTAssertEqual((a .< 2).asArray(Bool.self), [true, false, false])
        XCTAssertEqual((a .<= b).asArray(Bool.self), [true, true, false])
        XCTAssertEqual((a .<= 2).asArray(Bool.self), [true, true, false])
        XCTAssertEqual((a .> b).asArray(Bool.self), [false, false, true])
        XCTAssertEqual((a .> 2).asArray(Bool.self), [false, false, true])
        XCTAssertEqual((a .>= b).asArray(Bool.self), [false, true, true])
        XCTAssertEqual((a .>= 2).asArray(Bool.self), [false, true, true])
        XCTAssertEqual((a .== b).dtype, .bool)
    }

    func testLogicalOperators() {
        let a = MLXArray([true, true, false, false])
        let b = MLXArray([true, false, true, false])
        XCTAssertEqual((a .&& b).asArray(Bool.self), [true, false, false, false])
        XCTAssertEqual((a .|| b).asArray(Bool.self), [true, true, true, false])
        XCTAssertEqual((.!a).asArray(Bool.self), [false, false, true, true])
    }

    // MARK: - Bitwise operators and functions

    func testBitwiseOperators() {
        let a = arrI([12, 10, 5])
        let b = arrI([10, 6, 3])

        XCTAssertEqual((a & b).asArray(Int32.self), [8, 2, 1])
        XCTAssertEqual((a & 4).asArray(Int32.self), [4, 0, 4])
        XCTAssertEqual((4 & a).asArray(Int32.self), [4, 0, 4])
        XCTAssertEqual((a | b).asArray(Int32.self), [14, 14, 7])
        XCTAssertEqual((a | 1).asArray(Int32.self), [13, 11, 5])
        XCTAssertEqual((1 | a).asArray(Int32.self), [13, 11, 5])
        XCTAssertEqual((a ^ b).asArray(Int32.self), [6, 12, 6])
        XCTAssertEqual((a ^ 1).asArray(Int32.self), [13, 11, 4])
        XCTAssertEqual((1 ^ a).asArray(Int32.self), [13, 11, 4])
        XCTAssertEqual((~a).asArray(Int32.self), [-13, -11, -6])

        let s = arrI([1, 2, 3])
        XCTAssertEqual((s << s).asArray(Int32.self), [2, 8, 24])
        XCTAssertEqual((s << 2).asArray(Int32.self), [4, 8, 12])
        XCTAssertEqual((1 << s).asArray(Int32.self), [2, 4, 8])
        XCTAssertEqual((a >> s).asArray(Int32.self), [6, 2, 0])
        XCTAssertEqual((a >> 1).asArray(Int32.self), [6, 5, 2])
        XCTAssertEqual((64 >> s).asArray(Int32.self), [32, 16, 8])
    }

    func testBitwiseFunctions() {
        let a = arrI([12, 10, 5])
        let b = arrI([10, 6, 3])
        XCTAssertEqual(bitwiseAnd(a, b).asArray(Int32.self), [8, 2, 1])
        XCTAssertEqual(bitwiseOr(a, b).asArray(Int32.self), [14, 14, 7])
        XCTAssertEqual(bitwiseXOr(a, b).asArray(Int32.self), [6, 12, 6])
        XCTAssertEqual(bitwiseAnd(a, 8).asArray(Int32.self), [8, 8, 0])
        XCTAssertEqual(leftShift(a, 1).asArray(Int32.self), [24, 20, 10])
        XCTAssertEqual(rightShift(a, 2).asArray(Int32.self), [3, 2, 1])

        let u = MLXArray([0, 15, 255] as [UInt8])
        let inv = bitwiseInvert(u)
        XCTAssertEqual(inv.dtype, .uint8)
        XCTAssertEqual(inv.asArray(UInt8.self), [255, 240, 0])
    }

    // MARK: - Unary math, free function and method

    func testExpLogFunctions() {
        let v: [Float] = [0.5, 1, 2, 4]
        let a = arrF(v)
        let tol = (rtol: 1e-5, atol: 1e-6)

        assertEqual(exp(a), arrMap(v) { Foundation.exp($0) }, rtol: tol.rtol, atol: tol.atol)
        assertEqual(a.exp(), arrMap(v) { Foundation.exp($0) }, rtol: tol.rtol, atol: tol.atol)
        assertEqual(log(a), arrMap(v) { Foundation.log($0) }, rtol: tol.rtol, atol: tol.atol)
        assertEqual(a.log(), arrMap(v) { Foundation.log($0) }, rtol: tol.rtol, atol: tol.atol)
        assertEqual(log2(a), arrMap(v) { Foundation.log2($0) }, rtol: tol.rtol, atol: tol.atol)
        assertEqual(a.log2(), arrMap(v) { Foundation.log2($0) }, rtol: tol.rtol, atol: tol.atol)
        assertEqual(log10(a), arrMap(v) { Foundation.log10($0) }, rtol: tol.rtol, atol: tol.atol)
        assertEqual(a.log10(), arrMap(v) { Foundation.log10($0) }, rtol: tol.rtol, atol: tol.atol)
        assertEqual(log1p(a), arrMap(v) { Foundation.log1p($0) }, rtol: tol.rtol, atol: tol.atol)
        assertEqual(a.log1p(), arrMap(v) { Foundation.log1p($0) }, rtol: tol.rtol, atol: tol.atol)
    }

    func testRootsAndPowers() {
        let v: [Float] = [1, 4, 9, 16]
        let a = arrF(v)
        XCTAssertEqual(sqrt(a).asArray(Float.self), [1, 2, 3, 4])
        XCTAssertEqual(a.sqrt().asArray(Float.self), [1, 2, 3, 4])
        assertEqual(rsqrt(a), arrF([1, 0.5, 1.0 / 3, 0.25]), rtol: 1e-5, atol: 1e-6)
        assertEqual(a.rsqrt(), arrF([1, 0.5, 1.0 / 3, 0.25]), rtol: 1e-5, atol: 1e-6)
        XCTAssertEqual(square(arrF([1, -2, 3])).asArray(Float.self), [1, 4, 9])
        XCTAssertEqual(arrF([1, -2, 3]).square().asArray(Float.self), [1, 4, 9])
        assertEqual(reciprocal(a), arrF([1, 0.25, 1.0 / 9, 0.0625]), rtol: 1e-5, atol: 1e-6)
        assertEqual(a.reciprocal(), arrF([1, 0.25, 1.0 / 9, 0.0625]), rtol: 1e-5, atol: 1e-6)

        let b = arrF([1, 2, 3])
        let e = arrF([3, 2, 1])
        assertEqual(pow(b, e), arrF([1, 4, 3]), rtol: 1e-5)
        assertEqual(pow(b, 2), arrF([1, 4, 9]), rtol: 1e-5)
        assertEqual(pow(2, b), arrF([2, 4, 8]), rtol: 1e-5)
        assertEqual(b.pow(3), arrF([1, 8, 27]), rtol: 1e-5)
        assertEqual(b.pow(e), arrF([1, 4, 3]), rtol: 1e-5)
    }

    func testTrigonometric() {
        let v: [Float] = [0, 0.5, 1, -2]
        let a = arrF(v)
        assertEqual(sin(a), arrMap(v) { Foundation.sin($0) }, rtol: 1e-5, atol: 1e-6)
        assertEqual(a.sin(), arrMap(v) { Foundation.sin($0) }, rtol: 1e-5, atol: 1e-6)
        assertEqual(cos(a), arrMap(v) { Foundation.cos($0) }, rtol: 1e-5, atol: 1e-6)
        assertEqual(a.cos(), arrMap(v) { Foundation.cos($0) }, rtol: 1e-5, atol: 1e-6)
    }

    func testAbsFloorRound() {
        let a = arrF([-1.6, -0.4, 0.4, 1.6, 2.5])
        XCTAssertEqual(abs(a).asArray(Float.self), [1.6, 0.4, 0.4, 1.6, 2.5])
        XCTAssertEqual(a.abs().asArray(Float.self), [1.6, 0.4, 0.4, 1.6, 2.5])
        XCTAssertEqual(floor(a).asArray(Float.self), [-2, -1, 0, 1, 2])
        XCTAssertEqual(a.floor().asArray(Float.self), [-2, -1, 0, 1, 2])

        let r = arrF([-1.6, -0.4, 0.4, 1.6])
        XCTAssertEqual(round(r).asArray(Float.self), [-2, 0, 0, 2])
        XCTAssertEqual(r.round().asArray(Float.self), [-2, 0, 0, 2])
        let d = arrF([1.24, 2.56])
        assertEqual(round(d, decimals: 1), arrF([1.2, 2.6]), rtol: 1e-5, atol: 1e-5)
        assertEqual(d.round(decimals: 1), arrF([1.2, 2.6]), rtol: 1e-5, atol: 1e-5)
    }

    func testFloorDivide() {
        let a = arrF([7, -7, 9])
        XCTAssertEqual(floorDivide(a, 2).asArray(Float.self), [3, -4, 4])
        XCTAssertEqual(a.floorDivide(arrF([2, 2, 4])).asArray(Float.self), [3, -4, 2])
        let i = arrI([7, 9])
        let q = floorDivide(i, 2)
        XCTAssertEqual(q.dtype, .int32)
        XCTAssertEqual(q.asArray(Int32.self), [3, 4])
        XCTAssertEqual(i.floorDivide(4).asArray(Int32.self), [1, 2])
    }

    func testConjugate() {
        let c = MLXArray(real: 1, imaginary: 2)
        let f = conjugate(c)
        XCTAssertEqual(f.dtype, .complex64)
        XCTAssertEqual(f.realPart().item(Float.self), 1)
        XCTAssertEqual(f.imaginaryPart().item(Float.self), -2)
        let m = c.conjugate()
        XCTAssertEqual(m.imaginaryPart().item(Float.self), -2)
    }

    // MARK: - Reductions

    func testAllAny() {
        let a = MLXArray([true, true, false, true], [2, 2])

        XCTAssertEqual(all(a).item(Bool.self), false)
        XCTAssertEqual(a.all().item(Bool.self), false)
        XCTAssertEqual(all(a, keepDims: true).shape, [1, 1])
        XCTAssertEqual(all(a, axis: 1).asArray(Bool.self), [true, false])
        XCTAssertEqual(a.all(axis: 0).asArray(Bool.self), [false, true])
        XCTAssertEqual(all(a, axes: [0, 1]).item(Bool.self), false)
        XCTAssertEqual(a.all(axes: [1], keepDims: true).shape, [2, 1])

        XCTAssertEqual(any(a).item(Bool.self), true)
        XCTAssertEqual(a.any().item(Bool.self), true)
        let n = MLXArray([false, false, false, true], [2, 2])
        XCTAssertEqual(any(n, axis: 1).asArray(Bool.self), [false, true])
        XCTAssertEqual(n.any(axis: 0).asArray(Bool.self), [false, true])
        XCTAssertEqual(any(n, axes: [0]).asArray(Bool.self), [false, true])
        XCTAssertEqual(n.any(axes: [0, 1]).item(Bool.self), true)
        XCTAssertEqual(any(n, keepDims: true).shape, [1, 1])
    }

    func testSumProductMean() {
        let a = arrF([1, 2, 3, 4, 5, 6], [2, 3])

        XCTAssertEqual(sum(a).item(Float.self), 21)
        XCTAssertEqual(a.sum().item(Float.self), 21)
        XCTAssertEqual(sum(a, axis: 0).asArray(Float.self), [5, 7, 9])
        XCTAssertEqual(a.sum(axis: 1).asArray(Float.self), [6, 15])
        XCTAssertEqual(sum(a, axes: [0, 1], keepDims: true).shape, [1, 1])
        XCTAssertEqual(a.sum(axes: [1]).asArray(Float.self), [6, 15])

        XCTAssertEqual(product(a).item(Float.self), 720)
        XCTAssertEqual(a.product().item(Float.self), 720)
        XCTAssertEqual(product(a, axis: 0).asArray(Float.self), [4, 10, 18])
        XCTAssertEqual(a.product(axis: 1).asArray(Float.self), [6, 120])
        XCTAssertEqual(product(a, axes: [1]).asArray(Float.self), [6, 120])
        XCTAssertEqual(a.product(axes: [0, 1], keepDims: true).shape, [1, 1])

        XCTAssertEqual(mean(a).item(Float.self), 3.5)
        XCTAssertEqual(a.mean().item(Float.self), 3.5)
        XCTAssertEqual(mean(a, axis: 0).asArray(Float.self), [2.5, 3.5, 4.5])
        XCTAssertEqual(a.mean(axis: 1).asArray(Float.self), [2, 5])
        XCTAssertEqual(mean(a, axes: [1], keepDims: true).shape, [2, 1])
        XCTAssertEqual(a.mean(axes: [0, 1]).item(Float.self), 3.5)
    }

    func testMaxMinArgMaxArgMin() {
        let a = arrF([3, 9, 1, 7, 2, 8], [2, 3])

        XCTAssertEqual(max(a).item(Float.self), 9)
        XCTAssertEqual(a.max().item(Float.self), 9)
        XCTAssertEqual(max(a, axis: 0).asArray(Float.self), [7, 9, 8])
        XCTAssertEqual(a.max(axis: 1).asArray(Float.self), [9, 8])
        XCTAssertEqual(max(a, axes: [1], keepDims: true).shape, [2, 1])
        XCTAssertEqual(a.max(axes: [0, 1]).item(Float.self), 9)

        XCTAssertEqual(min(a).item(Float.self), 1)
        XCTAssertEqual(a.min().item(Float.self), 1)
        XCTAssertEqual(min(a, axis: 0).asArray(Float.self), [3, 2, 1])
        XCTAssertEqual(a.min(axis: 1).asArray(Float.self), [1, 2])
        XCTAssertEqual(min(a, axes: [0, 1], keepDims: true).shape, [1, 1])
        XCTAssertEqual(a.min(axes: [1]).asArray(Float.self), [1, 2])

        let mx = argMax(a)
        XCTAssertEqual(mx.dtype, .uint32)
        XCTAssertEqual(mx.item(UInt32.self), 1)
        XCTAssertEqual(a.argMax().item(UInt32.self), 1)
        XCTAssertEqual(argMax(a, axis: 1).asArray(UInt32.self), [1, 2])
        XCTAssertEqual(a.argMax(axis: 0, keepDims: true).asArray(UInt32.self), [1, 0, 1])
        XCTAssertEqual(a.argMax(axis: 0, keepDims: true).shape, [1, 3])

        XCTAssertEqual(argMin(a).item(UInt32.self), 2)
        XCTAssertEqual(a.argMin().item(UInt32.self), 2)
        XCTAssertEqual(argMin(a, axis: 1).asArray(UInt32.self), [2, 1])
        XCTAssertEqual(a.argMin(axis: 0).asArray(UInt32.self), [0, 1, 0])
    }

    func testLogSumExp() {
        let v: [Float] = [0, 1, 2, 3]
        let a = arrF(v, [2, 2])
        func lse(_ x: [Float]) -> Float {
            Foundation.log(x.map { Foundation.exp($0) }.reduce(0, +))
        }
        let all = lse(v)
        XCTAssertEqual(logSumExp(a).item(Float.self), all, accuracy: 1e-5)
        XCTAssertEqual(a.logSumExp().item(Float.self), all, accuracy: 1e-5)
        XCTAssertEqual(logSumExp(a, keepDims: true).shape, [1, 1])

        let rows = arrF([lse([0, 1]), lse([2, 3])])
        assertEqual(logSumExp(a, axis: 1), rows, rtol: 1e-5, atol: 1e-6)
        assertEqual(a.logSumExp(axis: 1), rows, rtol: 1e-5, atol: 1e-6)
        assertEqual(logSumExp(a, axes: [1]), rows, rtol: 1e-5, atol: 1e-6)
        assertEqual(a.logSumExp(axes: [0, 1]), MLXArray(all), rtol: 1e-5, atol: 1e-6)
    }

    func testVariance() {
        let a = arrF([1, 2, 3, 4, 2, 2, 2, 2], [2, 4])
        // row 0: mean 2.5, population variance 1.25, sample variance 5/3
        assertEqual(variance(a, axis: 1), arrF([1.25, 0]), rtol: 1e-5, atol: 1e-6)
        assertEqual(a.variance(axis: 1, ddof: 1), arrF([5.0 / 3.0, 0]), rtol: 1e-5, atol: 1e-6)
        XCTAssertEqual(variance(a, axis: 1, keepDims: true).shape, [2, 1])
        // all 8 values: mean 2.25, sum of squared deviations 5.5
        assertEqual(variance(a), MLXArray(Float(5.5 / 8)), rtol: 1e-5, atol: 1e-6)
        assertEqual(a.variance(), MLXArray(Float(5.5 / 8)), rtol: 1e-5, atol: 1e-6)
        assertEqual(
            variance(a, axes: [0, 1], ddof: 1), MLXArray(Float(0.5)), rtol: 1e-5, atol: 1e-6)
        assertEqual(a.variance(axes: [1]), arrF([1.25, 0]), rtol: 1e-5, atol: 1e-6)
        XCTAssertEqual(variance(a, keepDims: true).shape, [1, 1])
    }

    func testAllCloseAndArrayEqual() {
        let a = arrF([1, 2, 3])
        let b = arrF([1, 2, 3.0001])
        XCTAssertFalse(allClose(a, b).item(Bool.self))
        XCTAssertTrue(allClose(a, b, atol: 1e-3).item(Bool.self))
        XCTAssertTrue(a.allClose(b, rtol: 1e-3).item(Bool.self))
        XCTAssertTrue(allClose(arrF([2, 2]), 2).item(Bool.self))
        let n = arrF([1, Float.nan])
        XCTAssertFalse(allClose(n, n).item(Bool.self))
        XCTAssertTrue(n.allClose(n, equalNaN: true).item(Bool.self))

        XCTAssertTrue(arrayEqual(a, arrF([1, 2, 3])).item(Bool.self))
        XCTAssertFalse(arrayEqual(a, b).item(Bool.self))
        XCTAssertFalse(arrayEqual(a, arrF([1, 2])).item(Bool.self))
        XCTAssertFalse(a.arrayEqual(b).item(Bool.self))
        XCTAssertFalse(arrayEqual(n, n).item(Bool.self))
        XCTAssertTrue(n.arrayEqual(n, equalNAN: true).item(Bool.self))
    }

    // MARK: - Cumulative

    func testCumulativeFunctions() {
        let a = arrF([1, 3, 2, 4], [2, 2])

        XCTAssertEqual(cumsum(a).asArray(Float.self), [1, 4, 6, 10])
        XCTAssertEqual(a.cumsum().asArray(Float.self), [1, 4, 6, 10])
        XCTAssertEqual(cumsum(a, axis: 0).asArray(Float.self), [1, 3, 3, 7])
        XCTAssertEqual(a.cumsum(axis: 1).asArray(Float.self), [1, 4, 2, 6])
        XCTAssertEqual(cumsum(a, axis: 1, reverse: true).asArray(Float.self), [4, 3, 6, 4])
        XCTAssertEqual(cumsum(a, axis: 1, inclusive: false).asArray(Float.self), [0, 1, 0, 2])
        XCTAssertEqual(a.cumsum(reverse: true).asArray(Float.self), [10, 9, 6, 4])

        XCTAssertEqual(cumprod(a).asArray(Float.self), [1, 3, 6, 24])
        XCTAssertEqual(a.cumprod().asArray(Float.self), [1, 3, 6, 24])
        XCTAssertEqual(cumprod(a, axis: 0).asArray(Float.self), [1, 3, 2, 12])
        XCTAssertEqual(a.cumprod(axis: 1).asArray(Float.self), [1, 3, 2, 8])

        XCTAssertEqual(cummax(a).asArray(Float.self), [1, 3, 3, 4])
        XCTAssertEqual(a.cummax().asArray(Float.self), [1, 3, 3, 4])
        XCTAssertEqual(cummax(a, axis: 0).asArray(Float.self), [1, 3, 2, 4])
        XCTAssertEqual(a.cummax(axis: 1, reverse: true).asArray(Float.self), [3, 3, 4, 4])

        XCTAssertEqual(cummin(a).asArray(Float.self), [1, 1, 1, 1])
        XCTAssertEqual(a.cummin().asArray(Float.self), [1, 1, 1, 1])
        XCTAssertEqual(cummin(a, axis: 0).asArray(Float.self), [1, 3, 1, 3])
        XCTAssertEqual(a.cummin(axis: 1).asArray(Float.self), [1, 1, 2, 2])

        // log(cumsum(exp(x)))
        let z = arrF([0, 0, 0])
        let lc = arrF([0, Foundation.log(Float(2)), Foundation.log(Float(3))])
        assertEqual(logCumsumExp(z), lc, rtol: 1e-5, atol: 1e-6)
        assertEqual(z.logCumsumExp(), lc, rtol: 1e-5, atol: 1e-6)
        assertEqual(logCumsumExp(z, axis: 0), lc, rtol: 1e-5, atol: 1e-6)
        assertEqual(z.logCumsumExp(axis: 0), lc, rtol: 1e-5, atol: 1e-6)
        let lr = arrF([Foundation.log(Float(3)), Foundation.log(Float(2)), 0])
        assertEqual(logCumsumExp(z, reverse: true), lr, rtol: 1e-5, atol: 1e-6)
    }

    // MARK: - Shapes

    func testReshapeFlattenSqueeze() {
        let a = MLXArray(0 ..< 6)
        XCTAssertEqual(reshaped(a, [2, 3]).shape, [2, 3])
        XCTAssertEqual(reshaped(a, 3, 2).shape, [3, 2])
        XCTAssertEqual(a.reshaped([3, -1]).shape, [3, 2])
        XCTAssertEqual(a.reshaped(1, 6).asArray(Int32.self), [0, 1, 2, 3, 4, 5])

        let b = MLXArray(0 ..< 24, [2, 3, 4])
        XCTAssertEqual(flattened(b).shape, [24])
        XCTAssertEqual(flattened(b, start: 1).shape, [2, 12])
        XCTAssertEqual(b.flattened(end: 1).shape, [6, 4])
        XCTAssertEqual(b.flattened().asArray(Int32.self), Array(0 ..< 24).map { Int32($0) })

        let c = MLXArray(0 ..< 3, [1, 3, 1])
        XCTAssertEqual(squeezed(c).shape, [3])
        XCTAssertEqual(c.squeezed().shape, [3])
        XCTAssertEqual(squeezed(c, axis: 0).shape, [3, 1])
        XCTAssertEqual(c.squeezed(axis: 2).shape, [1, 3])
        XCTAssertEqual(squeezed(c, axes: [0, 2]).shape, [3])
        XCTAssertEqual(c.squeezed(axes: [0]).shape, [3, 1])

        XCTAssertEqual(a.expandedDimensions(axis: 0).shape, [1, 6])
        XCTAssertEqual(a.expandedDimensions(axes: [0, 2]).shape, [1, 6, 1])
    }

    func testTransposes() {
        let a = MLXArray(0 ..< 6, [2, 3])
        let expectedT: [Int32] = [0, 3, 1, 4, 2, 5]
        XCTAssertEqual(transposed(a).shape, [3, 2])
        XCTAssertEqual(transposed(a).asArray(Int32.self), expectedT)
        XCTAssertEqual(a.transposed().asArray(Int32.self), expectedT)
        XCTAssertEqual(T(a).asArray(Int32.self), expectedT)
        XCTAssertEqual(a.T.asArray(Int32.self), expectedT)
        XCTAssertEqual(transposed(a, 1, 0).asArray(Int32.self), expectedT)
        XCTAssertEqual(a.transposed(1, 0).asArray(Int32.self), expectedT)
        XCTAssertEqual(transposed(a, axes: [1, 0]).asArray(Int32.self), expectedT)
        XCTAssertEqual(a.transposed(axes: [1, 0]).asArray(Int32.self), expectedT)

        let v = MLXArray(0 ..< 3)
        XCTAssertEqual(transposed(v, axis: 0).asArray(Int32.self), [0, 1, 2])
        XCTAssertEqual(v.transposed(axis: 0).shape, [3])

        let b = MLXArray(0 ..< 24, [2, 3, 4])
        XCTAssertEqual(swappedAxes(b, 0, 2).shape, [4, 3, 2])
        XCTAssertEqual(b.swappedAxes(1, 2).shape, [2, 4, 3])
        XCTAssertEqual(movedAxis(b, source: 0, destination: 2).shape, [3, 4, 2])
        XCTAssertEqual(b.movedAxis(source: 2, destination: 0).shape, [4, 2, 3])
        // element [i, j, k] of b is i * 12 + j * 4 + k
        let m = movedAxis(b, source: 0, destination: 2)
        XCTAssertEqual(m[1, 2, 1].item(Int32.self), 1 * 12 + 1 * 4 + 2)
        XCTAssertEqual(b.swappedAxes(0, 2)[3, 1, 1].item(Int32.self), 1 * 12 + 1 * 4 + 3)
    }

    func testSplit() {
        let a = MLXArray(0 ..< 6)

        let p = split(a, parts: 3)
        XCTAssertEqual(p.count, 3)
        XCTAssertEqual(p[1].asArray(Int32.self), [2, 3])
        let pm = a.split(parts: 2)
        XCTAssertEqual(pm[1].asArray(Int32.self), [3, 4, 5])

        let (l, r) = split(a)
        XCTAssertEqual(l.asArray(Int32.self), [0, 1, 2])
        XCTAssertEqual(r.asArray(Int32.self), [3, 4, 5])
        let m = MLXArray(0 ..< 6, [2, 3])
        let (top, bottom) = m.split(axis: 0)
        XCTAssertEqual(top.asArray(Int32.self), [0, 1, 2])
        XCTAssertEqual(bottom.asArray(Int32.self), [3, 4, 5])

        let s = split(a, indices: [1, 4])
        XCTAssertEqual(s.count, 3)
        XCTAssertEqual(s[0].asArray(Int32.self), [0])
        XCTAssertEqual(s[1].asArray(Int32.self), [1, 2, 3])
        XCTAssertEqual(s[2].asArray(Int32.self), [4, 5])
        let sm = m.split(indices: [2], axis: 1)
        XCTAssertEqual(sm[0].shape, [2, 2])
        XCTAssertEqual(sm[1].asArray(Int32.self), [2, 5])
    }

    func testTake() {
        let a = MLXArray(0 ..< 6, [2, 3])
        let idx = arrI([2, 0])
        XCTAssertEqual(take(a, idx, axis: 1).asArray(Int32.self), [2, 0, 5, 3])
        XCTAssertEqual(a.take(idx, axis: 0).shape, [2, 3])
        XCTAssertEqual(a.take(arrI([1]), axis: 0).asArray(Int32.self), [3, 4, 5])
        // no axis: index the flattened array
        XCTAssertEqual(take(a, arrI([5, 1])).asArray(Int32.self), [5, 1])
        XCTAssertEqual(a.take(arrI([4])).asArray(Int32.self), [4])
    }

    func testDiagAndDiagonal() {
        let v = arrI([1, 2])
        let d = diag(v)
        XCTAssertEqual(d.shape, [2, 2])
        XCTAssertEqual(d.asArray(Int32.self), [1, 0, 0, 2])
        let d1 = v.diag(k: 1)
        XCTAssertEqual(d1.shape, [3, 3])
        XCTAssertEqual(d1.asArray(Int32.self), [0, 1, 0, 0, 0, 2, 0, 0, 0])

        let m = MLXArray(0 ..< 9, [3, 3])
        XCTAssertEqual(diag(m).asArray(Int32.self), [0, 4, 8])
        XCTAssertEqual(m.diag(k: -1).asArray(Int32.self), [3, 7])
        XCTAssertEqual(diagonal(m).asArray(Int32.self), [0, 4, 8])
        XCTAssertEqual(diagonal(m, offset: 1).asArray(Int32.self), [1, 5])
        XCTAssertEqual(m.diagonal(offset: -1).asArray(Int32.self), [3, 7])
        XCTAssertEqual(m.diagonal(axis1: 1, axis2: 0).asArray(Int32.self), [0, 4, 8])
    }

    func testMatmulViewContiguous() {
        let a = arrF([1, 2, 3, 4], [2, 2])
        let b = arrF([5, 6, 7, 8], [2, 2])
        XCTAssertEqual(matmul(a, b).asArray(Float.self), [19, 22, 43, 50])
        XCTAssertEqual(a.matmul(b).asArray(Float.self), [19, 22, 43, 50])

        // 1.0 as float32 has the bit pattern 0x3F800000
        let one = arrF([1, 2])
        let bits = view(one, dtype: .int32)
        XCTAssertEqual(bits.dtype, .int32)
        XCTAssertEqual(bits.asArray(Int32.self), [0x3F80_0000, 0x4000_0000])
        let back = bits.view(dtype: .float32)
        XCTAssertEqual(back.asArray(Float.self), [1, 2])

        let t = MLXArray(0 ..< 6, [2, 3]).transposed()
        let c = t.contiguous()
        XCTAssertEqual(c.shape, [3, 2])
        XCTAssertEqual(c.asArray(Int32.self), [0, 3, 1, 4, 2, 5])
        XCTAssertEqual(t.contiguous(allowColMajor: true).asArray(Int32.self), [0, 3, 1, 4, 2, 5])
    }
}
