// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

// Tests for the free functions in Source/MLX/Ops.swift. Each test uses a small
// array and compares the result with values that are computed by hand or by a
// plain Swift loop.

/// Make a float32 array with the given values and shape.
private func opsF(_ values: [Float], _ shape: [Int]? = nil) -> MLXArray {
    MLXArray(values, shape ?? [values.count])
}

/// Make an int32 array with the given values and shape.
private func opsI(_ values: [Int32], _ shape: [Int]? = nil) -> MLXArray {
    MLXArray(values, shape ?? [values.count])
}

/// Apply `f` to each value and make a float32 array with the results.
private func opsMap(_ values: [Float], _ f: (Float) -> Float) -> MLXArray {
    opsF(values.map(f))
}

/// Full 1D convolution, as numpy.convolve(a, b, mode: "full") computes it.
private func opsFullConvolve(_ a: [Float], _ b: [Float]) -> [Float] {
    var out = [Float](repeating: 0, count: a.count + b.count - 1)
    for i in 0 ..< a.count {
        for j in 0 ..< b.count {
            out[i + j] += a[i] * b[j]
        }
    }
    return out
}

/// Single channel 2D cross-correlation with zero padding and a stride.
private func opsCorrelate2D(
    _ x: [[Float]], _ w: [[Float]], padding: Int = 0, stride: Int = 1
) -> [[Float]] {
    let h = x.count + 2 * padding
    let wd = x[0].count + 2 * padding
    var padded = [[Float]](repeating: [Float](repeating: 0, count: wd), count: h)
    for i in 0 ..< x.count {
        for j in 0 ..< x[0].count {
            padded[i + padding][j + padding] = x[i][j]
        }
    }
    let oh = (h - w.count) / stride + 1
    let ow = (wd - w[0].count) / stride + 1
    var out = [[Float]](repeating: [Float](repeating: 0, count: ow), count: oh)
    for i in 0 ..< oh {
        for j in 0 ..< ow {
            var s: Float = 0
            for ki in 0 ..< w.count {
                for kj in 0 ..< w[0].count {
                    s += padded[i * stride + ki][j * stride + kj] * w[ki][kj]
                }
            }
            out[i][j] = s
        }
    }
    return out
}

class OpsCoverageTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    // MARK: - Arithmetic

    func testAddSubtractMultiplyDivide() {
        // XCTestCase has an add(_:) method, so the MLX free function needs the module name.
        let a = opsF([1, 2, 3])
        let b = opsF([4, 5, 6])

        let sum = MLX.add(a, b)
        XCTAssertEqual(sum.dtype, .float32)
        XCTAssertEqual(sum.asArray(Float.self), [5, 7, 9])

        // both scalars: the result is an int32 array
        let scalarSum = MLX.add(1, 2)
        XCTAssertEqual(scalarSum.dtype, .int32)
        XCTAssertEqual(scalarSum.item(Int32.self), 3)

        XCTAssertEqual(subtract(a, b).asArray(Float.self), [-3, -3, -3])
        XCTAssertEqual(subtract(10, a).asArray(Float.self), [9, 8, 7])
        XCTAssertEqual(multiply(a, b).asArray(Float.self), [4, 10, 18])
        XCTAssertEqual(multiply(a, 2).asArray(Float.self), [2, 4, 6])
        XCTAssertEqual(divide(b, a).asArray(Float.self), [4, 2.5, 2])

        // integer division gives a float result
        let q = divide(opsI([7, 9]), 2)
        XCTAssertEqual(q.dtype, .float32)
        XCTAssertEqual(q.asArray(Float.self), [3.5, 4.5])

        XCTAssertEqual(negative(a).asArray(Float.self), [-1, -2, -3])
    }

    func testDivmodAndRemainder() {
        let a = opsI([7, -7, 9])
        let (q, r) = divmod(a, 3)
        XCTAssertEqual(q.shape, [3])
        XCTAssertEqual(r.shape, [3])
        // floor division and a remainder with the sign of the divisor
        XCTAssertEqual(q.asArray(Int32.self), [2, -3, 3])
        XCTAssertEqual(r.asArray(Int32.self), [1, 2, 0])

        XCTAssertEqual(remainder(a, 4).asArray(Int32.self), [3, 1, 1])
        XCTAssertEqual(remainder(opsF([5.5, 7]), opsF([2, 3])).asArray(Float.self), [1.5, 1])
    }

    func testAddMM() {
        // c + alpha * (a @ b) with beta = 1, then with alpha and beta
        let a = opsF([1, 2, 3, 4], [2, 2])
        let b = opsF([5, 6, 7, 8], [2, 2])
        let c = opsF([1, 1, 1, 1], [2, 2])
        // a @ b = [[19, 22], [43, 50]]
        let r1 = addMM(c, a, b)
        XCTAssertEqual(r1.shape, [2, 2])
        XCTAssertEqual(r1.asArray(Float.self), [20, 23, 44, 51])

        let r2 = addMM(c, a, b, alpha: 2, beta: 0.5)
        XCTAssertEqual(r2.asArray(Float.self), [38.5, 44.5, 86.5, 100.5])

        let r3 = addmm(c, a, b, alpha: 1, beta: 3)
        XCTAssertEqual(r3.asArray(Float.self), [22, 25, 46, 53])
    }

    func testMaximumMinimum() {
        let a = opsF([1, 5, 3])
        let b = opsF([4, 2, 3])
        XCTAssertEqual(maximum(a, b).asArray(Float.self), [4, 5, 3])
        XCTAssertEqual(minimum(a, b).asArray(Float.self), [1, 2, 3])
        XCTAssertEqual(maximum(a, 2).asArray(Float.self), [2, 5, 3])
        XCTAssertEqual(minimum(2, a).asArray(Float.self), [1, 2, 2])
    }

    func testClip() {
        let a = opsF([-2, 0, 3, 8])
        XCTAssertEqual(clip(a, min: 0).asArray(Float.self), [0, 0, 3, 8])
        XCTAssertEqual(clip(a, max: 3).asArray(Float.self), [-2, 0, 3, 3])
        XCTAssertEqual(clip(a, min: -1, max: 5).asArray(Float.self), [-1, 0, 3, 5])
        let lo = opsF([0, 1, 4, 0])
        let hi = opsF([1, 2, 5, 6])
        XCTAssertEqual(clip(a, min: lo, max: hi).asArray(Float.self), [0, 1, 4, 6])
    }

    func testCeilSignNanToNum() {
        let a = opsF([-1.5, -0.5, 0, 0.5, 2.1])
        XCTAssertEqual(ceil(a).asArray(Float.self), [-1, 0, 0, 1, 3])
        XCTAssertEqual(sign(a).asArray(Float.self), [-1, -1, 0, 1, 1])

        let special = opsF([Float.nan, Float.infinity, -Float.infinity, 1])
        let replaced = nanToNum(special, nan: 7, posInf: 100, negInf: -100)
        XCTAssertEqual(replaced.dtype, .float32)
        XCTAssertEqual(replaced.asArray(Float.self), [7, 100, -100, 1])

        // nil asks for the largest finite value of the dtype
        let largest = nanToNum(special, posInf: nil, negInf: nil)
        XCTAssertEqual(
            largest.asArray(Float.self),
            [0, Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, 1])
    }

    func testNanToNumDefaults() {
        // The doc comment of nanToNum (Source/MLX/Ops.swift:2092) says that if posInf and
        // negInf are not given, the largest finite value is used. Python MLX
        // (nan_to_num, posinf=None) does the same. The Swift defaults are 0, so a call
        // without them replaces +inf and -inf with 0.
        let special = opsF([Float.nan, Float.infinity, -Float.infinity, 1])
        let replaced = nanToNum(special)
        XCTAssertEqual(replaced.shape, [4])
        XCTAssertEqual(replaced[0].item(Float.self), 0)
        XCTAssertEqual(replaced[3].item(Float.self), 1)
        XCTExpectFailure(
            "nanToNum defaults posInf and negInf to 0, not to the largest finite value (Source/MLX/Ops.swift:2102)"
        ) {
            XCTAssertEqual(replaced[1].item(Float.self), Float.greatestFiniteMagnitude)
            XCTAssertEqual(replaced[2].item(Float.self), -Float.greatestFiniteMagnitude)
        }
    }

    // MARK: - Trigonometric and other unary functions

    func testInverseTrigonometric() {
        let unit: [Float] = [-0.5, 0, 0.5, 0.9]
        assertEqual(acos(opsF(unit)), opsMap(unit) { Foundation.acos($0) }, rtol: 1e-4, atol: 1e-5)
        assertEqual(asin(opsF(unit)), opsMap(unit) { Foundation.asin($0) }, rtol: 1e-4, atol: 1e-5)
        assertEqual(atan(opsF(unit)), opsMap(unit) { Foundation.atan($0) }, rtol: 1e-4, atol: 1e-5)
        assertEqual(
            atanh(opsF(unit)), opsMap(unit) { Foundation.atanh($0) }, rtol: 1e-4, atol: 1e-5)

        let big: [Float] = [1, 1.5, 3]
        assertEqual(acosh(opsF(big)), opsMap(big) { Foundation.acosh($0) }, rtol: 1e-4, atol: 1e-5)
        assertEqual(asinh(opsF(big)), opsMap(big) { Foundation.asinh($0) }, rtol: 1e-4, atol: 1e-5)

        let y = opsF([1, 1, -1, 0])
        let x = opsF([1, -1, -1, 2])
        let expected = opsF([
            Float.pi / 4, 3 * Float.pi / 4, -3 * Float.pi / 4, 0,
        ])
        assertEqual(atan2(y, x), expected, rtol: 1e-4, atol: 1e-5)
    }

    func testHyperbolicAndTangent() {
        let v: [Float] = [-1, 0, 0.5, 2]
        assertEqual(cosh(opsF(v)), opsMap(v) { Foundation.cosh($0) }, rtol: 1e-4, atol: 1e-5)
        assertEqual(sinh(opsF(v)), opsMap(v) { Foundation.sinh($0) }, rtol: 1e-4, atol: 1e-5)
        assertEqual(tanh(opsF(v)), opsMap(v) { Foundation.tanh($0) }, rtol: 1e-4, atol: 1e-5)
        assertEqual(tan(opsF(v)), opsMap(v) { Foundation.tan($0) }, rtol: 1e-4, atol: 1e-5)
        assertEqual(
            sigmoid(opsF(v)), opsMap(v) { 1 / (1 + Foundation.exp(-$0)) }, rtol: 1e-4, atol: 1e-5)
        assertEqual(expm1(opsF(v)), opsMap(v) { Foundation.expm1($0) }, rtol: 1e-4, atol: 1e-6)
        assertEqual(erf(opsF(v)), opsMap(v) { Foundation.erf($0) }, rtol: 1e-4, atol: 1e-5)
    }

    func testErfInverse() {
        let a = opsF([-0.5, 0, 0.5])
        let r = erfInverse(a)
        XCTAssertEqual(r.dtype, .float32)
        assertEqual(r, opsF([-0.4769363, 0, 0.4769363]), rtol: 1e-4, atol: 1e-5)
        // erf(erfinv(x)) == x
        assertEqual(erf(r), a, rtol: 1e-4, atol: 1e-5)
    }

    func testDegreesRadians() {
        let deg = opsF([0, 90, 180, -45])
        let rad = opsF([0, Float.pi / 2, Float.pi, -Float.pi / 4])
        assertEqual(radians(deg), rad, rtol: 1e-5, atol: 1e-6)
        assertEqual(degrees(rad), deg, rtol: 1e-5, atol: 1e-4)
    }

    func testLogAddExp() {
        let av: [Float] = [0, 1, -2]
        let bv: [Float] = [0, 2, 3]
        let expected = zip(av, bv).map { (x: Float, y: Float) -> Float in
            Foundation.log(Foundation.exp(x) + Foundation.exp(y))
        }
        assertEqual(logAddExp(opsF(av), opsF(bv)), opsF(expected), rtol: 1e-5, atol: 1e-6)
        // a scalar on one side
        XCTAssertEqual(logAddExp(opsF([0]), 0).item(Float.self), Foundation.log(2), accuracy: 1e-6)
    }

    func testIsChecks() {
        let a = opsF([1, Float.nan, Float.infinity, -Float.infinity])
        XCTAssertEqual(isNaN(a).asArray(Bool.self), [false, true, false, false])
        XCTAssertEqual(isInf(a).asArray(Bool.self), [false, false, true, true])
        XCTAssertEqual(isFinite(a).asArray(Bool.self), [true, false, false, false])
        XCTAssertEqual(isPosInf(a).asArray(Bool.self), [false, false, true, false])
        XCTAssertEqual(isNegInf(a).asArray(Bool.self), [false, false, false, true])
        XCTAssertEqual(isNaN(a).dtype, .bool)

        let x = opsF([1, 2, 3])
        let y = opsF([1.000001, 2.1, 3])
        XCTAssertEqual(isClose(x, y).asArray(Bool.self), [true, false, true])
        XCTAssertEqual(isClose(x, y, rtol: 0.1).asArray(Bool.self), [true, true, true])
        let n = opsF([Float.nan])
        XCTAssertEqual(isClose(n, n).asArray(Bool.self), [false])
        XCTAssertEqual(isClose(n, n, equalNaN: true).asArray(Bool.self), [true])
    }

    // MARK: - Comparison and logic

    func testComparisonFunctions() {
        let a = opsI([1, 2, 3])
        XCTAssertEqual(equal(a, 2).asArray(Bool.self), [false, true, false])
        XCTAssertEqual(notEqual(a, 2).asArray(Bool.self), [true, false, true])
        XCTAssertEqual(greater(a, 2).asArray(Bool.self), [false, false, true])
        XCTAssertEqual(greaterEqual(a, 2).asArray(Bool.self), [false, true, true])
        XCTAssertEqual(less(a, 2).asArray(Bool.self), [true, false, false])
        XCTAssertEqual(lessEqual(a, 2).asArray(Bool.self), [true, true, false])
        XCTAssertEqual(equal(2, a).dtype, .bool)
    }

    func testLogicalFunctions() {
        let a = MLXArray([true, true, false, false])
        let b = MLXArray([true, false, true, false])
        XCTAssertEqual(logicalAnd(a, b).asArray(Bool.self), [true, false, false, false])
        XCTAssertEqual(logicalOr(a, b).asArray(Bool.self), [true, true, true, false])
        XCTAssertEqual(logicalNot(a).asArray(Bool.self), [false, false, true, true])
    }

    func testWhereAndWhich() {
        let cond = MLXArray([true, false, true])
        let a = opsI([1, 2, 3])
        let b = opsI([10, 20, 30])
        let r1 = `where`(cond, a, b)
        XCTAssertEqual(r1.dtype, .int32)
        XCTAssertEqual(r1.asArray(Int32.self), [1, 20, 3])
        let r2 = which(cond, a, 0)
        XCTAssertEqual(r2.asArray(Int32.self), [1, 0, 3])
    }

    // MARK: - Shapes

    func testAtLeast() {
        let s = MLXArray(Float(5))
        XCTAssertEqual(atLeast1D(s).shape, [1])
        XCTAssertEqual(atLeast2D(s).shape, [1, 1])
        XCTAssertEqual(atLeast3D(s).shape, [1, 1, 1])

        let v = opsF([1, 2, 3])
        XCTAssertEqual(atLeast1D(v).shape, [3])
        XCTAssertEqual(atLeast2D(v).shape, [1, 3])
        XCTAssertEqual(atLeast3D(v).shape, [1, 3, 1])
        XCTAssertEqual(atLeast3D(v).asArray(Float.self), [1, 2, 3])
    }

    func testConcatenatedAndStacked() {
        let a = opsF([1, 2, 3, 4], [2, 2])
        let b = opsF([5, 6], [2, 1])
        let c = concatenated([a, b], axis: 1)
        XCTAssertEqual(c.shape, [2, 3])
        XCTAssertEqual(c.asArray(Float.self), [1, 2, 5, 3, 4, 6])

        let d = concatenated([opsF([1]), opsF([2, 3])])
        XCTAssertEqual(d.asArray(Float.self), [1, 2, 3])

        let s = stacked([opsF([1, 2]), opsF([3, 4])], axis: 1)
        XCTAssertEqual(s.shape, [2, 2])
        XCTAssertEqual(s.asArray(Float.self), [1, 3, 2, 4])
    }

    func testExpandedDimensions() {
        let a = opsF([1, 2])
        XCTAssertEqual(expandedDimensions(a, axis: 0).shape, [1, 2])
        XCTAssertEqual(expandedDimensions(a, axis: -1).shape, [2, 1])
        let b = expandedDimensions(a, axes: [0, 2])
        XCTAssertEqual(b.shape, [1, 2, 1])
        XCTAssertEqual(b.asArray(Float.self), [1, 2])
    }

    func testFlattenUnflatten() {
        let a = MLXArray(0 ..< 24, [2, 3, 4])
        let f = flatten(a, startAxis: 1)
        XCTAssertEqual(f.shape, [2, 12])
        XCTAssertEqual(flatten(a, startAxis: 0, endAxis: 1).shape, [6, 4])
        let u = unflatten(f, axis: 1, shape: [3, 4])
        XCTAssertEqual(u.shape, [2, 3, 4])
        XCTAssertEqual(u.asArray(Int32.self), Array(0 ..< 24).map { Int32($0) })
    }

    func testPadded() {
        let a = opsF([1, 2])
        let p1 = padded(a, width: 1)
        XCTAssertEqual(p1.asArray(Float.self), [0, 1, 2, 0])

        let p2 = padded(a, width: [1, 2], value: MLXArray(Float(9)))
        XCTAssertEqual(p2.asArray(Float.self), [9, 1, 2, 9, 9])

        let p3 = padded(a, width: [2, 1], mode: .edge)
        XCTAssertEqual(p3.asArray(Float.self), [1, 1, 1, 2, 2])

        let m = opsF([1, 2, 3, 4], [2, 2])
        let p4 = padded(m, widths: [[1, 0], [0, 1]])
        XCTAssertEqual(p4.shape, [3, 3])
        XCTAssertEqual(p4.asArray(Float.self), [0, 0, 0, 1, 2, 0, 3, 4, 0])

        let p5 = padded(m, widths: [0, 1], mode: .edge)
        XCTAssertEqual(p5.shape, [2, 4])
        XCTAssertEqual(p5.asArray(Float.self), [1, 1, 2, 2, 3, 3, 4, 4])
    }

    func testTiled() {
        let a = opsI([1, 2])
        XCTAssertEqual(tiled(a, repetitions: 2).asArray(Int32.self), [1, 2, 1, 2])
        let b = tiled(opsI([1, 2], [1, 2]), repetitions: [2, 1])
        XCTAssertEqual(b.shape, [2, 2])
        XCTAssertEqual(b.asArray(Int32.self), [1, 2, 1, 2])
    }

    func testRoll() {
        let a = MLXArray(0 ..< 6, [2, 3])
        // no axes: roll the flattened array and keep the shape
        let r1 = roll(a, shift: 1)
        XCTAssertEqual(r1.shape, [2, 3])
        XCTAssertEqual(r1.asArray(Int32.self), [5, 0, 1, 2, 3, 4])

        let r2 = roll(a, shift: 1, axis: 1)
        XCTAssertEqual(r2.asArray(Int32.self), [2, 0, 1, 5, 3, 4])

        let r3 = roll(a, shift: 1, axes: [1])
        XCTAssertEqual(r3.asArray(Int32.self), [2, 0, 1, 5, 3, 4])

        let r4 = roll(a, shift: -1, axes: [0])
        XCTAssertEqual(r4.asArray(Int32.self), [3, 4, 5, 0, 1, 2])
    }

    func testRollMultipleAxes() {
        // Python MLX mx.roll(a, 1, axis=(0, 1)) applies the shift to each axis.
        // The Swift wrapper passes one shift value for all the axes
        // (Source/MLX/Ops.swift:2595), and MLX raises "[roll] At least one shift
        // value per axis is required".
        let a = MLXArray(0 ..< 6, [2, 3])
        XCTExpectFailure(
            "roll(_:shift:axes:) passes one shift value for more than one axis (Source/MLX/Ops.swift:2595)"
        ) {
            do {
                let r = try withError { () -> MLXArray in
                    roll(a, shift: 1, axes: [0, 1])
                }
                XCTAssertEqual(r.shape, [2, 3])
                XCTAssertEqual(r.asArray(Int32.self), [5, 3, 4, 2, 0, 1])
            } catch {
                XCTFail("roll over two axes raised an MLX error: \(error)")
            }
        }
    }

    func testTrilTriu() {
        let a = MLXArray(1 ..< 10, [3, 3])
        XCTAssertEqual(tril(a).asArray(Int32.self), [1, 0, 0, 4, 5, 0, 7, 8, 9])
        XCTAssertEqual(tril(a, k: -1).asArray(Int32.self), [0, 0, 0, 4, 0, 0, 7, 8, 0])
        XCTAssertEqual(triu(a).asArray(Int32.self), [1, 2, 3, 0, 5, 6, 0, 0, 9])
        XCTAssertEqual(triu(a, k: 1).asArray(Int32.self), [0, 2, 3, 0, 0, 6, 0, 0, 0])
    }

    func testContiguousAndStopGradient() {
        let a = MLXArray(0 ..< 6, [2, 3]).transposed()
        let c = contiguous(a)
        XCTAssertEqual(c.shape, [3, 2])
        XCTAssertEqual(c.asArray(Int32.self), [0, 3, 1, 4, 2, 5])
        let c2 = contiguous(a, allowColMajor: true)
        XCTAssertEqual(c2.asArray(Int32.self), [0, 3, 1, 4, 2, 5])

        let x = opsF([1, 2, 3])
        XCTAssertEqual(stopGradient(x).asArray(Float.self), [1, 2, 3])
        // the gradient does not flow through stopGradient
        let g = grad { (x: MLXArray) -> MLXArray in
            (stopGradient(x) * x).sum()
        }(x)
        XCTAssertEqual(g.asArray(Float.self), [1, 2, 3])
    }

    func testDepends() {
        let a = opsF([1, 2])
        let b = opsF([3])
        let r = depends(input: a, dependencies: [b])
        XCTAssertEqual(r.shape, [2])
        XCTAssertEqual(r.asArray(Float.self), [1, 2])

        let rs = depends(inputs: [a, b], dependencies: [opsF([5])])
        XCTAssertEqual(rs.count, 2)
        XCTAssertEqual(rs[0].asArray(Float.self), [1, 2])
        XCTAssertEqual(rs[1].asArray(Float.self), [3])
    }

    func testMeshGrid() {
        let x = opsF([1, 2, 3])
        let y = opsF([4, 5])

        let xy = meshGrid([x, y])
        XCTAssertEqual(xy.count, 2)
        XCTAssertEqual(xy[0].shape, [2, 3])
        XCTAssertEqual(xy[0].asArray(Float.self), [1, 2, 3, 1, 2, 3])
        XCTAssertEqual(xy[1].asArray(Float.self), [4, 4, 4, 5, 5, 5])

        let ij = meshGrid([x, y], indexing: .ij)
        XCTAssertEqual(ij[0].shape, [3, 2])
        XCTAssertEqual(ij[0].asArray(Float.self), [1, 1, 2, 2, 3, 3])
        XCTAssertEqual(ij[1].asArray(Float.self), [4, 5, 4, 5, 4, 5])

        let sparse = meshGrid([x, y], sparse: true)
        XCTAssertEqual(sparse[0].shape, [1, 3])
        XCTAssertEqual(sparse[1].shape, [2, 1])
    }

    // MARK: - Sort, partition, select

    func testArgSortAndSorted() {
        let a = opsF([3, 1, 2, 0, 5, 4], [2, 3])
        let s1 = argSort(a, axis: 1)
        XCTAssertEqual(s1.dtype, .uint32)
        XCTAssertEqual(s1.asArray(UInt32.self), [1, 2, 0, 0, 2, 1])

        // no axis: sort the flattened array
        let s2 = argSort(a)
        XCTAssertEqual(s2.shape, [6])
        XCTAssertEqual(s2.asArray(UInt32.self), [3, 1, 2, 0, 5, 4])

        XCTAssertEqual(sorted(a, axis: 0).asArray(Float.self), [0, 1, 2, 3, 5, 4])
        XCTAssertEqual(sorted(a).asArray(Float.self), [0, 1, 2, 3, 4, 5])
    }

    /// Check that `values` is partitioned at `kth`, and that it has the same
    /// elements as `source`.
    private func checkPartition(
        _ values: [Float], kth: Int, source: [Float], file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let pivot = source.sorted()[kth]
        XCTAssertEqual(values[kth], pivot, file: file, line: line)
        for v in values[..<kth] {
            XCTAssertLessThanOrEqual(v, pivot, file: file, line: line)
        }
        for v in values[(kth + 1)...] {
            XCTAssertGreaterThanOrEqual(v, pivot, file: file, line: line)
        }
        XCTAssertEqual(values.sorted(), source.sorted(), file: file, line: line)
    }

    func testPartition() {
        let values: [Float] = [3, 1, 2, 5, 4]
        let a = opsF(values)

        checkPartition(partitioned(a, kth: 2).asArray(Float.self), kth: 2, source: values)
        checkPartition(partitioned(a, kth: 1, axis: 0).asArray(Float.self), kth: 1, source: values)

        let i1 = argPartition(a, kth: 2)
        XCTAssertEqual(i1.dtype, .uint32)
        checkPartition(take(a, i1).asArray(Float.self), kth: 2, source: values)
        let i2 = argPartition(a, kth: 3, axis: 0)
        checkPartition(take(a, i2).asArray(Float.self), kth: 3, source: values)
    }

    func testTop() {
        let a = opsF([5, 1, 4, 2, 3])
        let t1 = top(a, k: 2)
        XCTAssertEqual(t1.shape, [2])
        XCTAssertEqual(t1.asArray(Float.self).sorted(), [4, 5])

        let m = opsF([1, 9, 3, 5, 3, 7], [2, 3])
        let t2 = top(m, k: 1, axis: 1)
        XCTAssertEqual(t2.shape, [2, 1])
        XCTAssertEqual(t2.asArray(Float.self), [9, 7])
        let t3 = top(m, k: 1, axis: 0)
        XCTAssertEqual(t3.shape, [1, 3])
        XCTAssertEqual(t3.asArray(Float.self), [5, 9, 7])
    }

    func testTakeAlongAndPutAlong() {
        let a = opsI([10, 30, 20, 60, 50, 40], [2, 3])
        let idx = opsI([1, 0], [2, 1])
        let t1 = takeAlong(a, idx, axis: 1)
        XCTAssertEqual(t1.shape, [2, 1])
        XCTAssertEqual(t1.asArray(Int32.self), [30, 60])

        // no axis: index the flattened array
        let t2 = takeAlong(a, opsI([5, 0]))
        XCTAssertEqual(t2.asArray(Int32.self), [40, 10])

        let z = MLXArray.zeros([2, 3], dtype: .int32)
        let p1 = putAlong(z, opsI([1, 2], [2, 1]), values: opsI([5, 6], [2, 1]), axis: 1)
        XCTAssertEqual(p1.shape, [2, 3])
        XCTAssertEqual(p1.asArray(Int32.self), [0, 5, 0, 0, 0, 6])

        let z2 = MLXArray.zeros([2, 2], dtype: .int32)
        let p2 = putAlong(z2, opsI([3, 0]), values: opsI([7, 8]))
        XCTAssertEqual(p2.shape, [2, 2])
        XCTAssertEqual(p2.asArray(Int32.self), [8, 0, 0, 7])
    }

    // MARK: - Reductions

    func testMedian() {
        let a = opsF([3, 1, 2, 9, 7, 8, 4, 6, 5], [3, 3])
        let m0 = median(a, axis: 1)
        XCTAssertEqual(m0.shape, [3])
        XCTAssertEqual(m0.asArray(Float.self), [2, 8, 5])

        let m1 = median(a, axis: 0, keepDims: true)
        XCTAssertEqual(m1.shape, [1, 3])
        XCTAssertEqual(m1.asArray(Float.self), [4, 6, 5])

        let m2 = median(a, axes: [0, 1])
        XCTAssertEqual(m2.shape, [])
        XCTAssertEqual(m2.item(Float.self), 5)
    }

    func testMedianAllAxes() {
        // The doc comment says median(_:keepDims:) computes the median over the full
        // array, as Python MLX mx.median(a) does. The Swift wrapper passes an empty
        // axes list (Source/MLX/Ops.swift:2032), and MLX raises "[flatten] start_axis
        // must be less than or equal to end_axis" for an array with 2 dimensions.
        let a = opsF([3, 1, 2, 9, 7, 8, 4, 6, 5], [3, 3])
        XCTExpectFailure(
            "median(_:keepDims:) passes no axes to mlx_median (Source/MLX/Ops.swift:2032)"
        ) {
            do {
                let m = try withError { () -> MLXArray in
                    median(a)
                }
                XCTAssertEqual(m.shape, [])
                XCTAssertEqual(m.item(Float.self), 5)

                let k = try withError { () -> MLXArray in
                    median(a, keepDims: true)
                }
                XCTAssertEqual(k.shape, [1, 1])
            } catch {
                XCTFail("median over all axes raised an MLX error: \(error)")
            }
        }
    }

    func testStd() {
        let a = opsF([1, 2, 3, 4, 2, 2, 2, 2], [2, 4])
        // row 0: mean 2.5, population variance 1.25, sample variance 5/3
        let s1 = std(a, axis: 1)
        XCTAssertEqual(s1.shape, [2])
        assertEqual(s1, opsF([Foundation.sqrt(1.25), 0]), rtol: 1e-5, atol: 1e-6)

        let s2 = std(a, axis: 1, keepDims: true, ddof: 1)
        XCTAssertEqual(s2.shape, [2, 1])
        assertEqual(s2, opsF([Foundation.sqrt(5.0 / 3.0), 0], [2, 1]), rtol: 1e-5, atol: 1e-6)

        let s3 = std(a, axes: [0, 1])
        // all 8 values: mean 2.25, sum of squared deviations 3.5
        assertEqual(s3, MLXArray(Foundation.sqrt(Float(3.5 / 8))), rtol: 1e-5, atol: 1e-6)

        let s4 = std(a)
        assertEqual(s4, MLXArray(Foundation.sqrt(Float(3.5 / 8))), rtol: 1e-5, atol: 1e-6)
        let s5 = std(a, ddof: 1)
        assertEqual(s5, MLXArray(Foundation.sqrt(Float(3.5 / 7))), rtol: 1e-5, atol: 1e-6)
    }

    func testTrace() {
        let a = opsI([1, 2, 3, 4], [2, 2])
        let t = trace(a)
        XCTAssertEqual(t.dtype, .int32)
        XCTAssertEqual(t.item(Int32.self), 5)
        XCTAssertEqual(trace(a, offset: 1).item(Int32.self), 2)
        XCTAssertEqual(trace(a, offset: -1).item(Int32.self), 3)
        let f = trace(a, dtype: .float32)
        XCTAssertEqual(f.dtype, .float32)
        XCTAssertEqual(f.item(Float.self), 5)

        let b = MLXArray(0 ..< 8, [2, 2, 2])
        // trace over axes 1 and 2 of each 2x2 block
        let tb = trace(b, axis1: 1, axis2: 2)
        XCTAssertEqual(tb.asArray(Int32.self), [3, 11])
    }

    // MARK: - Products

    func testInnerOuterKron() {
        let a = opsF([1, 2, 3])
        let b = opsF([4, 5, 6])
        XCTAssertEqual(inner(a, b).item(Float.self), 32)

        let o = outer(opsF([1, 2]), opsF([3, 4, 5]))
        XCTAssertEqual(o.shape, [2, 3])
        XCTAssertEqual(o.asArray(Float.self), [3, 4, 5, 6, 8, 10])

        let x: [Float] = [1, 2, 3, 4]
        let y: [Float] = [0, 1, 1, 0]
        let k = kron(opsF(x, [2, 2]), opsF(y, [2, 2]))
        XCTAssertEqual(k.shape, [4, 4])
        var expected = [Float](repeating: 0, count: 16)
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                for p in 0 ..< 2 {
                    for q in 0 ..< 2 {
                        expected[(i * 2 + p) * 4 + (j * 2 + q)] = x[i * 2 + j] * y[p * 2 + q]
                    }
                }
            }
        }
        XCTAssertEqual(k.asArray(Float.self), expected)
    }

    func testTensordotOverloads() {
        let a = MLXArray(0 ..< 6, [2, 3]).asType(.float32)
        let b = MLXArray(0 ..< 6, [3, 2]).asType(.float32)
        // axes: 1 is a matrix product
        let r1 = tensordot(a, b)
        XCTAssertEqual(r1.shape, [2, 2])
        XCTAssertEqual(r1.asArray(Float.self), [10, 13, 28, 40])

        // contract both axes: sum over i, j of a[i, j] * c[i, j]
        let c = MLXArray(0 ..< 6, [2, 3]).asType(.float32)
        let r2 = tensordot(a, c, axes: 2)
        XCTAssertEqual(r2.item(Float.self), 55)

        let r3 = tensordot(a, c, axes: ((0, 1), (0, 1)))
        XCTAssertEqual(r3.item(Float.self), 55)

        // a[i, j] * c[i, j] with the axes of c swapped: c^T is [3, 2]
        let ct = c.transposed()
        let r4 = tensordot(a, ct, axes: ((0, 1), (1, 0)))
        XCTAssertEqual(r4.item(Float.self), 55)
    }

    func testEinsum() {
        let a = opsF([1, 2, 3, 4], [2, 2])
        let b = opsF([5, 6, 7, 8], [2, 2])
        let r1 = einsum("ij,jk->ik", a, b)
        XCTAssertEqual(r1.shape, [2, 2])
        XCTAssertEqual(r1.asArray(Float.self), [19, 22, 43, 50])

        let r2 = einsum("ij->", operands: [a])
        XCTAssertEqual(r2.item(Float.self), 10)

        let r3 = einsum("ij->ji", operands: [b])
        XCTAssertEqual(r3.asArray(Float.self), [5, 7, 6, 8])
    }

    func testBlockMaskedMM() {
        let a = MLXArray.ones([64, 64], dtype: .float32)
        let b = MLXArray.ones([64, 64], dtype: .float32)

        let full = blockMaskedMM(a, b, blockSize: 32)
        XCTAssertEqual(full.shape, [64, 64])
        XCTAssertEqual(full.min().item(Float.self), 64)
        XCTAssertEqual(full.max().item(Float.self), 64)

        // the output mask keeps the two diagonal 32x32 blocks
        let maskOut = MLXArray([true, false, false, true], [2, 2])
        let masked = blockMaskedMM(a, b, blockSize: 32, maskOut: maskOut)
        XCTAssertEqual(masked.shape, [64, 64])
        XCTAssertEqual(masked[0, 0].item(Float.self), 64)
        XCTAssertEqual(masked[0, 40].item(Float.self), 0)
        XCTAssertEqual(masked[40, 0].item(Float.self), 0)
        XCTAssertEqual(masked[63, 63].item(Float.self), 64)
        XCTAssertEqual(masked.sum().item(Float.self), 64 * 32 * 32 * 2)
    }

    func testGatherMatmul() {
        // a: batch 2 of [1, 3], b: batch 2 of [3, 2]
        let av: [Float] = [1, 2, 3, 4, 5, 6]
        let bv: [Float] = [1, 0, 0, 1, 1, 1, 2, 1, 0, 3, 1, 0]
        let a = opsF(av, [2, 1, 3])
        let b = opsF(bv, [2, 3, 2])
        let lhs = MLXArray([1, 0] as [UInt32])
        let rhs = MLXArray([0, 1] as [UInt32])

        // out[i] = a[lhs[i]] @ b[rhs[i]]
        var expected = [Float]()
        for (l, r) in [(1, 0), (0, 1)] {
            for n in 0 ..< 2 {
                var s: Float = 0
                for k in 0 ..< 3 {
                    s += av[l * 3 + k] * bv[r * 6 + k * 2 + n]
                }
                expected.append(s)
            }
        }

        let r1 = gatherMatmul(a, b, lhsIndices: lhs, rhsIndices: rhs)
        XCTAssertEqual(r1.shape, [2, 1, 2])
        XCTAssertEqual(r1.asArray(Float.self), expected)

        let r2 = gatherMM(a, b, lhsIndices: lhs, rhsIndices: rhs)
        XCTAssertEqual(r2.asArray(Float.self), expected)
    }

    func testSegmentedMM() {
        // out[s] = a[:, k0 ..< k1] @ b[k0 ..< k1, :] for each segment (k0, k1)
        let av: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let bv: [Float] = (0 ..< 12).map { Float($0) }
        let a = opsF(av, [2, 4])
        let b = opsF(bv, [4, 3])
        let segments = opsI([0, 2, 2, 4], [2, 2])

        var expected = [Float]()
        for (k0, k1) in [(0, 2), (2, 4)] {
            for m in 0 ..< 2 {
                for n in 0 ..< 3 {
                    var s: Float = 0
                    for k in k0 ..< k1 {
                        s += av[m * 4 + k] * bv[k * 3 + n]
                    }
                    expected.append(s)
                }
            }
        }

        let r = segmentedMM(a, b, segments: segments)
        XCTAssertEqual(r.shape, [2, 2, 3])
        XCTAssertEqual(r.dtype, .float32)
        XCTAssertEqual(r.asArray(Float.self), expected)
    }

    // MARK: - Quantized

    func testDeprecatedQuantizedMatmulWrappers() {
        let k = 64
        let n = 16
        let wValues = (0 ..< (n * k)).map { Float(($0 * 7) % 11 - 5) / 10 }
        let w = opsF(wValues, [n, k])
        let (wq, scales, biases) = quantized(w)
        let dq = dequantized(wq, scales: scales, biases: biases)
        XCTAssertEqual(dq.shape, [n, k])

        let xValues = (0 ..< k).map { Float($0 % 5 - 2) / 4 }
        let x = opsF(xValues, [1, k])

        let reference = matmul(x, dq.transposed())
        let r1 = quantizedMatmul(x, wq, scales: scales, biases: biases)
        XCTAssertEqual(r1.shape, [1, n])
        assertEqual(r1, reference, rtol: 1e-3, atol: 1e-3)

        // two experts: the same weights and the weights times -1
        let w2 = concatenated(
            [expandedDimensions(w, axis: 0), expandedDimensions(-w, axis: 0)], axis: 0)
        let (wq2, scales2, biases2) = quantized(w2)
        let dq2 = dequantized(wq2, scales: scales2, biases: biases2)
        let xs = concatenated(
            [expandedDimensions(x, axis: 0), expandedDimensions(x * 2, axis: 0)], axis: 0)
        let rhs = MLXArray([1, 0] as [UInt32])
        let r2 = gatherQuantizedMatmul(
            xs, wq2, scales: scales2, biases: biases2, rhsIndices: rhs)
        XCTAssertEqual(r2.shape, [2, 1, n])
        let expected0 = matmul(x, dq2[1].transposed())
        let expected1 = matmul(x * 2, dq2[0].transposed())
        assertEqual(r2[0], expected0, rtol: 1e-3, atol: 1e-3)
        assertEqual(r2[1], expected1, rtol: 1e-3, atol: 1e-3)
    }

    // MARK: - Windows

    func testWindowFunctions() {
        let m = 5
        let d = Float(m - 1)
        let n = (0 ..< m).map { Float($0) }

        let bartlettRef = n.map { 1 - Swift.abs(2 * $0 / d - 1) }
        let hanningRef = n.map { 0.5 - 0.5 * Foundation.cos(2 * Float.pi * $0 / d) }
        let hammingRef = n.map { 0.54 - 0.46 * Foundation.cos(2 * Float.pi * $0 / d) }
        let blackmanRef = n.map {
            0.42 - 0.5 * Foundation.cos(2 * Float.pi * $0 / d)
                + 0.08 * Foundation.cos(4 * Float.pi * $0 / d)
        }

        let b = bartlett(m)
        XCTAssertEqual(b.shape, [m])
        XCTAssertEqual(b.dtype, .float32)
        assertEqual(b, opsF(bartlettRef), rtol: 1e-4, atol: 1e-5)
        assertEqual(hanning(m), opsF(hanningRef), rtol: 1e-4, atol: 1e-5)
        assertEqual(hamming(m), opsF(hammingRef), rtol: 1e-4, atol: 1e-5)
        assertEqual(blackman(m), opsF(blackmanRef), rtol: 1e-4, atol: 1e-5)
    }

    // MARK: - Softmax

    func testSoftmaxOverloads() {
        let v: [Float] = [1, 2, 3, 1, 1, 1]
        let a = opsF(v, [2, 3])

        func rowSoftmax(_ row: [Float]) -> [Float] {
            let m = row.max()!
            let e = row.map { Foundation.exp($0 - m) }
            let s = e.reduce(0, +)
            return e.map { $0 / s }
        }
        let rows = opsF(rowSoftmax(Array(v[0 ..< 3])) + rowSoftmax(Array(v[3 ..< 6])), [2, 3])
        let all = opsF(rowSoftmax(v), [2, 3])

        assertEqual(softmax(a, axis: 1), rows, rtol: 1e-5, atol: 1e-6)
        assertEqual(softmax(a, axis: -1, precise: true), rows, rtol: 1e-5, atol: 1e-6)
        assertEqual(softmax(a, axes: [0, 1]), all, rtol: 1e-5, atol: 1e-6)
        assertEqual(softmax(a), all, rtol: 1e-5, atol: 1e-6)

        assertEqual(softMax(a, axis: 1), rows, rtol: 1e-5, atol: 1e-6)
        assertEqual(softMax(a, axes: [0, 1]), all, rtol: 1e-5, atol: 1e-6)
        assertEqual(softMax(a), all, rtol: 1e-5, atol: 1e-6)
    }

    // MARK: - Convolution

    func testConv1dAndConv3d() {
        let x: [Float] = [1, 2, 3, 4, 5]
        let w: [Float] = [1, 0, -1]
        // stride 2, padding 1: the padded input is [0, 1, 2, 3, 4, 5, 0]
        let r = conv1d(opsF(x, [1, 5, 1]), opsF(w, [1, 3, 1]), stride: 2, padding: 1)
        XCTAssertEqual(r.shape, [1, 3, 1])
        XCTAssertEqual(r.asArray(Float.self), [-2, -2, 4])

        // conv3d with a [1, 1, 2] kernel along the last spatial axis
        let r3 = conv3d(opsF([1, 2, 3], [1, 1, 1, 3, 1]), opsF([1, 10], [1, 1, 1, 2, 1]))
        XCTAssertEqual(r3.shape, [1, 1, 1, 2, 1])
        XCTAssertEqual(r3.asArray(Float.self), [21, 32])
    }

    func testConv2dAndConvGeneral() {
        let x: [[Float]] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        let w: [[Float]] = [[1, 2], [3, 4]]
        let input = opsF(x.flatMap { $0 }, [1, 3, 3, 1])
        let weight = opsF(w.flatMap { $0 }, [1, 2, 2, 1])

        let r1 = conv2d(input, weight)
        XCTAssertEqual(r1.shape, [1, 2, 2, 1])
        XCTAssertEqual(r1.asArray(Float.self), opsCorrelate2D(x, w).flatMap { $0 })

        let r2 = conv2d(input, weight, stride: 2, padding: 1)
        let e2 = opsCorrelate2D(x, w, padding: 1, stride: 2)
        XCTAssertEqual(r2.shape, [1, e2.count, e2[0].count, 1])
        XCTAssertEqual(r2.asArray(Float.self), e2.flatMap { $0 })

        let r3 = convGeneral(input, weight, strides: 1, padding: 1)
        let e3 = opsCorrelate2D(x, w, padding: 1)
        XCTAssertEqual(r3.shape, [1, 4, 4, 1])
        XCTAssertEqual(r3.asArray(Float.self), e3.flatMap { $0 })

        // 1D with low padding 1, high padding 0, and a flipped kernel (true convolution)
        let x1: [Float] = [1, 2, 3, 4]
        let w1: [Float] = [1, 10]
        let r4 = convGeneral(
            opsF(x1, [1, 4, 1]), opsF(w1, [1, 2, 1]), padding: (1, 0), flip: true)
        // padded input [0, 1, 2, 3, 4], kernel used as [10, 1]
        XCTAssertEqual(r4.shape, [1, 4, 1])
        XCTAssertEqual(r4.asArray(Float.self), [1, 12, 23, 34])
    }

    func testConvTransposed() {
        let x: [Float] = [1, 2, 3]
        let w: [Float] = [1, 10]
        let r1 = convTransposed1d(opsF(x, [1, 3, 1]), opsF(w, [1, 2, 1]))
        XCTAssertEqual(r1.shape, [1, 4, 1])
        XCTAssertEqual(r1.asArray(Float.self), opsFullConvolve(x, w))

        let r2 = convTransposed2d(
            opsF([1, 2, 3, 4], [1, 2, 2, 1]), MLXArray.ones([1, 2, 2, 1], dtype: .float32))
        XCTAssertEqual(r2.shape, [1, 3, 3, 1])
        XCTAssertEqual(r2.asArray(Float.self), [1, 3, 2, 4, 10, 6, 3, 7, 4])

        let r3 = convTransposed3d(
            opsF([1, 2], [1, 1, 1, 2, 1]), MLXArray.ones([1, 1, 1, 2, 1], dtype: .float32))
        XCTAssertEqual(r3.shape, [1, 1, 1, 3, 1])
        XCTAssertEqual(r3.asArray(Float.self), [1, 3, 2])
    }

    func testConvolveModes() {
        let a: [Float] = [1, 2, 3, 4]
        let b: [Float] = [1, 2, 3]
        let full = opsFullConvolve(a, b)
        XCTAssertEqual(full, [1, 4, 10, 16, 17, 12])

        let r1 = convolve(opsF(a), opsF(b))
        XCTAssertEqual(r1.shape, [6])
        XCTAssertEqual(r1.asArray(Float.self), full)

        // plain Swift arrays, and the shorter one first: the result is the same
        let r2 = convolve(b, a, mode: .full)
        XCTAssertEqual(r2.asArray(Float.self), full)

        let r3 = convolve(opsF(a), opsF(b), mode: .valid)
        XCTAssertEqual(r3.asArray(Float.self), Array(full[2 ..< 4]))

        // same, odd kernel: length max(4, 3), start at (3 - 1) / 2
        let r4 = convolve(opsF(a), opsF(b), mode: .same)
        XCTAssertEqual(r4.asArray(Float.self), Array(full[1 ..< 5]))

        // same, kernel of size 2: start at (2 - 1) / 2 = 0
        let k2: [Float] = [1, 1]
        let r5 = convolve(opsF([1, 2, 3]), opsF(k2), mode: .same)
        XCTAssertEqual(r5.asArray(Float.self), Array(opsFullConvolve([1, 2, 3], k2)[0 ..< 3]))
    }

    func testConvolveSameEvenKernel() {
        // numpy.convolve(a, v, "same") and Python MLX give an output of length
        // max(len(a), len(v)). For an even kernel, Python MLX pads the input with
        // pad_l = size / 2 on the left and pad_r = pad_l - 1 on the right. The Swift
        // code uses padLeft / 2 - 1 for the right side (Source/MLX/Ops.swift:1006),
        // so a kernel of size 4 gives an output that is one element too short.
        let a: [Float] = [1, 2, 3, 4, 5]
        let k: [Float] = [1, 1, 1, 1]
        let full = opsFullConvolve(a, k)
        let expected = Array(full[1 ..< 6])
        XCTAssertEqual(expected, [3, 6, 10, 14, 12])

        let r = convolve(opsF(a), opsF(k), mode: .same)
        XCTExpectFailure(
            "convolve .same pads the right side with padLeft / 2 - 1, not padLeft - 1 (Source/MLX/Ops.swift:1006)"
        ) {
            XCTAssertEqual(r.shape, [5])
            if r.shape == [5] {
                XCTAssertEqual(r.asArray(Float.self), expected)
            }
        }
    }
}
