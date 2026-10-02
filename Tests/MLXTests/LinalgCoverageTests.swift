// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

/// Tests for the `MLXLinalg` functions and the free linalg functions in
/// `Source/MLX/Linalg.swift`. Each test rebuilds the input from the
/// factorization, or compares with values worked out by hand.
///
/// The factorizations run only on the CPU in MLX, so these tests pass
/// `stream: .cpu` to them.
class LinalgCoverageTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    private func matrix(_ values: [Float], _ shape: [Int]) -> MLXArray {
        MLXArray(values, shape)
    }

    // MARK: - norm

    func testVectorNormOrders() {
        // v = [3, -4, 0, 12]
        let v = matrix([3, -4, 0, 12], [4])
        let p3 = Float(pow(27.0 + 64.0 + 0.0 + 1728.0, 1.0 / 3.0))

        // free functions, no axis
        XCTAssertEqual(norm(v).item(Float.self), 13, accuracy: 1e-4)
        XCTAssertEqual(norm(v, ord: 1).item(Float.self), 19, accuracy: 1e-4)
        XCTAssertEqual(norm(v, ord: 2).item(Float.self), 13, accuracy: 1e-4)
        XCTAssertEqual(norm(v, ord: .infinity).item(Float.self), 12, accuracy: 1e-4)
        XCTAssertEqual(norm(v, ord: -.infinity).item(Float.self), 0, accuracy: 1e-4)
        XCTAssertEqual(norm(v, ord: 0).item(Float.self), 3, accuracy: 1e-4)
        XCTAssertEqual(norm(v, ord: 3).item(Float.self), p3, accuracy: 1e-3)

        // free functions, one Int axis
        XCTAssertEqual(norm(v, axis: 0).item(Float.self), 13, accuracy: 1e-4)
        XCTAssertEqual(norm(v, ord: 1, axis: 0).item(Float.self), 19, accuracy: 1e-4)
        let kept = norm(v, axis: 0, keepDims: true)
        XCTAssertEqual(kept.shape, [1])
        XCTAssertEqual(kept.item(Float.self), 13, accuracy: 1e-4)
        let keptOrd = norm(v, ord: .infinity, axis: 0, keepDims: true)
        XCTAssertEqual(keptOrd.shape, [1])
        XCTAssertEqual(keptOrd.item(Float.self), 12, accuracy: 1e-4)

        // MLXLinalg with ord and a list of axes
        XCTAssertEqual(MLXLinalg.norm(v, ord: 1, axes: [0]).item(Float.self), 19, accuracy: 1e-4)
        XCTAssertEqual(MLXLinalg.norm(v, ord: 2, axis: 0).item(Float.self), 13, accuracy: 1e-4)
    }

    func testVectorNormAlongAxis() {
        // rows [3, 4] and [6, 8] have 2-norms 5 and 10
        let m = matrix([3, 4, 6, 8], [2, 2])

        assertEqual(norm(m, axis: 1), matrix([5, 10], [2]), atol: 1e-4)
        assertEqual(norm(m, ord: 1, axis: 1), matrix([7, 14], [2]), atol: 1e-4)
        assertEqual(norm(m, ord: .infinity, axis: 0), matrix([6, 8], [2]), atol: 1e-4)

        let axis: IntOrArray = 1
        assertEqual(norm(m, axis: axis), matrix([5, 10], [2]), atol: 1e-4)
        assertEqual(norm(m, ord: -.infinity, axis: axis), matrix([3, 6], [2]), atol: 1e-4)

        let kept = norm(m, axis: axis, keepDims: true)
        XCTAssertEqual(kept.shape, [2, 1])
        assertEqual(kept, matrix([5, 10], [2, 1]), atol: 1e-4)
    }

    func testMatrixNormOrders() {
        // m = [[1, -2], [3, 4]]
        // fro = sqrt(30); row sums of |m| = 3, 7; column sums of |m| = 4, 6
        // m^T m = [[10, 10], [10, 20]] has eigenvalues 15 +- sqrt(125), so the
        // singular values are sqrt(15 +- sqrt(125)); their sum is sqrt(50).
        let m = matrix([1, -2, 3, 4], [2, 2])
        let fro = Float(30.0.squareRoot())
        let sigmaMax = Float((15 + 125.0.squareRoot()).squareRoot())
        let sigmaMin = Float((15 - 125.0.squareRoot()).squareRoot())
        let nuc = Float(50.0.squareRoot())

        XCTAssertEqual(norm(m, axes: [0, 1]).item(Float.self), fro, accuracy: 1e-4)
        XCTAssertEqual(norm(m, ord: .fro, axes: [0, 1]).item(Float.self), fro, accuracy: 1e-4)
        XCTAssertEqual(norm(m, ord: .fro).item(Float.self), fro, accuracy: 1e-4)
        XCTAssertEqual(
            MLXLinalg.norm(m, ord: .nuc, axes: [0, 1], stream: .cpu).item(Float.self), nuc,
            accuracy: 1e-4)

        let both: IntOrArray = [0, 1]
        XCTAssertEqual(norm(m, ord: .fro, axis: both).item(Float.self), fro, accuracy: 1e-4)
        XCTAssertEqual(
            norm(m, ord: .nuc, axis: both, stream: .cpu).item(Float.self), nuc, accuracy: 1e-4)
        XCTAssertEqual(norm(m, ord: 1, axis: both).item(Float.self), 6, accuracy: 1e-4)
        XCTAssertEqual(norm(m, ord: -1, axis: both).item(Float.self), 4, accuracy: 1e-4)
        XCTAssertEqual(norm(m, ord: .infinity).item(Float.self), 7, accuracy: 1e-4)
        XCTAssertEqual(norm(m, ord: -.infinity).item(Float.self), 3, accuracy: 1e-4)
        XCTAssertEqual(
            norm(m, ord: 2, axis: both, stream: .cpu).item(Float.self), sigmaMax, accuracy: 1e-4)
        XCTAssertEqual(
            MLXLinalg.norm(m, ord: -2, axes: [0, 1], stream: .cpu).item(Float.self), sigmaMin,
            accuracy: 1e-4)

        let kept = norm(m, ord: .fro, axes: [0, 1], keepDims: true)
        XCTAssertEqual(kept.shape, [1, 1])
        XCTAssertEqual(kept.item(Float.self), fro, accuracy: 1e-4)
    }

    func testMatrixNormBatched() {
        // two matrices: [[0, 1], [2, 3]] and [[4, 5], [6, 7]]
        let m = MLXArray(0 ..< 8, [2, 2, 2]).asType(.float32)
        let expected = matrix(
            [Float(14.0.squareRoot()), Float(126.0.squareRoot())], [2])
        assertEqual(norm(m, ord: .fro, axes: [1, 2]), expected, atol: 1e-4)
        assertEqual(MLXLinalg.norm(m, ord: .fro, axes: [1, 2]), expected, atol: 1e-4)
    }

    func testNormKindNeedsTwoAxes() {
        // The "fro" and "nuc" norms are only defined for matrices.
        let v = matrix([3, 4], [2])

        XCTAssertThrowsError(
            try withError {
                _ = MLXLinalg.norm(v, ord: .fro, axis: 0)
            }
        ) { error in
            XCTAssertTrue("\(error)".contains("only supported for matrices"), "\(error)")
        }
        XCTAssertThrowsError(
            try withError {
                _ = norm(v, ord: .nuc, axis: 0)
            }
        ) { error in
            XCTAssertTrue("\(error)".contains("only supported for matrices"), "\(error)")
        }
    }

    // MARK: - qr

    func testQRSquare() {
        let a = matrix([12, -51, 4, 6, 167, -68, -4, 24, -41], [3, 3])
        let (q, r) = qr(a, stream: .cpu)

        XCTAssertEqual(q.shape, [3, 3])
        XCTAssertEqual(r.shape, [3, 3])

        // Q * R == A
        assertEqual(matmul(q, r), a, rtol: 1e-4, atol: 1e-3)

        // Q is orthonormal
        assertEqual(matmul(q.T, q), MLXArray.eye(3), atol: 1e-5)

        // R is upper triangular with |diag(R)| = [14, 175, 35]
        assertEqual(tril(r, k: -1), MLXArray.zeros([3, 3]), atol: 1e-4)
        XCTAssertEqual(Swift.abs(r[0, 0].item(Float.self)), 14, accuracy: 1e-3)
        XCTAssertEqual(Swift.abs(r[1, 1].item(Float.self)), 175, accuracy: 1e-2)
        XCTAssertEqual(Swift.abs(r[2, 2].item(Float.self)), 35, accuracy: 1e-3)
    }

    func testQRTallAndBatched() {
        // tall: [3, 2] gives Q [3, 2] and R [2, 2]
        let a = matrix([1, 2, 3, 4, 5, 6], [3, 2])
        let (q, r) = MLXLinalg.qr(a, stream: .cpu)
        XCTAssertEqual(q.shape, [3, 2])
        XCTAssertEqual(r.shape, [2, 2])
        assertEqual(matmul(q, r), a, atol: 1e-4)
        assertEqual(matmul(q.T, q), MLXArray.eye(2), atol: 1e-5)
        XCTAssertEqual(r[1, 0].item(Float.self), 0, accuracy: 1e-5)

        // batched: two [2, 2] matrices
        let b = matrix([2, 3, 1, 2, 4, 0, 3, 5], [2, 2, 2])
        let (qb, rb) = qr(b, stream: .cpu)
        XCTAssertEqual(qb.shape, [2, 2, 2])
        XCTAssertEqual(rb.shape, [2, 2, 2])
        assertEqual(matmul(qb, rb), b, atol: 1e-4)
    }

    // MARK: - svd

    func testSVDSquare() {
        // a = [[3, 0], [4, 5]], a^T a = [[25, 20], [20, 25]] with
        // eigenvalues 45 and 5, so S = [sqrt(45), sqrt(5)].
        let a = matrix([3, 0, 4, 5], [2, 2])
        let expectedS = matrix([Float(45.0.squareRoot()), Float(5.0.squareRoot())], [2])

        let (u, s, vt) = MLXLinalg.svd(a, stream: .cpu)
        XCTAssertEqual(u.shape, [2, 2])
        XCTAssertEqual(s.shape, [2])
        XCTAssertEqual(vt.shape, [2, 2])
        assertEqual(s, expectedS, atol: 1e-4)

        // U * diag(S) * Vt == A
        let rebuilt = matmul(u * s.expandedDimensions(axis: 0), vt)
        assertEqual(rebuilt, a, atol: 1e-4)

        // U and Vt are orthonormal
        assertEqual(matmul(u.T, u), MLXArray.eye(2), atol: 1e-5)
        assertEqual(matmul(vt, vt.T), MLXArray.eye(2), atol: 1e-5)

        // the overload that returns only S
        let sOnly: MLXArray = MLXLinalg.svd(a, stream: .cpu)
        assertEqual(sOnly, expectedS, atol: 1e-4)
    }

    func testSVDTall() {
        // a is [3, 2]: U is [3, 3], S is [2], Vt is [2, 2]
        let a = matrix([1, 2, 3, 4, 5, 6], [3, 2])
        let (u, s, vt) = svd(a, stream: .cpu)
        XCTAssertEqual(u.shape, [3, 3])
        XCTAssertEqual(s.shape, [2])
        XCTAssertEqual(vt.shape, [2, 2])

        let rebuilt = matmul(u[0..., 0 ..< 2] * s.expandedDimensions(axis: 0), vt)
        assertEqual(rebuilt, a, atol: 1e-4)

        // S is sorted from large to small and its squares sum to |a|_F^2 = 91
        let sv = s.asArray(Float.self)
        XCTAssertGreaterThan(sv[0], sv[1])
        XCTAssertEqual(sv[0] * sv[0] + sv[1] * sv[1], 91, accuracy: 1e-2)
    }

    // MARK: - inverses

    func testInverse() {
        // [[2, 1], [1, 1]] has inverse [[1, -1], [-1, 2]]
        let a = matrix([2, 1, 1, 1], [2, 2])
        let expected = matrix([1, -1, -1, 2], [2, 2])

        let i1 = MLXLinalg.inv(a, stream: .cpu)
        let i2 = inv(a, stream: .cpu)
        assertEqual(i1, expected, atol: 1e-5)
        assertEqual(i2, expected, atol: 1e-5)
        assertEqual(matmul(a, i2), MLXArray.eye(2), atol: 1e-5)
        assertEqual(matmul(i2, a), MLXArray.eye(2), atol: 1e-5)

        // batched: each matrix is inverted
        let b = matrix([2, 1, 1, 1, 4, 0, 0, 2], [2, 2, 2])
        let expectedB = matrix([1, -1, -1, 2, 0.25, 0, 0, 0.5], [2, 2, 2])
        assertEqual(inv(b, stream: .cpu), expectedB, atol: 1e-5)
    }

    func testTriangularInverse() {
        // lower [[2, 0], [1, 1]] -> [[0.5, 0], [-0.5, 1]]
        // upper [[2, 1], [0, 1]] -> [[0.5, -0.5], [0, 1]]
        let lower = matrix([2, 0, 1, 1], [2, 2])
        let upper = matrix([2, 1, 0, 1], [2, 2])
        let lowerInv = matrix([0.5, 0, -0.5, 1], [2, 2])
        let upperInv = matrix([0.5, -0.5, 0, 1], [2, 2])

        assertEqual(MLXLinalg.triInv(lower, stream: .cpu), lowerInv, atol: 1e-6)
        assertEqual(MLXLinalg.triInv(upper, upper: true, stream: .cpu), upperInv, atol: 1e-6)
        assertEqual(triInv(lower, stream: .cpu), lowerInv, atol: 1e-6)
        assertEqual(matmul(upper, upperInv), MLXArray.eye(2), atol: 1e-6)

        assertEqual(triInv(upper, upper: true, stream: .cpu), upperInv, atol: 1e-6)
    }

    func testCholesky() {
        // s = [[4, 2], [2, 3]] = L L^T with L = [[2, 0], [1, sqrt(2)]]
        let s = matrix([4, 2, 2, 3], [2, 2])
        let r2 = Float(2.0.squareRoot())
        let expectedL = matrix([2, 0, 1, r2], [2, 2])
        let expectedU = matrix([2, 1, 0, r2], [2, 2])

        let l = MLXLinalg.cholesky(s, stream: .cpu)
        assertEqual(l, expectedL, atol: 1e-5)
        assertEqual(matmul(l, l.T), s, atol: 1e-5)

        let l2 = cholesky(s, stream: .cpu)
        assertEqual(l2, expectedL, atol: 1e-5)

        let u = cholesky(s, upper: true, stream: .cpu)
        assertEqual(u, expectedU, atol: 1e-5)
        assertEqual(matmul(u.T, u), s, atol: 1e-5)
        assertEqual(MLXLinalg.cholesky(s, upper: true, stream: .cpu), expectedU, atol: 1e-5)
    }

    func testCholeskyInverse() {
        // inv([[4, 2], [2, 3]]) = 1/8 * [[3, -2], [-2, 4]]
        let s = matrix([4, 2, 2, 3], [2, 2])
        let expected = matrix([0.375, -0.25, -0.25, 0.5], [2, 2])
        let r2 = Float(2.0.squareRoot())
        let l = matrix([2, 0, 1, r2], [2, 2])
        let u = matrix([2, 1, 0, r2], [2, 2])

        assertEqual(MLXLinalg.choleskyInv(l, stream: .cpu), expected, atol: 1e-5)
        assertEqual(choleskyInv(l, stream: .cpu), expected, atol: 1e-5)
        assertEqual(choleskyInv(u, upper: true, stream: .cpu), expected, atol: 1e-5)
        assertEqual(
            MLXLinalg.choleskyInv(u, upper: true, stream: .cpu), expected, atol: 1e-5)
        assertEqual(matmul(s, choleskyInv(l, stream: .cpu)), MLXArray.eye(2), atol: 1e-5)
    }

    func testPseudoInverse() {
        // pinv(a) = (a^T a)^-1 a^T, worked out by hand:
        // a^T a = [[35, 44], [44, 56]], det = 24
        let a = matrix([1, 2, 3, 4, 5, 6], [3, 2])
        let expected = matrix(
            [-32.0 / 24, -8.0 / 24, 16.0 / 24, 26.0 / 24, 8.0 / 24, -10.0 / 24], [2, 3])

        let p1 = MLXLinalg.pinv(a, stream: .cpu)
        let p2 = pinv(a, stream: .cpu)
        XCTAssertEqual(p1.shape, [2, 3])
        assertEqual(p1, expected, atol: 1e-4)
        assertEqual(p2, expected, atol: 1e-4)

        // a * pinv(a) * a == a and pinv(a) * a == I
        assertEqual(matmul(matmul(a, p2), a), a, atol: 1e-4)
        assertEqual(matmul(p2, a), MLXArray.eye(2), atol: 1e-4)
    }

    // MARK: - cross

    func testCross() {
        // [1, 2, 3] x [4, 5, 6] = [-3, 6, -3]
        let a = matrix([1, 2, 3], [3])
        let b = matrix([4, 5, 6], [3])
        assertEqual(MLXLinalg.cross(a, b), matrix([-3, 6, -3], [3]))
        assertEqual(cross(a, b), matrix([-3, 6, -3], [3]))

        // b x a = -(a x b)
        assertEqual(cross(b, a), matrix([3, -6, 3], [3]))

        // vectors of size 2 have an implied third value of 0:
        // [1, 2, 0] x [3, 4, 0] = [0, 0, -2]
        assertEqual(cross(matrix([1, 2], [2]), matrix([3, 4], [2])), matrix([0, 0, -2], [3]))

        // the vectors are the columns (axis 0)
        // column 0: [1, 2, 3] x [4, 5, 6] = [-3, 6, -3]
        // column 1: [1, 0, 0] x [0, 1, 0] = [0, 0, 1]
        let ca = matrix([1, 1, 2, 0, 3, 0], [3, 2])
        let cb = matrix([4, 0, 5, 1, 6, 0], [3, 2])
        let expected = matrix([-3, 0, 6, 0, -3, 1], [3, 2])
        assertEqual(cross(ca, cb, axis: 0), expected)
        assertEqual(MLXLinalg.cross(ca, cb, axis: 0), expected)
    }

    // MARK: - lu

    func testLU() {
        // a = [[1, 2], [3, 4]]; partial pivoting takes row 1 first.
        // L = [[1, 0], [1/3, 1]], U = [[3, 4], [0, 2/3]], P = [1, 0]
        let a = matrix([1, 2, 3, 4], [2, 2])
        let (p, l, u) = lu(a, stream: .cpu)

        XCTAssertEqual(p.dtype, .uint32)
        XCTAssertEqual(p.asArray(UInt32.self), [1, 0])
        assertEqual(l, matrix([1, 0, 1.0 / 3, 1], [2, 2]), atol: 1e-5)
        assertEqual(u, matrix([3, 4, 0, 2.0 / 3], [2, 2]), atol: 1e-5)

        // L[P] * U == A
        assertEqual(matmul(l[p], u), a, atol: 1e-5)
    }

    func testLU3x3() {
        let a = matrix([2, 1, 1, 4, 3, 3, 8, 7, 9], [3, 3])
        let (p, l, u) = MLXLinalg.lu(a, stream: .cpu)

        XCTAssertEqual(p.shape, [3])
        XCTAssertEqual(l.shape, [3, 3])
        XCTAssertEqual(u.shape, [3, 3])

        // P is a permutation of 0 ..< 3
        XCTAssertEqual(p.asArray(UInt32.self).sorted(), [0, 1, 2])

        // L is unit lower triangular and U is upper triangular
        assertEqual(triu(l, k: 1), MLXArray.zeros([3, 3]), atol: 1e-6)
        assertEqual(l.diagonal(), MLXArray.ones([3]), atol: 1e-6)
        assertEqual(tril(u, k: -1), MLXArray.zeros([3, 3]), atol: 1e-6)

        assertEqual(matmul(l[p], u), a, atol: 1e-4)
    }

    func testLUFactor() {
        // compact form of the factors above: LU = [[3, 4], [1/3, 2/3]],
        // pivots (0-based row swaps) = [1, 1]
        let a = matrix([1, 2, 3, 4], [2, 2])
        let expectedLU = matrix([3, 4, 1.0 / 3, 2.0 / 3], [2, 2])

        let (lu1, piv1) = MLXLinalg.lu_factor(a, stream: .cpu)
        assertEqual(lu1, expectedLU, atol: 1e-5)
        XCTAssertEqual(piv1.dtype, .uint32)
        XCTAssertEqual(piv1.asArray(UInt32.self), [1, 1])

        let (lu2, piv2) = lu_factor(a, stream: .cpu)
        assertEqual(lu2, expectedLU, atol: 1e-5)
        XCTAssertEqual(piv2.asArray(UInt32.self), [1, 1])
    }

    // MARK: - solve

    func testSolve() {
        // 3x + y = 9, x + 2y = 8 -> x = 2, y = 3
        let a = matrix([3, 1, 1, 2], [2, 2])
        let b = matrix([9, 8], [2])
        assertEqual(MLXLinalg.solve(a, b, stream: .cpu), matrix([2, 3], [2]), atol: 1e-5)
        assertEqual(solve(a, b, stream: .cpu), matrix([2, 3], [2]), atol: 1e-5)

        // two right hand sides as columns: [9, 8] and [1, 2]
        // the second gives x = 0, y = 1
        let b2 = matrix([9, 1, 8, 2], [2, 2])
        let x2 = solve(a, b2, stream: .cpu)
        assertEqual(x2, matrix([2, 0, 3, 1], [2, 2]), atol: 1e-5)
        assertEqual(matmul(a, x2), b2, atol: 1e-5)
    }

    func testSolveTriangular() {
        // lower: 2x = 4, x + y = 5 -> [2, 3]
        let lower = matrix([2, 0, 1, 1], [2, 2])
        let bl = matrix([4, 5], [2])
        assertEqual(
            MLXLinalg.solveTriangular(lower, bl, stream: .cpu), matrix([2, 3], [2]), atol: 1e-5)
        assertEqual(solveTriangular(lower, bl, stream: .cpu), matrix([2, 3], [2]), atol: 1e-5)

        // upper: 2x + y = 7, y = 3 -> [2, 3]
        let upper = matrix([2, 1, 0, 1], [2, 2])
        let bu = matrix([7, 3], [2])
        assertEqual(
            solveTriangular(upper, bu, upper: true, stream: .cpu), matrix([2, 3], [2]),
            atol: 1e-5)
        assertEqual(
            MLXLinalg.solveTriangular(upper, bu, upper: true, stream: .cpu), matrix([2, 3], [2]),
            atol: 1e-5)
    }

    // MARK: - eigen

    func testEig() {
        // upper triangular, so the eigenvalues are the diagonal: 2 and 3
        let a = matrix([2, 1, 0, 3], [2, 2])

        let (w, v) = MLXLinalg.eig(a, stream: .cpu)
        XCTAssertEqual(w.dtype, .complex64)
        XCTAssertEqual(v.dtype, .complex64)
        XCTAssertEqual(w.shape, [2])
        XCTAssertEqual(v.shape, [2, 2])

        let wr = w.realPart()
        assertEqual(sorted(wr), matrix([2, 3], [2]), atol: 1e-5)
        assertEqual(w.imaginaryPart(), MLXArray.zeros([2]), atol: 1e-6)

        // A * v[:, i] == w[i] * v[:, i]
        let vr = v.realPart()
        assertEqual(matmul(a, vr), vr * wr.expandedDimensions(axis: 0), atol: 1e-5)

        let (w2, _) = eig(a, stream: .cpu)
        assertEqual(sorted(w2.realPart()), matrix([2, 3], [2]), atol: 1e-5)

        let ev1 = MLXLinalg.eigvals(a, stream: .cpu)
        let ev2 = eigvals(a, stream: .cpu)
        XCTAssertEqual(ev1.dtype, .complex64)
        assertEqual(sorted(ev1.realPart()), matrix([2, 3], [2]), atol: 1e-5)
        assertEqual(sorted(ev2.realPart()), matrix([2, 3], [2]), atol: 1e-5)
    }

    func testEigh() {
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3 (ascending)
        let s = matrix([2, 1, 1, 2], [2, 2])

        let (w, v) = MLXLinalg.eigh(s, stream: .cpu)
        assertEqual(w, matrix([1, 3], [2]), atol: 1e-5)
        assertEqual(matmul(s, v), v * w.expandedDimensions(axis: 0), atol: 1e-5)
        assertEqual(matmul(v.T, v), MLXArray.eye(2), atol: 1e-5)

        let (w2, v2) = eigh(s, stream: .cpu)
        assertEqual(w2, matrix([1, 3], [2]), atol: 1e-5)
        assertEqual(matmul(s, v2), v2 * w2.expandedDimensions(axis: 0), atol: 1e-5)

        // UPLO picks the triangle that is read. The other triangle holds 99,
        // which must be ignored.
        let upperOnly = matrix([2, 1, 99, 2], [2, 2])
        let lowerOnly = matrix([2, 99, 1, 2], [2, 2])
        let (wu, _) = eigh(upperOnly, UPLO: "U", stream: .cpu)
        assertEqual(wu, matrix([1, 3], [2]), atol: 1e-5)
        let (wl, _) = MLXLinalg.eigh(lowerOnly, UPLO: "L", stream: .cpu)
        assertEqual(wl, matrix([1, 3], [2]), atol: 1e-5)

        assertEqual(MLXLinalg.eigvalsh(s, stream: .cpu), matrix([1, 3], [2]), atol: 1e-5)
        assertEqual(eigvalsh(lowerOnly, stream: .cpu), matrix([1, 3], [2]), atol: 1e-5)
        assertEqual(
            eigvalsh(upperOnly, UPLO: "U", stream: .cpu), matrix([1, 3], [2]), atol: 1e-5)
        assertEqual(
            MLXLinalg.eigvalsh(upperOnly, UPLO: "U", stream: .cpu), matrix([1, 3], [2]),
            atol: 1e-5)
    }
}
