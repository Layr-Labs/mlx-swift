// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

/// Tests for `Source/MLX/Random.swift`: the `MLXRandom` functions and the
/// free functions with the same names.
///
/// Each sampler is checked for:
/// - seeded determinism (the same key gives the same values, a different key
///   gives different values)
/// - shape and dtype
/// - range
/// - mean and variance of a few thousand samples, with a wide tolerance
class RandomCoverageTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    private let sampleCount = 4000

    private func values(_ array: MLXArray) -> [Double] {
        array.asArray(Float.self).map { Double($0) }
    }

    private func moments(_ xs: [Double]) -> (mean: Double, variance: Double) {
        let mean = xs.reduce(0, +) / Double(xs.count)
        let variance = xs.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(xs.count)
        return (mean, variance)
    }

    // MARK: - seed, key, split

    func testSeedFreeFunction() {
        MLX.seed(42)
        let a = MLX.uniform(Float(0) ..< Float(1), [16])
        MLX.seed(42)
        let b = MLX.uniform(Float(0) ..< Float(1), [16])
        MLX.seed(43)
        let c = MLX.uniform(Float(0) ..< Float(1), [16])

        assertEqual(a, b)
        assertNotEqual(a, c)
    }

    func testKeyFreeFunction() {
        let k1 = MLX.key(5)
        let k2 = MLX.key(5)
        let k3 = MLX.key(6)

        XCTAssertEqual(k1.shape, [2])
        XCTAssertEqual(k1.dtype, .uint32)
        XCTAssertEqual(k1.asArray(UInt32.self), k2.asArray(UInt32.self))
        XCTAssertNotEqual(k1.asArray(UInt32.self), k3.asArray(UInt32.self))
        XCTAssertEqual(k1.asArray(UInt32.self), MLXRandom.key(5).asArray(UInt32.self))
    }

    func testSplitFreeFunctions() {
        let key = MLXRandom.key(9)

        let keys = MLX.split(key: key, into: 3)
        XCTAssertEqual(keys.count, 3)
        for k in keys {
            XCTAssertEqual(k.shape, [2])
            XCTAssertEqual(k.dtype, .uint32)
        }
        let raw = keys.map { $0.asArray(UInt32.self) }
        XCTAssertEqual(Set(raw.map { "\($0)" }).count, 3, "keys are not distinct: \(raw)")

        // the free function and the MLXRandom function agree
        let keys2 = MLXRandom.split(key: key, into: 3)
        XCTAssertEqual(raw, keys2.map { $0.asArray(UInt32.self) })

        let (a, b) = MLX.split(key: key)
        let (a2, b2) = MLXRandom.split(key: key)
        XCTAssertEqual(a.asArray(UInt32.self), a2.asArray(UInt32.self))
        XCTAssertEqual(b.asArray(UInt32.self), b2.asArray(UInt32.self))
        XCTAssertNotEqual(a.asArray(UInt32.self), b.asArray(UInt32.self))
    }

    func testRandomStateDeterminism() {
        // two states with the same seed give the same sequence of values
        let s1 = MLXRandom.RandomState(seed: 3)
        let s2 = MLXRandom.RandomState(seed: 3)
        let first1 = MLX.uniform(Float(0) ..< Float(1), [8], key: s1)
        let first2 = MLX.uniform(Float(0) ..< Float(1), [8], key: s2)
        let second1 = MLX.uniform(Float(0) ..< Float(1), [8], key: s1)
        assertEqual(first1, first2)

        // the state moves on after each use
        assertNotEqual(first1, second1)
    }

    // MARK: - uniform

    func testUniformFloatRange() {
        let key = MLXRandom.key(1)
        let a = MLXRandom.uniform(Float(2) ..< Float(5), [sampleCount], key: key)
        let b = MLX.uniform(Float(2) ..< Float(5), [sampleCount], key: key)
        let c = MLX.uniform(Float(2) ..< Float(5), [sampleCount], key: MLXRandom.key(2))

        XCTAssertEqual(a.shape, [sampleCount])
        XCTAssertEqual(a.dtype, .float32)
        assertEqual(a, b)
        assertNotEqual(a, c)

        let xs = values(a)
        XCTAssertTrue(xs.allSatisfy { $0 >= 2 && $0 < 5 })
        // uniform on [2, 5): mean 3.5, variance 9 / 12 = 0.75
        let (mean, variance) = moments(xs)
        XCTAssertEqual(mean, 3.5, accuracy: 0.1)
        XCTAssertEqual(variance, 0.75, accuracy: 0.1)
    }

    func testUniformGenericRange() {
        // a Range<Double> uses the generic overload
        let key = MLXRandom.key(3)
        let a = MLXRandom.uniform(Double(-1) ..< Double(1), [sampleCount], key: key)
        let b = MLX.uniform(Double(-1) ..< Double(1), [sampleCount], key: key)

        XCTAssertEqual(a.dtype, .float32)
        assertEqual(a, b)

        let xs = values(a)
        XCTAssertTrue(xs.allSatisfy { $0 >= -1 && $0 < 1 })
        let (mean, variance) = moments(xs)
        XCTAssertEqual(mean, 0, accuracy: 0.1)
        XCTAssertEqual(variance, 4.0 / 12, accuracy: 0.05)

        // the type argument sets the dtype of the result
        let h = MLX.uniform(Double(0) ..< Double(1), [8], type: Float16.self, key: key)
        XCTAssertEqual(h.dtype, .float16)
        XCTAssertEqual(h.shape, [8])
    }

    func testUniformLowHigh() {
        let key = MLXRandom.key(4)
        let low = MLXArray([Float(0), Float(10)])
        let high = MLXArray([Float(1), Float(20)])

        // shape comes from low
        let a = MLXRandom.uniform(low: low, high: high, key: key)
        XCTAssertEqual(a.shape, [2])
        let av = values(a)
        XCTAssertTrue(av[0] >= 0 && av[0] < 1, "\(av)")
        XCTAssertTrue(av[1] >= 10 && av[1] < 20, "\(av)")

        // the free function gives the same values
        assertEqual(MLX.uniform(low: low, high: high, key: key), a)

        // explicit shape: the bounds broadcast over the rows
        let b = MLX.uniform(low: low, high: high, [sampleCount, 2], key: key)
        XCTAssertEqual(b.shape, [sampleCount, 2])
        XCTAssertEqual(b.dtype, .float32)
        let col0 = values(b[0..., 0])
        let col1 = values(b[0..., 1])
        XCTAssertTrue(col0.allSatisfy { $0 >= 0 && $0 < 1 })
        XCTAssertTrue(col1.allSatisfy { $0 >= 10 && $0 < 20 })
        XCTAssertEqual(moments(col0).mean, 0.5, accuracy: 0.05)
        XCTAssertEqual(moments(col1).mean, 15, accuracy: 0.5)

        // explicit dtype
        let c = MLXRandom.uniform(
            low: Float(-3), high: Float(3), [sampleCount], dtype: .float16, key: key)
        XCTAssertEqual(c.dtype, .float16)
        XCTAssertEqual(c.shape, [sampleCount])
        let cv = values(c)
        XCTAssertTrue(cv.allSatisfy { $0 >= -3 && $0 <= 3 })
        XCTAssertEqual(moments(cv).mean, 0, accuracy: 0.2)

        let d = MLX.uniform(low: Float(-3), high: Float(3), [4], dtype: .bfloat16, key: key)
        XCTAssertEqual(d.dtype, .bfloat16)
        XCTAssertEqual(d.shape, [4])

        let e = MLX.uniform(low: low, high: high, [3, 2], type: Float16.self, key: key)
        XCTAssertEqual(e.dtype, .float16)
        XCTAssertEqual(e.shape, [3, 2])
    }

    // MARK: - normal

    func testNormal() {
        let key = MLXRandom.key(5)
        let a = MLXRandom.normal([sampleCount], loc: 2, scale: 3, key: key)
        let b = MLX.normal([sampleCount], loc: 2, scale: 3, key: key)
        let c = MLX.normal([sampleCount], loc: 2, scale: 3, key: MLXRandom.key(6))

        XCTAssertEqual(a.shape, [sampleCount])
        XCTAssertEqual(a.dtype, .float32)
        assertEqual(a, b)
        assertNotEqual(a, c)

        let (mean, variance) = moments(values(a))
        XCTAssertEqual(mean, 2, accuracy: 0.3)
        XCTAssertEqual(variance.squareRoot(), 3, accuracy: 0.3)

        // dtype forms
        let h = MLXRandom.normal([2, 3], dtype: .float16, key: key)
        XCTAssertEqual(h.dtype, .float16)
        XCTAssertEqual(h.shape, [2, 3])

        let bf = MLX.normal([5], dtype: .bfloat16, loc: 1, scale: 0.5, key: key)
        XCTAssertEqual(bf.dtype, .bfloat16)
        XCTAssertEqual(bf.shape, [5])

        let t = MLX.normal([7], type: Float16.self, key: key)
        XCTAssertEqual(t.dtype, .float16)
        XCTAssertEqual(t.shape, [7])
    }

    func testMultivariateNormal() {
        let key = MLXRandom.key(7)
        let mean = MLXArray([Float(1), Float(-1)])
        let cov = MLXArray([Float(1), 0.5, 0.5, 2], [2, 2])

        let a = MLXRandom.multivariateNormal(
            mean: mean, covariance: cov, shape: [sampleCount], dtype: .float32, key: key,
            stream: .cpu)
        let b = MLX.multivariateNormal(
            mean: mean, covariance: cov, shape: [sampleCount], dtype: .float32, key: key,
            stream: .cpu)

        XCTAssertEqual(a.shape, [sampleCount, 2])
        XCTAssertEqual(a.dtype, .float32)
        assertEqual(a, b)

        let x = values(a[0..., 0])
        let y = values(a[0..., 1])
        let mx = moments(x)
        let my = moments(y)
        XCTAssertEqual(mx.mean, 1, accuracy: 0.15)
        XCTAssertEqual(my.mean, -1, accuracy: 0.15)
        XCTAssertEqual(mx.variance, 1, accuracy: 0.2)
        XCTAssertEqual(my.variance, 2, accuracy: 0.3)

        var covXY = 0.0
        for i in 0 ..< x.count {
            covXY += (x[i] - mx.mean) * (y[i] - my.mean)
        }
        covXY /= Double(x.count)
        XCTAssertEqual(covXY, 0.5, accuracy: 0.15)
    }

    // MARK: - permutation

    func testPermutationOfCount() {
        let key = MLXRandom.key(8)
        let p = MLXRandom.permutation(10, key: key)
        let p2 = MLXRandom.permutation(10, key: key)

        XCTAssertEqual(p.shape, [10])
        let pv = p.asArray(Int32.self)
        XCTAssertEqual(pv.sorted(), Array(0 ..< 10).map { Int32($0) })
        XCTAssertEqual(pv, p2.asArray(Int32.self))
    }

    func testPermutationOfArray() {
        // rows are [10 * i, 10 * i + 1]; a row permutation keeps each row whole
        let rows: [Int32] = [0, 1, 10, 11, 20, 21, 30, 31, 40, 41]
        let a = MLXArray(rows, [5, 2])
        let key = MLXRandom.key(9)
        let p = MLXRandom.permutation(a, key: key)

        XCTAssertEqual(p.shape, [5, 2])
        let first = p[0..., 0].asArray(Int32.self)
        let second = p[0..., 1].asArray(Int32.self)
        XCTAssertEqual(first.sorted(), [0, 10, 20, 30, 40])
        XCTAssertEqual(zip(first, second).map { $1 - $0 }, [1, 1, 1, 1, 1])

        // axis 1 swaps (or keeps) the two columns of all rows together
        let q = MLXRandom.permutation(a, axis: 1, key: key)
        let qv = q.asArray(Int32.self)
        XCTAssertTrue(qv == rows || qv == [1, 0, 11, 10, 21, 20, 31, 30, 41, 40], "\(qv)")
    }

    // MARK: - randInt

    func testRandIntRange() {
        let key = MLXRandom.key(10)
        let a = MLXRandom.randInt(Int32(3) ..< Int32(9), [sampleCount], key: key)
        let b = MLX.randInt(Int32(3) ..< Int32(9), [sampleCount], key: key)
        let c = MLX.randInt(Int32(3) ..< Int32(9), [sampleCount], key: MLXRandom.key(11))

        XCTAssertEqual(a.dtype, .int32)
        XCTAssertEqual(a.shape, [sampleCount])
        assertEqual(a, b)
        assertNotEqual(a, c)

        let xs = a.asArray(Int32.self)
        XCTAssertEqual(xs.min(), 3)
        XCTAssertEqual(xs.max(), 8)
        // uniform on 3 ... 8: mean 5.5, variance (6 * 6 - 1) / 12
        let (mean, variance) = moments(xs.map { Double($0) })
        XCTAssertEqual(mean, 5.5, accuracy: 0.15)
        XCTAssertEqual(variance, 35.0 / 12, accuracy: 0.25)

        // other integer types
        let u8 = MLX.randInt(UInt8(0) ..< UInt8(4), [10], key: key)
        XCTAssertEqual(u8.dtype, .uint8)
        XCTAssertTrue(u8.asArray(UInt8.self).allSatisfy { $0 < 4 })
    }

    func testRandIntLowHigh() {
        let key = MLXRandom.key(12)

        let a = MLX.randInt(low: 0, high: 5, [4, 3], key: key)
        XCTAssertEqual(a.shape, [4, 3])
        XCTAssertEqual(a.dtype, .int32)
        XCTAssertTrue(a.asArray(Int32.self).allSatisfy { $0 >= 0 && $0 < 5 })
        assertEqual(MLXRandom.randInt(low: 0, high: 5, [4, 3], key: key), a)

        let low = MLXArray([Int32(0), Int32(100)])
        let high = MLXArray([Int32(10), Int32(110)])
        let b = MLX.randInt(low: low, high: high, key: key)
        XCTAssertEqual(b.shape, [2])
        let bv = b.asArray(Int32.self)
        XCTAssertTrue(bv[0] >= 0 && bv[0] < 10, "\(bv)")
        XCTAssertTrue(bv[1] >= 100 && bv[1] < 110, "\(bv)")

        let c = MLX.randInt(low: low, high: high, [6, 2], type: Int16.self, key: key)
        XCTAssertEqual(c.dtype, .int16)
        XCTAssertEqual(c.shape, [6, 2])
        let cv = c.asArray(Int16.self)
        for row in 0 ..< 6 {
            XCTAssertTrue(cv[row * 2] >= 0 && cv[row * 2] < 10, "\(cv)")
            XCTAssertTrue(cv[row * 2 + 1] >= 100 && cv[row * 2 + 1] < 110, "\(cv)")
        }
    }

    // MARK: - bernoulli

    func testBernoulli() {
        let key = MLXRandom.key(13)

        let a = MLX.bernoulli([sampleCount], key: key)
        XCTAssertEqual(a.dtype, .bool)
        XCTAssertEqual(a.shape, [sampleCount])
        assertEqual(a, MLXRandom.bernoulli([sampleCount], key: key))
        let fa = Double(a.asArray(Bool.self).filter { $0 }.count) / Double(sampleCount)
        XCTAssertEqual(fa, 0.5, accuracy: 0.05)

        let b = MLX.bernoulli(Float(0.3), [sampleCount], key: key)
        XCTAssertEqual(b.shape, [sampleCount])
        let fb = Double(b.asArray(Bool.self).filter { $0 }.count) / Double(sampleCount)
        XCTAssertEqual(fb, 0.3, accuracy: 0.05)

        // p = 0 is never true and p = 1 is always true; the shape comes from p
        let p = MLXArray([Float(0), Float(1), Float(0), Float(1)])
        let c = MLX.bernoulli(p, key: key)
        XCTAssertEqual(c.shape, [4])
        XCTAssertEqual(c.asArray(Bool.self), [false, true, false, true])
    }

    // MARK: - truncatedNormal

    func testTruncatedNormal() {
        let key = MLXRandom.key(14)

        // generic overload with a Range<Double>
        let a = MLXRandom.truncatedNormal(Double(-1) ..< Double(1), [sampleCount], key: key)
        let b = MLX.truncatedNormal(Double(-1) ..< Double(1), [sampleCount], key: key)
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.shape, [sampleCount])
        assertEqual(a, b)
        let xs = values(a)
        XCTAssertTrue(xs.allSatisfy { $0 >= -1 && $0 <= 1 })
        // symmetric bounds: mean 0; the variance of N(0, 1) cut to [-1, 1]
        // is 1 - 2 * phi(1) / (2 * Phi(1) - 1) = 0.2911
        let (mean, variance) = moments(xs)
        XCTAssertEqual(mean, 0, accuracy: 0.05)
        XCTAssertEqual(variance, 0.2911, accuracy: 0.04)

        // Range<Float> overload
        let c = MLX.truncatedNormal(Float(0) ..< Float(0.5), [100], key: key)
        XCTAssertEqual(c.shape, [100])
        XCTAssertTrue(values(c).allSatisfy { $0 >= 0 && $0 <= 0.5 })

        let h = MLX.truncatedNormal(Float(0) ..< Float(0.5), [3], type: Float16.self, key: key)
        XCTAssertEqual(h.dtype, .float16)
    }

    func testTruncatedNormalLowHigh() {
        let key = MLXRandom.key(15)
        let low = MLXArray([Float(-1), Float(2)])
        let high = MLXArray([Float(0), Float(3)])

        // shape comes from low
        let a = MLX.truncatedNormal(low: low, high: high, key: key)
        XCTAssertEqual(a.shape, [2])
        let av = values(a)
        XCTAssertTrue(av[0] >= -1 && av[0] <= 0, "\(av)")
        XCTAssertTrue(av[1] >= 2 && av[1] <= 3, "\(av)")

        let b = MLX.truncatedNormal(low: low, high: high, [50, 2], key: key)
        XCTAssertEqual(b.shape, [50, 2])
        XCTAssertTrue(values(b[0..., 0]).allSatisfy { $0 >= -1 && $0 <= 0 })
        XCTAssertTrue(values(b[0..., 1]).allSatisfy { $0 >= 2 && $0 <= 3 })

        let c = MLXRandom.truncatedNormal(
            low: Float(-2), high: Float(2), [20], dtype: .float16, key: key)
        XCTAssertEqual(c.dtype, .float16)
        XCTAssertEqual(c.shape, [20])
        XCTAssertTrue(values(c).allSatisfy { $0 >= -2 && $0 <= 2 })

        let d = MLX.truncatedNormal(low: Float(-2), high: Float(2), [20], dtype: .float32, key: key)
        XCTAssertEqual(d.dtype, .float32)
        XCTAssertTrue(values(d).allSatisfy { $0 >= -2 && $0 <= 2 })
        assertEqual(
            d,
            MLXRandom.truncatedNormal(
                low: Float(-2), high: Float(2), [20], dtype: .float32, key: key))

        let e = MLXRandom.truncatedNormal(
            low: low, high: high, [4, 2], type: Float16.self, key: key)
        XCTAssertEqual(e.dtype, .float16)
        XCTAssertEqual(e.shape, [4, 2])
    }

    // MARK: - gumbel

    func testGumbel() {
        let key = MLXRandom.key(16)
        let a = MLX.gumbel([sampleCount], key: key)
        let b = MLXRandom.gumbel([sampleCount], key: key)
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.shape, [sampleCount])
        assertEqual(a, b)
        assertNotEqual(a, MLX.gumbel([sampleCount], key: MLXRandom.key(17)))

        // standard Gumbel: mean = Euler's constant, variance = pi^2 / 6
        let (mean, variance) = moments(values(a))
        XCTAssertEqual(mean, 0.5772, accuracy: 0.1)
        XCTAssertEqual(variance, Double.pi * Double.pi / 6, accuracy: 0.3)

        let h = MLX.gumbel([3, 2], dtype: .float16, key: key)
        XCTAssertEqual(h.dtype, .float16)
        XCTAssertEqual(h.shape, [3, 2])
        let h2 = MLXRandom.gumbel([3, 2], dtype: .float16, key: key)
        assertEqual(h, h2)

        let t = MLX.gumbel([4], type: Float16.self, key: key)
        XCTAssertEqual(t.dtype, .float16)
    }

    // MARK: - categorical

    func testCategoricalWithMaskedLogits() {
        // -inf logits can never be picked, so each row has one possible class
        let ninf = -Float.infinity
        let logits = MLXArray([Float(0), ninf, ninf, ninf, ninf, Float(0)], [2, 3])
        let key = MLXRandom.key(18)

        let a = MLX.categorical(logits, key: key)
        XCTAssertEqual(a.dtype, .uint32)
        XCTAssertEqual(a.asArray(UInt32.self), [0, 2])

        // the distribution is along axis 0 of the transposed logits
        let b = MLX.categorical(logits.T, axis: 0, key: key)
        XCTAssertEqual(b.asArray(UInt32.self), [0, 2])

        let c = MLX.categorical(logits, count: 5, key: key)
        XCTAssertEqual(c.shape, [2, 5])
        XCTAssertEqual(c.asArray(UInt32.self), [0, 0, 0, 0, 0, 2, 2, 2, 2, 2])

        let d = MLX.categorical(logits, shape: [3, 2], key: key)
        XCTAssertEqual(d.shape, [3, 2])
        XCTAssertEqual(d.asArray(UInt32.self), [0, 2, 0, 2, 0, 2])

        let e = MLXRandom.categorical(logits, shape: [4, 2], key: key)
        XCTAssertEqual(e.shape, [4, 2])
        XCTAssertEqual(e.asArray(UInt32.self), [0, 2, 0, 2, 0, 2, 0, 2])
    }

    func testCategoricalFrequencies() {
        // probabilities 0.1, 0.2, 0.7
        let logits = log(MLXArray([Float(0.1), Float(0.2), Float(0.7)]))
        let key = MLXRandom.key(19)

        let a = MLX.categorical(logits, shape: [sampleCount], key: key)
        XCTAssertEqual(a.shape, [sampleCount])
        assertEqual(a, MLXRandom.categorical(logits, shape: [sampleCount], key: key))

        let xs = a.asArray(UInt32.self)
        XCTAssertTrue(xs.allSatisfy { $0 < 3 })
        let n = Double(sampleCount)
        XCTAssertEqual(Double(xs.filter { $0 == 0 }.count) / n, 0.1, accuracy: 0.03)
        XCTAssertEqual(Double(xs.filter { $0 == 1 }.count) / n, 0.2, accuracy: 0.04)
        XCTAssertEqual(Double(xs.filter { $0 == 2 }.count) / n, 0.7, accuracy: 0.04)

        let b = MLX.categorical(logits, count: sampleCount, key: key)
        XCTAssertEqual(b.shape, [sampleCount])
        let bs = b.asArray(UInt32.self)
        XCTAssertEqual(Double(bs.filter { $0 == 2 }.count) / n, 0.7, accuracy: 0.04)
    }

    // MARK: - laplace

    func testLaplace() {
        let key = MLXRandom.key(20)
        let a = MLX.laplace([sampleCount], dtype: .float32, loc: 1, scale: 2, key: key)
        let b = MLXRandom.laplace([sampleCount], dtype: .float32, loc: 1, scale: 2, key: key)
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.shape, [sampleCount])
        assertEqual(a, b)
        assertNotEqual(
            a, MLX.laplace([sampleCount], dtype: .float32, loc: 1, scale: 2, key: MLXRandom.key(21))
        )

        // Laplace(loc, b): mean = loc, variance = 2 * b^2 = 8
        let (mean, variance) = moments(values(a))
        XCTAssertEqual(mean, 1, accuracy: 0.25)
        XCTAssertEqual(variance, 8, accuracy: 1.5)

        let h = MLXRandom.laplace([2, 2], dtype: .float16, key: key)
        XCTAssertEqual(h.dtype, .float16)
        XCTAssertEqual(h.shape, [2, 2])
    }
}
