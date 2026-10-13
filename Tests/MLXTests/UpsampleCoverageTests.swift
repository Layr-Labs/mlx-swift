// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import XCTest

/// One output position: the source indices and their weights.
private typealias Taps = [(index: Int, weight: Float)]

/// Source position of output `i` (PyTorch convention).
///
/// With `alignCorners` the first and last samples of input and output match.
/// Otherwise the pixel centers are matched: `(i + 0.5) / scale - 0.5`.
private func sourcePosition(_ i: Int, n: Int, m: Int, scale: Float, alignCorners: Bool) -> Float {
    if alignCorners {
        return Float(i) * Float(n - 1) / Float(m - 1)
    }
    return (Float(i) + 0.5) / scale - 0.5
}

private func nearestTaps(n: Int, scale: Float) -> [Taps] {
    let m = Int(scale * Float(n))
    return (0 ..< m).map { i -> Taps in
        let index: Int
        if m > n {
            // the nearest source center, rounded
            index = Int(((Float(i) + 0.5) * Float(n) / Float(m) - 0.5).rounded())
        } else {
            index = Int(Float(i) * Float(n) / Float(m))
        }
        return [(Swift.max(index, 0), 1)]
    }
}

private func linearTaps(n: Int, scale: Float, alignCorners: Bool) -> [Taps] {
    let m = Int(scale * Float(n))
    return (0 ..< m).map { i -> Taps in
        var p = sourcePosition(i, n: n, m: m, scale: scale, alignCorners: alignCorners)
        p = Swift.min(Swift.max(p, 0), Float(n - 1))
        let left = Int(p.rounded(.down))
        let right = Int(p.rounded(.up))
        let w = p - Float(left)
        return [(left, 1 - w), (right, w)]
    }
}

/// The Keys cubic convolution kernel with a = -0.75.
private func keys(_ t: Float) -> Float {
    let a: Float = -0.75
    let x = Swift.abs(t)
    if x <= 1 {
        return (a + 2) * x * x * x - (a + 3) * x * x + 1
    } else if x < 2 {
        return a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a
    }
    return 0
}

private func cubicTaps(n: Int, scale: Float, alignCorners: Bool) -> [Taps] {
    let m = Int(scale * Float(n))
    return (0 ..< m).map { i -> Taps in
        let p = sourcePosition(i, n: n, m: m, scale: scale, alignCorners: alignCorners)
        let f = p.rounded(.down)
        // four neighbours; indices outside the input repeat the border value
        return (-1 ... 2).map { k -> (index: Int, weight: Float) in
            let index = Swift.min(Swift.max(Int(f) + k, 0), n - 1)
            return (index, keys(p - (f + Float(k))))
        }
    }
}

/// Interpolate a [1, N, C] signal with the given taps. Result is [M * C] values.
private func apply1D(_ x: [Float], channels c: Int, _ taps: [Taps]) -> [Float] {
    var out = [Float]()
    for t in taps {
        for ch in 0 ..< c {
            out.append(t.reduce(0) { $0 + $1.weight * x[$1.index * c + ch] })
        }
    }
    return out
}

/// Interpolate a [1, H, W, C] image with the given taps per axis.
private func apply2D(_ x: [Float], width w: Int, channels c: Int, _ rows: [Taps], _ cols: [Taps])
    -> [Float]
{
    var out = [Float]()
    for r in rows {
        for col in cols {
            for ch in 0 ..< c {
                var v: Float = 0
                for (ri, rw) in r {
                    for (ci, cw) in col {
                        v += rw * cw * x[(ri * w + ci) * c + ch]
                    }
                }
                out.append(v)
            }
        }
    }
    return out
}

private func signal(_ count: Int) -> [Float] {
    (0 ..< count).map { Float(($0 * 7) % 11) - 3.5 }
}

/// Tests for Upsample in MLXNN/Upsample.swift: nearest, linear and cubic modes, with
/// and without alignCorners, on 1D and 2D inputs. Each result is compared with a plain
/// Swift interpolation loop.
class UpsampleCoverageTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    private func check(
        _ actual: MLXArray, _ expected: [Float], shape: [Int], file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, shape, file: file, line: line)
        XCTAssertEqual(actual.dtype, .float32, file: file, line: line)
        let values = actual.asArray(Float.self)
        XCTAssertEqual(values.count, expected.count, file: file, line: line)
        for (i, (v, e)) in zip(values, expected).enumerated() {
            XCTAssertEqual(
                v, e, accuracy: 1e-4 + 1e-5 * Swift.abs(e), "index \(i)", file: file, line: line)
        }
    }

    // MARK: - Fields

    func testFields() {
        let up = Upsample(scaleFactor: [2.0, 1.5], mode: .cubic(alignCorners: true))
        XCTAssertEqual(up.scaleFactor.asArray(dimensions: 2), [2.0, 1.5])
        if case .cubic(let alignCorners) = up.mode {
            XCTAssertTrue(alignCorners)
        } else {
            XCTFail("mode is not cubic")
        }

        let defaults = Upsample(scaleFactor: 3.0)
        XCTAssertEqual(defaults.scaleFactor.asArray(dimensions: 2), [3.0, 3.0])
        guard case .nearest = defaults.mode else {
            XCTFail("the default mode is not nearest")
            return
        }
    }

    // MARK: - Nearest

    func testNearestIntegerScale1D() {
        let x = signal(3 * 2)
        let up = Upsample(scaleFactor: 3.0, mode: .nearest)
        check(
            up(MLXArray(x, [1, 3, 2])), apply1D(x, channels: 2, nearestTaps(n: 3, scale: 3)),
            shape: [1, 9, 2])
    }

    func testNearestIntegerScalePerAxis2D() {
        let x = signal(2 * 3)
        let up = Upsample(scaleFactor: [2.0, 3.0], mode: .nearest)
        let expected = apply2D(
            x, width: 3, channels: 1, nearestTaps(n: 2, scale: 2), nearestTaps(n: 3, scale: 3))
        check(up(MLXArray(x, [1, 2, 3, 1])), expected, shape: [1, 4, 9, 1])

        // a batch of two: each item is upsampled alone
        let batch = MLXArray(x + x.map { $0 * 10 }, [2, 2, 3, 1])
        check(up(batch), expected + expected.map { $0 * 10 }, shape: [2, 4, 9, 1])
    }

    func testNearestFloatScale1D() {
        // up: 3 -> 4
        let x = signal(3 * 2)
        let taps = nearestTaps(n: 3, scale: 1.5)
        XCTAssertEqual(taps.map { $0[0].index }, [0, 1, 1, 2])
        check(
            Upsample(scaleFactor: 1.5)(MLXArray(x, [1, 3, 2])),
            apply1D(x, channels: 2, taps), shape: [1, 4, 2])

        // down: 5 -> 2
        let y = signal(5 * 2)
        let down = nearestTaps(n: 5, scale: 0.5)
        XCTAssertEqual(down.map { $0[0].index }, [0, 2])
        check(
            Upsample(scaleFactor: 0.5)(MLXArray(y, [1, 5, 2])),
            apply1D(y, channels: 2, down), shape: [1, 2, 2])
    }

    func testNearestFloatScale2D() {
        // one scale is not a whole number, so both axes use the index path
        let x = signal(3 * 2 * 2)
        let up = Upsample(scaleFactor: [1.5, 2.0], mode: .nearest)
        let expected = apply2D(
            x, width: 2, channels: 2, nearestTaps(n: 3, scale: 1.5), nearestTaps(n: 2, scale: 2))
        check(up(MLXArray(x, [1, 3, 2, 2])), expected, shape: [1, 4, 4, 2])
    }

    // MARK: - Linear

    func testLinear1D() {
        let x = signal(4 * 2)
        let input = MLXArray(x, [1, 4, 2])
        for scale: Float in [2, 1.5] {
            for alignCorners in [false, true] {
                let up = Upsample(
                    scaleFactor: FloatOrArray(scale), mode: .linear(alignCorners: alignCorners))
                let m = Int(scale * 4)
                check(
                    up(input),
                    apply1D(
                        x, channels: 2,
                        linearTaps(n: 4, scale: scale, alignCorners: alignCorners)),
                    shape: [1, m, 2])
            }
        }
        // the default is alignCorners false
        check(
            Upsample(scaleFactor: 2.0, mode: .linear())(input),
            apply1D(x, channels: 2, linearTaps(n: 4, scale: 2, alignCorners: false)),
            shape: [1, 8, 2])
    }

    func testLinear1DHandValues() {
        // [0, 4] scaled by 2 with alignCorners: positions 0, 1/3, 2/3, 1
        let input = MLXArray([0, 4] as [Float], [1, 2, 1])
        check(
            Upsample(scaleFactor: 2.0, mode: .linear(alignCorners: true))(input),
            [0, 4.0 / 3, 8.0 / 3, 4], shape: [1, 4, 1])
        // without alignCorners: positions -0.25, 0.25, 0.75, 1.25, clipped to [0, 1]
        check(
            Upsample(scaleFactor: 2.0, mode: .linear(alignCorners: false))(input),
            [0, 1, 3, 4], shape: [1, 4, 1])
    }

    func testLinear2D() {
        let x = signal(3 * 4 * 2)
        let input = MLXArray(x, [1, 3, 4, 2])
        for alignCorners in [false, true] {
            let up = Upsample(scaleFactor: [2.0, 1.5], mode: .linear(alignCorners: alignCorners))
            let expected = apply2D(
                x, width: 4, channels: 2,
                linearTaps(n: 3, scale: 2, alignCorners: alignCorners),
                linearTaps(n: 4, scale: 1.5, alignCorners: alignCorners))
            check(up(input), expected, shape: [1, 6, 6, 2])
        }
    }

    // MARK: - Cubic

    func testCubic1D() {
        let x = signal(4 * 2)
        let input = MLXArray(x, [1, 4, 2])
        for scale: Float in [2, 1.5] {
            for alignCorners in [false, true] {
                let up = Upsample(
                    scaleFactor: FloatOrArray(scale), mode: .cubic(alignCorners: alignCorners))
                let m = Int(scale * 4)
                check(
                    up(input),
                    apply1D(
                        x, channels: 2,
                        cubicTaps(n: 4, scale: scale, alignCorners: alignCorners)),
                    shape: [1, m, 2])
            }
        }
        // the default is alignCorners false
        check(
            Upsample(scaleFactor: 2.0, mode: .cubic())(input),
            apply1D(x, channels: 2, cubicTaps(n: 4, scale: 2, alignCorners: false)),
            shape: [1, 8, 2])
    }

    func testCubic2D() {
        let x = signal(3 * 4)
        let input = MLXArray(x, [1, 3, 4, 1])
        for alignCorners in [false, true] {
            let up = Upsample(scaleFactor: [2.0, 1.5], mode: .cubic(alignCorners: alignCorners))
            let expected = apply2D(
                x, width: 4, channels: 1,
                cubicTaps(n: 3, scale: 2, alignCorners: alignCorners),
                cubicTaps(n: 4, scale: 1.5, alignCorners: alignCorners))
            check(up(input), expected, shape: [1, 6, 6, 1])
        }
    }

    func testCubicKeepsConstant() {
        // the cubic weights add up to 1, so a constant input stays constant
        let input = MLXArray.full([1, 3, 3, 2], values: MLXArray(Float(5)))
        let up = Upsample(scaleFactor: 2.0, mode: .cubic(alignCorners: false))
        check(up(input), [Float](repeating: 5, count: 6 * 6 * 2), shape: [1, 6, 6, 2])
    }
}
