// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import XCTest

/// The input values for the element-wise activation tests.
private let activationInputs: [Float] = [-3, -1, -0.25, 0, 0.5, 2, 7]

private func sigmoidRef(_ x: Float) -> Float {
    1 / (1 + exp(-x))
}

private func softplusRef(_ x: Float) -> Float {
    log(1 + exp(x))
}

private func eluRef(_ x: Float, _ alpha: Float) -> Float {
    x > 0 ? x : alpha * (exp(x) - 1)
}

private func softmaxRef(_ v: [Float]) -> [Float] {
    let e = v.map { exp($0) }
    let s = e.reduce(0, +)
    return e.map { $0 / s }
}

/// Tests for each activation in MLXNN/Activations.swift: the free function and
/// the Module class. Each result is compared with a plain Swift formula.
class ActivationCoverageTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    // MARK: - Helpers

    private func input() -> MLXArray {
        MLXArray(activationInputs, [activationInputs.count])
    }

    /// Check an element-wise result against `f` applied to each input value.
    private func check(
        _ actual: MLXArray, _ f: (Float) -> Float, file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, [activationInputs.count], file: file, line: line)
        XCTAssertEqual(actual.dtype, .float32, file: file, line: line)
        let values = actual.asArray(Float.self)
        for (v, x) in zip(values, activationInputs) {
            let e = f(x)
            XCTAssertEqual(
                v, e, accuracy: 1e-5 + 1e-4 * Swift.abs(e), "x = \(x)", file: file, line: line)
        }
    }

    private func check(
        _ actual: MLXArray, _ expected: [Float], shape: [Int], file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, shape, file: file, line: line)
        let values = actual.asArray(Float.self)
        XCTAssertEqual(values.count, expected.count, file: file, line: line)
        for (v, e) in zip(values, expected) {
            XCTAssertEqual(v, e, accuracy: 1e-5 + 1e-4 * Swift.abs(e), file: file, line: line)
        }
    }

    // MARK: - Sigmoid, ReLU family

    func testSigmoid() {
        let x = input()
        check(MLXNN.sigmoid(x), sigmoidRef)
        check(Sigmoid()(x), sigmoidRef)
    }

    func testReLU() {
        let x = input()
        let ref: (Float) -> Float = { Swift.max($0, 0) }
        check(MLXNN.relu(x), ref)
        check(ReLU()(x), ref)
    }

    func testLeakyReLU() {
        let x = input()
        check(MLXNN.leakyRelu(x), { Swift.max(0.01 * $0, $0) })
        check(MLXNN.leakyRelu(x, negativeSlope: 0.2), { Swift.max(0.2 * $0, $0) })

        let layer = LeakyReLU(negativeSlope: 0.3)
        XCTAssertEqual(layer.negativeSlope, 0.3)
        check(layer(x), { Swift.max(0.3 * $0, $0) })
        check(LeakyReLU()(x), { Swift.max(0.01 * $0, $0) })
    }

    func testReLU6() {
        let x = input()
        let ref: (Float) -> Float = { Swift.min(Swift.max($0, 0), 6) }
        check(MLXNN.relu6(x), ref)
        check(ReLU6()(x), ref)
    }

    func testReLUSquared() {
        let x = input()
        let ref: (Float) -> Float = { Swift.max($0, 0) * Swift.max($0, 0) }
        check(MLXNN.reluSquared(x), ref)
        check(ReLUSquared()(x), ref)
    }

    func testPReLU() {
        let x = input()
        let ref: (Float) -> Float = { Swift.max(0, $0) + 0.25 * Swift.min(0, $0) }
        check(MLXNN.prelu(x, alpha: MLXArray([0.25] as [Float], [1])), ref)

        let layer = PReLU()
        XCTAssertEqual(layer.weight.shape, [1])
        check(layer(x), ref)

        // one alpha per channel
        let channels = PReLU(count: 3, value: 0.1)
        XCTAssertEqual(channels.weight.shape, [3])
        XCTAssertEqual(channels.weight.asArray(Float.self), [0.1, 0.1, 0.1])
        let values: [Float] = [-1, 2, -3, 4, -5, 6]
        let alphas: [Float] = [0.5, 1, 2]
        let x2 = MLXArray(values, [2, 3])
        check(
            MLXNN.prelu(x2, alpha: MLXArray(alphas, [3])),
            values.enumerated().map { i, v in v > 0 ? v : alphas[i % 3] * v }, shape: [2, 3])
        check(
            channels(x2), values.map { $0 > 0 ? $0 : 0.1 * $0 }, shape: [2, 3])
    }

    // MARK: - Exponential family

    func testELU() {
        let x = input()
        check(MLXNN.elu(x), { eluRef($0, 1) })
        check(MLXNN.elu(x, alpha: 0.5), { eluRef($0, 0.5) })

        let layer = ELU(alpha: 2)
        XCTAssertEqual(layer.alpha, 2)
        check(layer(x), { eluRef($0, 2) })
        check(ELU()(x), { eluRef($0, 1) })
    }

    func testCELU() {
        let ref: (Float, Float) -> Float = { x, a in
            Swift.max(x, 0) + a * (exp(Swift.min(x, 0) / a) - 1)
        }
        let x = input()
        check(MLXNN.celu(x), { ref($0, 1) })
        check(MLXNN.celu(x, alpha: 2), { ref($0, 2) })

        let layer = CELU(alpha: 0.5)
        XCTAssertEqual(layer.alpha, 0.5)
        check(layer(x), { ref($0, 0.5) })
        check(CELU()(x), { ref($0, 1) })
    }

    func testSELU() {
        let x = input()
        let ref: (Float) -> Float = { eluRef($0, 1.67326) * 1.0507 }
        check(MLXNN.selu(x), ref)
        check(SELU()(x), ref)
    }

    // MARK: - Softplus, Softsign, shrink

    func testSoftplus() {
        let x = input()
        check(MLXNN.softplus(x), softplusRef)
        check(Softplus()(x), softplusRef)
    }

    @available(*, deprecated)
    func testSoftplusDeprecatedNames() {
        let x = input()
        check(MLXNN.softPlus(x), softplusRef)
        check(SoftPlus()(x), softplusRef)
    }

    func testSoftsign() {
        let x = input()
        let ref: (Float) -> Float = { $0 / (1 + Swift.abs($0)) }
        check(MLXNN.softsign(x), ref)
        check(Softsign()(x), ref)
    }

    @available(*, deprecated)
    func testSoftsignDeprecatedNames() {
        let x = input()
        let ref: (Float) -> Float = { $0 / (1 + Swift.abs($0)) }
        check(MLXNN.softSign(x), ref)
        check(SoftSign()(x), ref)
    }

    func testSoftshrink() {
        let ref: (Float, Float) -> Float = { x, l in
            x > l ? x - l : (x < -l ? x + l : 0)
        }
        let x = input()
        check(MLXNN.softshrink(x), { ref($0, 0.5) })
        check(MLXNN.softshrink(x, lambda: 1), { ref($0, 1) })

        let layer = Softshrink(lambda: 2)
        XCTAssertEqual(layer.lambda, 2)
        check(layer(x), { ref($0, 2) })
        check(Softshrink()(x), { ref($0, 0.5) })
    }

    func testHardShrink() {
        let ref: (Float, Float) -> Float = { x, l in Swift.abs(x) > l ? x : 0 }
        let x = input()
        check(MLXNN.hardShrink(x), { ref($0, 0.5) })
        check(MLXNN.hardShrink(x, lambda: 1.5), { ref($0, 1.5) })

        let layer = HardShrink(lambda: 0.1)
        XCTAssertEqual(layer.lambda, 0.1)
        check(layer(x), { ref($0, 0.1) })
        check(HardShrink()(x), { ref($0, 0.5) })
    }

    func testHardTanh() {
        let ref: (Float, Float, Float) -> Float = { x, lo, hi in Swift.min(Swift.max(x, lo), hi) }
        let x = input()
        check(MLXNN.hardTanH(x), { ref($0, -1, 1) })
        check(MLXNN.hardTanH(x, min: -2, max: 3), { ref($0, -2, 3) })

        let layer = HardTanh(min: -0.5, max: 4)
        XCTAssertEqual(layer.min, -0.5)
        XCTAssertEqual(layer.max, 4)
        check(layer(x), { ref($0, -0.5, 4) })
        check(HardTanh()(x), { ref($0, -1, 1) })
    }

    func testHardSwish() {
        let x = input()
        let ref: (Float) -> Float = { $0 * Swift.min(Swift.max($0 + 3, 0), 6) / 6 }
        check(MLXNN.hardSwish(x), ref)
        check(HardSwish()(x), ref)
    }

    // MARK: - SiLU, LogSigmoid, Mish, Tanh

    func testSiLU() {
        let x = input()
        let ref: (Float) -> Float = { $0 * sigmoidRef($0) }
        check(MLXNN.silu(x), ref)
        check(SiLU()(x), ref)
    }

    func testLogSigmoid() {
        let x = input()
        let ref: (Float) -> Float = { -softplusRef(-$0) }
        check(MLXNN.logSigmoid(x), ref)
        check(LogSigmoid()(x), ref)
    }

    func testMish() {
        let x = input()
        let ref: (Float) -> Float = { $0 * tanh(softplusRef($0)) }
        check(MLXNN.mish(x), ref)
        check(Mish()(x), ref)
    }

    func testTanh() {
        check(Tanh()(input()), { tanh($0) })
    }

    // MARK: - GELU

    func testGELU() {
        let x = input()
        let exact: (Float) -> Float = { $0 * (1 + erf($0 / Float(2).squareRoot())) / 2 }
        let approximate: (Float) -> Float = { x in
            0.5 * x * (1 + tanh((2 / Float.pi).squareRoot() * (x + 0.044715 * x * x * x)))
        }
        let fast: (Float) -> Float = { $0 * sigmoidRef(1.702 * $0) }

        check(MLXNN.gelu(x), exact)
        check(MLXNN.geluApproximate(x), approximate)
        check(MLXNN.geluFastApproximate(x), fast)

        check(GELU()(x), exact)
        check(GELU(approximation: .none)(x), exact)
        check(GELU(approximation: .precise)(x), approximate)
        check(GELU(approximation: .tanh)(x), approximate)
        check(GELU(approximation: .fast)(x), fast)
    }

    // MARK: - Step

    func testStep() {
        let x = input()
        check(MLXNN.step(x).asType(.float32), { $0 > 0 ? 1 : 0 })
        check(MLXNN.step(x, threshold: 1).asType(.float32), { $0 > 1 ? 1 : 0 })

        let layer = Step(threshold: -0.5)
        XCTAssertEqual(layer.threshold, -0.5)
        check(layer(x).asType(.float32), { $0 > -0.5 ? 1 : 0 })
        check(Step()(x).asType(.float32), { $0 > 0 ? 1 : 0 })
    }

    // MARK: - GLU

    func testGLU() {
        // shape [2, 4]: a is columns 0..<2, b is columns 2..<4
        let rows: [[Float]] = [[1, 2, -1, 0.5], [-2, 0, 3, 1]]
        let x = MLXArray(rows.flatMap { $0 }, [2, 4])
        let expected = rows.flatMap { r in (0 ..< 2).map { r[$0] * sigmoidRef(r[$0 + 2]) } }
        check(MLXNN.glu(x), expected, shape: [2, 2])
        check(GLU()(x), expected, shape: [2, 2])

        // split on axis 0: shape [4, 2], a is rows 0..<2, b is rows 2..<4
        let cols: [[Float]] = [[1, -2], [2, 0], [-1, 3], [0.5, 1]]
        let x0 = MLXArray(cols.flatMap { $0 }, [4, 2])
        let expected0 = (0 ..< 2).flatMap { i in
            (0 ..< 2).map { j in cols[i][j] * sigmoidRef(cols[i + 2][j]) }
        }
        check(MLXNN.glu(x0, axis: 0), expected0, shape: [2, 2])
        let layer = GLU(axis: 0)
        XCTAssertEqual(layer.axis, 0)
        check(layer(x0), expected0, shape: [2, 2])
    }

    // MARK: - Softmax family

    private let matrix: [[Float]] = [[1, 2, 3], [-1, 0, 4]]

    /// Apply `f` to each row (axis -1) or each column (axis 0) of `matrix`.
    private func applyRef(axis: Int, _ f: ([Float]) -> [Float]) -> [Float] {
        if axis == 0 {
            let columns = (0 ..< 3).map { j in f(matrix.map { $0[j] }) }
            return (0 ..< 2).flatMap { i in columns.map { $0[i] } }
        }
        return matrix.flatMap { f($0) }
    }

    func testSoftmax() {
        let x = MLXArray(matrix.flatMap { $0 }, [2, 3])
        check(Softmax()(x), applyRef(axis: -1, softmaxRef), shape: [2, 3])
        let layer = Softmax(axis: 0)
        XCTAssertEqual(layer.axis, 0)
        check(layer(x), applyRef(axis: 0, softmaxRef), shape: [2, 3])
    }

    func testSoftmin() {
        let x = MLXArray(matrix.flatMap { $0 }, [2, 3])
        let ref: ([Float]) -> [Float] = { softmaxRef($0.map { -$0 }) }
        check(MLXNN.softmin(x), applyRef(axis: -1, ref), shape: [2, 3])
        check(MLXNN.softmin(x, axis: 0), applyRef(axis: 0, ref), shape: [2, 3])
        check(Softmin()(x), applyRef(axis: -1, ref), shape: [2, 3])
        let layer = Softmin(axis: 0)
        XCTAssertEqual(layer.axis, 0)
        check(layer(x), applyRef(axis: 0, ref), shape: [2, 3])
    }

    func testLogSoftmax() {
        let x = MLXArray(matrix.flatMap { $0 }, [2, 3])
        let ref: ([Float]) -> [Float] = { softmaxRef($0).map { log($0) } }
        check(MLXNN.logSoftmax(x), applyRef(axis: -1, ref), shape: [2, 3])
        check(MLXNN.logSoftmax(x, axis: 0), applyRef(axis: 0, ref), shape: [2, 3])
        check(LogSoftmax()(x), applyRef(axis: -1, ref), shape: [2, 3])
        let layer = LogSoftmax(axis: 0)
        XCTAssertEqual(layer.axis, 0)
        check(layer(x), applyRef(axis: 0, ref), shape: [2, 3])
    }

    @available(*, deprecated)
    func testSoftmaxDeprecatedNames() {
        let x = MLXArray(matrix.flatMap { $0 }, [2, 3])
        let logRef: ([Float]) -> [Float] = { softmaxRef($0).map { log($0) } }
        check(SoftMax()(x), applyRef(axis: -1, softmaxRef), shape: [2, 3])
        check(LogSoftMax()(x), applyRef(axis: -1, logRef), shape: [2, 3])
        check(MLXNN.logSoftMax(x), applyRef(axis: -1, logRef), shape: [2, 3])
        check(MLXNN.logSoftMax(x, axis: 0), applyRef(axis: 0, logRef), shape: [2, 3])
    }
}
