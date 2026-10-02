// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import XCTest

/// Tests for each loss function in MLXNN/Losses.swift. Each test compares the
/// result with a value from a plain Swift loop, for each reduction.
class LossCoverageTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    // MARK: - Helpers

    private func check(
        _ actual: MLXArray, _ expected: [Float], shape: [Int], tolerance: Float = 1e-5,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, shape, file: file, line: line)
        XCTAssertEqual(actual.dtype, .float32, file: file, line: line)
        let values = actual.asArray(Float.self)
        XCTAssertEqual(values.count, expected.count, file: file, line: line)
        for (v, e) in zip(values, expected) {
            XCTAssertEqual(
                v, e, accuracy: tolerance + 1e-5 * Swift.abs(e), file: file, line: line)
        }
    }

    /// Check `.none`, `.mean` and `.sum` against the per-element values.
    private func checkReductions(
        _ loss: (LossReduction) -> MLXArray, _ perElement: [Float], shape: [Int],
        tolerance: Float = 1e-5, file: StaticString = #filePath, line: UInt = #line
    ) {
        let total = perElement.reduce(0, +)
        check(loss(.none), perElement, shape: shape, tolerance: tolerance, file: file, line: line)
        check(loss(.sum), [total], shape: [], tolerance: tolerance, file: file, line: line)
        check(
            loss(.mean), [total / Float(perElement.count)], shape: [], tolerance: tolerance,
            file: file, line: line)
    }

    private func logSumExpRef(_ row: [Float]) -> Float {
        log(row.map { exp($0) }.reduce(0, +))
    }

    private func transpose(_ rows: [[Float]]) -> [Float] {
        var result = [Float]()
        for j in 0 ..< rows[0].count {
            for i in 0 ..< rows.count {
                result.append(rows[i][j])
            }
        }
        return result
    }

    // MARK: - LossReduction

    func testLossReduction() {
        let loss = MLXArray([1, 2, 3, 6] as [Float], [2, 2])
        check(LossReduction.none.reduce(loss: loss), [1, 2, 3, 6], shape: [2, 2])
        check(LossReduction.sum.reduce(loss: loss), [12], shape: [])
        check(LossReduction.mean.reduce(loss: loss), [3], shape: [])
        XCTAssertEqual(LossReduction(rawValue: "mean"), .mean)
        XCTAssertEqual(LossReduction.sum.rawValue, "sum")
    }

    // MARK: - crossEntropy

    private let ceRows: [[Float]] = [[1, 2, 3], [1, 0, -1]]

    func testCrossEntropyClassIndices() {
        let logits = MLXArray(ceRows.flatMap { $0 }, [2, 3])
        let targets = MLXArray([2, 0] as [Int32], [2])
        let classes = [2, 0]
        let expected = (0 ..< 2).map { logSumExpRef(ceRows[$0]) - ceRows[$0][classes[$0]] }
        checkReductions(
            { crossEntropy(logits: logits, targets: targets, reduction: $0) }, expected,
            shape: [2])
        // the default reduction is .none
        check(crossEntropy(logits: logits, targets: targets), expected, shape: [2])
    }

    func testCrossEntropyProbabilities() {
        let probs: [[Float]] = [[0.2, 0.3, 0.5], [1, 0, 0]]
        let logits = MLXArray(ceRows.flatMap { $0 }, [2, 3])
        let targets = MLXArray(probs.flatMap { $0 }, [2, 3])
        let expected = (0 ..< 2).map { i in
            logSumExpRef(ceRows[i]) - zip(ceRows[i], probs[i]).map { $0 * $1 }.reduce(0, +)
        }
        checkReductions(
            { crossEntropy(logits: logits, targets: targets, reduction: $0) }, expected,
            shape: [2])

        // the same data with the classes on axis 0
        let logitsT = MLXArray(transpose(ceRows), [3, 2])
        let targetsT = MLXArray(transpose(probs), [3, 2])
        check(
            crossEntropy(logits: logitsT, targets: targetsT, axis: 0), expected, shape: [2])
    }

    func testCrossEntropyWeights() {
        let logits = MLXArray(ceRows.flatMap { $0 }, [2, 3])
        let targets = MLXArray([1, 2] as [Int32], [2])
        let weights: [Float] = [0.5, 2]
        let classes = [1, 2]
        let expected = (0 ..< 2).map {
            (logSumExpRef(ceRows[$0]) - ceRows[$0][classes[$0]]) * weights[$0]
        }
        checkReductions(
            {
                crossEntropy(
                    logits: logits, targets: targets, weights: MLXArray(weights, [2]),
                    reduction: $0)
            }, expected, shape: [2])
    }

    func testCrossEntropyLabelSmoothing() {
        let logits = MLXArray(ceRows.flatMap { $0 }, [2, 3])
        let targets = MLXArray([2, 1] as [Int32], [2])
        let classes = [2, 1]
        let smoothing: Float = 0.1
        let expected = (0 ..< 2).map { i -> Float in
            let row = ceRows[i]
            let mean = row.reduce(0, +) / Float(row.count)
            return logSumExpRef(row) - (1 - smoothing) * row[classes[i]] - smoothing * mean
        }
        checkReductions(
            {
                crossEntropy(
                    logits: logits, targets: targets, labelSmoothing: smoothing, reduction: $0)
            }, expected, shape: [2])

        // label smoothing with weights
        let weights: [Float] = [3, 0.25]
        check(
            crossEntropy(
                logits: logits, targets: targets, weights: MLXArray(weights, [2]),
                labelSmoothing: smoothing),
            zip(expected, weights).map { $0 * $1 }, shape: [2])
    }

    // MARK: - binaryCrossEntropy

    func testBinaryCrossEntropyWithLogits() {
        let x: [Float] = [0.5, -1, 2, 0]
        let t: [Float] = [1, 0, 1, 0]
        let logits = MLXArray(x, [4])
        let targets = MLXArray(t, [4])
        let expected = zip(x, t).map { log(1 + exp($0)) - $1 * $0 }
        checkReductions(
            { binaryCrossEntropy(logits: logits, targets: targets, reduction: $0) }, expected,
            shape: [4])

        // the default reduction is .mean
        check(
            binaryCrossEntropy(logits: logits, targets: targets),
            [expected.reduce(0, +) / 4], shape: [])

        // weights
        let w: [Float] = [1, 2, 0.5, 3]
        check(
            binaryCrossEntropy(
                logits: logits, targets: targets, weights: MLXArray(w, [4]), reduction: .none),
            zip(expected, w).map { $0 * $1 }, shape: [4])
    }

    func testBinaryCrossEntropyWithProbabilities() {
        let p: [Float] = [0.9, 0.2, 0.6, 0.5]
        let t: [Float] = [1, 0, 1, 0]
        let probs = MLXArray(p, [4])
        let targets = MLXArray(t, [4])
        let expected = zip(p, t).map { -($1 * log($0) + (1 - $1) * log(1 - $0)) }
        checkReductions(
            {
                binaryCrossEntropy(
                    logits: probs, targets: targets, withLogits: false, reduction: $0)
            }, expected, shape: [4])

        // the log values are clipped at -100
        let edge = MLXArray([0, 1, 0, 1] as [Float], [4])
        let edgeTargets = MLXArray([1, 1, 0, 0] as [Float], [4])
        check(
            binaryCrossEntropy(
                logits: edge, targets: edgeTargets, withLogits: false, reduction: .none),
            [100, 0, 0, 100], shape: [4])

        // weights with probabilities
        let w: [Float] = [2, 1, 0, 4]
        check(
            binaryCrossEntropy(
                logits: probs, targets: targets, weights: MLXArray(w, [4]), withLogits: false,
                reduction: .sum),
            [zip(expected, w).map { $0 * $1 }.reduce(0, +)], shape: [])
    }

    // MARK: - l1Loss, mseLoss

    func testL1Loss() {
        let predictions = MLXArray([1, 2, 3, 4] as [Float], [2, 2])
        let targets = MLXArray([2, 2, 1, 8] as [Float], [2, 2])
        checkReductions(
            { l1Loss(predictions: predictions, targets: targets, reduction: $0) },
            [1, 0, 2, 4], shape: [2, 2])
        check(l1Loss(predictions: predictions, targets: targets), [7.0 / 4], shape: [])
    }

    func testMSELoss() {
        let predictions = MLXArray([1, 2, 3, 4] as [Float], [2, 2])
        let targets = MLXArray([2, 2, 1, 8] as [Float], [2, 2])
        checkReductions(
            { mseLoss(predictions: predictions, targets: targets, reduction: $0) },
            [1, 0, 4, 16], shape: [2, 2])
        check(mseLoss(predictions: predictions, targets: targets), [21.0 / 4], shape: [])
    }

    // MARK: - nllLoss

    func testNLLLoss() {
        let inputs = MLXArray([-1, -2, -3, -0.5, -1.5, -2.5] as [Float], [2, 3])
        let targets = MLXArray([1, 2] as [Int32], [2])
        checkReductions(
            { nllLoss(inputs: inputs, targets: targets, reduction: $0) }, [2, 2.5], shape: [2])
        check(nllLoss(inputs: inputs, targets: targets), [2, 2.5], shape: [2])
    }

    // MARK: - klDivLoss

    func testKLDivLoss() {
        let p: [[Float]] = [[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]]
        let q: [[Float]] = [[0.25, 0.25, 0.5], [0.3, 0.3, 0.4]]
        let logP = p.map { $0.map { log($0) } }
        let logQ = q.map { $0.map { log($0) } }
        let expected = (0 ..< 2).map { i in
            (0 ..< 3).map { j in q[i][j] * (logQ[i][j] - logP[i][j]) }.reduce(0, +)
        }
        let inputs = MLXArray(logP.flatMap { $0 }, [2, 3])
        let targets = MLXArray(logQ.flatMap { $0 }, [2, 3])
        checkReductions(
            { klDivLoss(inputs: inputs, targets: targets, reduction: $0) }, expected,
            shape: [2])

        // the distribution on axis 0
        check(
            klDivLoss(
                inputs: MLXArray(transpose(logP), [3, 2]),
                targets: MLXArray(transpose(logQ), [3, 2]), axis: 0),
            expected, shape: [2])
    }

    // MARK: - smoothL1Loss

    func testSmoothL1Loss() {
        let predictions = MLXArray([0, 0.5, 2, -3] as [Float], [4])
        let targets = MLXArray.zeros([4])

        // beta 1: 0.5 * d^2 / beta below beta, d - 0.5 * beta above
        checkReductions(
            { smoothL1Loss(predictions: predictions, targets: targets, reduction: $0) },
            [0, 0.125, 1.5, 2.5], shape: [4])
        check(
            smoothL1Loss(predictions: predictions, targets: targets), [4.125 / 4], shape: [])

        // beta 2
        check(
            smoothL1Loss(predictions: predictions, targets: targets, beta: 2, reduction: .none),
            [0, 0.0625, 1, 2], shape: [4])
    }

    // MARK: - tripletLoss

    private func tripletRef(
        _ a: [[Float]], _ p: [[Float]], _ n: [[Float]], power: Float, margin: Float,
        eps: Float
    ) -> [Float] {
        (0 ..< a.count).map { i in
            let dp = zip(a[i], p[i]).map { pow($0 - $1, power) }.reduce(0, +)
            let dn = zip(a[i], n[i]).map { pow($0 - $1, power) }.reduce(0, +)
            return Swift.max((dp + eps).squareRoot() - (dn + eps).squareRoot() + margin, 0)
        }
    }

    func testTripletLoss() {
        let a: [[Float]] = [[0, 0], [1, 1], [2, 0]]
        let p: [[Float]] = [[1, 0], [1, 2], [2, 1]]
        let n: [[Float]] = [[3, 4], [1, 1], [2, 2]]
        let anchors = MLXArray(a.flatMap { $0 }, [3, 2])
        let positives = MLXArray(p.flatMap { $0 }, [3, 2])
        let negatives = MLXArray(n.flatMap { $0 }, [3, 2])

        let expected = tripletRef(a, p, n, power: 2, margin: 1, eps: 1e-6)
        checkReductions(
            {
                tripletLoss(
                    anchors: anchors, positives: positives, negatives: negatives, reduction: $0)
            }, expected, shape: [3], tolerance: 1e-4)

        // margin, eps and p
        let expected2 = tripletRef(a, p, n, power: 4, margin: 5, eps: 0.01)
        check(
            tripletLoss(
                anchors: anchors, positives: positives, negatives: negatives, p: 4, margin: 5,
                eps: 0.01),
            expected2, shape: [3], tolerance: 1e-4)

        // samples on axis 0
        check(
            tripletLoss(
                anchors: MLXArray(transpose(a), [2, 3]),
                positives: MLXArray(transpose(p), [2, 3]),
                negatives: MLXArray(transpose(n), [2, 3]), axis: 0),
            expected, shape: [3], tolerance: 1e-4)
    }

    // MARK: - hingeLoss

    func testHingeLoss() {
        let inputs = MLXArray([0.5, -2, 2, 0, 3] as [Float], [5])
        let targets = MLXArray([1, 1, -1, -1, 1] as [Float], [5])
        checkReductions(
            { hingeLoss(inputs: inputs, targets: targets, reduction: $0) },
            [0.5, 3, 3, 1, 0], shape: [5])
        check(hingeLoss(inputs: inputs, targets: targets), [0.5, 3, 3, 1, 0], shape: [5])
    }

    // MARK: - huberLoss

    func testHuberLoss() {
        let inputs = MLXArray([0, 0.5, 2, -3] as [Float], [4])
        let targets = MLXArray.zeros([4])

        // delta 1: 0.5 * e^2 for |e| <= delta, delta * (|e| - 0.5 * delta) above
        checkReductions(
            { huberLoss(inputs: inputs, targets: targets, reduction: $0) },
            [0, 0.125, 1.5, 2.5], shape: [4])

        // delta 2
        check(
            huberLoss(inputs: inputs, targets: targets, delta: 2),
            [0, 0.125, 2, 4], shape: [4])
    }

    // MARK: - logCoshLoss

    func testLogCoshLoss() {
        let e: [Float] = [0, 0.5, -1, 2]
        let inputs = MLXArray(e, [4]) + 1
        let targets = MLXArray.ones([4])
        let expected = e.map { log(cosh($0)) }
        checkReductions(
            { logCoshLoss(inputs: inputs, targets: targets, reduction: $0) }, expected,
            shape: [4])
        check(logCoshLoss(inputs: inputs, targets: targets), expected, shape: [4])
    }

    // MARK: - cosineSimilarityLoss

    func testCosineSimilarityLoss() {
        let a: [[Float]] = [[1, 0, 0], [1, 2, 2], [3, 4, 0]]
        let b: [[Float]] = [[0, 1, 0], [2, 4, 4], [4, 3, 0]]
        let x1 = MLXArray(a.flatMap { $0 }, [3, 3])
        let x2 = MLXArray(b.flatMap { $0 }, [3, 3])
        let expected: [Float] = [0, 1, 24.0 / 25.0]

        checkReductions(
            { cosineSimilarityLoss(x1: x1, x2: x2, reduction: $0) }, expected, shape: [3])

        // the embedding on axis 0
        check(
            cosineSimilarityLoss(
                x1: MLXArray(transpose(a), [3, 3]), x2: MLXArray(transpose(b), [3, 3]), axis: 0),
            expected, shape: [3])

        // eps is the minimum of the denominator: 0.001 / max(0.001, 1)
        let small = MLXArray([0.001, 0, 0] as [Float], [1, 3])
        let unit = MLXArray([1, 0, 0] as [Float], [1, 3])
        check(cosineSimilarityLoss(x1: small, x2: unit, eps: 1), [0.001], shape: [1])
        check(cosineSimilarityLoss(x1: small, x2: unit), [1], shape: [1])
    }

    func testCosineSimilarityLossComplex() {
        // complex inputs with a zero imaginary part give the real result
        let a: [Float] = [1, 2, 2, 3, 4, 0]
        let b: [Float] = [2, 4, 4, 4, 3, 0]
        let x1 = MLXArray(a, [2, 3]).asType(.complex64)
        let x2 = MLXArray(b, [2, 3]).asType(.complex64)
        let loss = cosineSimilarityLoss(x1: x1, x2: x2)
        XCTAssertEqual(loss.dtype, .complex64)
        check(loss.realPart(), [1, 24.0 / 25.0], shape: [2])
    }
}
