// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

@testable import MLXNN

/// Tests for `RNN`, `GRU` and `LSTM` in `Recurrent.swift`.
///
/// Each layer is compared with a reference that uses the layer's own weights
/// and plain ops, one time step at a time. Other tests check output shapes,
/// that seeded weights give the same output, and that the state can be carried
/// from one call to the next.
class RecurrentTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    private let inputSize = 3
    private let hiddenSize = 4

    /// A seeded [2, 5, 3] input: batch 2, sequence length 5, input size 3.
    private func input(batch: Int = 2, length: Int = 5, seed: UInt64 = 1) -> MLXArray {
        MLXRandom.normal([batch, length, inputSize], key: MLXRandom.key(seed))
    }

    /// Splits a [N, L, D] array into L arrays of shape [N, D].
    private func steps(_ x: MLXArray) -> [MLXArray] {
        split(x, parts: x.dim(-2), axis: -2).map { $0.squeezed(axis: -2) }
    }

    private func assertClose(
        _ a: MLXArray, _ b: MLXArray, _ message: String = "",
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(a.shape, b.shape, message, file: file, line: line)
        XCTAssertTrue(
            a.allClose(b, rtol: 1e-5, atol: 1e-5).item(Bool.self),
            "\(message)\n\(a)\n\(b)", file: file, line: line)
    }

    // MARK: - References with plain ops

    /// h = tanh(x Wxh^T + b + h Whh^T). The first step has no h term when `hidden` is nil.
    private func referenceRNN(_ rnn: RNN, _ x: MLXArray, hidden: MLXArray? = nil) -> MLXArray {
        var h = hidden
        var out = [MLXArray]()
        for xt in steps(x) {
            var pre = matmul(xt, rnn.wxh.T)
            if let b = rnn.bias {
                pre = pre + b
            }
            if let h {
                pre = pre + matmul(h, rnn.whh.T)
            }
            h = tanh(pre)
            out.append(h!)
        }
        return stacked(out, axis: -2)
    }

    /// The GRU as upstream MLX defines it. When `hidden` is nil the first step
    /// has no hidden term, and the `bhn` bias is part of that term.
    private func referenceGRU(_ gru: GRU, _ x: MLXArray, hidden: MLXArray? = nil) -> MLXArray {
        let h3 = gru.hiddenSize
        var h = hidden
        var out = [MLXArray]()
        for xt in steps(x) {
            var xp = matmul(xt, gru.wx.T)
            if let b = gru.b {
                xp = xp + b
            }
            let xParts = split(xp, indices: [h3, 2 * h3], axis: -1)
            var r = xParts[0]
            var z = xParts[1]
            var n = xParts[2]
            if let h {
                let hp = matmul(h, gru.wh.T)
                let hParts = split(hp, indices: [h3, 2 * h3], axis: -1)
                r = r + hParts[0]
                z = z + hParts[1]
                var hn = hParts[2]
                if let bhn = gru.bhn {
                    hn = hn + bhn
                }
                n = n + sigmoid(r) * hn
            }
            n = tanh(n)
            let zs = sigmoid(z)
            let next = (1 - zs) * n
            h = h.map { next + zs * $0 } ?? next
            out.append(h!)
        }
        return stacked(out, axis: -2)
    }

    /// i, f, g, o gates; c = f * c + i * g; h = o * tanh(c).
    private func referenceLSTM(
        _ lstm: LSTM, _ x: MLXArray, hidden: MLXArray? = nil, cell: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let size = lstm.hiddenSize
        var h = hidden
        var c = cell
        var hs = [MLXArray]()
        var cs = [MLXArray]()
        for xt in steps(x) {
            var pre = matmul(xt, lstm.wx.T)
            if let b = lstm.bias {
                pre = pre + b
            }
            if let h {
                pre = pre + matmul(h, lstm.wh.T)
            }
            let g4 = split(pre, indices: [size, 2 * size, 3 * size], axis: -1)
            let i = sigmoid(g4[0])
            let f = sigmoid(g4[1])
            let g = tanh(g4[2])
            let o = sigmoid(g4[3])
            let nextC = c.map { f * $0 + i * g } ?? i * g
            c = nextC
            h = o * tanh(nextC)
            hs.append(h!)
            cs.append(nextC)
        }
        return (stacked(hs, axis: -2), stacked(cs, axis: -2))
    }

    // MARK: - RNN

    func testRNNMatchesReference() {
        for bias in [true, false] {
            MLXRandom.seed(10)
            let rnn = RNN(inputSize: inputSize, hiddenSize: hiddenSize, bias: bias)
            let x = input()
            let y = rnn(x)
            XCTAssertEqual(y.shape, [2, 5, hiddenSize])
            assertClose(y, referenceRNN(rnn, x), "bias: \(bias)")

            let h0 = MLXRandom.normal([2, hiddenSize], key: MLXRandom.key(2))
            assertClose(rnn(x, hidden: h0), referenceRNN(rnn, x, hidden: h0), "bias: \(bias), h0")
        }
    }

    func testRNNCustomNonLinearity() {
        MLXRandom.seed(11)
        let rnn = RNN(inputSize: inputSize, hiddenSize: hiddenSize) { x, _ in maximum(x, 0) }
        let x = input()
        let y = rnn(x)
        // A ReLU output has no negative values, and a tanh output would.
        XCTAssertGreaterThanOrEqual(y.min().item(Float.self), 0)

        var h: MLXArray? = nil
        var out = [MLXArray]()
        for xt in steps(x) {
            var pre = matmul(xt, rnn.wxh.T) + rnn.bias!
            if let h {
                pre = pre + matmul(h, rnn.whh.T)
            }
            h = maximum(pre, 0)
            out.append(h!)
        }
        assertClose(y, stacked(out, axis: -2))
    }

    // MARK: - GRU

    func testGRUMatchesReference() {
        for bias in [true, false] {
            MLXRandom.seed(20)
            let gru = GRU(inputSize: inputSize, hiddenSize: hiddenSize, bias: bias)
            XCTAssertEqual(gru.wx.shape, [3 * hiddenSize, inputSize])
            XCTAssertEqual(gru.wh.shape, [3 * hiddenSize, hiddenSize])
            let x = input()
            let y = gru(x)
            XCTAssertEqual(y.shape, [2, 5, hiddenSize])
            assertClose(y, referenceGRU(gru, x), "bias: \(bias)")

            let h0 = MLXRandom.normal([2, hiddenSize], key: MLXRandom.key(3))
            assertClose(gru(x, hidden: h0), referenceGRU(gru, x, hidden: h0), "bias: \(bias), h0")
        }
    }

    // MARK: - LSTM

    func testLSTMMatchesReference() {
        for bias in [true, false] {
            MLXRandom.seed(30)
            let lstm = LSTM(inputSize: inputSize, hiddenSize: hiddenSize, bias: bias)
            XCTAssertEqual(lstm.wx.shape, [4 * hiddenSize, inputSize])
            let x = input()
            let (h, c) = lstm(x)
            XCTAssertEqual(h.shape, [2, 5, hiddenSize])
            XCTAssertEqual(c.shape, [2, 5, hiddenSize])
            let (rh, rc) = referenceLSTM(lstm, x)
            assertClose(h, rh, "hidden, bias: \(bias)")
            assertClose(c, rc, "cell, bias: \(bias)")

            let h0 = MLXRandom.normal([2, hiddenSize], key: MLXRandom.key(4))
            let c0 = MLXRandom.normal([2, hiddenSize], key: MLXRandom.key(5))
            let (h2, c2) = lstm(x, hidden: h0, cell: c0)
            let (rh2, rc2) = referenceLSTM(lstm, x, hidden: h0, cell: c0)
            assertClose(h2, rh2, "hidden with h0 and c0, bias: \(bias)")
            assertClose(c2, rc2, "cell with h0 and c0, bias: \(bias)")
        }
    }

    // MARK: - Properties of all three layers

    /// The output has one state per time step, for any sequence length,
    /// with or without the batch axis. An unbatched input gives the same
    /// values as the same input in a batch.
    func testShapesForEachSequenceLength() {
        MLXRandom.seed(40)
        let rnn = RNN(inputSize: inputSize, hiddenSize: hiddenSize)
        let gru = GRU(inputSize: inputSize, hiddenSize: hiddenSize)
        let lstm = LSTM(inputSize: inputSize, hiddenSize: hiddenSize)

        for length in [1, 3, 7] {
            let x = input(length: length)
            XCTAssertEqual(rnn(x).shape, [2, length, hiddenSize])
            XCTAssertEqual(gru(x).shape, [2, length, hiddenSize])
            let (h, c) = lstm(x)
            XCTAssertEqual(h.shape, [2, length, hiddenSize])
            XCTAssertEqual(c.shape, [2, length, hiddenSize])

            // The last state has the same shape for every sequence length.
            XCTAssertEqual(rnn(x)[0..., length - 1].shape, [2, hiddenSize])

            let row = x[1]
            XCTAssertEqual(rnn(row).shape, [length, hiddenSize])
            assertClose(rnn(row), rnn(x)[1], "RNN unbatched, length \(length)")
            assertClose(gru(row), gru(x)[1], "GRU unbatched, length \(length)")
            assertClose(lstm(row).0, h[1], "LSTM unbatched, length \(length)")
            assertClose(lstm(row).1, c[1], "LSTM cell unbatched, length \(length)")
        }
    }

    /// The same seed gives the same weights and the same output.
    /// A different seed gives a different output.
    func testSeededWeightsAreDeterministic() {
        func run(_ seed: UInt64) -> [MLXArray] {
            MLXRandom.seed(seed)
            let rnn = RNN(inputSize: inputSize, hiddenSize: hiddenSize)
            let gru = GRU(inputSize: inputSize, hiddenSize: hiddenSize)
            let lstm = LSTM(inputSize: inputSize, hiddenSize: hiddenSize)
            let x = input()
            let (h, c) = lstm(x)
            return [rnn(x), gru(x), h, c]
        }
        let a = run(50)
        let b = run(50)
        let other = run(51)
        for (i, (x, y)) in zip(a, b).enumerated() {
            XCTAssertTrue(x.arrayEqual(y).item(Bool.self), "output \(i) differs for the same seed")
        }
        for (i, (x, y)) in zip(a, other).enumerated() {
            XCTAssertFalse(
                x.allClose(y).item(Bool.self), "output \(i) is the same for a different seed")
        }
    }

    /// Running the sequence in two calls, with the last state of the first call
    /// passed to the second, gives the same result as one call.
    func testStateCarriesAcrossCalls() {
        MLXRandom.seed(60)
        let rnn = RNN(inputSize: inputSize, hiddenSize: hiddenSize)
        let gru = GRU(inputSize: inputSize, hiddenSize: hiddenSize)
        let lstm = LSTM(inputSize: inputSize, hiddenSize: hiddenSize)
        let x = input(length: 6)
        let first = x[0..., ..<2]
        let second = x[0..., 2...]

        let rnnAll = rnn(x)
        let rnnFirst = rnn(first)
        let rnnSecond = rnn(second, hidden: rnnFirst[0..., 1])
        assertClose(concatenated([rnnFirst, rnnSecond], axis: 1), rnnAll, "RNN")

        let gruAll = gru(x)
        let gruFirst = gru(first)
        let gruSecond = gru(second, hidden: gruFirst[0..., 1])
        assertClose(concatenated([gruFirst, gruSecond], axis: 1), gruAll, "GRU")

        let (hAll, cAll) = lstm(x)
        let (hFirst, cFirst) = lstm(first)
        let (hSecond, cSecond) = lstm(second, hidden: hFirst[0..., 1], cell: cFirst[0..., 1])
        assertClose(concatenated([hFirst, hSecond], axis: 1), hAll, "LSTM hidden")
        assertClose(concatenated([cFirst, cSecond], axis: 1), cAll, "LSTM cell")

        // A later input does not change an earlier output.
        let changed = concatenated([first, second + 1], axis: 1)
        assertClose(rnn(changed)[0..., ..<2], rnnFirst, "RNN prefix")
        assertClose(gru(changed)[0..., ..<2], gruFirst, "GRU prefix")
        assertClose(lstm(changed).0[0..., ..<2], hFirst, "LSTM prefix")
    }
}
