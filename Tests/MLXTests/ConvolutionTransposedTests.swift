// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

@testable import MLXNN

/// Tests for `ConvTransposed1d`, `ConvTransposed2d` and `ConvTransposed3d`.
///
/// The 1D and 2D layers are compared with a direct loop in plain Swift:
/// each input value, times each kernel tap, is added to the output at
/// `i * stride + k * dilation - padding`. The 3D layer is compared with the
/// same loops along the depth axis alone and along height and width alone.
class ConvolutionTransposedTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
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

    /// Sets a bias with different values per channel, so that the test sees
    /// that the layer adds the bias.
    private func setBias(_ layer: Module, channels: Int) {
        layer.update(
            parameters: ModuleParameters.unflattened([
                ("bias", MLXArray((0 ..< channels).map { Float($0) + 1 }))
            ]))
    }

    /// Output length of a transposed convolution along one axis.
    private func outputLength(
        _ length: Int, kernel: Int, stride: Int, padding: Int, dilation: Int, outputPadding: Int
    ) -> Int {
        (length - 1) * stride - 2 * padding + dilation * (kernel - 1) + outputPadding + 1
    }

    /// x: [N, L, C], w: [O, K, C / groups].
    private func reference1d(
        _ x: MLXArray, _ w: MLXArray, bias: MLXArray?, stride: Int, padding: Int, dilation: Int,
        outputPadding: Int, groups: Int
    ) -> MLXArray {
        let (n, l, cin) = (x.dim(0), x.dim(1), x.dim(2))
        let (cout, k, cg) = (w.dim(0), w.dim(1), w.dim(2))
        let lout = outputLength(
            l, kernel: k, stride: stride, padding: padding, dilation: dilation,
            outputPadding: outputPadding)
        let xv = x.asArray(Float.self)
        let wv = w.asArray(Float.self)
        let og = cout / groups
        var y = [Float](repeating: 0, count: n * lout * cout)
        for b in 0 ..< n {
            for i in 0 ..< l {
                for c in 0 ..< cin {
                    let group = c / cg
                    for o in group * og ..< (group + 1) * og {
                        for kk in 0 ..< k {
                            let pos = i * stride + kk * dilation - padding
                            guard pos >= 0 && pos < lout else { continue }
                            y[(b * lout + pos) * cout + o] +=
                                xv[(b * l + i) * cin + c] * wv[(o * k + kk) * cg + c % cg]
                        }
                    }
                }
            }
        }
        var result = MLXArray(y, [n, lout, cout])
        if let bias {
            result = result + bias
        }
        return result
    }

    /// x: [N, H, W, C], w: [O, KH, KW, C], one group.
    private func reference2d(
        _ x: MLXArray, _ w: MLXArray, bias: MLXArray?, stride: (Int, Int), padding: (Int, Int),
        dilation: (Int, Int), outputPadding: (Int, Int)
    ) -> MLXArray {
        let (n, h, wd, cin) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let (cout, kh, kw) = (w.dim(0), w.dim(1), w.dim(2))
        let hout = outputLength(
            h, kernel: kh, stride: stride.0, padding: padding.0, dilation: dilation.0,
            outputPadding: outputPadding.0)
        let wout = outputLength(
            wd, kernel: kw, stride: stride.1, padding: padding.1, dilation: dilation.1,
            outputPadding: outputPadding.1)
        let xv = x.asArray(Float.self)
        let wv = w.asArray(Float.self)
        var y = [Float](repeating: 0, count: n * hout * wout * cout)
        for b in 0 ..< n {
            for i in 0 ..< h {
                for j in 0 ..< wd {
                    for c in 0 ..< cin {
                        let xval = xv[((b * h + i) * wd + j) * cin + c]
                        for o in 0 ..< cout {
                            for ki in 0 ..< kh {
                                let pi = i * stride.0 + ki * dilation.0 - padding.0
                                guard pi >= 0 && pi < hout else { continue }
                                for kj in 0 ..< kw {
                                    let pj = j * stride.1 + kj * dilation.1 - padding.1
                                    guard pj >= 0 && pj < wout else { continue }
                                    y[((b * hout + pi) * wout + pj) * cout + o] +=
                                        xval * wv[((o * kh + ki) * kw + kj) * cin + c]
                                }
                            }
                        }
                    }
                }
            }
        }
        var result = MLXArray(y, [n, hout, wout, cout])
        if let bias {
            result = result + bias
        }
        return result
    }

    func testConvTransposed1dMatchesReference() {
        let cases: [(stride: Int, padding: Int, dilation: Int, outputPadding: Int, groups: Int)] =
            [
                (1, 0, 1, 0, 1),
                (2, 1, 1, 1, 1),
                (3, 0, 2, 0, 1),
                (2, 0, 1, 0, 2),
            ]
        for (index, p) in cases.enumerated() {
            MLXRandom.seed(UInt64(index))
            let layer = ConvTransposed1d(
                inputChannels: 4, outputChannels: 6, kernelSize: 3, stride: p.stride,
                padding: p.padding, outputPadding: p.outputPadding, dilation: p.dilation,
                groups: p.groups)
            XCTAssertEqual(layer.weight.shape, [6, 3, 4 / p.groups])
            setBias(layer, channels: 6)
            let x = MLXRandom.normal([2, 5, 4], key: MLXRandom.key(100 + UInt64(index)))
            let expected = reference1d(
                x, layer.weight, bias: layer.bias, stride: p.stride, padding: p.padding,
                dilation: p.dilation, outputPadding: p.outputPadding, groups: p.groups)
            assertClose(layer(x), expected, "case \(index): \(p)")
        }
    }

    func testConvTransposed1dWithoutBias() {
        let layer = ConvTransposed1d(
            inputChannels: 2, outputChannels: 3, kernelSize: 2, bias: false)
        XCTAssertNil(layer.bias)
        let x = MLXRandom.normal([1, 4, 2], key: MLXRandom.key(7))
        assertClose(
            layer(x),
            reference1d(
                x, layer.weight, bias: nil, stride: 1, padding: 0, dilation: 1, outputPadding: 0,
                groups: 1))
    }

    /// Height and width use different settings, so a swap of the two shows.
    func testConvTransposed2dMatchesReference() {
        MLXRandom.seed(20)
        let layer = ConvTransposed2d(
            inputChannels: 3, outputChannels: 4, kernelSize: [2, 3], stride: [2, 1],
            padding: [0, 1], outputPadding: [1, 0], dilation: [1, 2])
        XCTAssertEqual(layer.weight.shape, [4, 2, 3, 3])
        setBias(layer, channels: 4)
        let x = MLXRandom.normal([2, 3, 4, 3], key: MLXRandom.key(21))
        let y = layer(x)
        XCTAssertEqual(y.shape, [2, 7, 6, 4])
        assertClose(
            y,
            reference2d(
                x, layer.weight, bias: layer.bias, stride: (2, 1), padding: (0, 1),
                dilation: (1, 2), outputPadding: (1, 0)))
    }

    /// With a depth of 1 and a depth kernel of 1, the 3D layer is the 2D reference.
    func testConvTransposed3dMatchesReferenceOverHeightAndWidth() {
        MLXRandom.seed(30)
        let layer = ConvTransposed3d(
            inputChannels: 2, outputChannels: 3, kernelSize: [1, 2, 3], stride: [1, 2, 1],
            padding: [0, 0, 1], outputPadding: [0, 1, 0], dilation: [1, 1, 2])
        XCTAssertEqual(layer.weight.shape, [3, 1, 2, 3, 2])
        setBias(layer, channels: 3)
        let x = MLXRandom.normal([1, 1, 3, 4, 2], key: MLXRandom.key(31))
        let expected = reference2d(
            x.squeezed(axis: 1), layer.weight.squeezed(axis: 1), bias: layer.bias,
            stride: (2, 1), padding: (0, 1), dilation: (1, 2), outputPadding: (1, 0))
        assertClose(layer(x), expandedDimensions(expected, axis: 1))
    }

    /// With height and width of 1 and kernels of 1 there, the 3D layer is the 1D reference
    /// along the depth axis.
    func testConvTransposed3dMatchesReferenceOverDepth() {
        MLXRandom.seed(40)
        let layer = ConvTransposed3d(
            inputChannels: 2, outputChannels: 3, kernelSize: [3, 1, 1], stride: [2, 1, 1],
            padding: [1, 0, 0], outputPadding: [1, 0, 0], dilation: [2, 1, 1])
        setBias(layer, channels: 3)
        let x = MLXRandom.normal([2, 4, 1, 1, 2], key: MLXRandom.key(41))
        let expected = reference1d(
            x.reshaped([2, 4, 2]), layer.weight.reshaped([3, 3, 2]), bias: layer.bias,
            stride: 2, padding: 1, dilation: 2, outputPadding: 1, groups: 1)
        let y = layer(x)
        XCTAssertEqual(y.shape, [2, expected.dim(1), 1, 1, 3])
        assertClose(y.reshaped(expected.shape), expected)
    }

    /// The same seed gives the same weights and the same output.
    func testSeededWeightsAreDeterministic() {
        func run(_ seed: UInt64) -> MLXArray {
            MLXRandom.seed(seed)
            let layer = ConvTransposed3d(inputChannels: 2, outputChannels: 2, kernelSize: 2)
            return layer(MLXArray.ones([1, 2, 2, 2, 2]))
        }
        XCTAssertEqual(run(50).shape, [1, 3, 3, 3, 2])
        XCTAssertTrue(run(50).arrayEqual(run(50)).item(Bool.self))
        XCTAssertFalse(run(50).allClose(run(51)).item(Bool.self))
    }
}
