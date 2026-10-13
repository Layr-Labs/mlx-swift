// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

/// Tests for `CustomFunction`, `Forward` and `VJP` in `MLXCustomFunction.swift`.
///
/// Each test compares the forward output and the gradient of a custom function
/// with a reference built from plain ops.
class CustomFunctionTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    private func assertClose(
        _ a: MLXArray, _ b: MLXArray, _ message: String = "",
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(a.shape, b.shape, message, file: file, line: line)
        XCTAssertTrue(
            a.allClose(b, rtol: 1e-5, atol: 1e-6).item(Bool.self),
            "\(message)\n\(a)\n\(b)", file: file, line: line)
    }

    private let x = MLXArray([-2, -0.5, 0, 0.5, 1, 3] as [Float], [2, 3])

    /// softplus(x) = log(1 + exp(x)), with plain ops.
    private static func softplus(_ x: MLXArray) -> MLXArray {
        log(1 + exp(x))
    }

    /// The forward output is the output of the Forward block.
    /// The gradient comes from the VJP block and matches the true derivative, sigmoid(x).
    func testForwardAndCustomGradientMatchReference() {
        let f = CustomFunction {
            Forward { inputs in [Self.softplus(inputs[0])] }
            VJP { primals, cotangents in [cotangents[0] * sigmoid(primals[0])] }
        }

        let y = f([x])
        XCTAssertEqual(y.count, 1)
        XCTAssertTrue(y[0].arrayEqual(Self.softplus(x)).item(Bool.self), "\(y[0])")

        let customGradFunction = grad { (x: MLXArray) -> MLXArray in f([x])[0].sum() }
        let customGrad = customGradFunction(x)
        let plainGradFunction = grad { (x: MLXArray) -> MLXArray in Self.softplus(x).sum() }
        let plainGrad = plainGradFunction(x)
        assertClose(customGrad, sigmoid(x), "custom gradient vs sigmoid(x)")
        assertClose(customGrad, plainGrad, "custom gradient vs autodiff of plain ops")
    }

    /// The gradient comes from the VJP block, not from autodiff of the Forward block.
    /// The VJP here is on purpose not the true derivative (2x).
    func testVJPBlockReplacesAutodiff() {
        let f = CustomFunction {
            Forward { inputs in [inputs[0] * inputs[0]] }
            VJP { _, cotangents in [cotangents[0] * 5] }
        }
        let gFunction = grad { (x: MLXArray) -> MLXArray in f([x])[0].sum() }
        let g = gFunction(x)
        XCTAssertTrue(
            g.arrayEqual(MLXArray.ones(x.shape) * 5).item(Bool.self), "\(g)"
        )
    }

    /// The VJP block gets the inputs as primals and the given cotangents.
    func testVJPBlockReceivesPrimalsAndCotangents() {
        var seenPrimals = [MLXArray]()
        var seenCotangents = [MLXArray]()
        let f = CustomFunction {
            Forward { inputs in [inputs[0] * 2] }
            VJP { primals, cotangents in
                seenPrimals = primals
                seenCotangents = cotangents
                return [cotangents[0] * 2]
            }
        }
        let cotangent = MLXArray([1, 2, 3, 4, 5, 6] as [Float], [2, 3])
        let (outputs, vjps) = vjp(f, primals: [x], cotangents: [cotangent])

        XCTAssertTrue(outputs[0].arrayEqual(x * 2).item(Bool.self))
        XCTAssertTrue(vjps[0].arrayEqual(cotangent * 2).item(Bool.self))
        XCTAssertEqual(seenPrimals.count, 1)
        XCTAssertEqual(seenCotangents.count, 1)
        XCTAssertTrue(seenPrimals[0].arrayEqual(x).item(Bool.self))
        XCTAssertTrue(seenCotangents[0].arrayEqual(cotangent).item(Bool.self))
    }

    /// Two inputs and two outputs: [a * b, a + b].
    /// The custom VJP must match the VJP of the same function in plain ops.
    func testTwoInputsTwoOutputs() {
        let plain: ([MLXArray]) -> [MLXArray] = { xs in [xs[0] * xs[1], xs[0] + xs[1]] }
        let f = CustomFunction {
            Forward(plain)
            VJP { p, c in [c[0] * p[1] + c[1], c[0] * p[0] + c[1]] }
        }
        let a = MLXRandom.normal([3, 2], key: MLXRandom.key(1))
        let b = MLXRandom.normal([3, 2], key: MLXRandom.key(2))
        let c0 = MLXRandom.normal([3, 2], key: MLXRandom.key(3))
        let c1 = MLXRandom.normal([3, 2], key: MLXRandom.key(4))

        let (customOut, customVJP) = vjp(f, primals: [a, b], cotangents: [c0, c1])
        let (plainOut, plainVJP) = vjp(plain, primals: [a, b], cotangents: [c0, c1])
        XCTAssertEqual(customOut.count, 2)
        XCTAssertEqual(customVJP.count, 2)
        for i in 0 ..< 2 {
            XCTAssertTrue(customOut[i].arrayEqual(plainOut[i]).item(Bool.self), "output \(i)")
            assertClose(customVJP[i], plainVJP[i], "gradient for input \(i)")
        }
    }

    /// Without a VJP block, the gradient is autodiff of the Forward block: d(x^3)/dx = 3x^2.
    func testWithoutVJPBlockUsesAutodiff() {
        let f = CustomFunction {
            Forward { inputs in [inputs[0] * inputs[0] * inputs[0]] }
        }
        XCTAssertTrue(f([x])[0].arrayEqual(x * x * x).item(Bool.self))
        let gFunction = grad { (x: MLXArray) -> MLXArray in f([x])[0].sum() }
        let g = gFunction(x)
        assertClose(g, 3 * x * x)
    }
}
