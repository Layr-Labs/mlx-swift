// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

/// Tests for the typed `compile` overloads in `Transforms+CompileOverloads.swift`.
///
/// Each overload takes a fixed number of arrays and returns a tuple of arrays.
/// A test compiles `reference(_:outputs:)` through one overload, calls it, and
/// compares each output with the same function called without `compile`.
/// Each test also counts the traces: the second call must use the compiled
/// graph and not run the function again.
/// The inputs hold small whole numbers, so the float32 results are exact.
class CompileOverloadsTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    /// Output `j` is the sum of `x[i] * (i + j + 1)`, minus `j`.
    ///
    /// Each input has its own weight and each output has its own offset.
    /// So a swapped input or a swapped output gives a different result.
    private static func reference(_ x: [MLXArray], outputs: Int) -> [MLXArray] {
        (0 ..< outputs).map { j in
            var result = MLXArray(Float(-j))
            for (i, xi) in x.enumerated() {
                result = result + xi * Float(i + j + 1)
            }
            return result
        }
    }

    /// Makes `count` different [2, 3] float32 arrays that hold whole numbers.
    private static func makeInputs(_ count: Int, offset: Float, shape: [Int] = [2, 3])
        -> [MLXArray]
    {
        let size = shape.reduce(1, *)
        return (0 ..< count).map { i in
            MLXArray((0 ..< size).map { Float($0) * Float(i + 1) + offset }, shape)
        }
    }

    /// Calls `compiled` two times with different values of the same shape.
    /// Each output must be equal to the uncompiled reference.
    private func check(
        inputs: Int, outputs: Int,
        _ compiled: ([MLXArray]) -> [MLXArray],
        file: StaticString = #filePath, line: UInt = #line
    ) {
        for offset: Float in [0, 7] {
            let x = Self.makeInputs(inputs, offset: offset)
            let got = compiled(x)
            let want = Self.reference(x, outputs: outputs)
            XCTAssertEqual(got.count, outputs, file: file, line: line)
            for (j, (g, w)) in zip(got, want).enumerated() {
                XCTAssertEqual(g.shape, w.shape, "output \(j)", file: file, line: line)
                XCTAssertEqual(g.dtype, .float32, "output \(j)", file: file, line: line)
                XCTAssertTrue(
                    g.arrayEqual(w).item(Bool.self),
                    "output \(j) of the \(inputs)-input overload differs:\n\(g)\n\(w)",
                    file: file, line: line)
            }
        }
    }

    func testOneInput() {
        var n2 = 0
        let c2 = compile { (a: MLXArray) -> (MLXArray, MLXArray) in
            n2 += 1
            let r = Self.reference([a], outputs: 2)
            return (r[0], r[1])
        }
        check(inputs: 1, outputs: 2) { x in
            let r = c2(x[0])
            return [r.0, r.1]
        }
        XCTAssertEqual(n2, 1, "the second call traced the function again")

        var n3 = 0
        let c3 = compile { (a: MLXArray) -> (MLXArray, MLXArray, MLXArray) in
            n3 += 1
            let r = Self.reference([a], outputs: 3)
            return (r[0], r[1], r[2])
        }
        check(inputs: 1, outputs: 3) { x in
            let r = c3(x[0])
            return [r.0, r.1, r.2]
        }
        XCTAssertEqual(n3, 1, "the second call traced the function again")

        var n4 = 0
        let c4 = compile { (a: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) in
            n4 += 1
            let r = Self.reference([a], outputs: 4)
            return (r[0], r[1], r[2], r[3])
        }
        check(inputs: 1, outputs: 4) { x in
            let r = c4(x[0])
            return [r.0, r.1, r.2, r.3]
        }
        XCTAssertEqual(n4, 1, "the second call traced the function again")
    }

    func testTwoInputs() {
        var n2 = 0
        let c2 = compile { (a: MLXArray, b: MLXArray) -> (MLXArray, MLXArray) in
            n2 += 1
            let r = Self.reference([a, b], outputs: 2)
            return (r[0], r[1])
        }
        check(inputs: 2, outputs: 2) { x in
            let r = c2(x[0], x[1])
            return [r.0, r.1]
        }
        XCTAssertEqual(n2, 1, "the second call traced the function again")

        var n3 = 0
        let c3 = compile { (a: MLXArray, b: MLXArray) -> (MLXArray, MLXArray, MLXArray) in
            n3 += 1
            let r = Self.reference([a, b], outputs: 3)
            return (r[0], r[1], r[2])
        }
        check(inputs: 2, outputs: 3) { x in
            let r = c3(x[0], x[1])
            return [r.0, r.1, r.2]
        }
        XCTAssertEqual(n3, 1, "the second call traced the function again")

        var n4 = 0
        let c4 = compile {
            (a: MLXArray, b: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) in
            n4 += 1
            let r = Self.reference([a, b], outputs: 4)
            return (r[0], r[1], r[2], r[3])
        }
        check(inputs: 2, outputs: 4) { x in
            let r = c4(x[0], x[1])
            return [r.0, r.1, r.2, r.3]
        }
        XCTAssertEqual(n4, 1, "the second call traced the function again")
    }

    func testThreeInputs() {
        var n2 = 0
        let c2 = compile { (a: MLXArray, b: MLXArray, c: MLXArray) -> (MLXArray, MLXArray) in
            n2 += 1
            let r = Self.reference([a, b, c], outputs: 2)
            return (r[0], r[1])
        }
        check(inputs: 3, outputs: 2) { x in
            let r = c2(x[0], x[1], x[2])
            return [r.0, r.1]
        }
        XCTAssertEqual(n2, 1, "the second call traced the function again")

        var n3 = 0
        let c3 = compile {
            (a: MLXArray, b: MLXArray, c: MLXArray) -> (MLXArray, MLXArray, MLXArray) in
            n3 += 1
            let r = Self.reference([a, b, c], outputs: 3)
            return (r[0], r[1], r[2])
        }
        check(inputs: 3, outputs: 3) { x in
            let r = c3(x[0], x[1], x[2])
            return [r.0, r.1, r.2]
        }
        XCTAssertEqual(n3, 1, "the second call traced the function again")

        var n4 = 0
        let c4 = compile {
            (a: MLXArray, b: MLXArray, c: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) in
            n4 += 1
            let r = Self.reference([a, b, c], outputs: 4)
            return (r[0], r[1], r[2], r[3])
        }
        check(inputs: 3, outputs: 4) { x in
            let r = c4(x[0], x[1], x[2])
            return [r.0, r.1, r.2, r.3]
        }
        XCTAssertEqual(n4, 1, "the second call traced the function again")
    }

    func testFourInputs() {
        var n1 = 0
        let c1 = compile { (a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray) -> MLXArray in
            n1 += 1
            return Self.reference([a, b, c, d], outputs: 1)[0]
        }
        check(inputs: 4, outputs: 1) { x in
            [c1(x[0], x[1], x[2], x[3])]
        }
        XCTAssertEqual(n1, 1, "the second call traced the function again")

        var n2 = 0
        let c2 = compile {
            (a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray) -> (MLXArray, MLXArray) in
            n2 += 1
            let r = Self.reference([a, b, c, d], outputs: 2)
            return (r[0], r[1])
        }
        check(inputs: 4, outputs: 2) { x in
            let r = c2(x[0], x[1], x[2], x[3])
            return [r.0, r.1]
        }
        XCTAssertEqual(n2, 1, "the second call traced the function again")

        var n3 = 0
        let c3 = compile {
            (a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray) -> (MLXArray, MLXArray, MLXArray)
            in
            n3 += 1
            let r = Self.reference([a, b, c, d], outputs: 3)
            return (r[0], r[1], r[2])
        }
        check(inputs: 4, outputs: 3) { x in
            let r = c3(x[0], x[1], x[2], x[3])
            return [r.0, r.1, r.2]
        }
        XCTAssertEqual(n3, 1, "the second call traced the function again")

        var n4 = 0
        let c4 = compile {
            (a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray) -> (
                MLXArray, MLXArray, MLXArray, MLXArray
            ) in
            n4 += 1
            let r = Self.reference([a, b, c, d], outputs: 4)
            return (r[0], r[1], r[2], r[3])
        }
        check(inputs: 4, outputs: 4) { x in
            let r = c4(x[0], x[1], x[2], x[3])
            return [r.0, r.1, r.2, r.3]
        }
        XCTAssertEqual(n4, 1, "the second call traced the function again")
    }

    func testFiveInputs() {
        var n1 = 0
        let c1 = compile {
            (a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray, e: MLXArray) -> MLXArray in
            n1 += 1
            return Self.reference([a, b, c, d, e], outputs: 1)[0]
        }
        check(inputs: 5, outputs: 1) { x in
            [c1(x[0], x[1], x[2], x[3], x[4])]
        }
        XCTAssertEqual(n1, 1, "the second call traced the function again")

        var n2 = 0
        let c2 = compile {
            (a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray, e: MLXArray) -> (
                MLXArray, MLXArray
            ) in
            n2 += 1
            let r = Self.reference([a, b, c, d, e], outputs: 2)
            return (r[0], r[1])
        }
        check(inputs: 5, outputs: 2) { x in
            let r = c2(x[0], x[1], x[2], x[3], x[4])
            return [r.0, r.1]
        }
        XCTAssertEqual(n2, 1, "the second call traced the function again")

        var n3 = 0
        let c3 = compile {
            (a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray, e: MLXArray) -> (
                MLXArray, MLXArray, MLXArray
            ) in
            n3 += 1
            let r = Self.reference([a, b, c, d, e], outputs: 3)
            return (r[0], r[1], r[2])
        }
        check(inputs: 5, outputs: 3) { x in
            let r = c3(x[0], x[1], x[2], x[3], x[4])
            return [r.0, r.1, r.2]
        }
        XCTAssertEqual(n3, 1, "the second call traced the function again")

        var n4 = 0
        let c4 = compile {
            (a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray, e: MLXArray) -> (
                MLXArray, MLXArray, MLXArray, MLXArray
            ) in
            n4 += 1
            let r = Self.reference([a, b, c, d, e], outputs: 4)
            return (r[0], r[1], r[2], r[3])
        }
        check(inputs: 5, outputs: 4) { x in
            let r = c4(x[0], x[1], x[2], x[3], x[4])
            return [r.0, r.1, r.2, r.3]
        }
        XCTAssertEqual(n4, 1, "the second call traced the function again")
    }

    func testSixInputs() {
        var n1 = 0
        let c1 = compile {
            (a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray, e: MLXArray, f: MLXArray)
                -> MLXArray in
            n1 += 1
            return Self.reference([a, b, c, d, e, f], outputs: 1)[0]
        }
        check(inputs: 6, outputs: 1) { x in
            [c1(x[0], x[1], x[2], x[3], x[4], x[5])]
        }
        XCTAssertEqual(n1, 1, "the second call traced the function again")

        var n2 = 0
        let c2 = compile {
            (a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray, e: MLXArray, f: MLXArray)
                -> (MLXArray, MLXArray) in
            n2 += 1
            let r = Self.reference([a, b, c, d, e, f], outputs: 2)
            return (r[0], r[1])
        }
        check(inputs: 6, outputs: 2) { x in
            let r = c2(x[0], x[1], x[2], x[3], x[4], x[5])
            return [r.0, r.1]
        }
        XCTAssertEqual(n2, 1, "the second call traced the function again")

        var n3 = 0
        let c3 = compile {
            (a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray, e: MLXArray, f: MLXArray)
                -> (MLXArray, MLXArray, MLXArray) in
            n3 += 1
            let r = Self.reference([a, b, c, d, e, f], outputs: 3)
            return (r[0], r[1], r[2])
        }
        check(inputs: 6, outputs: 3) { x in
            let r = c3(x[0], x[1], x[2], x[3], x[4], x[5])
            return [r.0, r.1, r.2]
        }
        XCTAssertEqual(n3, 1, "the second call traced the function again")

        var n4 = 0
        let c4 = compile {
            (a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray, e: MLXArray, f: MLXArray)
                -> (MLXArray, MLXArray, MLXArray, MLXArray) in
            n4 += 1
            let r = Self.reference([a, b, c, d, e, f], outputs: 4)
            return (r[0], r[1], r[2], r[3])
        }
        check(inputs: 6, outputs: 4) { x in
            let r = c4(x[0], x[1], x[2], x[3], x[4], x[5])
            return [r.0, r.1, r.2, r.3]
        }
        XCTAssertEqual(n4, 1, "the second call traced the function again")
    }

    func testSevenInputs() {
        var n1 = 0
        let c1 = compile {
            (
                a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray, e: MLXArray, f: MLXArray,
                g: MLXArray
            ) -> MLXArray in
            n1 += 1
            return Self.reference([a, b, c, d, e, f, g], outputs: 1)[0]
        }
        check(inputs: 7, outputs: 1) { x in
            [c1(x[0], x[1], x[2], x[3], x[4], x[5], x[6])]
        }
        XCTAssertEqual(n1, 1, "the second call traced the function again")

        var n2 = 0
        let c2 = compile {
            (
                a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray, e: MLXArray, f: MLXArray,
                g: MLXArray
            ) -> (MLXArray, MLXArray) in
            n2 += 1
            let r = Self.reference([a, b, c, d, e, f, g], outputs: 2)
            return (r[0], r[1])
        }
        check(inputs: 7, outputs: 2) { x in
            let r = c2(x[0], x[1], x[2], x[3], x[4], x[5], x[6])
            return [r.0, r.1]
        }
        XCTAssertEqual(n2, 1, "the second call traced the function again")

        var n3 = 0
        let c3 = compile {
            (
                a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray, e: MLXArray, f: MLXArray,
                g: MLXArray
            ) -> (MLXArray, MLXArray, MLXArray) in
            n3 += 1
            let r = Self.reference([a, b, c, d, e, f, g], outputs: 3)
            return (r[0], r[1], r[2])
        }
        check(inputs: 7, outputs: 3) { x in
            let r = c3(x[0], x[1], x[2], x[3], x[4], x[5], x[6])
            return [r.0, r.1, r.2]
        }
        XCTAssertEqual(n3, 1, "the second call traced the function again")

        var n4 = 0
        let c4 = compile {
            (
                a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray, e: MLXArray, f: MLXArray,
                g: MLXArray
            ) -> (MLXArray, MLXArray, MLXArray, MLXArray) in
            n4 += 1
            let r = Self.reference([a, b, c, d, e, f, g], outputs: 4)
            return (r[0], r[1], r[2], r[3])
        }
        check(inputs: 7, outputs: 4) { x in
            let r = c4(x[0], x[1], x[2], x[3], x[4], x[5], x[6])
            return [r.0, r.1, r.2, r.3]
        }
        XCTAssertEqual(n4, 1, "the second call traced the function again")
    }

    func testEightInputs() {
        var n1 = 0
        let c1 = compile {
            (
                a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray, e: MLXArray, f: MLXArray,
                g: MLXArray, h: MLXArray
            ) -> MLXArray in
            n1 += 1
            return Self.reference([a, b, c, d, e, f, g, h], outputs: 1)[0]
        }
        check(inputs: 8, outputs: 1) { x in
            [c1(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])]
        }
        XCTAssertEqual(n1, 1, "the second call traced the function again")

        var n2 = 0
        let c2 = compile {
            (
                a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray, e: MLXArray, f: MLXArray,
                g: MLXArray, h: MLXArray
            ) -> (MLXArray, MLXArray) in
            n2 += 1
            let r = Self.reference([a, b, c, d, e, f, g, h], outputs: 2)
            return (r[0], r[1])
        }
        check(inputs: 8, outputs: 2) { x in
            let r = c2(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])
            return [r.0, r.1]
        }
        XCTAssertEqual(n2, 1, "the second call traced the function again")

        var n3 = 0
        let c3 = compile {
            (
                a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray, e: MLXArray, f: MLXArray,
                g: MLXArray, h: MLXArray
            ) -> (MLXArray, MLXArray, MLXArray) in
            n3 += 1
            let r = Self.reference([a, b, c, d, e, f, g, h], outputs: 3)
            return (r[0], r[1], r[2])
        }
        check(inputs: 8, outputs: 3) { x in
            let r = c3(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])
            return [r.0, r.1, r.2]
        }
        XCTAssertEqual(n3, 1, "the second call traced the function again")

        var n4 = 0
        let c4 = compile {
            (
                a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray, e: MLXArray, f: MLXArray,
                g: MLXArray, h: MLXArray
            ) -> (MLXArray, MLXArray, MLXArray, MLXArray) in
            n4 += 1
            let r = Self.reference([a, b, c, d, e, f, g, h], outputs: 4)
            return (r[0], r[1], r[2], r[3])
        }
        check(inputs: 8, outputs: 4) { x in
            let r = c4(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])
            return [r.0, r.1, r.2, r.3]
        }
        XCTAssertEqual(n4, 1, "the second call traced the function again")
    }

    /// The `shapeless` flag reaches the compiler through the overloads.
    /// Without it, a new input shape traces the function again.
    /// With it, the first trace serves the new shape.
    func testShapelessFlag() {
        for shapeless in [false, true] {
            var traces = 0
            let compiled = compile(shapeless: shapeless) {
                (a: MLXArray, b: MLXArray) -> (MLXArray, MLXArray) in
                traces += 1
                let r = Self.reference([a, b], outputs: 2)
                return (r[0], r[1])
            }
            for shape in [[2, 3], [4, 5]] {
                let x = Self.makeInputs(2, offset: 1, shape: shape)
                let got = compiled(x[0], x[1])
                let want = Self.reference(x, outputs: 2)
                XCTAssertEqual(got.0.shape, shape)
                XCTAssertTrue(got.0.arrayEqual(want[0]).item(Bool.self))
                XCTAssertTrue(got.1.arrayEqual(want[1]).item(Bool.self))
            }
            XCTAssertEqual(traces, shapeless ? 1 : 2, "shapeless: \(shapeless)")
        }
    }
}
