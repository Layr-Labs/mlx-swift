// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

/// Tests for the overloads in `Transforms+Eval.swift` and `Transforms+Grad.swift`.
class EvalGradOverloadTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    /// Lazy arrays: `base * k` for k in 1 ... count.
    private func lazyArrays(_ count: Int) -> [MLXArray] {
        let base = MLXArray([1, 2, 3] as [Int32])
        return (1 ... count).map { base * Int32($0) }
    }

    private func checkValues(
        _ arrays: [MLXArray], file: StaticString = #filePath, line: UInt = #line
    ) {
        for (index, array) in arrays.enumerated() {
            let k = Int32(index + 1)
            XCTAssertEqual(array.asArray(Int32.self), [k, 2 * k, 3 * k], file: file, line: line)
        }
    }

    // MARK: - eval

    func testEvalVariadicAndCollection() {
        let a = lazyArrays(2)
        eval(a[0], a[1])
        checkValues(a)

        let b = lazyArrays(3)
        eval(b)
        checkValues(b)
    }

    func testEvalStructuredValues() {
        let a = lazyArrays(12)
        var nested = NestedDictionary<String, MLXArray>()
        nested["n"] = .array([.value(a[11])])

        eval(
            a[0],  // MLXArray
            [a[1]],  // [MLXArray]
            ["k": a[2]],  // dictionary
            (a[3], a[4], a[5]),  // 3-tuple
            (a[6], a[7], "label", 3),  // 4-tuple with ignored values
            (a[8], a[9], a[10], 2.5, "x"),  // 5-tuple with ignored values
            nested  // NestedDictionary
        )
        checkValues(a)
    }

    func testEvalSequenceOfAny() {
        let a = lazyArrays(3)
        let values: [Any] = [a[0], [a[1]], ("name", a[2])]
        eval(values)
        checkValues(a)
    }

    func testAsyncEvalOverloads() {
        let a = lazyArrays(2)
        asyncEval(a[0], [a[1]])
        checkValues(a)

        let b = lazyArrays(2)
        let values: [Any] = [b[0], (b[1], "label")]
        asyncEval(values)
        checkValues(b)

        let c = lazyArrays(2)
        asyncEval(c)
        checkValues(c)
    }

    func testCheckedEval() throws {
        let a = lazyArrays(2)
        try checkedEval(a[0], ["k": a[1]])
        checkValues(a)

        let b = lazyArrays(2)
        let values: [Any] = [b[0], [b[1]]]
        try checkedEval(values)
        checkValues(b)
    }

    func testCheckedEvalThrowsOnBrokenArray() {
        // a broadcast error makes an empty array; the handler keeps the process alive
        let ignore: @Sendable (String) -> Void = { _ in }
        let broken = withErrorHandler(ignore) {
            MLXArray(0 ..< 10, [2, 5]) + MLXArray(0 ..< 15, [3, 5])
        }

        XCTAssertThrowsError(try checkedEval(broken)) { error in
            XCTAssertTrue(error is MLXError, "\(error)")
        }
        let values: [Any] = [broken]
        XCTAssertThrowsError(try checkedEval(values)) { error in
            XCTAssertTrue(error is MLXError, "\(error)")
        }
    }

    // MARK: - grad

    func testGradArraysToArrays() {
        // f(x, y) = sum(x * y); df/dx = y, df/dy = x
        let f: ([MLXArray]) -> [MLXArray] = { arrays in
            [(arrays[0] * arrays[1]).sum()]
        }
        let x = MLXArray([1, 2, 3] as [Float])
        let y = MLXArray([4, 5, 6] as [Float])

        let dx = grad(f)([x, y])
        XCTAssertEqual(dx.count, 1)
        assertEqual(dx[0], y)

        let dxy = grad(f, argumentNumbers: [0, 1])([x, y])
        XCTAssertEqual(dxy.count, 2)
        assertEqual(dxy[0], y)
        assertEqual(dxy[1], x)
    }

    func testGradArraysToArray() {
        // f(x, y) = sum(x * x * y); df/dx = 2 * x * y, df/dy = x * x
        let f: ([MLXArray]) -> MLXArray = { arrays in
            (arrays[0] * arrays[0] * arrays[1]).sum()
        }
        let x = MLXArray([1, 2, 3] as [Float])
        let y = MLXArray([4, 5, 6] as [Float])

        let dx = grad(f)([x, y])
        assertEqual(dx, MLXArray([8, 20, 36] as [Float]))

        let dy = grad(f, argumentNumbers: [1])([x, y])
        assertEqual(dy, MLXArray([1, 4, 9] as [Float]))
    }

    func testGradArrayToArrays() {
        // f(x) = sum(x^3); df/dx = 3 * x^2
        let f: (MLXArray) -> [MLXArray] = { x in
            [(x * x * x).sum()]
        }
        let x = MLXArray([1, 2, 3] as [Float])

        let dx = grad(f)(x)
        XCTAssertEqual(dx.count, 1)
        assertEqual(dx[0], MLXArray([3, 12, 27] as [Float]))
    }

    func testValueAndGradArgumentNumbers() {
        // f(x, y) = sum(x * y)
        let f: ([MLXArray]) -> [MLXArray] = { arrays in
            [(arrays[0] * arrays[1]).sum()]
        }
        let x = MLXArray([1, 2] as [Float])
        let y = MLXArray([3, 4] as [Float])

        let (value, grads) = valueAndGrad(f, argumentNumbers: [0, 1])([x, y])
        XCTAssertEqual(value.count, 1)
        XCTAssertEqual(value[0].item(Float.self), 11)
        XCTAssertEqual(grads.count, 2)
        assertEqual(grads[0], y)
        assertEqual(grads[1], x)
    }
}
