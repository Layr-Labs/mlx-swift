// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import Numerics
import XCTest

/// Tests for the default arguments of `nanToNum` in `Ops.swift`.
class NanToNumDefaultsTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    /// Without posInf and negInf, infinities become the largest finite values of the
    /// dtype, as the doc comment and Python `mx.nan_to_num` say.
    func testNanToNumDefaults() {
        let special = MLXArray([Float.nan, Float.infinity, -Float.infinity, 1], [4])
        let r = nanToNum(special)
        XCTAssertEqual(r.dtype, .float32)
        XCTAssertEqual(
            r.asArray(Float.self),
            [0, Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, 1])

        // an explicit value still replaces the infinities
        let z = nanToNum(special, nan: 2, posInf: 0, negInf: 0)
        XCTAssertEqual(z.asArray(Float.self), [2, 0, 0, 1])
    }

    /// The C++ default throws for float64 and complex64. The Swift function passes the
    /// largest finite Float for these types. float64 is not allowed on the GPU.
    func testNanToNumDefaultsFloat64() {
        let special = MLXArray([Double.nan, Double.infinity, -Double.infinity, 1])
        XCTAssertEqual(special.dtype, .float64)
        let r = nanToNum(special, stream: .cpu)
        XCTAssertEqual(r.dtype, .float64)
        let m = Double(Float.greatestFiniteMagnitude)
        XCTAssertEqual(r.asArray(Double.self), [0, m, -m, 1])
    }

    func testNanToNumDefaultsComplex64() {
        let special = MLXArray([
            Complex<Float>(.nan, 0), Complex<Float>(.infinity, 0),
            Complex<Float>(-.infinity, 0), Complex<Float>(1, 2),
        ])
        XCTAssertEqual(special.dtype, .complex64)
        let r = nanToNum(special)
        XCTAssertEqual(r.dtype, .complex64)
        let m = Float.greatestFiniteMagnitude
        XCTAssertEqual(
            r.asArray(Complex<Float>.self),
            [Complex(0, 0), Complex(m, 0), Complex(-m, 0), Complex(1, 2)])
    }
}
