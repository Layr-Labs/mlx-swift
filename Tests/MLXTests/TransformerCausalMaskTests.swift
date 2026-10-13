// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

@testable import MLXNN

/// Tests for `MultiHeadAttention.createAdditiveCausalMask` in `Transformer.swift`.
class TransformerCausalMaskTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    /// A float16 mask must still be 0 where a position may attend.
    func testCausalMaskFloat16() {
        let mask = MultiHeadAttention.createAdditiveCausalMask(4, dtype: .float16)
        XCTAssertEqual(mask.dtype, .float16)
        let values = mask.asType(.float32)
        let indices = MLXArray(0 ..< 4)
        let blocked = expandedDimensions(indices, axis: 1) .< expandedDimensions(indices, axis: 0)
        XCTAssertFalse(isNaN(values).any().item(Bool.self), "\(values)")
        // Blocked positions hold the most negative finite float16 value, -65504.
        let float16Min: Float = -65504
        let ok = which(blocked, values .== float16Min, values .== 0)
        XCTAssertTrue(ok.all().item(Bool.self), "\(values)")
    }
}
