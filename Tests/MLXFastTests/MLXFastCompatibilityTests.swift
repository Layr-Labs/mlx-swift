// Copyright © 2026 Apple Inc.

import XCTest

import class MLX.MLXArray
import struct MLX.StreamOrDevice

@testable import MLXFast

final class MLXFastCompatibilityTests: XCTestCase {

    private typealias TopLevelFunction = (
        MLXArray, MLXArray, MLXArray, Float, MLXArray?, Int?, StreamOrDevice
    ) -> MLXArray

    func testFormerDeprecatedScaledDotProductAttentionFunctionType() {
        let topLevel: TopLevelFunction = scaledDotProductAttention
        _ = topLevel
    }
}
