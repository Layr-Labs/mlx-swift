// Copyright © 2026 Apple Inc.

import MLX
import XCTest

final class MLXFastCompatibilityTests: XCTestCase {

    private typealias ArrayMaskFunction = (
        MLXArray, MLXArray, MLXArray, Float, MLXArray?, MLXArray?, Int?, StreamOrDevice
    ) -> MLXArray
    private typealias MaskModeFunction = (
        MLXArray, MLXArray, MLXArray, Float, MLXFast.ScaledDotProductAttentionMaskMode,
        MLXArray?, StreamOrDevice
    ) -> MLXArray
    private typealias TopLevelFunction = (
        MLXArray, MLXArray, MLXArray, Float, MLXArray?, Int?, StreamOrDevice
    ) -> MLXArray

    func testFormerScaledDotProductAttentionFunctionTypes() {
        let arrayMask: ArrayMaskFunction = MLXFast.scaledDotProductAttention
        let maskMode: MaskModeFunction = MLXFast.scaledDotProductAttention
        let topLevel: TopLevelFunction = scaledDotProductAttention

        _ = (arrayMask, maskMode, topLevel)
    }
}
