// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

class MLXFastKernelTests: XCTestCase {

    func testCustomKernelBasic() {
        // based on def test_custom_kernel_basic
        MLXRandom.seed(7)
        let a = normal([2, 2])
        let kernel = MLXFast.metalKernel(
            name: "basic",
            inputNames: ["a"],
            outputNames: ["out1"],
            source: """
                    uint elem = thread_position_in_grid.x;
                    out1[elem] = a[elem];
                """)

        let out = kernel(
            [a],
            grid: (4, 1, 1),
            threadGroup: (2, 1, 1),
            outputShapes: [[2, 2]],
            outputDTypes: [.float32])

        XCTAssertTrue(allClose(out[0], a).all().item())
    }

    func testCustomKernelArgs() {
        // based on def test_custom_kernel_args
        MLXRandom.seed(7)
        let a = normal([3, 6])
        let c = normal([2, 2]).asType(.bfloat16)

        let kernel = MLXFast.metalKernel(
            name: "arg_test",
            inputNames: ["a", "b", "c", "d"],
            outputNames: ["out1", "out2"],
            source: """
                    uint elem = thread_position_in_grid.x;
                    T tmp = a[0];
                    if (e) {
                        out1[elem] = a[1] + b[2] + c[3] + d + f;
                    } else {
                        out1[elem] = 1;
                    }
                    out2[elem] = a[1] + b[2] + c[1] - d;
                """)

        let out = kernel(
            [
                a,
                MLXArray([3, 4, 5]),
                c,
                7.3,
            ],
            template: [
                ("e", true),
                ("f", 3),
                ("T", DType.float16),
            ],
            grid: (6, 1, 1),
            threadGroup: (2, 1, 1),
            outputShapes: [[2, 2], [3, 2]],
            outputDTypes: [.float32, .int32])

        XCTAssertTrue(allClose(out[0], full([2, 2], values: 14.0484)).all().item())
        XCTAssertTrue(allClose(out[1], full([3, 2], values: -2)).all().item())
    }

    func testFastSDPA() {
        // https://github.com/ml-explore/mlx-swift/issues/172
        // this will just make sure the MLXFast.scaled_dot_product_attention is
        // callable in the various cases, based on
        // https://github.com/ml-explore/mlx/blob/main/python/tests/test_fast_sdpa.py#L65-L87

        let Dk = 64
        let scale = 1.0 / sqrt(Float(Dk))
        let dTypes = [DType.float32, DType.float16]
        for SEQUENCE_LENGTH in [63, 129, 400] {
            for dtype in dTypes {
                let B = 2
                let H = 24
                let q = MLXRandom.normal([B, H, SEQUENCE_LENGTH, Dk]).asType(dtype)
                let k = MLXRandom.normal([B, H, SEQUENCE_LENGTH, Dk]).asType(dtype)
                let v = MLXRandom.normal([B, H, SEQUENCE_LENGTH, Dk]).asType(dtype)

                let result = MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: scale, mask: nil,
                    memoryEfficientThreshold: 2)

                eval(result)
            }
        }
    }

    func testFastSDPAOutput() {
        MLXRandom.seed(0)
        let queries = MLXRandom.uniform(0.0 ..< 1.0, [1, 32, 1, 80])
        let keys = MLXRandom.uniform(0.0 ..< 1.0, [1, 32, 9, 80])
        let values = MLXRandom.uniform(0.0 ..< 1.0, [1, 32, 9, 80])
        let result = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: 0.1118, mask: .none)
        print(result.shape)
        print(result.sum().item(Float.self))
        XCTAssertEqual(result.shape, [1, 32, 1, 80])
        XCTAssertEqual(result.sum().item(Float.self), 1281.9253, accuracy: 0.01)
    }

    func testFastSDPAForceFusedD256() {
        MLXRandom.seed(7)
        let headDimension = 256
        let scale = 1.0 / sqrt(Float(headDimension))
        let queries = MLXRandom.normal([1, 2, 9, headDimension]).asType(.float16)
        let keys = MLXRandom.normal([1, 2, 17, headDimension]).asType(.float16)
        let values = MLXRandom.normal([1, 2, 17, headDimension]).asType(.float16)
        let reference = softmax(
            (queries * scale).matmul(keys.transposed(0, 1, 3, 2)), axis: -1
        ).matmul(values)

        let defaultResult = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: nil)
        let forcedResult = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: .none,
            forceFused: true)

        XCTAssertTrue(defaultResult.allClose(reference, rtol: 1e-2, atol: 1e-2).item(Bool.self))
        XCTAssertTrue(forcedResult.allClose(reference, rtol: 1e-2, atol: 1e-2).item(Bool.self))
    }

    func testFastSDPAForceFusedRejectsUnsupportedD256Float32() {
        let queries = MLXRandom.normal([1, 2, 9, 256])
        let keys = MLXRandom.normal([1, 2, 17, 256])
        let values = MLXRandom.normal([1, 2, 17, 256])

        XCTAssertThrowsError(
            try withError {
                _ = MLXFast.scaledDotProductAttention(
                    queries: queries, keys: keys, values: values, scale: 1.0 / 16.0,
                    mask: nil, forceFused: true)
            }
        ) { error in
            guard case MLXError.caught(let message) = error else {
                return XCTFail("expected MLXError, got \(error)")
            }
            XCTAssertTrue(message.contains("53760 bytes"), message)
            XCTAssertTrue(message.contains("32 KiB"), message)
        }
    }

    func testRoPEOutput() {
        // https://github.com/ml-explore/mlx-swift/issues/315
        MLXRandom.seed(0)
        let rope = RoPE(dimensions: 32, traditional: false, base: 10_000, scale: 1)
        let queries = MLXRandom.uniform(0.0 ..< 1.0, [1, 32, 1, 80])
        print(queries.shape)
        let result = rope(queries, offset: 8)
        XCTAssertEqual(result.sum().item(Float.self), 1079.7894, accuracy: 0.01)
    }
}
