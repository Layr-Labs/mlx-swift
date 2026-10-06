// Copyright © 2026 Apple Inc.

import MLX
import XCTest

final class FloorDivideTests: XCTestCase {
    func testSignedFloorDivideEagerAndCompiled() {
        let numerators: [Int32] = [-7, 7, -7, 7, -6, 6, -6, 6, 0, 0, -1, 1, -1, 1]
        let denominators: [Int32] = [2, 2, -2, -2, 3, 3, -3, -3, 5, -5, 4, -4, -4, 4]
        let quotients: [Int32] = [-4, 3, 3, -4, -2, 2, 2, -2, 0, 0, -1, -1, 0, 0]
        var devices: [Device] = [.cpu]
        #if canImport(Metal)
            devices.append(.gpu)
        #endif

        for device in devices {
            Device.withDefaultDevice(device) {
                for dtype in [DType.int8, .int16, .int32, .int64] {
                    // Include full SIMD blocks and a tail for every signed width.
                    let x = MLXArray(Array(repeating: numerators, count: 3).flatMap { $0 })
                        .asType(dtype)
                    let y = MLXArray(Array(repeating: denominators, count: 3).flatMap { $0 })
                        .asType(dtype)
                    let expected = Array(repeating: quotients, count: 3).flatMap { $0 }
                    let eager = x.floorDivide(y)
                    XCTAssertEqual(eager.dtype, dtype)
                    XCTAssertEqual(eager.asArray(Int32.self), expected)

                    let compiled = compile { (inputs: [MLXArray]) -> [MLXArray] in
                        [inputs[0].floorDivide(inputs[1]) + 1]
                    }
                    let fused = compiled([x, y])[0]
                    XCTAssertEqual(fused.dtype, dtype)
                    XCTAssertEqual(fused.asArray(Int32.self), expected.map { $0 + 1 })
                }
            }
        }
    }

    func testCPUFloorDivideZeroAndMinimumOverflow() {
        // Metal leaves these cases undefined; CPU preserves zero and wrapping results.
        Device.withDefaultDevice(.cpu) {
            let cases: [(DType, Int64)] = [
                (.int8, Int64(Int8.min)), (.int16, Int64(Int16.min)),
                (.int32, Int64(Int32.min)), (.int64, Int64.min),
            ]
            for (dtype, minimum) in cases {
                let x = MLXArray(
                    Array(repeating: [Int64(-7), 7, 0, minimum], count: 9)
                        .flatMap { $0 }
                ).asType(dtype)
                let y = MLXArray(
                    Array(repeating: [Int64(0), 0, 0, -1], count: 9)
                        .flatMap { $0 }
                ).asType(dtype)
                let expected: [Int64] = Array(repeating: [Int64(0), 0, 0, minimum], count: 9)
                    .flatMap { $0 }
                XCTAssertEqual(x.floorDivide(y).asArray(Int64.self), expected)

                let compiled = compile { (inputs: [MLXArray]) -> [MLXArray] in
                    [inputs[0].floorDivide(inputs[1]) + 1]
                }
                XCTAssertEqual(compiled([x, y])[0].asArray(Int64.self), expected.map { $0 + 1 })
            }
        }
    }
}
