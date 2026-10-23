// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import XCTest

/// A compiled function keeps one cached trace per task default stream.
///
/// mlx-swift scopes the default device and stream to the task
/// (`Device.withDefaultDevice`, `Stream.withNewDefaultStream`), while MLX keys
/// its trace cache by the C++ thread default stream. A trace recorded in a CPU
/// scope must not be replayed when the same thread later calls the function in
/// a GPU scope.
class CompileStreamKeyTests: XCTestCase {

    private static func differing(_ a: MLXArray, _ b: MLXArray) -> Int {
        (a.view(dtype: .uint32) .!= b.view(dtype: .uint32)).sum().item(Int.self)
    }

    private static func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

    func testCPUScopedTraceIsNotReplayedOnTheGPU() {
        let x = (MLXRandom.normal([1, 8192], key: MLXRandom.key(4)) * 3).asType(.float32)
        eval(x)

        // CPU reference: a fresh compiled function in a CPU scope.
        let cpuReference = Device.withDefaultDevice(.cpu) { () -> MLXArray in
            let y = compile(shapeless: true) { (x: MLXArray) in Self.silu(x) }(x)
            eval(y)
            return y
        }
        // GPU reference: eager evaluation on the default (GPU) device.
        let gpuEager = Self.silu(x)
        eval(gpuEager)
        // The devices must round some elements apart; the result values then
        // show which device ran the compiled function.
        XCTAssertGreaterThan(Self.differing(cpuReference, gpuEager), 0)

        // One compiled function: first traced in a CPU scope, then called on the GPU.
        let shared = compile(shapeless: true) { (x: MLXArray) in Self.silu(x) }
        let onCPU = Device.withDefaultDevice(.cpu) { () -> MLXArray in
            let y = shared(x)
            eval(y)
            return y
        }
        let onGPU = shared(x)
        eval(onGPU)

        XCTAssertEqual(Self.differing(onCPU, cpuReference), 0)
        // Bitwise equal to GPU eager, and not the CPU result: it ran on the GPU.
        XCTAssertEqual(Self.differing(onGPU, gpuEager), 0)
        XCTAssertGreaterThan(Self.differing(onGPU, cpuReference), 0)
    }
}
