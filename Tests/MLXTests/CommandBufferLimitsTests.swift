// Copyright © 2026 Eigen Labs.

import Cmlx
import Foundation
import XCTest

@testable import MLX

/// The calling thread's command-buffer limits. Setting new limits returns the previous ones, so
/// the query writes them back at once.
private func threadLimits() -> GPU.CommandBufferLimits {
    var operations: Int32 = 0
    var megabytes: Int32 = 0
    mlx_metal_set_thread_command_buffer_limits(0, 0, &operations, &megabytes)
    mlx_metal_set_thread_command_buffer_limits(operations, megabytes, nil, nil)
    return .init(operations: Int(operations), megabytes: Int(megabytes))
}

private struct ScopeError: Error {}

/// Tests for `GPU.withCommandBufferLimits(_:_:)`: the scope sets the limits of the calling
/// thread only, restores the previous limits on return and on throw, and does not change results.
class CommandBufferLimitsTests: XCTestCase {

    private static let defaults = GPU.CommandBufferLimits(operations: 0, megabytes: 0)

    override class func setUp() {
        setDefaultDevice()
    }

    func testNestedScopesRestoreThePreviousLimits() {
        XCTAssertEqual(threadLimits(), Self.defaults)
        GPU.withCommandBufferLimits(.init(operations: 100, megabytes: 0)) {
            XCTAssertEqual(threadLimits(), .init(operations: 100, megabytes: 0))
            GPU.withCommandBufferLimits(.init(operations: 7, megabytes: 3)) {
                XCTAssertEqual(threadLimits(), .init(operations: 7, megabytes: 3))
            }
            XCTAssertEqual(threadLimits(), .init(operations: 100, megabytes: 0))
        }
        XCTAssertEqual(threadLimits(), Self.defaults)
    }

    func testThrowingBodyRestoresTheLimits() {
        XCTAssertThrowsError(
            try GPU.withCommandBufferLimits(.init(operations: 9, megabytes: 9)) {
                XCTAssertEqual(threadLimits(), .init(operations: 9, megabytes: 9))
                throw ScopeError()
            })
        XCTAssertEqual(threadLimits(), Self.defaults)
    }

    func testNegativeLimitsKeepTheDefaults() {
        GPU.withCommandBufferLimits(.init(operations: -5, megabytes: -1)) {
            XCTAssertEqual(threadLimits(), Self.defaults)
        }
    }

    func testLimitsApplyOnlyToTheCallingThread() {
        final class Observation: @unchecked Sendable {
            var limits: GPU.CommandBufferLimits?
        }
        GPU.withCommandBufferLimits(.init(operations: 11, megabytes: 13)) {
            let observation = Observation()
            let done = DispatchSemaphore(value: 0)
            Thread {
                observation.limits = threadLimits()
                done.signal()
            }.start()
            done.wait()
            XCTAssertEqual(observation.limits, Self.defaults)
            XCTAssertEqual(threadLimits(), .init(operations: 11, megabytes: 13))
        }
    }

    func testResultsDoNotDependOnTheLimits() {
        func compute() -> MLXArray {
            var x = MLXRandom.normal([64, 64], key: MLXRandom.key(7))
            for _ in 0 ..< 40 {
                x = tanh(matmul(x, x.T) / 64) + 0.5 * x
            }
            eval(x)
            return x
        }
        let reference = compute()
        let oneOperation = GPU.withCommandBufferLimits(.init(operations: 1, megabytes: 1)) {
            compute()
        }
        let large = GPU.withCommandBufferLimits(.init(operations: 1000, megabytes: 4000)) {
            compute()
        }
        XCTAssertTrue(arrayEqual(reference, oneOperation).item(Bool.self))
        XCTAssertTrue(arrayEqual(reference, large).item(Bool.self))
    }
}
