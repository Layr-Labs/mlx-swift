// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation
import MLX
import XCTest

/// Counts calls of a `MLXArray(rawPointer:)` finalizer. MLX can call it on
/// another thread, after the GPU commands that use the buffer complete.
final class FinalizerCount: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0

    func increment() {
        lock.lock()
        value += 1
        lock.unlock()
    }

    var current: Int {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    /// Waits up to 5 s until `condition` is true.
    static func wait(until condition: () throws -> Bool) rethrows {
        let deadline = Date().addingTimeInterval(5)
        while try !condition(), Date() < deadline { usleep(1_000) }
    }
}

/// One page-aligned host page of Float 1.0. Metal wraps it without a copy.
func makeWrappableHostPage() -> (base: UnsafeMutableRawPointer, bytes: Int, count: Int) {
    let bytes = Int(getpagesize())
    let count = bytes / MemoryLayout<Float>.size
    let base = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: bytes)
    base.initializeMemory(as: Float.self, repeating: 1, count: count)
    return (base, bytes, count)
}

final class AllocationFootprintTests: XCTestCase {
    /// Upstream mlx #4634 marks a wrapped external buffer as not owned by MLX.
    /// `isUnique` must still be true while no other array refers to the buffer:
    /// mlx-swift-lm requires it for wrapped paged-KV pages.
    func testWrappedExternalBufferIsUniqueUntilShared() throws {
        let (base, bytes, count) = makeWrappableHostPage()
        let finalizer = FinalizerCount()
        do {
            let array = MLXArray(rawPointer: base, [count], dtype: .float32) {
                finalizer.increment()
            }
            // If Metal refuses the wrap, MLX copies and calls the finalizer at once.
            XCTAssertEqual(finalizer.current, 0, "MLX copied the page instead of wrapping it")
            XCTAssertEqual(
                UnsafeRawPointer(mlx_array_data_uint8(array.ctx)), UnsafeRawPointer(base))
            let info = try XCTUnwrap(array.evaluatedBufferInfo())
            XCTAssertTrue(info.isUnique)
            XCTAssertTrue(info.isRowContiguous)
            XCTAssertEqual(info.allocatedBytes, bytes)

            let allocationStream = StreamOrDevice.default
            let view = array[1 ..< count]
            try withError {
                eval(view)
                allocationStream.stream.synchronize()
            }
            XCTAssertFalse(try XCTUnwrap(array.evaluatedBufferInfo()).isUnique)
            XCTAssertFalse(try XCTUnwrap(view.evaluatedBufferInfo()).isUnique)
            withExtendedLifetime((array, view)) {}
        }
        FinalizerCount.wait { finalizer.current > 0 }
        XCTAssertEqual(finalizer.current, 1)
        // Keep the page if MLX can still use it.
        if finalizer.current == 1 { base.deallocate() }
    }

    /// Upstream mlx #4634: MLX does not donate a buffer that it does not own.
    /// An op on a wrapped array writes a new allocation and the caller's page
    /// does not change, also when the graph holds the only reference to it.
    func testOpOnWrappedExternalBufferDoesNotWriteTheCallerPage() throws {
        let (base, _, count) = makeWrappableHostPage()
        let finalizer = FinalizerCount()
        let result: MLXArray
        do {
            let wrapped = MLXArray(rawPointer: base, [count], dtype: .float32) {
                finalizer.increment()
            }
            XCTAssertEqual(finalizer.current, 0, "MLX copied the page instead of wrapping it")
            result = wrapped + 1
        }
        // An owned input with no other reference is donated to the output here.
        let allocationStream = StreamOrDevice.default
        try withError {
            eval(result)
            allocationStream.stream.synchronize()
        }
        XCTAssertNotEqual(
            UnsafeRawPointer(mlx_array_data_uint8(result.ctx)), UnsafeRawPointer(base))
        XCTAssertEqual(result.asArray(Float.self), [Float](repeating: 2, count: count))
        let page = UnsafeBufferPointer(
            start: base.assumingMemoryBound(to: Float.self), count: count)
        XCTAssertEqual(Array(page), [Float](repeating: 1, count: count))
        FinalizerCount.wait { finalizer.current > 0 }
        XCTAssertEqual(finalizer.current, 1)
        if finalizer.current == 1 { base.deallocate() }
    }

    func testUnscheduledMetadataQueryDoesNotEvaluate() throws {
        let allocationStream = StreamOrDevice.default
        let array = MLXArray.zeros([24_576], dtype: .uint8, stream: allocationStream)
        let before = Memory.snapshot()
        XCTAssertNil(try array.evaluatedBufferInfo())
        let after = Memory.snapshot()
        XCTAssertEqual(after.activeMemory, before.activeMemory)
        XCTAssertEqual(after.cacheMemory, before.cacheMemory)
        try withError {
            eval(array)
            allocationStream.stream.synchronize()
        }
        let info = try XCTUnwrap(array.evaluatedBufferInfo())
        XCTAssertTrue(info.isUnique)
        XCTAssertTrue(info.isRowContiguous)
        XCTAssertEqual(info.dataElements, array.size)
        XCTAssertEqual(info.dataOffset, 0)
        XCTAssertGreaterThanOrEqual(info.allocatedBytes, array.nbytes)
        XCTAssertLessThanOrEqual(
            info.allocatedBytes,
            try Memory.allocationFootprintUpperBound(byteCount: array.nbytes))
    }

    func testViewsReportSharedFullBackingWithoutBecomingOwners() throws {
        let allocationStream = StreamOrDevice.default
        let array = MLXArray.zeros([24_576], dtype: .uint8, stream: allocationStream)
        try withError { eval(array) }
        let original = try XCTUnwrap(array.evaluatedBufferInfo())
        let view = array[1024 ..< 2048]
        try withError {
            eval(view)
            allocationStream.stream.synchronize()
        }
        let sliced = try XCTUnwrap(view.evaluatedBufferInfo())
        XCTAssertEqual(sliced.allocatedBytes, original.allocatedBytes)
        XCTAssertGreaterThan(sliced.allocatedBytes, view.nbytes)
        XCTAssertEqual(sliced.dataOffset, 1024)
        XCTAssertFalse(sliced.isUnique)
        XCTAssertFalse(try XCTUnwrap(array.evaluatedBufferInfo()).isUnique)
        withExtendedLifetime((array, view)) {}
    }

    func testBoundCoversFreshAndLargerCachedBuffers() throws {
        let previousLimit = Memory.cacheLimit
        Memory.cacheLimit = 2 << 20
        defer {
            Memory.clearCache()
            Memory.cacheLimit = previousLimit
        }
        for (cachedBytes, requestedBytes) in [
            (48 << 10, 24_576), (8192, 6000), (64 << 10, 48 << 10),
        ] {
            Memory.clearCache()
            try autoreleasepool {
                let cached = MLXArray.zeros([cachedBytes], dtype: .uint8)
                try withError { eval(cached) }
            }
            let array = MLXArray.zeros([requestedBytes], dtype: .uint8)
            try withError { eval(array) }
            let info = try XCTUnwrap(array.evaluatedBufferInfo())
            XCTAssertGreaterThanOrEqual(info.allocatedBytes, array.nbytes)
            XCTAssertLessThanOrEqual(
                info.allocatedBytes,
                try Memory.allocationFootprintUpperBound(byteCount: requestedBytes))
        }
    }

    func testPredictionRejectsNegativeAndOverflowWithoutAllocation() throws {
        XCTAssertEqual(try Memory.allocationFootprintUpperBound(byteCount: 0), 0)
        XCTAssertThrowsError(try Memory.allocationFootprintUpperBound(byteCount: -1))
        XCTAssertThrowsError(try Memory.allocationFootprintUpperBound(byteCount: Int.max))
    }

    func testDetachedPolicyMatchesAllocatorAtSizeClassBoundaries() throws {
        let policy = try XCTUnwrap(Memory.allocationFootprintPolicy())
        let extra = try XCTUnwrap(policy.maximumExtraBytes)
        for bytes in [
            0, 1, 2, 4, 7, 8, 9, 4095, 4096, 4097, 8191, 8192, 8193,
            16_383, 16_384, 16_385, 24_576, 32_767, 32_768, (64 << 20) + 1,
        ] {
            let bound = try XCTUnwrap(policy.upperBound(byteCount: bytes))
            XCTAssertEqual(bound, try Memory.allocationFootprintUpperBound(byteCount: bytes))
            XCTAssertLessThanOrEqual(bound - bytes, extra)
        }
    }

    func testDetachedPolicyOverflowDoesNotInvokeMLXErrorHandler() throws {
        let policy = try XCTUnwrap(Memory.allocationFootprintPolicy())
        try withError {
            XCTAssertNil(policy.upperBound(byteCount: -1))
            XCTAssertNil(policy.upperBound(byteCount: Int.max))
            XCTAssertEqual(policy.upperBound(byteCount: 0), 0)
        }
    }
}
