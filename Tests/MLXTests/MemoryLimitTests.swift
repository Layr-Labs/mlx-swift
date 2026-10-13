// Copyright © 2026 Eigen Labs.

import Foundation
import XCTest

@testable import MLX

/// Tests for `Memory.swift`: snapshots and the cache and memory limits.
///
/// The limits are global state. Each test restores every value it changes in a `defer`.
class MemoryLimitTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    func testSnapshotDelta() {
        let start = Memory.Snapshot(activeMemory: 100, cacheMemory: 50, peakMemory: 200)
        let end = Memory.Snapshot(activeMemory: 160, cacheMemory: 20, peakMemory: 300)

        let delta = start.delta(end)
        XCTAssertEqual(delta.activeMemory, 60)
        XCTAssertEqual(delta.cacheMemory, -30)
        XCTAssertEqual(delta.peakMemory, 100)
    }

    func testSnapshotDescription() {
        // values up to 10 MiB show in K, larger values show in M
        let active = 5 * 1024
        let cache = 20 * 1024 * 1024
        let peak = 11 * 1024 * 1024
        let snapshot = Memory.Snapshot(activeMemory: active, cacheMemory: cache, peakMemory: peak)

        func padded(_ s: String) -> String {
            s + String(repeating: " ", count: 12 - s.count)
        }
        let expected =
            "Peak:   \(padded("11M")) (\(peak))\n"
            + "Active: \(padded("5K")) (\(active))\n"
            + "Cache:  \(padded("20M")) (\(cache))"
        XCTAssertEqual(snapshot.description, expected)
    }

    func testSnapshotCodableRoundTrip() throws {
        let snapshot = Memory.Snapshot(activeMemory: 1, cacheMemory: 2, peakMemory: 3)
        let data = try JSONEncoder().encode(snapshot)
        let decoded = try JSONDecoder().decode(Memory.Snapshot.self, from: data)
        XCTAssertEqual(decoded.activeMemory, 1)
        XCTAssertEqual(decoded.cacheMemory, 2)
        XCTAssertEqual(decoded.peakMemory, 3)
    }

    func testLiveSnapshotMatchesGetters() {
        let snapshot = Memory.snapshot()
        XCTAssertGreaterThanOrEqual(snapshot.peakMemory, snapshot.activeMemory)
        XCTAssertEqual(snapshot.activeMemory, Memory.activeMemory)
        XCTAssertEqual(snapshot.cacheMemory, Memory.cacheMemory)
        XCTAssertEqual(snapshot.peakMemory, Memory.peakMemory)
    }

    func testCacheLimitGetAndSet() {
        let savedCached = Memory._cacheLimit
        let savedLimit = Memory.cacheLimit
        defer {
            Memory.cacheLimit = savedLimit
            Memory._cacheLimit = savedCached
        }

        // with no stored value the getter reads the limit from MLX
        Memory._cacheLimit = nil
        XCTAssertEqual(Memory.cacheLimit, savedLimit)
        XCTAssertEqual(Memory._cacheLimit, savedLimit)

        // set a new value: the stored value and the MLX value change
        let newLimit = 3 * 1024 * 1024
        Memory.cacheLimit = newLimit
        XCTAssertEqual(Memory.cacheLimit, newLimit)

        Memory._cacheLimit = nil
        XCTAssertEqual(Memory.cacheLimit, newLimit)
    }

    func testMemoryLimitGetAndSet() {
        let savedStored = Memory._memoryLimit
        let savedLimit = Memory.memoryLimit
        defer {
            Memory.memoryLimit = savedLimit
            Memory._memoryLimit = savedStored
        }
        XCTAssertGreaterThan(savedLimit, 0)

        let newLimit = savedLimit / 2
        Memory.memoryLimit = newLimit
        XCTAssertEqual(Memory.memoryLimit, newLimit)
        XCTAssertEqual(Memory._memoryLimit, newLimit)

        // work still runs under the lower limit
        let a = MLXArray([1, 2, 3] as [Int32]) * 2
        XCTAssertEqual(a.asArray(Int32.self), [2, 4, 6])

        Memory.memoryLimit = savedLimit
        XCTAssertEqual(Memory.memoryLimit, savedLimit)
    }
}
