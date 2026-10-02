// Copyright © 2026 Eigen Labs.

import Foundation
import XCTest

@testable import MLXNN

/// Tests for the small keyed cache in `MLXNN/Cache.swift` (used by `RoPE`).
class CacheTests: XCTestCase {

    func testStoreReadAndRemove() {
        let cache = Cache<String, Int>(maxSize: 3)
        XCTAssertNil(cache["a"])
        cache["a"] = 1
        cache["b"] = 2
        XCTAssertEqual(cache["a"], 1)
        XCTAssertEqual(cache["b"], 2)

        cache["a"] = 10
        XCTAssertEqual(cache["a"], 10)
        XCTAssertEqual(cache.contents.count, 2)

        cache["a"] = nil
        XCTAssertNil(cache["a"])
        XCTAssertEqual(cache["b"], 2)
        XCTAssertEqual(cache.contents.count, 1)
    }

    func testDefaultMaxSize() {
        XCTAssertEqual(Cache<Int, Int>().maxSize, 10)
    }

    /// When a new key makes the cache too large, the oldest entry goes.
    func testRemovesOldestEntryWhenFull() {
        let cache = Cache<Int, String>(maxSize: 2)
        cache[1] = "one"
        cache[2] = "two"
        cache[3] = "three"
        XCTAssertEqual(cache.contents.count, 2)
        XCTAssertNil(cache[1])
        XCTAssertEqual(cache[2], "two")
        XCTAssertEqual(cache[3], "three")
    }

    /// Setting a key again makes it the newest entry. Reading a key does not.
    func testSetAgainMakesEntryNewest() {
        let cache = Cache<Int, String>(maxSize: 2)
        cache[1] = "one"
        cache[2] = "two"
        _ = cache[1]
        cache[1] = "one again"
        cache[3] = "three"
        XCTAssertEqual(cache[1], "one again")
        XCTAssertNil(cache[2])
        XCTAssertEqual(cache[3], "three")
    }

    /// When the serial number reaches Int.max, the cache empties and starts again at 0.
    func testSerialWrapEmptiesCache() {
        let cache = Cache<Int, Int>(maxSize: 4)
        cache[1] = 1
        cache[2] = 2
        cache.serial = Int.max
        cache[3] = 3
        XCTAssertNil(cache[1])
        XCTAssertNil(cache[2])
        XCTAssertEqual(cache[3], 3)
        XCTAssertEqual(cache.contents.count, 1)
        XCTAssertEqual(cache.contents[3]?.serial, 0)
        XCTAssertEqual(cache.serial, 1)
    }

    /// Writes from many threads keep the size limit.
    func testConcurrentWritesKeepTheLimit() {
        let cache = Cache<Int, Int>(maxSize: 5)
        DispatchQueue.concurrentPerform(iterations: 200) { i in
            cache[i % 20] = i
            _ = cache[(i + 7) % 20]
        }
        XCTAssertEqual(cache.contents.count, 5)
        XCTAssertEqual(cache.serial, 200)
    }
}
