// Copyright © 2026 Eigen Labs.
import Foundation
@_spi(GLMCalibration) import MLX
import XCTest

final class ProcessAllocationCapTests: XCTestCase {
    func testFreshProcessSwiftCap() throws {
        guard ProcessInfo.processInfo.environment["MLX_TEST_PROCESS_CAP"] == "1" else {
            throw XCTSkip("requires a dedicated fresh process")
        }
        try Memory.installProcessAllocationCap(bytes: 8 << 20)
        let initial = try Memory.processAllocationCapSnapshot()
        XCTAssertTrue(initial.installed)
        XCTAssertEqual(initial.chargedBytes, initial.heapBytes)
        XCTAssertThrowsError(try Memory.installProcessAllocationCap(bytes: 16 << 20))
        try autoreleasepool {
            let x = MLXArray((0 ..< 1024).map(Float.init))
            let y = x * 2 + 3
            eval(y)
            XCTAssertEqual(y.asArray(Float.self), (0 ..< 1024).map { Float($0 * 2 + 3) })
            XCTAssertGreaterThan(try Memory.processAllocationCapSnapshot().liveBytes, 0)
        }
        Memory.clearCache()
        let final = try Memory.processAllocationCapSnapshot()
        XCTAssertEqual(final.pendingBytes, 0)
        XCTAssertLessThanOrEqual(final.chargedBytes, final.limitBytes)

        // A wrapped external page is charged while MLX holds it and refunded
        // once when MLX releases it. MLX does not own it (upstream mlx #4634),
        // so the release runs the wrapper deleter, not the allocator free.
        try FinalizerCount.wait { try Memory.processAllocationCapSnapshot().unreclaimedBytes == 0 }
        let baseline = try Memory.processAllocationCapSnapshot()
        XCTAssertEqual(baseline.unreclaimedBytes, 0)
        let (base, bytes, count) = makeWrappableHostPage()
        let finalizer = FinalizerCount()
        try autoreleasepool {
            let wrapped = MLXArray(rawPointer: base, [count], dtype: .float32) {
                finalizer.increment()
            }
            let held = try Memory.processAllocationCapSnapshot()
            XCTAssertEqual(held.chargedBytes, baseline.chargedBytes + bytes)
            XCTAssertEqual(held.liveBytes, baseline.liveBytes + bytes)
            let doubled = wrapped * 2
            eval(doubled)
            XCTAssertEqual(doubled.asArray(Float.self), [Float](repeating: 2, count: count))
        }
        Memory.clearCache()
        try FinalizerCount.wait {
            let charged = try Memory.processAllocationCapSnapshot().chargedBytes
            return finalizer.current > 0 && charged == baseline.chargedBytes
        }
        let released = try Memory.processAllocationCapSnapshot()
        XCTAssertEqual(finalizer.current, 1)
        XCTAssertEqual(released.chargedBytes, baseline.chargedBytes)
        XCTAssertEqual(released.liveBytes, baseline.liveBytes)
        XCTAssertEqual(released.cachedBytes, baseline.cachedBytes)
        XCTAssertEqual(released.pendingBytes, 0)
        XCTAssertEqual(released.unreclaimedBytes, 0)
        // Keep the page if MLX can still use it.
        if finalizer.current == 1 { base.deallocate() }
    }
}
