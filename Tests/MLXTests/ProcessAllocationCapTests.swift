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
    }
}
