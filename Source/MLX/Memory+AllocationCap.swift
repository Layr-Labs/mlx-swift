// Copyright © 2026 Eigen Labs.
import Cmlx

extension Memory {
    /// Install once before any buffer allocation. Includes the existing Metal heap.
    /// Caps allocator backing only; host memory and driver overhead are separate.
    @_spi(GLMCalibration)
    public static func installProcessAllocationCap(bytes: Int) throws {
        guard bytes > 0 else { throw MLXError.caught("allocation cap must be positive") }
        try withSerializedEvaluation {
            try withError {
                guard mlx_install_process_allocation_cap(bytes) == 0 else {
                    throw MLXError.caught("allocation cap installation refused")
                }
            }
        }
    }

    @_spi(GLMCalibration)
    public struct ProcessAllocationCapSnapshot: Sendable {
        public let installed: Bool
        public let limitBytes: Int
        public let heapBytes: Int
        public let liveBytes: Int
        public let cachedBytes: Int
        public let pendingBytes: Int
        /// Awaiting Metal deallocation, or quarantined after failed page release.
        public let unreclaimedBytes: Int
        public let chargedBytes: Int
        public let peakBytes: Int
    }

    /// Does not synchronize or reclaim backing. Unsupported backends throw.
    @_spi(GLMCalibration)
    public static func processAllocationCapSnapshot() throws -> ProcessAllocationCapSnapshot {
        try withError {
            var value = mlx_process_allocation_cap_snapshot()
            guard mlx_get_process_allocation_cap_snapshot(&value) == 0 else {
                throw MLXError.caught("allocation cap snapshot unavailable")
            }
            return ProcessAllocationCapSnapshot(
                installed: value.installed,
                limitBytes: value.limit_bytes, heapBytes: value.heap_bytes,
                liveBytes: value.live_bytes, cachedBytes: value.cached_bytes,
                pendingBytes: value.pending_bytes, unreclaimedBytes: value.unreclaimed_bytes,
                chargedBytes: value.charged_bytes, peakBytes: value.peak_bytes)
        }
    }
}
