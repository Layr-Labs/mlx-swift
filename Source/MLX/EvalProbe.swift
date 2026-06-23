// Copyright © 2026 Eigen Labs.
//
// EvalProbe — process-global, MEASUREMENT-ONLY instrumentation of the blocking
// `eval` path. This is the first-token "wedge" smoking gun: the first content
// token requires a blocking `mlx_eval` / `mlx_array_eval` under the single
// process-global `evalLock` (see Transforms+Eval.swift and MLXArray.eval). If
// that call hangs (a stuck Metal command buffer / wedged first-use kernel) it
// never returns while holding `evalLock`, so EVERY model's token production
// freezes while accept/preamble/heartbeats keep running. This probe lets the
// provider report "the current eval has been running 11s+" — direct confirmation,
// not the inferred signature.
//
// Threading: `beginEval()` / `endEval()` are called from inside
// `evalLock.withLock { ... }` (free `eval` + `MLXArray.eval`), so writes are
// additionally serialized by that lock and the `depth` counter handles the
// recursive lock's nesting. The cross-thread hazard is the heartbeat thread
// READING the state while an eval thread writes it — so the state lives behind an
// `OSAllocatedUnfairLock` rather than `nonisolated(unsafe)` scalars (which only
// silence the isolation check and leave a real torn-read data race).
//
// Why a lock is safe on the hot path (and the heartbeat is never blocked by a
// wedge): the unfair lock is held ONLY for the few nanoseconds of begin/end
// bookkeeping — NEVER across the actual `mlx_eval` call. So when an eval wedges,
// the probe lock is free and the heartbeat reader acquires it immediately. The
// lock is also uncontended on writes (begin/end are serialized by `evalLock`).
// `OSAllocatedUnfairLock` (macOS 13+/iOS 16+) is available at the package's
// macOS 14 / iOS 17 floor; `Synchronization.Atomic` would require macOS 15.
//
// PRIVACY: only durations/counters — never prompt/response content.

import Foundation
import os

public enum EvalProbe {
    /// All probe state under one lock so cross-thread reads get a consistent,
    /// non-torn snapshot. Value type ⇒ the lock box is Sendable.
    private struct State {
        /// Recursion depth of nested locked evals (writer-only; serialized by
        /// `evalLock`). The outermost 0→1 stamps `sinceNanos`; 1→0 clears it.
        var depth: Int = 0
        /// Monotonic start (DispatchTime uptime ns) of the in-flight outermost
        /// eval; 0 means none in flight.
        var sinceNanos: UInt64 = 0
        /// Cumulative completed (returned) evals.
        var completed: Int = 0
        /// Longest completed eval observed (ms).
        var longestMs: Int64 = 0
    }

    private static let state = OSAllocatedUnfairLock(initialState: State())

    // MARK: - Writers (called inside evalLock, via begin / defer end)

    /// Mark the start of a locked eval. Only the outermost (depth 0→1) entry
    /// stamps the start, so a recursive eval reports the OUTER call's elapsed time
    /// (the one that can wedge).
    public static func beginEval() {
        let now = DispatchTime.now().uptimeNanoseconds
        state.withLock { s in
            if s.depth == 0 { s.sinceNanos = now }
            s.depth += 1
        }
    }

    /// Mark the end of a locked eval (via `defer`). On a wedge this never runs, so
    /// `sinceNanos` stays set and `currentEvalElapsedMs` keeps climbing — exactly
    /// the signal we want. Updates `longest`/`completed` only on the outermost end.
    public static func endEval() {
        let now = DispatchTime.now().uptimeNanoseconds
        state.withLock { s in
            s.depth -= 1
            guard s.depth <= 0 else { return }
            s.depth = 0
            if s.sinceNanos != 0, now > s.sinceNanos {
                let elapsed = Int64((now &- s.sinceNanos) / 1_000_000)
                if elapsed > s.longestMs { s.longestMs = elapsed }
            }
            s.completed += 1
            s.sinceNanos = 0
        }
    }

    // MARK: - Reads (heartbeat / WedgeMonitor) — never take evalLock

    /// True while a blocking eval is executing under `evalLock`.
    public static var inFlight: Bool {
        state.withLock { $0.sinceNanos != 0 }
    }

    /// How long the CURRENT eval has been running (ms), or 0 if none is in
    /// flight. A value in the seconds range is the wedge smoking gun.
    public static var currentEvalElapsedMs: Int64 {
        let now = DispatchTime.now().uptimeNanoseconds
        return state.withLock { s in
            guard s.sinceNanos != 0, now > s.sinceNanos else { return 0 }
            return Int64((now &- s.sinceNanos) / 1_000_000)
        }
    }

    /// Cumulative count of completed (returned) evals.
    public static var evalsCompleted: Int {
        state.withLock { $0.completed }
    }

    /// Longest completed eval observed (ms). A wedged eval never completes, so it
    /// is captured by `currentEvalElapsedMs`, not here.
    public static var longestEvalMs: Int64 {
        state.withLock { $0.longestMs }
    }
}
