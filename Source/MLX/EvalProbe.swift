// Copyright © 2026 Eigen Labs.
//
// EvalProbe — process-global, MEASUREMENT-ONLY instrumentation of the blocking
// `eval` path. This is the first-token "wedge" smoking gun: the first content
// token requires a blocking `mlx_eval` under the single process-global
// `evalLock` (see Transforms+Eval.swift). If that call hangs (a stuck Metal
// command buffer / wedged first-use kernel), it never returns while holding the
// lock, so EVERY model's token production freezes while accept/preamble/
// heartbeats keep running. This probe lets the provider report "the current eval
// has been running 11s+" — direct confirmation, not the inferred signature.
//
// Threading: `beginEval()` / `endEval()` are called ONLY from inside
// `evalLock.withLock { ... }`, so all WRITES are serialized by that lock and the
// `depth` counter correctly handles the lock's recursion. READS
// (`currentEvalElapsedMs`, etc.) are intentionally LOCK-FREE — they must NEVER
// take `evalLock`, because a wedged eval holds it forever and the reader (the
// heartbeat thread) has to stay alive to report the wedge. The read/write race
// on these scalars is the same benign-diagnostic pattern as
// `EngineCore.stepsExecuted`; on 64-bit aligned words it does not tear.
//
// PRIVACY: only durations/counters — never prompt/response content.

import Foundation

public enum EvalProbe {
    // Mutated only under `evalLock`; `nonisolated(unsafe)` mirrors the existing
    // global mutable state in Device.swift / Memory.swift.
    nonisolated(unsafe) private static var depth: Int = 0
    nonisolated(unsafe) private static var currentStartNanos: UInt64 = 0
    nonisolated(unsafe) private static var inFlightFlag: Bool = false
    nonisolated(unsafe) private static var completedCount: Int = 0
    nonisolated(unsafe) private static var longestMillis: Int64 = 0

    /// Mark the start of a locked eval. MUST be called inside `evalLock`.
    /// Only the outermost (depth 0→1) entry stamps the start, so a recursive
    /// eval reports the OUTER call's elapsed time (the one that can wedge).
    public static func beginEval() {
        if depth == 0 {
            currentStartNanos = DispatchTime.now().uptimeNanoseconds
            inFlightFlag = true
        }
        depth += 1
    }

    /// Mark the end of a locked eval. MUST be called inside `evalLock` (via
    /// `defer`). On a wedge this never runs, so `inFlight` stays true and
    /// `currentEvalElapsedMs` keeps climbing — exactly the signal we want.
    public static func endEval() {
        depth -= 1
        if depth <= 0 {
            depth = 0
            let elapsed = elapsedMs(since: currentStartNanos)
            if elapsed > longestMillis { longestMillis = elapsed }
            completedCount += 1
            inFlightFlag = false
            currentStartNanos = 0
        }
    }

    // MARK: - Lock-free reads (heartbeat / WedgeMonitor)

    /// True while a blocking eval is executing under `evalLock`.
    public static var inFlight: Bool { inFlightFlag }

    /// How long the CURRENT eval has been running (ms), or 0 if none is in
    /// flight. A value in the seconds range is the wedge smoking gun.
    public static var currentEvalElapsedMs: Int64 {
        guard inFlightFlag else { return 0 }
        return elapsedMs(since: currentStartNanos)
    }

    /// Cumulative count of completed (returned) evals.
    public static var evalsCompleted: Int { completedCount }

    /// Longest completed eval observed (ms). A wedged eval never completes, so it
    /// is captured by `currentEvalElapsedMs`, not here.
    public static var longestEvalMs: Int64 { longestMillis }

    private static func elapsedMs(since startNanos: UInt64) -> Int64 {
        guard startNanos != 0 else { return 0 }
        let now = DispatchTime.now().uptimeNanoseconds
        guard now > startNanos else { return 0 }
        return Int64((now &- startNanos) / 1_000_000)
    }
}
