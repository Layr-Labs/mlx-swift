# Bounded JACCL startup readiness

Cmlx mirrors `mlx_distributed_group_begin_startup_wait_budget(group, timeout_ms)`
and `mlx_distributed_group_end_startup_wait_budget(group)` from the C dependency.
Begin accepts 1..900000 ms on an initialized JACCL group. The deadline begins
once and progress cannot renew it. Drain queued work and serialize scheduling
around begin/end; end requires an idle, open, unexpired budget and restores the
ordinary idle policy. Invalid/null/unsupported/closed groups return failure via
the installed C error handler. The caller retains hard cancellation and publishes
READY only after successful end. Memory caps and authentication are unchanged.

The mirrored header is byte-equal to the C startup-wait header. The pinned core
and C commits carry this API, and both gitlinks move together when either one
moves. Native compile, actual RDMA and GPU qualification remain separate
requirements.
