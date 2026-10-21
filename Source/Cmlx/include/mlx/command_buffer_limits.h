// Copyright © 2026 Apple Inc.

#ifndef MLX_COMMAND_BUFFER_LIMITS_H
#define MLX_COMMAND_BUFFER_LIMITS_H

#if defined(__APPLE__)
#ifdef __cplusplus
extern "C" {
#endif

/// Commit limits for the Metal command buffers that the calling thread
/// encodes (operations per buffer, megabytes of distinct inputs per buffer);
/// zero keeps the device default. Stores the thread's previous values when
/// the out pointers are not null.
void mlx_metal_set_thread_command_buffer_limits(
    int max_ops,
    int max_mb,
    int* previous_ops,
    int* previous_mb);

#ifdef __cplusplus
}
#endif
#endif

#endif
