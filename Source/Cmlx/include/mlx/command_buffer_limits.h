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

/// Output tile (32, 64, 6432, 3264 or 6464: rows x columns[, K depth]) of the
/// steel affine quantized matmul for the
/// products the calling thread encodes on a GPU without neural accelerators;
/// any other value restores MLX_QMM_STEEL_TILE. Stores the thread's previous
/// override (0 for none) when the out pointer is not null. Every element of
/// a product keeps its bits at either tile.
void mlx_metal_set_thread_qmm_steel_tile(int tile, int* previous_tile);

/// Reference row count for the quantized matmuls and attention ops that the
/// calling thread builds from now on; 0 ends the scope. Stores the thread's
/// previous value when the out pointer is not null. An op keeps the value of
/// the thread that builds it. On the GPU, a quantized x @ w.T of two or more
/// rows built with rows > 0 splits K as a product of `rows` rows would, so a
/// row's bits do not depend on how many rows share the call; head-dim 256
/// causal or masked attention takes the fused NAX kernel from 16 queries
/// instead of 1024. Ops built with 0 are unchanged.
void mlx_metal_set_thread_prefill_rows(int rows, int* previous_rows);

/// The calling thread's prefill reference rows (0 outside a scope).
int mlx_metal_thread_prefill_rows(void);

#ifdef __cplusplus
}
#endif
#endif

#endif
