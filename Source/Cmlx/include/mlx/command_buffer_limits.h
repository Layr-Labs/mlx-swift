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

#ifdef __cplusplus
}
#endif
#endif

#endif
