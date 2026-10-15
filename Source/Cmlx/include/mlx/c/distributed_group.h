/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_DISTRIBUTED_GROUP_H
#define MLX_DISTRIBUTED_GROUP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mlx/c/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_distributed_group MLX distributed
 */
/**@{*/

/**
 * A MLX distributed group object.
 */
typedef struct mlx_distributed_group_ {
  void* ctx;
} mlx_distributed_group;

/**
 * Create an empty group.
 */
mlx_distributed_group mlx_distributed_group_new(void);

/**
 * Free the group.
 */
int mlx_distributed_group_free(mlx_distributed_group group);

/**
 * Initialize distributed.
 */
int mlx_distributed_init(
    mlx_distributed_group* res,
    bool strict,
    const char* bk /* may be null */);

/**
 * Synchronous owner-supplied JACCL bootstrap exchange. Return zero only after
 * writing exactly dst_bytes, in rank order. Pointers must not escape the call.
 * Callbacks must not throw or reenter MLX. The owner bounds waits and authenticates
 * its channel, epoch, rank, sequence, lengths and native topology.
 */
typedef int (*mlx_distributed_bootstrap_gather)(
    void* context, int rank, int size, uint64_t sequence,
    const char* src, size_t src_bytes, char* dst, size_t dst_bytes);
typedef void (*mlx_distributed_bootstrap_release)(void* context);

/**
 * Strict JACCL initialization through an owner callback, without TCPAllGather.
 * Existing cached JACCL initialization is rejected; it is not reauthenticated.
 * Both callbacks are required. When release_context is non-null, this call
 * consumes one context reference on every path. Release can occur only when
 * the native backend cache is destroyed, not when one group handle is freed.
 * Callback failure is permanent. This API itself does not authenticate bytes.
 * Bounds: 2..64 ranks, 1..1MiB per rank, at most 8MiB total; the total bound
 * must fit all ranks at their per-rank bound. Use a fresh worker per epoch.
 */
int mlx_distributed_init_jaccl_with_bootstrap(
    mlx_distributed_group* res, int expected_rank, int expected_size,
    size_t maximum_rank_bytes, size_t maximum_total_bytes,
    mlx_distributed_bootstrap_gather gather, void* context,
    mlx_distributed_bootstrap_release release_context);

/**
 * Get the rank.
 */
int mlx_distributed_group_rank(mlx_distributed_group group);

/**
 * Get the group size.
 */
int mlx_distributed_group_size(mlx_distributed_group group);

/**
 * Split the group.
 */
int mlx_distributed_group_split(
    mlx_distributed_group* res,
    mlx_distributed_group group,
    int color,
    int key);

/**
 * Check if distributed is available.
 */
bool mlx_distributed_is_available(const char* bk /* may be null */);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
