/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_DISTRIBUTED_GROUP_H
#define MLX_DISTRIBUTED_GROUP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mlx/c/array.h"
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
 * Sealed all-sum (Darkbloom fork, mlx/distributed/sealed.h): the two ranks'
 * partials cross the link only as frames the installed codec seals and opens
 * (a frame is plaintext bytes + overhead). Callbacks run on the CPU stream's
 * thread, must not reenter MLX and must not throw; nonzero refuses (the
 * evaluation fails). counter advances by one per sealed all-sum, identically
 * on both ranks. A null seal removes the codec.
 */
typedef int (*mlx_distributed_seal_fn)(
    void* context, const void* plaintext, size_t bytes, void* frame, uint64_t counter);
typedef int (*mlx_distributed_open_fn)(
    void* context, const void* frame, size_t bytes, void* plaintext, uint64_t counter, int peer);
int mlx_distributed_set_seal_codec(
    mlx_distributed_seal_fn seal, mlx_distributed_open_fn open, void* context, size_t overhead);
/**
 * The same codec with per-piece callbacks. On a two-rank JACCL mesh a partial
 * then crosses as pieces of piece_bytes plaintext bytes (the last can be
 * shorter), each sealed as its ciphertext then a 16-byte tag, in order. One
 * CPU-stream task sends each piece when it is sealed and opens and adds each
 * peer piece when it lands. Other groups use seal and open. The piece
 * callbacks run at the same time on worker threads, in any order within one
 * counter; they must not reenter MLX or throw; nonzero refuses (the
 * evaluation fails). piece_bytes: a multiple of 16 in [4 KiB, 16 MiB].
 */
typedef int (*mlx_distributed_seal_piece_fn)(
    void* context, const void* plaintext, size_t length, void* sealed,
    uint64_t counter, uint32_t piece, uint32_t pieces, size_t total);
typedef int (*mlx_distributed_open_piece_fn)(
    void* context, const void* sealed, size_t length, void* plaintext,
    uint64_t counter, uint32_t piece, uint32_t pieces, size_t total, int peer);
int mlx_distributed_set_streamed_seal_codec(
    mlx_distributed_seal_fn seal, mlx_distributed_open_fn open,
    mlx_distributed_seal_piece_fn seal_piece, mlx_distributed_open_piece_fn open_piece,
    void* context, size_t overhead, size_t piece_bytes);
int mlx_distributed_all_sum_sealed(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s);

/**
 * Opt in to one bounded JACCL startup wait on this initialized group.
 * timeout_ms must be 1..900000; progress cannot renew its absolute deadline.
 * Drain queued MLX work and serialize scheduling around begin/end; require
 * no running local group operation. Unsupported backends, repeated begin,
 * and begin after end refuse. This does not change the global
 * progress timeout, memory caps, authentication, or failed-group state.
 */
int mlx_distributed_group_begin_startup_wait_budget(
    mlx_distributed_group group,
    int64_t timeout_ms);

/**
 * End the one-shot startup budget before publishing READY. Subsequent calls
 * use the original progress guard. A closed group is never reopened.
 */
int mlx_distributed_group_end_startup_wait_budget(mlx_distributed_group group);

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
