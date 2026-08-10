// Copyright © 2026 Apple Inc.

#ifndef MLX_GEMMA4_EXPERT_QMM_H
#define MLX_GEMMA4_EXPERT_QMM_H

#include <stdint.h>

#if defined(__APPLE__)
#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlx_metal_gemma4_expert_qmm_diagnostics {
  uint8_t requested;
  uint8_t aot_available;
  uint8_t nax_available;
  uint8_t armed;
  uint64_t attempts;
  uint64_t hits;
  uint64_t fallback_nax;
  uint64_t fallback_outer_route;
  uint64_t fallback_quantization;
  uint64_t fallback_topology;
  uint64_t fallback_assignment_count;
  uint64_t fallback_geometry;
  uint64_t fallback_metallib_unavailable;
} mlx_metal_gemma4_expert_qmm_diagnostics;

void mlx_metal_gemma4_expert_qmm_diagnostics_snapshot(
    mlx_metal_gemma4_expert_qmm_diagnostics* diagnostics);
void mlx_metal_gemma4_expert_qmm_diagnostics_reset(void);
void mlx_metal_gemma4_expert_qmm_diagnostics_clear_and_arm(void);
void mlx_metal_gemma4_expert_qmm_diagnostics_snapshot_and_disarm(
    mlx_metal_gemma4_expert_qmm_diagnostics* diagnostics);

#ifdef __cplusplus
}
#endif
#endif

#endif
