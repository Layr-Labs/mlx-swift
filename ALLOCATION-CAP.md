# Metal backing cap and calibration SPI

Install `Memory.installProcessAllocationCap(bytes:)` through
`@_spi(GLMCalibration) import MLX` before any positive buffer allocation attempt.
Installation is process-local, once-only, and cannot be reset, disabled, or raised.
It includes the existing Metal allocator heap in full. CPU-only and CUDA builds
refuse the API. Ordinary allocation and soft `Memory.memoryLimit` remain unchanged
when the cap is not installed.

This limits owned VM backing plus the fixed heap and conservatively charged
foreign wrappers. It is **not** a whole-process, host, driver-overhead, or resident
physical-memory ceiling. The calibration owner must separately authorize host
budget and OS headroom. Installing this cap grants no serving qualification.

## Backing ownership

Cap-mode allocations outside the fixed heap reserve page-rounded bytes under
the allocator lock before releasing that lock. A single anonymous VM mapping is
passed to Metal's no-copy buffer API with the existing shared/untracked resource
options. There is no copied-buffer, CPU, or precision fallback. The deallocator
block holds a shared lease and ledger; it unmaps the owned region before refunding
credit. The callback takes no allocator lock. Cache reuse transfers the same
charge. Cache eviction and reference release only move credit to `unreclaimed`;
actual Metal deallocation returns it. An unsuccessful `munmap` leaves the charge
quarantined until process exit.

The existing native graph/command ownership must still keep buffers alive while
GPU work uses them. This cap does not repair premature releases in that ownership
protocol. The Metal deallocator is the destruction boundary, not a command fence
or an arbitrary `release()` call.

`make_buffer` remains a wrapper over caller-owned memory. Cap mode requires a
nonempty page-aligned pointer and length, charges its length while Metal retains
the wrapper, and never unmaps/frees the caller's region. That wrapper charge is
returned at the Metal callback; external ownership remains the caller's duty.
Aliased wrappers are charged separately. A pointer's single-region validity is
still the caller's Metal API precondition.

Acquisition failure releases any owned mapping and rolls back its pending charge.
Residency registration is transactional: a thrown insertion removes its map and
wired-set membership before the allocator releases the buffer. This cleanup also
applies when the cap is disabled. A capped no-copy wrapper failure throws instead
of returning null, so the array constructor cannot copy and invoke the caller
deleter as a fallback.

Once a valid buffer exists, setup failure releases the buffer and waits for its
deallocator; it does not refund early. The shared lease makes null-result cleanup
and callback cleanup idempotent, including callbacks during construction failure.

## Interfaces

Public core: `install_process_allocation_cap(size_t)` and
`get_process_allocation_cap_snapshot()` in `mlx/memory.h`.

Public C: `mlx_install_process_allocation_cap(size_t)` and
`mlx_get_process_allocation_cap_snapshot(mlx_process_allocation_cap_snapshot*)`.
Both return zero on success and route exceptions to the existing error handler.
A failed snapshot leaves its output unchanged. The Swift copied header must stay
in sync with the C repository header.

Swift SPI: `Memory.installProcessAllocationCap(bytes:)` and
`Memory.processAllocationCapSnapshot()`. Snapshot fields distinguish heap, live,
cached, pending, unreclaimed, charged, peak, and installed limit. A snapshot neither
synchronizes GPU work nor establishes cleanup/peer retirement proof.

## Validation gates

Run `sh tools/check-process-allocation-cap.sh` for isolated CPU checks (serial compilation).
Core ledger coverage uses real VM mappings and copied Blocks ABI callbacks without
creating a Metal device. C adapter coverage links the actual adapter against a
CPU ledger. Metal and CPU backend sources are syntax checked; Swift is parsed.

Before integration, require full Swift/C/core compilation and isolated Metal tests:

- Install before model buffers; reject a second install and late install.
- Keep an extra Metal reference across allocator release and cache clearing:
  credit stays charged until the final reference/deallocator completes.
- Keep actual submitted GPU work and native array roots alive, then synchronize
  and retire them; verify correct output and callback-delayed credit recovery.
- Exercise repeated differently sized allocation churn, fixed-heap allocation,
  construction failure, external wrappers, and exact-cap refusal without fallback.
- Run selected-stage tiny GLM load/eval/retirement and then parent-authorized real
  stage calibration under independent host/headroom limits.

Canonical worker binding is intentionally unchanged. It must use the compiled
SPI, authenticate the authorized native limit, verify the installed snapshot,
and retain its separate host/headroom and process-exit-only completion contract.
No model-serving qualification follows from the standalone ledger tests.
