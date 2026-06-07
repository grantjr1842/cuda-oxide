# SM75 (Turing) Target Architecture Support

This document outlines the support for Turing architecture (`sm_75` / compute capability 7.5, e.g., RTX 2070-class hardware) in the `cuda-oxide` compiler and runtime.

## 1. Scope & Compilation Baseline

*   **Compilation Default**: Basic kernels compiled without target overrides default to `sm_75`. Because `sm_75` is the oldest supported virtual architecture in this toolchain, compiling with `sm_75` by default ensures **maximum JIT compatibility** across all subsequent architectures (Ampere `sm_80`, Ada `sm_89`, Hopper `sm_90`, Blackwell `sm_100`).
*   **Run Auto-detection**: `cargo oxide run` automatically detects if the host GPU is `sm_75` (compute capability 7.5) and sets the `CUDA_OXIDE_TARGET` variable dynamically to match. Implemented in `crates/cargo-oxide/src/commands.rs::detect_run_target_arch`, which reads `cuda_core::CudaContext::compute_capability()` for device 0. Falls through to the backend's `select_target` default when no GPU is available (e.g. CI runners).
*   **`cargo oxide doctor` SM75 check**: prints the host's compute capability and compares it against the sm_75 baseline. Reports `✓` on matching CC, `✓` on newer CCs (auto-detect will bump the target), and `✗` on older CCs. When no GPU is reachable (CI, headless boxes) it falls back to a `-` informational line — the backend's sm_75 default still applies.

## 2. Supported Baseline Features

The following CUDA core features are fully supported and validated on `sm_75`:

*   **Execution Hierarchy**: Grid, thread-block, and thread indexing (`threadIdx`, `blockIdx`, etc.).
*   **Memory Hierarchy**: Normal global memory, constant memory, and block-shared memory (static and dynamic).
*   **Synchronization**: Ordinary block-local barriers (`sync_threads`).
*   **Warp-Synchronous Operations**: Warp shuffle and vote instructions (e.g., `shuffle_xor_f32`, `shuffle_down_f32`, `ballot`).
*   **Atomics**: Floating-point and integer atomic operations.
*   **Debugging**: Console printf (`gpu_printf`) and assertion macros (`gpu_assert`).
*   **Safe Slicing**: Out-of-bounds safe indexing via `DisjointSlice`.

### 2.1. What's *not* in the SM75 baseline

The following CUDA core features are **explicitly absent on Turing**. The first column is the minimum architecture that supports the feature; the second column is whether the SM75 gate actually catches it today.

| Feature                         | Min arch | Gated by `check_target_compat`? | Notes |
|---------------------------------|----------|---------------------------------|-------|
| Async copy (`cp.async`)         | sm_80    | **No** — see note               | The `contains_*` detectors currently only look for `cp.async.bulk.tensor` (TMA, SM90+). A kernel that emits plain `cp.async` will compile silently to `sm_75` and JIT-fail at load time on Turing. **Workaround today:** compile to `sm_80` or higher when using `cp.async`. Tracked as a follow-up to add a `cp.async`-only detector. |
| Warp-aggregated barrier (`bar.sync` with named barrier counts) | sm_80 | **No** | Same status as `cp.async` — not detected today. |
| `cuda::pipeline` / HW pipeline barriers | sm_80 | **No** | Built on `cp.async`; unavailable on Turing hardware but not gated. |
| 64-bit atomics on global memory | sm_60    | No (falls under `Basic`)        | Available on Turing; no separate gate needed. |
| Distributed shared memory (`mapa.shared::cluster`) | sm_90 | **Yes** | Caught by `contains_cluster_features`. |
| TMA, WGMMA, tcgen05             | sm_90+   | **Yes**                          | See §3 — gated by the negative-feature mechanism. |
| `bar.warp.sync` (warp-specialisation barrier) | sm_80 | **No** | Ampere hardware feature; not detected today. |

**If a kernel compiles cleanly to `sm_75` (auto-detected `Basic`), it is guaranteed Turing-clean for the features listed with a "Yes" in the gated column.** Features marked "No" require manual target selection (e.g. `--arch sm_80`) to ensure correctness — relying on auto-detect for those is unsafe until the detector set is expanded.

## 3. Negative Gating & Gated Features

To prevent compilation leaks and JIT loading crashes on Turing hardware, the compiler enforces **strict negative feature gates** when the target architecture is `sm_75` or `compute_75`. The gate runs in two stages:

1.  **IR scan** — before target selection, the emitted `.ll` is scanned for forbidden intrinsic families by the `contains_*` helpers in `crates/mir-importer/src/pipeline/target_resolution.rs`. The result is collapsed into a `DetectedFeatures` value. (The submodule is co-located under `pipeline/` rather than at the conventional `src/target_resolution.rs` path because of a module-resolution interaction with this crate's `extern crate` block — see the comment in `pipeline.rs` for the workaround.)
2.  **Capability check** — if the resolved target is `sm_75` / `compute_75` and `detected != DetectedFeatures::Basic`, compilation is aborted with a human-readable error naming both the target and the offending feature.

Gated feature families (all SM90+ unless noted):

*   **TMA (Tensor Memory Accelerator)** (SM90+): Rejects `cp.async.bulk.tensor` and `mbarrier.*` / `fence.proxy.async` patterns.
*   **TMA Multicast** (SM100a): Rejects the `use_cta_mask` form of `cp.async.bulk.tensor.g2s.tile`.
*   **WGMMA (Warpgroup MMA)** (SM90a): Rejects `wgmma.fence` / `wgmma.commit_group` / `wgmma.wait_group` / `wgmma.mma_async`.
*   **tcgen05 / TMEM** (SM100a): Rejects `tcgen05.alloc` / `tcgen05.mma` / `tcgen05.cp` / etc.
*   **Thread Block Clusters** (SM90+): Rejects `cluster_ctaid` / `cluster.sync` / `mapa.shared::cluster`.

The `CUDA_OXIDE_TARGET=sm_75` override is the primary lever for users who want to force the gate and prove their kernel is Turing-clean.

## 4. Verification

Compile and run the baseline smoke example (`vecadd`):
```bash
cargo oxide run vecadd
```

Verify that advanced features fail to compile when forced targeting `sm_75`:
```bash
cargo oxide run tma_copy --arch sm_75
```
*(Expected compilation failure: `Architecture sm_75 does not support detected advanced features: Tma`)*
