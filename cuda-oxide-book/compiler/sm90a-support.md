# SM90 and SM90a (Hopper) Target Architecture Support

This document outlines the support for the Hopper architecture (`sm_90` / `sm_90a` / compute capability 9.0/9.0a, e.g., H100/H200-class hardware) in the `cuda-oxide` compiler and runtime.

## 1. Scope & Compilation Baseline

*   **SM90 vs SM90a**: Hopper introduces a split target architecture model:
    *   `sm_90` is the forward-compatible virtual architecture target. Features like Thread Block Clusters require `sm_90` or newer but remain compatible with subsequent architectures (e.g. Blackwell `sm_100`).
    *   `sm_90a` is the architecture-specific target that enables Warpgroup Matrix Multiply-Accumulate (WGMMA). WGMMA instructions are Hopper-only and **not forward-compatible** to newer hardware architectures like Blackwell (`sm_100a`).
*   **Run Auto-detection & Target Selection**: `cargo oxide run` automatically detects host capability. If a kernel uses WGMMA features, target selection resolves to `sm_90a`. If a kernel uses Cluster features but not WGMMA, it resolves to `sm_90`. If TMA features are detected, target selection defaults to `sm_100` (for Blackwell compatibility) unless overridden by setting `CUDA_OXIDE_TARGET=sm_90a`.

## 2. Advanced Feature Detection & Gating

The compiler enforces strict target resolution checks via `check_target_compat` to ensure that specific features are only targeted when the appropriate target architecture is selected.

| Feature / Detector | Required target | Forward-Compatible? | Description |
|--------------------|-----------------|---------------------|-------------|
| **Thread Block Clusters** (`contains_cluster_features` / `contains_cluster_sync`) | `sm_90`+ | **Yes** | Support for cooperative thread block groups within a cluster, cluster special registers, distributed shared memory, and cluster synchronization. |
| **TMA / mbarrier** (`contains_tma_features`) | `sm_100` (or `sm_90a` override) | **Yes** | Tensor Memory Accelerator bulk copies, transaction-tracking hardware asynchronous barriers (`mbarrier`), and proxy fences. |
| **WGMMA** (`contains_wgmma_features` / `contains_wgmma_mma_async` / `contains_wgmma_fence`) | `sm_90a` | **No** | Warpgroup Matrix Multiply-Accumulate instructions (128-thread). Gated strictly to `sm_90a`; will fail to compile if targeted to generic `sm_90` or Blackwell. |

## 3. Negative Gating Examples

If a target constraint is violated (for instance, compiling a kernel containing WGMMA instructions with `CUDA_OXIDE_TARGET=sm_75` or targeting a Blackwell-only feature with `sm_90`), the gate in `check_target_compat` aborts compilation with a descriptive error.

### Verification of WGMMA Gate
Forcing `sm_75` on a kernel using WGMMA features:
```bash
cargo oxide run wgmma --arch sm_75
```
*(Expected compilation failure: `Architecture sm_75 does not support detected advanced features: Wgmma`)*

Forcing generic `sm_90` on WGMMA code:
```bash
cargo oxide run wgmma --arch sm_90
```
*(Expected compilation failure: `Architecture sm_90 does not support detected advanced features: Wgmma`)*

## 4. Cross-References
*   [Matrix Multiply Accelerators](../advanced/matrix-multiply-accelerators.md)
*   [Cluster Programming](../advanced/cluster-programming.md)
