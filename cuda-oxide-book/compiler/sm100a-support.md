# SM100 and SM100a (Blackwell) Target Architecture Support

This document outlines the support for the Blackwell architecture (`sm_100` / `sm_100a` / compute capability 10.0/10.0a, e.g., B100/B200-class hardware) in the `cuda-oxide` compiler and runtime.

## 1. Scope & Compilation Baseline

*   **SM100 vs SM100a**: Blackwell introduces a split target architecture model similar to Hopper:
    *   `sm_100` is the forward-compatible virtual architecture target. It supports Tensor Memory Accelerator (TMA) bulk copies.
    *   `sm_100a` is the architecture-specific target that enables tcgen05/TMEM and TMA Multicast. These are Blackwell-only and **not forward-compatible** to older hardware architectures.
*   **Run Auto-detection & Target Selection**: `cargo oxide run` automatically detects host capability. If a kernel uses Blackwell tcgen05 or TMA multicast features, target selection resolves to `sm_100a`. If TMA features are detected, target selection defaults to `sm_100`.

## 2. Advanced Feature Detection & Gating

The compiler enforces strict target resolution checks via `check_target_compat` to ensure that Blackwell-specific features are only targeted when the appropriate target architecture is selected.

| Feature / Detector | Required target | Forward-Compatible? | Description |
|--------------------|-----------------|---------------------|-------------|
| **tcgen05 / TMEM** (`contains_blackwell_features` / `contains_tcgen05_alloc` / `contains_tcgen05_mma` / `contains_tcgen05_fence`) | `sm_100a` | **No** | Tensor Core Gen 5 (TMEM allocation, MMA, sync primitives). Utilizes Tensor Memory (TMEM) instead of registers, with a single-thread execution model. |
| **TMA Multicast** (`contains_tma_multicast`) | `sm_100a` | **No** | Architecture-specific extension allowing TMA bulk copy broadcast to all CTAs in a cluster via the `use_cta_mask` parameter. |

## 3. Negative Gating Examples

If a target constraint is violated (for instance, compiling a kernel containing tcgen05 instructions with `CUDA_OXIDE_TARGET=sm_75` or targeting a Blackwell-only feature with `sm_90`), the gate in `check_target_compat` aborts compilation with a descriptive error.

### Verification of Blackwell Gate
Forcing `sm_75` on a kernel using tcgen05 features:
```bash
cargo oxide run tcgen05_kernel --arch sm_75
```
*(Expected compilation failure: `Architecture sm_75 does not support detected advanced features: Blackwell`)*

Forcing `sm_90` on tcgen05 features:
```bash
cargo oxide run tcgen05_kernel --arch sm_90
```
*(Expected compilation failure: `Architecture sm_90 does not support detected advanced features: Blackwell`)*

Forcing `sm_100` (non-a) on TMA Multicast features:
```bash
cargo oxide run tma_multicast_kernel --arch sm_100
```
*(Expected compilation failure: `Architecture sm_100 does not support detected advanced features: TmaMulticast`)*

## 4. Cross-References
*   [Tensor Memory Accelerator](../advanced/tensor-memory-accelerator.md)
