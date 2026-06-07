# SM75 (Turing) Target Architecture Support

This document outlines the support for Turing architecture (`sm_75` / compute capability 7.5, e.g., RTX 2070-class hardware) in the `cuda-oxide` compiler and runtime.

## 1. Scope & Compilation Baseline

*   **Compilation Default**: Basic kernels compiled without target overrides default to `sm_75`. Because `sm_75` is the oldest supported virtual architecture in this toolchain, compiling with `sm_75` by default ensures **maximum JIT compatibility** across all subsequent architectures (Ampere `sm_80`, Ada `sm_89`, Hopper `sm_90`, Blackwell `sm_100`).
*   **Run Auto-detection**: `cargo oxide run` automatically detects if the host GPU is `sm_75` (compute capability 7.5) and sets the `CUDA_OXIDE_TARGET` variable dynamically to match.

## 2. Supported Baseline Features

The following CUDA core features are fully supported and validated on `sm_75`:

*   **Execution Hierarchy**: Grid, thread-block, and thread indexing (`threadIdx`, `blockIdx`, etc.).
*   **Memory Hierarchy**: Normal global memory, constant memory, and block-shared memory (static and dynamic).
*   **Synchronization**: Ordinary block-local barriers (`sync_threads`).
*   **Warp-Synchronous Operations**: Warp shuffle and vote instructions (e.g., `shuffle_xor_f32`, `shuffle_down_f32`, `ballot`).
*   **Atomics**: Floating-point and integer atomic operations.
*   **Debugging**: Console printf (`gpu_printf`) and assertion macros (`gpu_assert`).
*   **Safe Slicing**: Out-of-bounds safe indexing via `DisjointSlice`.

## 3. Negative Gating & Gated Features

To prevent compilation leaks and JIT loading crashes on Turing hardware, the compiler enforces **strict negative feature gates** when the target architecture is `sm_75` or `compute_75`. If the intermediate LLVM IR contains any advanced instructions belonging to newer architectures, compilation is aborted:

*   **TMA (Tensor Memory Accelerator)**: Gated on SM90+. Rejects `llvm.nvvm.cp.async.bulk.tensor` operations.
*   **WGMMA (Warpgroup MMA)**: Gated on SM90+. Rejects `llvm.nvvm.wgmma` operations.
*   **tcgen05 / TMEM**: Gated on SM100+. Rejects `tcgen05` operations.
*   **Thread Block Clusters**: Gated on SM90+. Rejects cluster registers or cluster-launch attributes.

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
