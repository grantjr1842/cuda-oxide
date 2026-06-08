# SM75 (Turing) Baseline & Gating Showcase

This example demonstrates the support for Turing architecture (`sm_75` / compute capability 7.5, e.g., RTX 2070-class hardware) baseline features in `cuda-oxide`, as well as the **negative feature gates** introduced in v0.2.0.

## Showcase Features

The kernel in `src/main.rs` utilizes only **Turing-clean** features that are fully supported by the default `sm_75` compilation baseline:
1. **Grid & Thread Indexing**: Linear and 3D index querying.
2. **Warp Shuffles**: Butterfly/down reduction using `warp::shuffle_down_f32`.
3. **Block-Shared Memory**: Thread-block cooperative sharing via static `SharedArray`.
4. **Ordinary Barriers**: Block-wide synchronization via `sync_threads()`.
5. **Floating-point Atomics**: Atomic block accumulation via `DeviceAtomicF32`.
6. **Console Printing**: Kernel-level debugging using the `gpu_printf!` macro.

---

## 1. Run the Baseline Showcase

To build and run the Turing baseline showcase on your local machine:
```bash
cargo oxide run sm75_showcase
```

### Expected Output
1. The compiler will auto-detect the host capability (e.g. `CC 7.5`).
2. The compiler targets `sm_75` by default (broadest JIT compatibility).
3. The kernel runs successfully and outputs:
```text
=== SM75 Baseline Showcase (Turing baseline support) ===
Detected host GPU compute capability: 7.5
Launching reduction_kernel on 1024 elements...
Hello from thread 0 inside the reduction_kernel!

Verification:
  Block sums: [128.0, 128.0, 128.0, 128.0, 128.0, 128.0, 128.0, 128.0]
  Global sum: 1024 (Expected: 1024)
✓ Showcase successfully completed!
```

---

## 2. Test the Negative Feature Gates

To prevent compilation leaks and JIT crashes on Turing hardware, the compiler enforces **strict negative feature gates** when targeting `sm_75`.

You can test this by forcing the compiler to compile the kernel with advanced (Ampere+) features enabled:

### A. Trigger AmpereAsync gate (Warp Barrier / cp.async)
By passing `--features trigger_gate_failure`, the code path containing `fence_proxy_async_shared_cta` (which uses `cp.async` and `bar.warp.sync` under the hood) is compiled:
```bash
cargo oxide run sm75_showcase --arch sm_75 --features trigger_gate_failure
```

**Expected Compilation Failure:**
```text
Error: Architecture sm_75 does not support detected advanced features: AmpereAsync
```

### B. Trigger TMA/mbarrier gate (Hopper+ Features)
Try running the Hopper Tensor Memory Accelerator example with the forced `sm_75` target:
```bash
cargo oxide run tma_copy --arch sm_75
```

**Expected Compilation Failure:**
```text
Error: Architecture sm_75 does not support detected advanced features: Tma
```

### C. Trigger WGMMA gate (Hopper-only Features)
Try running the Warpgroup MMA example with the forced `sm_75` target:
```bash
cargo oxide run wgmma --arch sm_75
```

**Expected Compilation Failure:**
```text
Error: Architecture sm_75 does not support detected advanced features: Wgmma
```

---

## 3. Verify Environment Readiness

Verify that the local environment matches the Turing baseline:
```bash
cargo oxide doctor
```
Look for the check mark next to:
`✓ SM75 baseline readiness... host CC 7.5 matches the sm_75 default target`
