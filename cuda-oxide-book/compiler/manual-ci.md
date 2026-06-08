# Manual CI: Hardware Verification

This page documents the **hardware-side** of the SM75 baseline CI
split. The CI workflows under `.github/workflows/` cover the **PTX-only**
side (build + gate check, no GPU needed). The hardware side needs a
real sm_75 GPU on the developer's machine; this file is the
reproducible recipe.

## Why this exists

`sm75-baseline.yml` runs the gate on a stock `ubuntu-latest` GitHub
runner: it builds `vecadd` to PTX (must succeed) and `tma_copy` with
`--arch sm_75` (must fail with the exact doc-advertised error). It
never needs a GPU.

The hardware side — actually launching a kernel on Turing — does.
The RTX 2070 (CC 7.5) on the maintainer's local box is the canonical
target. This doc is what someone with sm_75 hardware runs to verify
the full path end-to-end.

## Pre-flight

```bash
# 1. Doctor must be green. The libdevice path fix in this release
#    adds /usr/lib/cuda and /usr/lib/nvidia-cuda-toolkit to the
#    search list, so a stock nvidia-cuda-toolkit install on Ubuntu
#    22.04+ passes without an env-var override.
cargo oxide doctor
```

Expected output ends with:
```
✅ Environment looks good!
```

## PTX smoke (no GPU needed)

This is the same path the CI runs. Use it as a quick local check
before going to the hardware step.

```bash
# vecadd must build cleanly to PTX targeting sm_75.
cargo oxide build vecadd
# Inspect the target arch line:
grep -E '^\.version|^\.target' \
  crates/rustc-codegen-cuda/examples/vecadd/vecadd.ptx
# Expected: .version 6.3, .target sm_75

# tma_copy with --arch sm_75 must FAIL with the doc-advertised error.
cargo oxide build tma_copy --arch sm_75 2>&1 | grep -F \
  "Architecture sm_75 does not support detected advanced features: Tma"
# Expected: the line above must appear, and the build must exit non-zero.
```

## Hardware run (sm_75 GPU required)

The full kernel-launch + verification path. Run on a machine with an
RTX 2070 (or any CC 7.5 GPU):

```bash
# Build, link, and run. The auto-detect path reads
# cuda_core::CudaContext::compute_capability() and sets
# CUDA_OXIDE_TARGET=sm_75 dynamically, so no override is needed
# on Turing hardware.
cargo oxide run vecadd
```

Expected behaviour:
- The build step emits `vecadd.ptx` (same as the PTX smoke above).
- The runtime loads the PTX via libNVVM + nvJitLink.
- The kernel launches and runs the vecadd computation.
- The host process exits 0 with a "vecadd: 0 errors" line (or
  whatever the example's success indicator is — see
  `crates/rustc-codegen-cuda/examples/vecadd/`).

## Captured baselines

The PTX-smoke baselines captured in this release are stored in
`.remember/` at the repo root (memory only, not in CI):

- `vecadd-ptx-baseline.txt` — first 10 lines of `vecadd.ptx` and
  build metadata (target arch, timestamp).
- `tma_copy-sm75-rejection.txt` — the exact error string the
  tma_copy gate produces, captured 2026-06-07.

These files are diagnostic only. The CI workflow
`.github/workflows/sm75-baseline.yml` re-asserts the same checks on
every push and PR.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `cargo oxide doctor` reports `libdevice.10.bc not found` | Stale `cuda_oxide` install from before the libdevice path fix | Reinstall or set `CUDA_OXIDE_LIBDEVICE=<path>` |
| `tma_copy` builds with `--arch sm_75` | Stale `cuda_oxide` from before the gate landed (cp.async / bar.warp.sync were un-gated in v0.1) | Upgrade to v0.2.0 |
| `cargo oxide run vecadd` errors on RTX 2070 | Host CC != 7.5 (older Turing) or driver mismatch | Confirm `nvidia-smi` shows the 2070 at CC 7.5; reinstall NVIDIA driver if not |
