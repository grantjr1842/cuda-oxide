# cuda-oxide v0.2.0 — Release Notes

> **Status:** Tag pending user push (requires GitHub auth).
> All workspace crates are at version 0.2.0 already; the
> `git tag -a v0.2.0` invocation is the user's call.
>
> **Hardware baseline:** v0.2.0 ships the SM75 (Turing, RTX 2070-class)
> target support that was mid-flight on `feat/sm75-baseline-support`.
> The branch is the v0.2.0 release branch; mainline merges that
> land after v0.2.0 cut will become 0.2.1+.

## Breaking changes

These are the user-visible behavior changes for anyone upgrading
from a v0.1.x install. They all show up as **new** errors on sm_75
kernels that compiled silently in v0.1.x — the gate is now honest.

### `cp.async` (non-bulk) is now sm_80+ gated

In v0.1.x, kernels using the non-bulk form of `cp.async` (and
its pipeline-control pair `cp.async.commit_group` /
`cp.async.wait_group`) compiled to sm_75 by default and then
JIT-failed at load time on Turing hardware. v0.2.0 closes that
leak with the `contains_ampere_async_features` detector; on
`CUDA_OXIDE_TARGET=sm_75` (or auto-detected sm_75) these
kernels now fail compilation with:

```
Architecture sm_75 does not support detected advanced features: AmpereAsync
```

**Migration:** upgrade the auto-detected target. The doc-honesty
test `test_cp_async_gate_produces_doc_advertised_error` pins the
exact string format, so any future error-format drift is caught
at `cargo test -p mir-importer` time.

### `bar.warp.sync` is now sm_80+ gated

The sub-warp barrier backing `CoalescedThreads::sync` and
`WarpTile<N>::sync` in `cuda-device` was un-gated in v0.1.x.
v0.2.0 closes the gap with the same detector. The user-facing
error string is identical to the `cp.async` case (both fold
into the `AmpereAsync` variant). Migration is the same:
upgrade the target.

### Named-barrier `bar.sync` is now sm_80+ gated

This was an unannounced gap in v0.1.x that the v0.2.0 detector
surface analysis surfaced. The named-barrier form
(`bar.sync N, !"%named-barrier-N"`, where N is a non-zero
barrier index) is an Ampere addition used to back warp-aggregated
barriers. v0.2.0 adds the `contains_named_barrier_bar_sync`
helper, folded into the same `AmpereAsync` variant. Same
user-facing error string as above; the detector is a separate
helper (so it's testable in isolation) but the variant is
intentionally shared for ergonomic reasons.

**Migration:** same — upgrade the target. The sm_75-legal
`bar.sync 0` form (the block-wide barrier backing
`sync_threads` / `barrier`) is explicitly *not* detected by
the new helper; a kernel that uses only `bar.sync 0` continues
to compile to sm_75 cleanly. The negative test
`test_contains_named_barrier_bar_sync_ignores_bar_sync_zero`
pins that selectivity.

## New

### SM75 (Turing) is the default target

`cargo oxide build` and `cargo oxide run` now default to `sm_75`
(the oldest supported virtual architecture in the toolchain),
maximising JIT compatibility across Turing → Ampere → Ada →
Hopper → Blackwell. `cargo oxide doctor` reports the host's
compute capability and the sm_75 match.

### `cargo oxide doctor` checks the SM75 baseline

The doctor now has a dedicated `SM75 baseline readiness` line
that prints the host's CC and reports a `✓` on matching CC,
`✓` on newer CCs (auto-detect will bump the target), and `✗`
on older CCs. The doctor also probes the libdevice path on
Debian/Ubuntu package installs (the path fix in this release
adds `/usr/lib/cuda` and `/usr/lib/nvidia-cuda-toolkit` to the
search list — see the fix under "Bugs fixed" below).

### `sm75-baseline.yml` CI workflow

A new GitHub Actions workflow (`.github/workflows/sm75-baseline.yml`)
runs the end-to-end SM75 gate on every push and PR. It builds
`vecadd` to PTX (must succeed with `.target sm_75`) and builds
`tma_copy` with `--arch sm_75` (must fail with the exact
`Tma` error string). No GPU required — runs on a stock
`ubuntu-latest` runner with CUDA 13.1.1 installed.

### Named-barrier `bar.sync` detector

See "Breaking changes" above. The helper is
`contains_named_barrier_bar_sync` in
`crates/mir-importer/src/pipeline/target_resolution.rs`.

### Doc-honesty tests for the §3 error string

The gate's user-facing error string format
(`Architecture <target> does not support detected advanced
features: <Variant>`) is now pinned by unit tests for *every*
detector in the §3 family, not just TMA. Three new tests:
- `test_cp_async_gate_produces_doc_advertised_error`
- `test_bar_warp_sync_gate_produces_doc_advertised_error`
- `test_named_barrier_bar_sync_gate_produces_doc_advertised_error`

A drift in the gate's error format or variant label is caught
at `cargo test -p mir-importer` time, before the doc and the
code can disagree.

### §2.1 detector table reconciled

The §2.1 row in `cuda-oxide-book/compiler/sm75-support.md` that
previously conflated sm_90+ TMA, sm_90a WGMMA, and sm_100a
tcgen05/TMA Multicast is now split per-feature to match §3's
per-feature bullets. The doc-honesty test for TMA
(`test_sm75_gate_doc_verification_tma_copy_uses_plain_tma`) is
unchanged; the §2.1 row label is now consistent with §3.

## Verification

- `cargo test --workspace` is green. Test counts after this
  release (Phase A baseline = 348, Phase C added 12):
  - mir-importer:    33 →  37 (+4: 3 from N3 detector, +1 from X7)
  - mir-lower:       27 →  35 (+8: X1 closes 5 TODOs)
  - dialect-mir:      0 →   2 (+2: X3 dialect unit tests)
  - dialect-nvvm:     0 →   2 (+2: X3 dialect unit tests)
  - **Total: 348 → 360** (+12)
- `cargo oxide build vecadd` produces a PTX targeting
  `.target sm_75` (verified on the maintainer's RTX 2070).
- `cargo oxide build tma_copy --arch sm_75` fails with
  `Architecture sm_75 does not support detected advanced features: Tma`
  (the exact string the CI workflow asserts).
- `cargo oxide doctor` is green from a stock
  `nvidia-cuda-toolkit` install on Ubuntu 22.04+ (the libdevice
  path fix from this release adds the Debian package paths).

## Bugs fixed

- `find_libdevice` in `crates/cuda-host/src/ltoir.rs` only
  probed `/usr/local/cuda` and `/opt/cuda` (the NVIDIA runfile
  and Bazel-cache layouts). The Debian/Ubuntu
  `nvidia-cuda-toolkit` package splits the toolkit across
  `/usr/lib/cuda` (nvvm) and `/usr/lib/nvidia-cuda-toolkit`
  (libdevice), so a default install on Ubuntu 22.04+ hit
  the "Could not locate libdevice.10.bc" branch and needed
  `CUDA_OXIDE_LIBDEVICE` set. Fixed by adding both Debian
  package paths to the search list; the NVIDIA runfile path
  still matches first for users with the official toolkit.

## Out of scope for v0.2.0

These are v0.3.0+ candidates; some are explicitly called out
in `ROADMAP.md`'s Next / Later stages:

- N1 (push pending PR + first CI run on remote fork) — user action
  required; the prerequisite work is all in this release.
- X2 fuzzer CI — corpus strategy decision is a separate human
  conversation.
- L1 pliron upstream rev lockstep — requires pliron-upstream
  coordination; the L1 work is a rev-bump procedure, not a code change.
- L2 async-mlp e2e in CI — needs sm_80+ hardware; RTX 2070 is
  sm_75. The L2 work is an `async-mlp-baseline.yml` analogous
  to `sm75-baseline.yml`.
- L3a / L3b sm_90a / sm_100a detector parity — L-stage work, the
  sm_75 + sm_80 gates are the explicit focus of this branch.
- L4 `rustc-codegen-cuda` 1.0 conversation — depends on OKR1 KRs
  landing; v0.2.0 is the natural conversation starter.

## Thanks

This release landed detector work, doc work, test work, and a
doctor bug fix in 5 commits on top of the v0.2.0 base. The
detector + doc-honesty pattern (separate helper, fold into
existing variant, pin the user-facing error string) is now
the template for any future detector addition — apply it
identically to L3a (sm_90a) and L3b (sm_100a) when those
parity workhorses land.
