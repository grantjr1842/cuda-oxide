/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU target resolution: feature detection in emitted LLVM IR, target
//! selection, and the SM75 capability gate.
//!
//! The compilation pipeline must answer three questions before `llc` is
//! invoked:
//!
//! 1. **What features does the IR actually use?** The `contains_*` helpers
//!    scan the textual `.ll` for forbidden intrinsic families. The result
//!    collapses into a [`DetectedFeatures`].
//! 2. **What `sm_XX` should we target?** [`select_target`] maps the detected
//!    features to the minimum architecture that supports them.
//! 3. **Is the resolved target compatible with the features?** [`check_target_compat`]
//!    fires the negative gate when the user pinned an older target than the
//!    IR actually requires (e.g. `CUDA_OXIDE_TARGET=sm_75` against a WGMMA
//!    kernel).
//!
//! Keeping these three concerns in one module makes the gate's contract
//! auditable end-to-end: any test that exercises a `contains_*` helper
//! exercises the detector that feeds the gate.

use std::path::Path;

/// GPU features detected in LLVM IR that determine target selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DetectedFeatures {
    /// tcgen05/TMEM (Blackwell datacenter, sm_100a).
    Blackwell,
    /// TMA multicast (arch-specific extension, sm_100a).
    TmaMulticast,
    /// WGMMA (Hopper only, sm_90a - NOT forward-compatible).
    Wgmma,
    /// TMA/mbarrier (Hopper+ compatible).
    Tma,
    /// Thread Block Clusters (sm_90+, forward-compatible).
    Cluster,
    /// Ampere async-copy + warp barriers (sm_80+). Plain `cp.async`,
    /// `cp.async.commit_group`, `cp.async.wait_group`, and `bar.warp.sync`
    /// are all Ampere features. The bulk forms (`cp.async.bulk.*`) are TMA
    /// (sm_90+) and detected separately by `contains_tma_features`.
    AmpereAsync,
    /// No special features (maximum compatibility, sm_80).
    Basic,
}

/// Checks for WGMMA instructions (Hopper sm_90a only, NOT forward-compatible).
///
/// WGMMA (Warpgroup Matrix Multiply-Accumulate) requires sm_90a specifically.
/// These are NOT forward-compatible - only work on H100/H200.
fn contains_wgmma_features(ll_path: &Path) -> bool {
    if let Ok(contents) = std::fs::read_to_string(ll_path) {
        contents.contains("wgmma.fence")
            || contents.contains("wgmma.commit_group")
            || contents.contains("wgmma.wait_group")
            || contents.contains("wgmma.mma_async")
    } else {
        false
    }
}

/// Checks for Thread Block Cluster instructions (sm_90+).
///
/// Cluster features require Hopper (sm_90) or newer:
/// - Cluster special registers (%cluster_ctaid, %cluster_nctaid)
/// - Cluster synchronization (cluster.sync)
/// - Distributed shared memory (mapa.shared::cluster)
fn contains_cluster_features(ll_path: &Path) -> bool {
    if let Ok(contents) = std::fs::read_to_string(ll_path) {
        // Cluster special registers
        contents.contains("cluster_ctaid")
            || contents.contains("cluster_nctaid")
            // Cluster synchronization
            || contents.contains("cluster.sync")
            // Distributed shared memory
            || contents.contains("mapa.shared::cluster")
    } else {
        false
    }
}

/// Checks for TMA/mbarrier instructions (Hopper+ compatible with Blackwell).
///
/// These instructions work on BOTH Hopper and Blackwell:
/// - TMA: Tensor Memory Accelerator bulk copies
/// - mbarrier: Async hardware barriers with transaction tracking
///
/// Use sm_90 (not sm_90a) for forward compatibility with sm_120 (Blackwell).
fn contains_tma_features(ll_path: &Path) -> bool {
    if let Ok(contents) = std::fs::read_to_string(ll_path) {
        // TMA instructions
        contents.contains("cp.async.bulk.tensor")
            // mbarrier with transaction tracking (Hopper+)
            || contents.contains("mbarrier.arrive.expect_tx")
            || contents.contains("mbarrier.try_wait")
            // Proxy fence for async operations
            || contents.contains("fence.proxy.async")
    } else {
        false
    }
}

/// Checks for Blackwell tcgen05 instructions (sm_100a+).
///
/// These instructions require sm_100a/sm_120a (Blackwell) or newer:
/// - tcgen05: Tensor Core Gen 5 (TMEM allocation, MMA, sync primitives)
///
/// Key differences from Hopper:
/// - tcgen05 MMA is single-thread (vs WGMMA's 128 threads)
/// - Uses Tensor Memory (TMEM) instead of registers
/// - Different synchronization model (mbarrier-based)
fn contains_blackwell_features(ll_path: &Path) -> bool {
    if let Ok(contents) = std::fs::read_to_string(ll_path) {
        // tcgen05 TMEM allocation/deallocation
        contents.contains("tcgen05.alloc")
            || contents.contains("tcgen05.dealloc")
            || contents.contains("tcgen05.relinquish_alloc_permit")
            // tcgen05 synchronization
            || contents.contains("tcgen05.fence")
            || contents.contains("tcgen05.commit")
            // tcgen05 MMA instructions (ws and non-ws/cta_group forms)
            || contents.contains("tcgen05.mma")
            // tcgen05 data movement
            || contents.contains("tcgen05.cp")
    } else {
        false
    }
}

/// Checks for TMA multicast in LLVM IR (requires sm_100a).
///
/// TMA multicast (`cp.async.bulk.tensor...multicast::cluster`) is an
/// architecture-specific extension that broadcasts a tile to all CTAs in a
/// cluster. In the LLVM intrinsic, this is controlled by the `use_cta_mask`
/// parameter (second-to-last i1 argument) being set to true.
fn contains_tma_multicast(ll_path: &Path) -> bool {
    if let Ok(contents) = std::fs::read_to_string(ll_path) {
        contents
            .lines()
            .any(|line| line.contains("g2s.tile") && line.contains(", i1 1, i1"))
    } else {
        false
    }
}

/// Checks for Ampere async-copy and warp-specialised barrier instructions
/// (sm_80+).
///
/// These are the intrinsics that the SM75 gate had a *known gap* on until
/// this detector was added (see `cuda-oxide-book/compiler/sm75-support.md`
/// §2.1). Without this detector, a kernel using `cp.async` or
/// `bar.warp.sync` would compile silently to `sm_75` and JIT-fail at
/// load time on Turing hardware.
///
/// Detected substrings (all sm_80+):
///
/// - `cp.async` (without the `.bulk.tensor` suffix) — Ampere async-memcpy
///   engine. The bulk forms are TMA (sm_90+) and detected separately by
///   `contains_tma_features` / `contains_tma_multicast`.
/// - `cp.async.commit_group` / `cp.async.wait_group` — the non-bulk
///   pipeline-control pair for Ampere async copy.
/// - `bar.warp.sync` — sub-warp barrier primitive backing
///   `CoalescedThreads::sync` and `WarpTile<N>::sync` in `cuda-device`.
/// - `bar.sync N` (N != 0) with a named-barrier operand — the warp-
///   aggregated barrier form (Ampere `bar.sync` with a named-barrier
///   index). Detected by [`contains_named_barrier_bar_sync`] and folded
///   into the same `DetectedFeatures::AmpereAsync` variant because
///   both target sm_80+ and share the same user-facing error message
///   (see `cuda-oxide-book/compiler/sm75-support.md` §3, "Ampere
///   async-copy + warp barriers" family).
fn contains_ampere_async_features(ll_path: &Path) -> bool {
    if let Ok(contents) = std::fs::read_to_string(ll_path) {
        // Plain `cp.async` (non-bulk). Match the leading substring but
        // exclude the bulk forms to avoid double-counting with the TMA
        // detector.
        contents.contains("cp.async")
            && !contents.contains("cp.async.bulk.tensor")
            // Non-bulk pipeline-control primitives.
            || contents.contains("cp.async.commit_group")
            || contents.contains("cp.async.wait_group")
            // Warp-specialised barrier.
            || contents.contains("bar.warp.sync")
            // Named-barrier `bar.sync` (Ampere). `bar.sync 0` is the
            // sm_75-legal block-wide form, so this only fires for the
            // named variant (non-zero index or `"…"` name operand).
            || contains_named_barrier_bar_sync(ll_path)
    } else {
        false
    }
}

/// Checks for the Ampere named-barrier `bar.sync` form (sm_80+).
///
/// The sm_75-legal block-wide barrier is `bar.sync 0` (also known as
/// `llvm.nvvm.barrier0`); both compile cleanly to Turing. The *named*
/// form — `bar.sync N, !"name"` where `N` is a non-zero barrier index
/// tied to a `bar[name]`-style name operand — is an Ampere addition
/// (`sm_80+`) used to back warp-aggregated barriers for cooperative
/// groups. Letting it leak to `sm_75` produces a silent compile and
/// a JIT-load failure on Turing hardware.
///
/// This is a separate helper (folded into `contains_ampere_async_features`
/// rather than getting its own `DetectedFeatures` variant) so the
/// detection can be tested in isolation and the public `detect_features`
/// chain is unchanged. The detector's return is "is the named-barrier
/// form present?", which then collapses into `DetectedFeatures::AmpereAsync`
/// in the public chain.
///
/// The match is deliberately conservative: a file containing *only*
/// `bar.sync 0` does NOT match. A file containing `bar.sync` with a
/// non-zero barrier index (e.g. `bar.sync 1`) or a name-operand
/// string (e.g. `bar.sync ..., !"%named-barrier-1"`) does match.
fn contains_named_barrier_bar_sync(ll_path: &Path) -> bool {
    let Ok(contents) = std::fs::read_to_string(ll_path) else {
        return false;
    };
    // The named-barrier form is identified by either:
    //   (a) a non-zero barrier index in the IR (PTX: `bar.sync 1, "name"`),
    //   (b) the LLVM IR name-operand form, which the NVPTX backend lowers
    //       the name string to as a literal `%named-barrier-N` substring
    //       in the .ll text.
    //
    // `bar.sync 0` (the sm_75-legal block-wide form) does NOT match any
    // of the patterns below, which is the property the gate relies on
    // for the `Basic` detector to fall through correctly.
    //
    // The `%named-barrier` substring is itself sufficient evidence of the
    // named-barrier form (it is not a substring of any other CUDA
    // intrinsic), so no `bar.sync` guard is needed; the gate then
    // collapses the result into `DetectedFeatures::AmpereAsync` via
    // `contains_ampere_async_features`.
    contents.contains("bar.sync 1,")
        || contents.contains("bar.sync 2,")
        || contents.contains("bar.sync 3,")
        || contents.contains("bar.sync 4,")
        || contents.contains("bar.sync 5,")
        || contents.contains("bar.sync 6,")
        || contents.contains("bar.sync 7,")
        || contents.contains("bar.sync 8,")
        || contents.contains("bar.sync 9,")
        // LLVM IR name-operand form: NVPTX backend emits the PTX
        // `!"name"` string as a literal `%named-barrier-N` substring
        // in the .ll text.
        || contents.contains("named-barrier")
}

/// Maps detected features to GPU target architecture.
pub(crate) fn select_target(features: DetectedFeatures) -> &'static str {
    match features {
        DetectedFeatures::Blackwell => "sm_100a",
        DetectedFeatures::TmaMulticast => "sm_100a",
        DetectedFeatures::Wgmma => "sm_90a",
        // TMA needs PTX 8.0+ which requires sm_90a or sm_100+.
        // sm_90a is NOT forward-compatible to Blackwell, so use sm_100 which:
        // - Generates PTX 8.6 (supports all TMA features)
        // - Works on all Blackwell variants (sm_100, sm_120)
        // - Hopper users can override with CUDA_OXIDE_TARGET=sm_90a
        DetectedFeatures::Tma => "sm_100",
        // Cluster features require sm_90+ but are forward-compatible.
        // Use sm_90 for Hopper compatibility, works on Blackwell too.
        DetectedFeatures::Cluster => "sm_90",
        // Ampere async-copy + warp barriers require sm_80+. Use sm_80
        // for the broadest forward-compatibility — works on every arch
        // the toolchain supports, matching the sm_80 "minimum baseline"
        // we used to default to before the sm_75 broadening.
        DetectedFeatures::AmpereAsync => "sm_80",
        DetectedFeatures::Basic => "sm_75",
    }
}

/// Returns true if the target architecture is the Turing baseline.
///
/// Both `sm_75` (SASS) and `compute_75` (PTX virtual arch) qualify, since the
/// gate must fire regardless of which form the user passed via
/// `CUDA_OXIDE_TARGET` or the auto-detect path.
pub(crate) fn is_sm75_target(target: &str) -> bool {
    target == "sm_75" || target == "compute_75"
}

/// Validates that the resolved target architecture is compatible with the
/// features detected in the IR.
///
/// On Turing (`sm_75` / `compute_75`), only kernels with no advanced features
/// (`DetectedFeatures::Basic`) are allowed. The IR-scanning detectors in
/// `contains_*` populate `detected`, so this gate is grounded in actual
/// intrinsic presence — not symbol-name heuristics.
///
/// `sm_75a` is rejected with a helpful error: Turing has no
/// arch-specific extensions, so the `a` suffix is always a typo for `sm_75`
/// (or for a different arch's `a` form).
///
/// Returns `Ok(())` if the target/detected pair is compatible, or `Err(msg)`
/// with a human-readable reason otherwise.
pub(crate) fn check_target_compat(target: &str, detected: DetectedFeatures) -> Result<(), String> {
    if target == "sm_75a" || target == "compute_75a" {
        return Err(format!(
            "Architecture {target} is not a valid target: Turing (sm_75) has no \
             arch-specific extensions. Did you mean `sm_75`?"
        ));
    }
    if is_sm75_target(target) && detected != DetectedFeatures::Basic {
        return Err(format!(
            "Architecture {target} does not support detected advanced features: {detected:?}"
        ));
    }
    Ok(())
}

/// Run all `contains_*` detectors against `ll_path` and collapse into a
/// single `DetectedFeatures`. Order matters: most specific first.
pub(crate) fn detect_features(ll_path: &Path) -> DetectedFeatures {
    match (
        contains_blackwell_features(ll_path),
        contains_tma_multicast(ll_path),
        contains_wgmma_features(ll_path),
        contains_tma_features(ll_path),
        contains_cluster_features(ll_path),
        contains_ampere_async_features(ll_path),
    ) {
        (true, _, _, _, _, _) => DetectedFeatures::Blackwell,
        (_, true, _, _, _, _) => DetectedFeatures::TmaMulticast,
        (_, _, true, _, _, _) => DetectedFeatures::Wgmma,
        (_, _, _, true, _, _) => DetectedFeatures::Tma,
        (_, _, _, _, true, _) => DetectedFeatures::Cluster,
        (_, _, _, _, _, true) => DetectedFeatures::AmpereAsync,
        _ => DetectedFeatures::Basic,
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the target resolution pipeline. Organised (in order) into:
    //!
    //! - **detector tests**: per-intrinsic-family `contains_*` positive/negative.
    //! - **collapse tests**: `detect_features` priority and `Basic` fall-through.
    //! - **select_target tests**: the feature-to-arch mapping.
    //! - **gate tests**: the SM75 capability check, including the `sm_75a` typo guard.
    //! - **integration tests**: end-to-end chain (IR -> detect -> select -> gate),
    //!   including the doc-pin for `tma_copy --arch sm_75`.
    //!
    //! The grouping is comment-annotated rather than nested submodules
    //! because nested `mod` blocks would force the test names to be
    //! re-prefixed; the flat list is grep-friendly and matches the
    //! surrounding crate's style.

    use super::*;
    use std::{fs, path::PathBuf};

    fn write_temp_ll(name: &str, contents: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "cuda_oxide_target_resolution_{}_{}.ll",
            std::process::id(),
            name
        ));
        fs::write(&path, contents).expect("write temp LLVM IR");
        path
    }

    // =========================================================================
    // Detector tests: per-intrinsic-family contains_* positive/negative.
    // =========================================================================

    #[test]
    fn test_contains_wgmma_features_detects_intrinsic() {
        let path = write_temp_ll("wgmma", "call void @llvm.nvvm.wgmma.mma_async(...)");
        assert!(contains_wgmma_features(&path));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_contains_wgmma_features_ignores_unrelated() {
        let path = write_temp_ll("no_wgmma", "ret void");
        assert!(!contains_wgmma_features(&path));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_contains_tma_features_detects_intrinsic() {
        let path = write_temp_ll(
            "tma",
            "call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile(...)",
        );
        assert!(contains_tma_features(&path));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_contains_cluster_features_detects_intrinsic() {
        let path = write_temp_ll(
            "cluster",
            "%id = call i32 @llvm.nvvm.read.ptx.sreg.cluster_ctaid()",
        );
        assert!(contains_cluster_features(&path));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_contains_blackwell_features_detects_intrinsic() {
        let path = write_temp_ll("blackwell", "call void @llvm.nvvm.tcgen05.alloc(...)");
        assert!(contains_blackwell_features(&path));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_contains_tma_multicast_requires_cta_mask() {
        let multicast = write_temp_ll(
            "tma_multicast",
            "call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile(i32 0, i1 1, i1 false)",
        );
        let unicast = write_temp_ll(
            "tma_unicast",
            "call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile(i32 0, i1 0, i1 false)",
        );
        assert!(contains_tma_multicast(&multicast));
        assert!(!contains_tma_multicast(&unicast));
        let _ = fs::remove_file(multicast);
        let _ = fs::remove_file(unicast);
    }

    #[test]
    fn test_contains_ampere_async_detects_plain_cp_async() {
        // The non-bulk `cp.async` form is the Ampere async-memcpy engine.
        let path = write_temp_ll(
            "cp_async",
            "call void @llvm.nvvm.cp.async.cg.shared.global(...)",
        );
        assert!(
            contains_ampere_async_features(&path),
            "plain cp.async must be detected as Ampere async"
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_contains_ampere_async_detects_commit_wait_group() {
        let commit = write_temp_ll(
            "cp_async_commit",
            "call void @llvm.nvvm.cp.async.commit_group(...)",
        );
        let wait = write_temp_ll(
            "cp_async_wait",
            "call void @llvm.nvvm.cp.async.wait_group(i32 0)",
        );
        assert!(contains_ampere_async_features(&commit));
        assert!(contains_ampere_async_features(&wait));
        let _ = fs::remove_file(commit);
        let _ = fs::remove_file(wait);
    }

    #[test]
    fn test_contains_ampere_async_detects_bar_warp_sync() {
        let path = write_temp_ll(
            "bar_warp_sync",
            "call void @llvm.nvvm.bar.warp.sync(i32 -1)",
        );
        assert!(
            contains_ampere_async_features(&path),
            "bar.warp.sync must be detected as Ampere async"
        );
        let _ = fs::remove_file(path);
    }

    /// TMA bulk-form intrinsics must NOT be misclassified as AmpereAsync
    /// — the `cp.async.bulk.tensor` form belongs to TMA (sm_90+) and
    /// should be caught by `contains_tma_features` (not the Ampere
    /// detector). The priority chain in `detect_features` resolves the
    /// double-match, but the Ampere detector itself must stay clean.
    #[test]
    fn test_contains_ampere_async_does_not_match_bulk_forms() {
        let path = write_temp_ll(
            "tma_bulk",
            "call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile(i32 0, i1 0, i1 false)",
        );
        assert!(
            !contains_ampere_async_features(&path),
            "cp.async.bulk.tensor must not be classified as Ampere async"
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_contains_ampere_async_ignores_unrelated() {
        let path = write_temp_ll("plain", "ret void");
        assert!(!contains_ampere_async_features(&path));
        let _ = fs::remove_file(path);
    }

    // ----- N3: named-barrier `bar.sync` detector (Ampere sm_80+) ------------
    //
    // The named-barrier form is `bar.sync N, !"%named-barrier-N"` (PTX
    // `bar.sync N, "name"`), distinct from the sm_75-legal `bar.sync 0`
    // block-wide form. The detector fires only on the non-zero-index /
    // name-operand variants, so a kernel that uses the plain block-wide
    // barrier must still detect as `Basic` and compile cleanly to sm_75.
    //
    // See `cuda-oxide-book/compiler/sm75-support.md` §2.1 row 5 and §3
    // "Ampere async-copy + warp barriers" bullet for the design contract.

    #[test]
    fn test_contains_named_barrier_bar_sync_detects_non_zero_index() {
        // PTX literal: `bar.sync 1, "named_bar"`. The non-zero index
        // (the comma-separated name follows) is the named-barrier form.
        let path = write_temp_ll(
            "named_barrier_index",
            r#"call void asm sideeffect "bar.sync 1, \"\24named_bar\"", ""();"#,
        );
        assert!(
            contains_named_barrier_bar_sync(&path),
            "bar.sync 1, ... must be detected as the named-barrier form"
        );
        assert!(
            contains_ampere_async_features(&path),
            "named-barrier form must collapse into AmpereAsync via the ampere detector"
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_contains_named_barrier_bar_sync_detects_name_operand_substring() {
        // The NVPTX backend lowers the PTX name operand to a literal
        // `named-barrier-N` substring in the .ll text (the LLVM IR
        // symbol-name convention). The detector must fire on that
        // form even when the `bar.sync 1,` index prefix is absent
        // from the substring scan (e.g. a future backend lowering
        // that hoists the name to a metadata entry).
        let path = write_temp_ll(
            "named_barrier_substring",
            "  %named-barrier-3 = ... ; metadata carrier for the named barrier",
        );
        assert!(
            contains_named_barrier_bar_sync(&path),
            "the LLVM IR name-operand substring must be detected"
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_contains_named_barrier_bar_sync_ignores_bar_sync_zero() {
        // The block-wide `bar.sync 0` form is sm_75-legal and is the
        // form backing `sync_threads` / `barrier` in cuda-device. The
        // detector must NOT fire on it, otherwise every sm_75 baseline
        // kernel would falsely trip the gate. This is the negative
        // test that pins the detector's selectivity.
        let path = write_temp_ll(
            "bar_sync_zero",
            "call void @llvm.nvvm.barrier0() ; emits 'bar.sync 0'",
        );
        assert!(
            !contains_named_barrier_bar_sync(&path),
            "bar.sync 0 (sm_75-legal block-wide) must NOT trigger the named-barrier detector"
        );
        assert_eq!(
            detect_features(&path),
            DetectedFeatures::Basic,
            "a kernel with only bar.sync 0 must collapse to Basic"
        );
        let _ = fs::remove_file(path);
    }

    // =========================================================================
    // Collapse tests: detect_features priority and Basic fall-through.
    // =========================================================================

    #[test]
    fn test_detect_features_prefers_most_specific() {
        // A file with both blackwell and wgmma patterns must collapse to
        // Blackwell (most specific wins by the order in detect_features).
        let path = write_temp_ll(
            "blackwell_plus_wgmma",
            "call void @llvm.nvvm.tcgen05.alloc(...)\n\
             call void @llvm.nvvm.wgmma.mma_async(...)",
        );
        assert_eq!(detect_features(&path), DetectedFeatures::Blackwell);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_detect_features_falls_through_to_basic() {
        let path = write_temp_ll("plain", "define void @kernel() { ret void }");
        assert_eq!(detect_features(&path), DetectedFeatures::Basic);
        let _ = fs::remove_file(path);
    }

    // =========================================================================
    // select_target tests: feature-to-arch mapping (pure lookup table).
    // =========================================================================

    #[test]
    fn test_select_target_prefers_required_architecture() {
        assert_eq!(select_target(DetectedFeatures::Blackwell), "sm_100a");
        assert_eq!(select_target(DetectedFeatures::TmaMulticast), "sm_100a");
        assert_eq!(select_target(DetectedFeatures::Wgmma), "sm_90a");
        assert_eq!(select_target(DetectedFeatures::Tma), "sm_100");
        assert_eq!(select_target(DetectedFeatures::Cluster), "sm_90");
        assert_eq!(select_target(DetectedFeatures::AmpereAsync), "sm_80");
        assert_eq!(select_target(DetectedFeatures::Basic), "sm_75");
    }

    // =========================================================================
    // Gate tests: SM75 capability check, including the sm_75a typo guard.
    // =========================================================================

    #[test]
    fn test_sm75_gate_accepts_basic_on_sm75() {
        assert!(check_target_compat("sm_75", DetectedFeatures::Basic).is_ok());
        assert!(check_target_compat("compute_75", DetectedFeatures::Basic).is_ok());
    }

    #[test]
    fn test_sm75_gate_rejects_advanced_features() {
        assert!(check_target_compat("sm_75", DetectedFeatures::Wgmma).is_err());
        assert!(check_target_compat("sm_75", DetectedFeatures::Tma).is_err());
        assert!(check_target_compat("sm_75", DetectedFeatures::TmaMulticast).is_err());
        assert!(check_target_compat("sm_75", DetectedFeatures::Cluster).is_err());
        assert!(check_target_compat("sm_75", DetectedFeatures::Blackwell).is_err());
        assert!(check_target_compat("sm_75", DetectedFeatures::AmpereAsync).is_err());
        assert!(check_target_compat("compute_75", DetectedFeatures::Wgmma).is_err());
        assert!(check_target_compat("compute_75", DetectedFeatures::AmpereAsync).is_err());
    }

    #[test]
    fn test_sm75_gate_passes_through_other_targets() {
        assert!(check_target_compat("sm_80", DetectedFeatures::Wgmma).is_ok());
        assert!(check_target_compat("sm_80", DetectedFeatures::AmpereAsync).is_ok());
        assert!(check_target_compat("sm_90", DetectedFeatures::Wgmma).is_ok());
        assert!(check_target_compat("sm_90a", DetectedFeatures::Wgmma).is_ok());
        assert!(check_target_compat("sm_100", DetectedFeatures::Tma).is_ok());
        assert!(check_target_compat("sm_100a", DetectedFeatures::Blackwell).is_ok());
    }

    #[test]
    fn test_sm75_gate_error_message_mentions_target_and_feature() {
        let err = check_target_compat("sm_75", DetectedFeatures::Wgmma).unwrap_err();
        assert!(err.contains("sm_75"), "error must name the target: {err}");
        assert!(
            err.contains("Wgmma"),
            "error must name the offending feature: {err}"
        );
    }

    #[test]
    fn test_sm75_gate_rejects_75a_with_helpful_error() {
        let err = check_target_compat("sm_75a", DetectedFeatures::Basic).unwrap_err();
        assert!(
            err.contains("sm_75a"),
            "error must name the bad target: {err}"
        );
        assert!(
            err.contains("sm_75"),
            "error must suggest the correct target: {err}"
        );
        assert!(
            err.to_lowercase().contains("typo") || err.to_lowercase().contains("did you mean"),
            "error must hint at the typo: {err}"
        );
        assert!(check_target_compat("compute_75a", DetectedFeatures::Basic).is_err());
    }

    #[test]
    fn test_sm75_gate_treats_compute_75_identically() {
        assert!(check_target_compat("sm_75", DetectedFeatures::Basic).is_ok());
        assert!(check_target_compat("compute_75", DetectedFeatures::Basic).is_ok());
        for detected in [
            DetectedFeatures::Wgmma,
            DetectedFeatures::Tma,
            DetectedFeatures::TmaMulticast,
            DetectedFeatures::Cluster,
            DetectedFeatures::Blackwell,
            DetectedFeatures::AmpereAsync,
        ] {
            assert!(
                check_target_compat("sm_75", detected).is_err(),
                "sm_75 must reject {detected:?}"
            );
            assert!(
                check_target_compat("compute_75", detected).is_err(),
                "compute_75 must reject {detected:?}"
            );
        }
        assert!(is_sm75_target("sm_75"));
        assert!(is_sm75_target("compute_75"));
        assert!(!is_sm75_target("sm_80"));
        assert!(!is_sm75_target("sm_90"));
        assert!(!is_sm75_target("sm_100"));
        assert!(!is_sm75_target("sm_100a"));
    }

    // =========================================================================
    // Integration tests: end-to-end chain through real IR fixtures, plus
    // doc-pin tests that lock the cuda-oxide-book claims to specific output.
    // =========================================================================

    #[test]
    fn test_sm75_gate_catches_real_wgmma_intrinsic() {
        let path = write_temp_ll(
            "wgmma_sm75",
            r#"
declare void @llvm.nvvm.wgmma.mma_async(...) #0
define void @kernel() {
  call void @llvm.nvvm.wgmma.mma_async(...)
  ret void
}
"#,
        );
        assert!(
            contains_wgmma_features(&path),
            "WGMMA intrinsic must be detected in the IR"
        );
        let detected = detect_features(&path);
        assert_eq!(detected, DetectedFeatures::Wgmma);
        let target = select_target(detected);
        assert_eq!(target, "sm_90a", "WGMMA must auto-select sm_90a");
        assert!(check_target_compat(target, detected).is_ok());
        let err = check_target_compat("sm_75", detected).unwrap_err();
        assert!(err.contains("Wgmma"));
        let _ = fs::remove_file(path);
    }

    /// The four intrinsic families that the gate is supposed to reject on
    /// sm_75 must each be detectable from realistic LLVM IR snippets. This
    /// is the test matrix a CI run on a stock runner (no GPU, no `llc`)
    /// can lock in for the `cuda-oxide-book/compiler/sm75-support.md`
    /// "Verification" section.
    #[test]
    fn test_sm75_gate_full_chain_for_each_intrinsic_family() {
        // Each entry is a (intrinsic substring, expected DetectedFeatures).
        // A regression in any detector would flip the assertion below.
        let cases: &[(&str, DetectedFeatures)] = &[
            (
                "call void @llvm.nvvm.wgmma.mma_async(...)",
                DetectedFeatures::Wgmma,
            ),
            (
                "call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile(...)",
                DetectedFeatures::Tma,
            ),
            (
                "call void @llvm.nvvm.tcgen05.alloc(...)",
                DetectedFeatures::Blackwell,
            ),
            (
                "%id = call i32 @llvm.nvvm.read.ptx.sreg.cluster_ctaid()",
                DetectedFeatures::Cluster,
            ),
            (
                "call void @llvm.nvvm.cp.async.cg.shared.global(...)",
                DetectedFeatures::AmpereAsync,
            ),
            (
                "call void @llvm.nvvm.bar.warp.sync(i32 -1)",
                DetectedFeatures::AmpereAsync,
            ),
        ];

        for (snippet, expected) in cases {
            let path = write_temp_ll("family", snippet);
            let detected = detect_features(&path);
            assert_eq!(
                detected, *expected,
                "snippet {snippet:?} must detect as {expected:?}, got {detected:?}"
            );

            // With the IR's own detected target, the gate must pass through.
            let auto_target = select_target(detected);
            assert!(
                check_target_compat(auto_target, detected).is_ok(),
                "auto-selected target {auto_target} must accept {detected:?}"
            );

            // Forcing sm_75 must always fire the gate for these families.
            let err = check_target_compat("sm_75", detected)
                .expect_err(&format!("sm_75 must reject {detected:?}"));
            assert!(err.contains("sm_75"), "error must name the target: {err}");
            let _ = fs::remove_file(path);
        }
    }

    /// `tma_copy` (the example used in the doc's "Verification" section) emits
    /// plain TMA — not multicast — so it must collapse to `Tma`, not
    /// `TmaMulticast`. If the TMA detectors ever flip priority or substring
    /// matching drifts, the doc's `cargo oxide run tma_copy --arch sm_75`
    /// expected error would no longer match.
    #[test]
    fn test_sm75_gate_doc_verification_tma_copy_uses_plain_tma() {
        // A representative TMA-but-not-multicast snippet. The use_cta_mask
        // argument is `i1 0` (false), so multicast detection must NOT fire.
        let path = write_temp_ll(
            "tma_copy_doc",
            r#"
declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile(
  i32, i1, i1, i64, i64, i64, i64, i64, ptr, ptr
) #0
define void @kernel(ptr %dst, ptr %src) {
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile(
    i32 0, i1 0, i1 false,
    i64 0, i64 0, i64 0, i64 0, i64 0,
    ptr %src, ptr %dst
  )
  ret void
}
"#,
        );
        let detected = detect_features(&path);
        assert_eq!(
            detected,
            DetectedFeatures::Tma,
            "tma_copy IR must detect as Tma, got {detected:?}"
        );
        let err = check_target_compat("sm_75", detected).unwrap_err();
        // The doc claims: "Architecture sm_75 does not support detected
        // advanced features: Tma". Pin the exact error string so the
        // doc and the code cannot drift.
        assert!(
            err.contains("Architecture sm_75 does not support detected advanced features: Tma"),
            "error must match the doc's expected message: {err}"
        );
        let _ = fs::remove_file(path);
    }

    // =====================================================================
    // N5 + X7: doc-honesty tests.
    //
    // The sm75-baseline CI workflow (.github/workflows/sm75-baseline.yml)
    // asserts the EXACT `Architecture sm_75 does not support detected
    // advanced features: Tma` error string for tma_copy. The same hardening
    // needs to exist for the other detectors — cp.async, bar.warp.sync,
    // and the new named-barrier bar.sync form (N3). If the error format
    // or the user-facing variant label ever changes, these tests catch
    // the drift before the doc and the code can disagree.
    //
    // Each test:
    //   1. Writes a temp .ll with the detector's canonical intrinsic form.
    //   2. Runs `detect_features` to confirm the right variant collapses.
    //   3. Runs `check_target_compat("sm_75", ...)` to capture the error.
    //   4. Asserts the error string matches the doc-advertised format
    //      (`Architecture sm_75 does not support detected advanced
    //      features: <Variant>`).
    // =====================================================================

    #[test]
    fn test_cp_async_gate_produces_doc_advertised_error() {
        // N5: the cp.async non-bulk form is the canonical Ampere async
        // example. The doc (sm75-support.md §3) says: "Rejects the
        // non-bulk form of cp.async, cp.async.commit_group /
        // cp.async.wait_group, and bar.warp.sync." The user-facing
        // error must name the variant, not the underlying intrinsic.
        let path = write_temp_ll(
            "cp_async_doc",
            "call void @llvm.nvvm.cp.async.cg.shared.global(...)",
        );
        let detected = detect_features(&path);
        assert_eq!(
            detected,
            DetectedFeatures::AmpereAsync,
            "cp.async must collapse to AmpereAsync"
        );
        let err = check_target_compat("sm_75", detected).unwrap_err();
        assert!(
            err.contains(
                "Architecture sm_75 does not support detected advanced features: AmpereAsync"
            ),
            "cp.async gate error must match doc format, got: {err}"
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_bar_warp_sync_gate_produces_doc_advertised_error() {
        // N5: bar.warp.sync (CoalescedThreads::sync, WarpTile<N>::sync
        // backing) is the other half of the §3 "Ampere async-copy +
        // warp barriers" family. Same error string format expected.
        let path = write_temp_ll(
            "bar_warp_sync_doc",
            "call void @llvm.nvvm.bar.warp.sync(i32 -1)",
        );
        let detected = detect_features(&path);
        assert_eq!(
            detected,
            DetectedFeatures::AmpereAsync,
            "bar.warp.sync must collapse to AmpereAsync"
        );
        let err = check_target_compat("sm_75", detected).unwrap_err();
        assert!(
            err.contains(
                "Architecture sm_75 does not support detected advanced features: AmpereAsync"
            ),
            "bar.warp.sync gate error must match doc format, got: {err}"
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_named_barrier_bar_sync_gate_produces_doc_advertised_error() {
        // X7: the new N3 named-barrier detector must produce the same
        // error string format as cp.async and bar.warp.sync (it folds
        // into the same AmpereAsync variant). This test pins the
        // contract so the doc-honesty claim in §3 holds for all three
        // members of the "Ampere async-copy + warp barriers" family.
        let path = write_temp_ll(
            "named_barrier_doc",
            // Non-zero-index form (the PTX literal `bar.sync 1, "name"`).
            "call void asm sideeffect \"bar.sync 1, \\\"\\\\24named_bar\\\"\", \"\"() ;",
        );
        let detected = detect_features(&path);
        assert_eq!(
            detected,
            DetectedFeatures::AmpereAsync,
            "named-barrier bar.sync must collapse to AmpereAsync"
        );
        let err = check_target_compat("sm_75", detected).unwrap_err();
        assert!(
            err.contains(
                "Architecture sm_75 does not support detected advanced features: AmpereAsync"
            ),
            "named-barrier gate error must match doc format, got: {err}"
        );
        let _ = fs::remove_file(path);
    }
}
