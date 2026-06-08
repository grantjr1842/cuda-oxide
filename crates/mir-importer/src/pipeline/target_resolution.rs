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
    } else {
        false
    }
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
        let path = write_temp_ll("cluster", "%id = call i32 @llvm.nvvm.read.ptx.sreg.cluster_ctaid()");
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
        assert!(err.contains("Wgmma"), "error must name the offending feature: {err}");
    }

    #[test]
    fn test_sm75_gate_rejects_75a_with_helpful_error() {
        let err = check_target_compat("sm_75a", DetectedFeatures::Basic).unwrap_err();
        assert!(err.contains("sm_75a"), "error must name the bad target: {err}");
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
            ("call void @llvm.nvvm.wgmma.mma_async(...)", DetectedFeatures::Wgmma),
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
}
