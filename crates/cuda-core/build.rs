/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

fn main() {
    // Declare the custom cfg so rustc doesn't warn about unexpected configuration names
    println!("cargo::rustc-check-cfg=cfg(feature_f16)");

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let output = std::process::Command::new(rustc)
        .arg("--version")
        .output()
        .ok();

    let is_nightly = output
        .map(|out| {
            let version_str = String::from_utf8_lossy(&out.stdout);
            version_str.contains("nightly") || version_str.contains("dev")
        })
        .unwrap_or(false);

    if is_nightly {
        println!("cargo:rustc-cfg=feature_f16");
    }
}
