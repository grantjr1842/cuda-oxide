#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Verify that the pliron git revision in the workspace Cargo.toml
# matches the one in crates/rustc-codegen-cuda/Cargo.toml.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROOT_REV=$(grep -E 'pliron\s*=\s*\{' "${ROOT_DIR}/Cargo.toml" | grep -oE 'rev\s*=\s*"[a-f0-9]+"' | head -n1 | cut -d'"' -f2)
CODEGEN_REV=$(grep -E 'pliron\s*=\s*\{' "${ROOT_DIR}/crates/rustc-codegen-cuda/Cargo.toml" | grep -oE 'rev\s*=\s*"[a-f0-9]+"' | head -n1 | cut -d'"' -f2)

echo "Workspace pliron rev:          $ROOT_REV"
echo "rustc-codegen-cuda pliron rev: $CODEGEN_REV"

if [ "$ROOT_REV" != "$CODEGEN_REV" ]; then
    echo "ERROR: pliron git revisions are out of sync!"
    echo "This will cause duplicate-crate type-unification breakage at build time."
    exit 1
fi

echo "SUCCESS: pliron revisions are in sync."
