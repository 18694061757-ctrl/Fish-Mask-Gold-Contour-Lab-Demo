#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! -f "$1" ]]; then
  echo "Usage: $0 <FishMask_Diagnostics_*.zip>" >&2
  exit 2
fi

ZIP_PATH="$1"
VERIFY_DIR="$(mktemp -d)"
trap 'rm -rf "$VERIFY_DIR"' EXIT
unzip -q "$ZIP_PATH" -d "$VERIFY_DIR"
test -f "$VERIFY_DIR/SHA256SUMS"
(cd "$VERIFY_DIR" && sha256sum --check SHA256SUMS)
echo "Diagnostic ZIP checksums verified: $ZIP_PATH"
