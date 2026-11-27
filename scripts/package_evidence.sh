#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EVIDENCE_SRC="${EVIDENCE_SRC:-$ROOT_DIR/app/build/evidence/p0_v2_1}"
JUNIT_SRC="${JUNIT_SRC:-$ROOT_DIR/app/build/test-results/testDebugUnitTest}"
APK_PATH="${APK_PATH:-$ROOT_DIR/app/build/outputs/apk/debug/app-debug.apk}"
VERIFY_DIR="${VERIFY_DIR:-$ROOT_DIR/app/build/ci-verify}"
OUT_ZIP="${OUT_ZIP:-$ROOT_DIR/FISH_MASK_ANDROID_P0_V2_1_EVIDENCE.zip}"

VERIFIED_SOURCE_HEAD="${VERIFIED_SOURCE_HEAD:-$(git rev-parse HEAD 2>/dev/null || echo 'UNCOMMITTED_WORKSPACE')}"
STAGING_DIR="$(mktemp -d)"
CHECK_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGING_DIR" "$CHECK_DIR"' EXIT

test -d "$EVIDENCE_SRC" || { echo "Missing evidence directory: $EVIDENCE_SRC" >&2; exit 1; }
test -d "$JUNIT_SRC" || { echo "Missing JUnit directory: $JUNIT_SRC" >&2; exit 1; }
test -f "$APK_PATH" || { echo "Missing Debug APK: $APK_PATH" >&2; exit 1; }

# 1. Copy evidence outputs (photos, web_android_parity, lifecycle, manifests)
cp -R "$EVIDENCE_SRC"/. "$STAGING_DIR"/

# Verify all 4 photos and required files exist
for pid in preset_fish preset_catfish preset_snakehead fish_crop; do
  for fname in \
    original.jpg detection.json detection_overlay.png \
    candidate_0.png candidate_1.png candidate_2.png \
    selected_mask.png lcc_mask.png selected_mask_overlay.png \
    contour_overlay.png contour_points.json \
    processor_metrics.json encoder_metrics.json decoder_metrics.json \
    mask_metrics.json contour_metrics.json refocus_metrics.json pipeline_events.json
  do
    test -f "$STAGING_DIR/photos/$pid/$fname" || {
      echo "Missing evidence file: photos/$pid/$fname" >&2
      exit 1
    }
  done
done

test -f "$STAGING_DIR/web_android_parity/summary.json" || { echo "Missing web_android_parity/summary.json" >&2; exit 1; }
test -f "$STAGING_DIR/web_android_parity/parity_report.md" || { echo "Missing web_android_parity/parity_report.md" >&2; exit 1; }
test -f "$STAGING_DIR/lifecycle/native_resource_lifecycle.log" || { echo "Missing lifecycle log" >&2; exit 1; }
test -f "$STAGING_DIR/lifecycle/shutdown_test_results.json" || { echo "Missing shutdown_test_results.json" >&2; exit 1; }

# 2. Copy JUnit XML reports and generate tests/test-summary.json
mkdir -p "$STAGING_DIR/tests"
cp "$JUNIT_SRC"/*.xml "$STAGING_DIR/tests/"
python3 - "$STAGING_DIR/tests" <<'PYEOF'
import glob, json, os, sys, xml.etree.ElementTree as ET

tests_dir = sys.argv[1]
suites = []
total = failures = errors = skipped = 0
total_time = 0.0

for path in sorted(glob.glob(os.path.join(tests_dir, "*.xml"))):
    root = ET.parse(path).getroot()
    t = int(root.attrib.get("tests", 0))
    f = int(root.attrib.get("failures", 0))
    e = int(root.attrib.get("errors", 0))
    s = int(root.attrib.get("skipped", 0))
    tm = float(root.attrib.get("time", 0.0))
    total += t
    failures += f
    errors += e
    skipped += s
    total_time += tm
    suites.append({
        "name": root.attrib.get("name", os.path.basename(path)),
        "tests": t,
        "failures": f,
        "errors": e,
        "skipped": s,
        "timeSec": tm,
    })

summary = {
    "totalTests": total,
    "passed": total - failures - errors - skipped,
    "failures": failures,
    "errors": errors,
    "skipped": skipped,
    "totalTimeSec": round(total_time, 3),
    "suites": suites,
}
with open(os.path.join(tests_dir, "test-summary.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)
if failures > 0 or errors > 0:
    sys.exit(f"JUnit contains failures={failures}, errors={errors}")
PYEOF

# 3. Verify APK signature & assets and populate build/ metadata
mkdir -p "$VERIFY_DIR" "$STAGING_DIR/build"

SDK_ROOT="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-/opt/android/sdk}}"
APKSIGNER_BIN="${APKSIGNER_BIN:-$(find "$SDK_ROOT/build-tools" -maxdepth 2 -type f -name apksigner | sort -V | tail -n 1)}"
test -x "$APKSIGNER_BIN" || { echo "apksigner not found under $SDK_ROOT/build-tools" >&2; exit 1; }

"$APKSIGNER_BIN" verify --verbose --print-certs "$APK_PATH" > "$VERIFY_DIR/apksigner-verify.txt"

{
  echo "=== APK ONNX & Native Libraries ==="
  unzip -l "$APK_PATH" | grep -E "assets/models/|lib/.*onnxruntime"
  echo ""
  echo "=== Extracted ONNX SHA-256 inside APK ==="
  for asset in \
    "assets/models/fish_detector_yolox_nano_v0_1.onnx" \
    "assets/models/slimsam/onnx/vision_encoder_quantized.onnx" \
    "assets/models/slimsam/onnx/prompt_encoder_mask_decoder_quantized.onnx"
  do
    sum=$(unzip -p "$APK_PATH" "$asset" | sha256sum | awk '{print $1}')
    echo "$sum  $asset"
  done
} > "$VERIFY_DIR/apk-assets-verify.txt"

APK_SHA256=$(sha256sum "$APK_PATH" | awk '{print $1}')
APK_SIZE_BYTES=$(stat -c%s "$APK_PATH" 2>/dev/null || stat -f%z "$APK_PATH")
echo "$APK_SHA256  app/build/outputs/apk/debug/app-debug.apk" > "$VERIFY_DIR/apk-sha256.txt"

YOLOX_SHA=$(unzip -p "$APK_PATH" "assets/models/fish_detector_yolox_nano_v0_1.onnx" | sha256sum | awk '{print $1}')
ENC_SHA=$(unzip -p "$APK_PATH" "assets/models/slimsam/onnx/vision_encoder_quantized.onnx" | sha256sum | awk '{print $1}')
DEC_SHA=$(unzip -p "$APK_PATH" "assets/models/slimsam/onnx/prompt_encoder_mask_decoder_quantized.onnx" | sha256sum | awk '{print $1}')

cat > "$VERIFY_DIR/build-metadata.txt" <<EOF
task_id=FISH_MASK_ANDROID_P0_V2_1_CI_NATIVE_PARITY_EVIDENCE
application_id=com.yujian.fishmasklab
verified_source_head=$VERIFIED_SOURCE_HEAD
archive_commit_note=The archive commit adding FISH_MASK_ANDROID_P0_V2_1_EVIDENCE.zip is created after SHA256SUMS finalization; verified_source_head records the exact tested and built source commit SHA.
apk_path=app/build/outputs/apk/debug/app-debug.apk
apk_sha256=$APK_SHA256
apk_size_bytes=$APK_SIZE_BYTES
apksigner_verified=true
yolox_apk_sha256=$YOLOX_SHA
slimsam_encoder_apk_sha256=$ENC_SHA
slimsam_decoder_apk_sha256=$DEC_SHA
built_at_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

cp "$VERIFY_DIR/build-metadata.txt" "$STAGING_DIR/build/build-metadata.txt"
cp "$VERIFY_DIR/apk-sha256.txt" "$STAGING_DIR/build/apk-sha256.txt"
cp "$VERIFY_DIR/apksigner-verify.txt" "$STAGING_DIR/build/apksigner-verify.txt"
cp "$VERIFY_DIR/apk-assets-verify.txt" "$STAGING_DIR/build/apk-assets-verify.txt"

# 4. Generate SHA256SUMS for every file in staging
(
  cd "$STAGING_DIR"
  find . -type f ! -name "SHA256SUMS" | LC_ALL=C sort | sed 's|^\./||' | xargs sha256sum > SHA256SUMS
)

# 5. Pack ZIP and run full integrity + checksum verification
rm -f "$OUT_ZIP"
(
  cd "$STAGING_DIR"
  zip -r -9 -q "$OUT_ZIP" .
)

unzip -t -q "$OUT_ZIP"
unzip -q "$OUT_ZIP" -d "$CHECK_DIR"
(
  cd "$CHECK_DIR"
  sha256sum -c SHA256SUMS >/dev/null
)

ZIP_SHA256=$(sha256sum "$OUT_ZIP" | awk '{print $1}')
FILE_COUNT=$(wc -l < "$CHECK_DIR/SHA256SUMS")
echo "[EVIDENCE_ZIP_VERIFIED] path=$OUT_ZIP sha256=$ZIP_SHA256 files=$FILE_COUNT verified_source_head=$VERIFIED_SOURCE_HEAD"
