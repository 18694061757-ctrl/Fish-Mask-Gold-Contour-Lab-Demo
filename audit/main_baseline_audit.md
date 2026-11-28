# V2.2.2 Main Baseline Audit

## Scope and source lock

- Repository: `18694061757-ctrl/Fish-Mask-Gold-Contour-Lab-Demo`
- Audited Main/source SHA: `539a13b5a0ea850dca7f56547c2cbdac40bc37b0`
- Local branch: `fix/fish-crop-parity-v2-2-2`, created from that exact Main SHA.
- Remote URL: `https://github.com/18694061757-ctrl/Fish-Mask-Gold-Contour-Lab-Demo.git`
- Before this audit file, the worktree was clean. No production or test source has been changed.
- `git diff --check` on the baseline is clean.

## Frozen input verification

All four committed test fixtures and all three model assets match the task's frozen hashes.

| Input | SHA-256 | Result |
|---|---|---|
| `app/src/test/resources/preset_fish.jpg` | `4f0bda4d423239352ccc6f453ec7c02349203929e9ffde15cb547c6572bb7933` | PASS |
| `app/src/test/resources/preset_catfish.jpg` | `386bb6ac75cade6043c7b2a3cb965d779c7ca96efe703db9d1120a097d8ed763` | PASS |
| `app/src/test/resources/preset_snakehead.jpg` | `b24ab3a0841ed437369b5c893d1e08dab7d05fa3ca1f56333367a60ff99d2269` | PASS |
| `app/src/test/resources/fish_crop.jpg` | `fb4470a4ec1e330c312adc7c13220b75e80a1891ce612dc424df553ad635cf6a` | PASS |
| YOLOX asset | `98257b953b0e9f8aa673572fe4a9749a730ed8a2c014a42b7c492a0e3592c9a1` | PASS |
| SlimSAM encoder asset | `cce23c7b2e5d4f330932738fb67ba518e04b0d99ccdd1cccd22a7da4e01f2971` | PASS |
| SlimSAM decoder asset | `cb90b279f549d2cab7fd6d20c38522438c65d84bdcca3d2a764cff7d857fdce2` | PASS |

## Relevant implementation map

| Area | Current implementation | Audit finding |
|---|---|---|
| Android image decode / orientation | `app/src/main/java/com/yujian/fishmasklab/app/ExifPhotoLoader.kt` | Fixture manifest freezes the exact bytes, dimensions, and orientation. Existing production-path ONNX fixture test uses this loader. |
| YOLOX preprocessing/inference | `app/src/main/java/com/yujian/fishmasklab/runtime/YoloxFishDetector.kt` | Runtime contract is 416x416 BGR letterbox, fill 114, confidence 0.20, NMS 0.45. The baseline crop detector box agrees with Web above the 0.98 gate. |
| SAM resize, normalize, pad | `app/src/main/java/com/yujian/fishmasklab/runtime/SlimSamImageProcessor.kt` | Pixel-center (`align_corners=false`) bilinear resize; RGB, 1/255, ImageNet normalize, bottom/right zero pad. The evidence test recomputes the processor tensor after inference but does not retain the exact production call input. |
| SAM dimensions and prompts | `app/src/main/java/com/yujian/fishmasklab/domain/ImageCoordinateMapper.kt`, `app/src/main/java/com/yujian/fishmasklab/domain/MaskQualityGate.kt` | Resized dimensions and prompt mapping are independently reproduced in Web. Existing Web summary does not include prompt tensor hash, source hash, or an explicit Android/Web prompt identity check. |
| Encoder / decoder | `app/src/main/java/com/yujian/fishmasklab/runtime/SlimSamVisionEncoder.kt`, `SlimSamMaskDecoder.kt` | Encoder embeddings and decoder logits are available from the test inference result, but current evidence exports only aggregate encoder metrics and decoder IoU scores, not positional embeddings or raw logits. |
| Mask postprocess and candidate rank | `app/src/main/java/com/yujian/fishmasklab/runtime/SlimSamMaskPostProcessor.kt`, `app/src/main/java/com/yujian/fishmasklab/domain/MaskQualityGate.kt` | Android resizes 256x256 logits to padded 1024, crops to resized dimensions, resizes to original dimensions, thresholds `logit > 0`, clips to 45% expanded detector box, isolates LCC, then scores candidates. Web uses `SamImageProcessor.post_process_masks`, separate JavaScript clipping/LCC, and a separate app score. |
| Web reference path | `scripts/web_parity_runner.mjs` | Runs Transformers.js Node image preprocessing (Sharp path) and a manually computed bilinear reference. Saves processor values and embeddings, but not decoder raw logits, positional embeddings, exact prompt input hashes, or same-candidate postprocess cross-checks. |
| End-to-end regression and evidence | `app/src/test/java/com/yujian/fishmasklab/runtime/EvidenceAndParityExportTest.kt`, `SlimSamOnnxRuntimeTest.kt` | Real ONNX fixture test exists. Parity test asserts within the photo loop; a strict failure aborts later photos and prevents the final summary and evidence manifest from being written. Bilinear selected-mask comparison currently compares Android's selected candidate against Web bilinear's independently selected candidate, which is not a same-candidate comparison. |
| CI packaging | `.github/workflows/android-ci.yml`, `scripts/package_evidence.sh` | The single job stops at the failing unit-test step, so instrumentation compile, lint, APK, signature, official Evidence ZIP, and their uploads are skipped. JUnit is uploaded with `always()`, but Web tensor outputs are currently under `/tmp` and are not uploaded on failure. |

## Exact baseline CI evidence

Read-only GitHub Actions run `46104864932` ran on the audited Main SHA. The JUnit artifact reported 33 tests: 32 passed, 1 failed, 0 skipped. Kotlin debug compilation passed. The real ONNX regression `SlimSamOnnxRuntimeTest` passed and exercised all four frozen fixtures through production `ExifPhotoLoader`, coordinator, YOLOX, SlimSAM Q8, and tap refocus. The single failure is `EvidenceAndParityExportTest`:

`fish_crop selectedMaskIoU=0.8809659309390141 must be >= 0.94`

Baseline values captured in the JUnit system output:

| Photo | Web/Android detector box IoU | Android/Web selected index | Selected mask IoU | Whole-image pixel difference | Mean contour Chamfer |
|---|---:|---:|---:|---:|---:|
| `preset_fish` | 0.9964 | 0 / 0 | 0.9854 | 0.080% | 0.817 px |
| `preset_catfish` | 0.9869 | 1 / 1 | 0.9902 | 0.061% | 0.612 px |
| `preset_snakehead` | 0.9886 | 0 / 0 | 0.9927 | 0.097% | 0.860 px |
| `fish_crop` | 0.9937 | 1 / 1 | 0.8810 | 1.759% | 6.668 px |

For `fish_crop`, the Web Node and Android selected indices agree, and the pixel-difference gate passes. The IoU and contour-distance gates fail. The Android-vs-manual-bilinear preprocessing metrics pass (`pvMAE=0.000000`, max difference `0.00006`) and encoder cosine passes (`0.999357`), but the old bilinear mask line reports IoU `0.5931` while Web bilinear selected candidate 0 and Android selected candidate 1. That number compares different candidates and is invalid as root-cause evidence. It must be replaced by same-candidate cross-runtime comparisons.

## Required diagnostic work before a production fix

The preserved CI run proves the failure and verifies the model/fixture path; it does not identify the first divergent tensor. Before changing production behavior, the next diagnostic run must save and compare for all four photos, especially `fish_crop`:

1. Fixture SHA, decoded source dimensions/orientation, YOLOX input and decoded box.
2. Generated prompts, SAM-mapped coordinates/labels, shape and byte hash.
3. Android, Web Node, and Web manual-bilinear processor tensors; exact MAE/max/changed-element counts.
4. Encoder image and positional embeddings; cosine/MAE/max.
5. Decoder input identities and raw logits/IoU scores for candidates 0, 1, and 2; finite checks and near-zero/sign-flip counts.
6. Apply the same raw candidate logits to Kotlin production postprocessing, and compare the result against Web Node and Web bilinear postprocessing candidate-by-candidate. Record both before and after clipping/LCC where possible.
7. Record candidate score/trustworthiness inputs and selection reasons without changing the thresholds.

Only the first measured divergence can justify a minimal fix. Keep all frozen thresholds and model assets unchanged. Preserve diagnostics and run every fixture before reporting accumulated parity failures.

## Execution environment limits observed during audit

The active workspace has JDK 21 and Node 24, but `ANDROID_HOME` / `ANDROID_SDK_ROOT` are unset and `adb` is not installed. GitHub CLI is not authenticated in this execution environment. These limits do not change the baseline findings; they may block local Android execution and remote branch/PR operations unless the connected CI/repository write path becomes available.
