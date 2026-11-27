package com.yujian.fishmasklab.runtime

import android.content.res.AssetManager
import java.security.MessageDigest

enum class ModelRuntimeStatus {
  NOT_LOADED,
  WARMING_UP,
  READY,
  NOT_READY,
  ERROR,
}

enum class SlimSamModelState {
  NOT_READY,
  ASSET_VERIFYING,
  ASSET_READY,
  SESSION_LOADING,
  SESSION_VALIDATING,
  READY,
  INFERENCE_RUNNING,
  ERROR,
}

enum class SlimSamErrorCode {
  ASSET_MISSING,
  ASSET_HASH_MISMATCH,
  SESSION_CREATE_FAILED,
  TENSOR_CONTRACT_MISMATCH,
  UNSUPPORTED_OPERATOR,
  ENCODER_FAILED,
  DECODER_FAILED,
  POSTPROCESS_FAILED,
}

data class ModelAssetEntry(
  val logicalName: String,
  val assetPath: String,
  val present: Boolean,
  val sizeBytes: Long,
  val sha256: String?,
  val expectedSha256: String? = null,
) {
  val shaVerified: Boolean
    get() = present && (expectedSha256 == null || sha256.equals(expectedSha256, ignoreCase = true))
}

data class ModelAssetAuditReport(
  val yoloxAsset: ModelAssetEntry,
  val pipelineContractAsset: ModelAssetEntry,
  val slimSamEncoderAsset: ModelAssetEntry,
  val slimSamDecoderAsset: ModelAssetEntry,
  val slimSamProcessorAsset: ModelAssetEntry,
  val slimSamConfigAsset: ModelAssetEntry,
  val slimSamQuantizeConfigAsset: ModelAssetEntry,
  val slimSamManifestAsset: ModelAssetEntry,
  val missingAssets: List<String>,
  val shaMismatchedAssets: List<String>,
) {
  val isYoloxReady: Boolean get() = yoloxAsset.present && yoloxAsset.shaVerified
  val isSlimSamReady: Boolean
    get() = slimSamEncoderAsset.shaVerified &&
      slimSamDecoderAsset.shaVerified &&
      slimSamProcessorAsset.shaVerified &&
      slimSamConfigAsset.shaVerified &&
      slimSamQuantizeConfigAsset.shaVerified &&
      slimSamManifestAsset.present
}

object ModelAssetAudit {

  const val YOLOX_ASSET_PATH = "models/fish_detector_yolox_nano_v0_1.onnx"
  const val YOLOX_EXPECTED_SHA256 =
    "98257b953b0e9f8aa673572fe4a9749a730ed8a2c014a42b7c492a0e3592c9a1"

  const val PIPELINE_CONTRACT_PATH = "models/recognition_pipeline_v1.json"

  const val SLIMSAM_MODEL_ID = "Xenova/slimsam-77-uniform"
  const val SLIMSAM_REVISION = "5850ab45f587c112167512ffef949107115e26a0"

  const val SLIMSAM_ENCODER_PATH = "models/slimsam/onnx/vision_encoder_quantized.onnx"
  const val SLIMSAM_ENCODER_EXPECTED_SHA256 =
    "cce23c7b2e5d4f330932738fb67ba518e04b0d99ccdd1cccd22a7da4e01f2971"

  const val SLIMSAM_DECODER_PATH = "models/slimsam/onnx/prompt_encoder_mask_decoder_quantized.onnx"
  const val SLIMSAM_DECODER_EXPECTED_SHA256 =
    "cb90b279f549d2cab7fd6e20c38522438c65d84bdcca3d2a764cff7d857fdce2"

  const val SLIMSAM_PROCESSOR_PATH = "models/slimsam/preprocessor_config.json"
  const val SLIMSAM_PROCESSOR_EXPECTED_SHA256 =
    "225545a743c654e3c495ec6f545a0eaba57c8ba3fbbd8483b3cb1c0fc58db517"

  const val SLIMSAM_CONFIG_PATH = "models/slimsam/config.json"
  const val SLIMSAM_CONFIG_EXPECTED_SHA256 =
    "6339884f168658d3ca6473b486973913fb33e84e625e06ae2dd7b4a808187419"

  const val SLIMSAM_QUANTIZE_CONFIG_PATH = "models/slimsam/quantize_config.json"
  const val SLIMSAM_QUANTIZE_CONFIG_EXPECTED_SHA256 =
    "2a1e2927485ff8675da940616e77578321e2083477d3fb22d2cfc53c604fb8ff"

  const val SLIMSAM_MANIFEST_PATH = "models/slimsam/slimsam_model_manifest.json"

  fun runAudit(assets: AssetManager): ModelAssetAuditReport {
    val yolox = inspectAsset(
      assets = assets,
      logicalName = "YOLOX-Nano Fish Detector",
      assetPath = YOLOX_ASSET_PATH,
      expectedSha256 = YOLOX_EXPECTED_SHA256,
    )
    val contract = inspectAsset(
      assets = assets,
      logicalName = "Recognition Pipeline Contract v1",
      assetPath = PIPELINE_CONTRACT_PATH,
    )
    val samEncoder = inspectAsset(
      assets = assets,
      logicalName = "SlimSAM-77 Q8 Vision Encoder ONNX",
      assetPath = SLIMSAM_ENCODER_PATH,
      expectedSha256 = SLIMSAM_ENCODER_EXPECTED_SHA256,
    )
    val samDecoder = inspectAsset(
      assets = assets,
      logicalName = "SlimSAM-77 Q8 Prompt & Mask Decoder ONNX",
      assetPath = SLIMSAM_DECODER_PATH,
      expectedSha256 = SLIMSAM_DECODER_EXPECTED_SHA256,
    )
    val samProc = inspectAsset(
      assets = assets,
      logicalName = "SlimSAM-77 Preprocessor Config",
      assetPath = SLIMSAM_PROCESSOR_PATH,
      expectedSha256 = SLIMSAM_PROCESSOR_EXPECTED_SHA256,
    )
    val samConfig = inspectAsset(
      assets = assets,
      logicalName = "SlimSAM-77 Model Config",
      assetPath = SLIMSAM_CONFIG_PATH,
      expectedSha256 = SLIMSAM_CONFIG_EXPECTED_SHA256,
    )
    val samQuantize = inspectAsset(
      assets = assets,
      logicalName = "SlimSAM-77 Quantize Config",
      assetPath = SLIMSAM_QUANTIZE_CONFIG_PATH,
      expectedSha256 = SLIMSAM_QUANTIZE_CONFIG_EXPECTED_SHA256,
    )
    val samManifest = inspectAsset(
      assets = assets,
      logicalName = "SlimSAM-77 Model Manifest",
      assetPath = SLIMSAM_MANIFEST_PATH,
    )

    val allRequired = listOf(
      yolox,
      samEncoder,
      samDecoder,
      samProc,
      samConfig,
      samQuantize,
      samManifest,
    )
    val missing = allRequired.filter { !it.present }.map { it.assetPath }
    val mismatched = allRequired.filter { it.present && !it.shaVerified }.map { it.assetPath }

    return ModelAssetAuditReport(
      yoloxAsset = yolox,
      pipelineContractAsset = contract,
      slimSamEncoderAsset = samEncoder,
      slimSamDecoderAsset = samDecoder,
      slimSamProcessorAsset = samProc,
      slimSamConfigAsset = samConfig,
      slimSamQuantizeConfigAsset = samQuantize,
      slimSamManifestAsset = samManifest,
      missingAssets = missing,
      shaMismatchedAssets = mismatched,
    )
  }

  private fun inspectAsset(
    assets: AssetManager,
    logicalName: String,
    assetPath: String,
    expectedSha256: String? = null,
  ): ModelAssetEntry {
    return try {
      assets.open(assetPath).use { stream ->
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(16 * 1024)
        var totalBytes = 0L
        while (true) {
          val read = stream.read(buffer)
          if (read <= 0) break
          digest.update(buffer, 0, read)
          totalBytes += read
        }
        val sha = digest.digest().joinToString("") { "%02x".format(it) }
        ModelAssetEntry(
          logicalName = logicalName,
          assetPath = assetPath,
          present = totalBytes > 0L,
          sizeBytes = totalBytes,
          sha256 = sha,
          expectedSha256 = expectedSha256,
        )
      }
    } catch (_: Exception) {
      ModelAssetEntry(
        logicalName = logicalName,
        assetPath = assetPath,
        present = false,
        sizeBytes = 0L,
        sha256 = null,
        expectedSha256 = expectedSha256,
      )
    }
  }
}
