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
  val missingAssets: List<String>,
) {
  val isYoloxReady: Boolean get() = yoloxAsset.present && yoloxAsset.shaVerified
  val isSlimSamReady: Boolean
    get() = slimSamEncoderAsset.present && slimSamDecoderAsset.present
}

object ModelAssetAudit {

  const val YOLOX_ASSET_PATH = "models/fish_detector_yolox_nano_v0_1.onnx"
  const val YOLOX_EXPECTED_SHA256 =
    "98257b953b0e9f8aa673572fe4a9749a730ed8a2c014a42b7c492a0e3592c9a1"

  const val PIPELINE_CONTRACT_PATH = "models/recognition_pipeline_v1.json"
  const val SLIMSAM_ENCODER_PATH = "models/slimsam_vision_encoder.onnx"
  const val SLIMSAM_DECODER_PATH = "models/slimsam_prompt_mask_decoder.onnx"
  const val SLIMSAM_PROCESSOR_PATH = "models/slimsam_preprocessor_config.json"

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
      logicalName = "SlimSAM-77 Vision Encoder ONNX",
      assetPath = SLIMSAM_ENCODER_PATH,
    )
    val samDecoder = inspectAsset(
      assets = assets,
      logicalName = "SlimSAM-77 Prompt & Mask Decoder ONNX",
      assetPath = SLIMSAM_DECODER_PATH,
    )
    val samProc = inspectAsset(
      assets = assets,
      logicalName = "SlimSAM-77 Preprocessor Config",
      assetPath = SLIMSAM_PROCESSOR_PATH,
    )

    val missing = buildList {
      if (!yolox.present) add(YOLOX_ASSET_PATH)
      if (!samEncoder.present) add(SLIMSAM_ENCODER_PATH)
      if (!samDecoder.present) add(SLIMSAM_DECODER_PATH)
      if (!samProc.present) add(SLIMSAM_PROCESSOR_PATH)
    }

    return ModelAssetAuditReport(
      yoloxAsset = yolox,
      pipelineContractAsset = contract,
      slimSamEncoderAsset = samEncoder,
      slimSamDecoderAsset = samDecoder,
      slimSamProcessorAsset = samProc,
      missingAssets = missing,
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
