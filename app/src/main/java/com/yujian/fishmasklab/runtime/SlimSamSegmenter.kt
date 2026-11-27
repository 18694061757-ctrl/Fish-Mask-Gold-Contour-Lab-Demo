package com.yujian.fishmasklab.runtime

import android.content.Context
import android.graphics.Bitmap
import com.yujian.fishmasklab.domain.MaskResult
import com.yujian.fishmasklab.domain.PromptPoint
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

class ModelNotReadyException(
  val missingAssets: List<String>,
  message: String,
) : IllegalStateException(message)

/**
 * SlimSAM-77 local ONNX Runtime contract and lifecycle manager.
 *
 * Asset Audit Finding (from GitHub `18694061757-ctrl/Fish-Mask-Gold-Contour-Lab`):
 * The reference Web app downloaded `Xenova/slimsam-77-uniform` at runtime from HuggingFace
 * (`allowLocalModels = false`). Local SlimSAM ONNX encoder/decoder weights are NOT present in
 * `public/models/` in the source repository.
 *
 * Per strict engineering rules:
 * - Never download models over the network.
 * - Never synthesize fake/mock masks or substitute a BBox for a real segmentation mask.
 * - Remain `NOT_READY` until real local SlimSAM encoder + decoder ONNX files are bundled in `assets/models/`.
 */
class SlimSamSegmenter(
  private val context: Context,
) {
  companion object {
    const val SAM_INPUT_SIZE = 1024
  }

  private val mutex = Mutex()

  @Volatile
  var status: ModelRuntimeStatus = ModelRuntimeStatus.NOT_READY
    private set

  @Volatile
  var auditReport: ModelAssetAuditReport? = null
    private set

  @Volatile
  var lastError: String? = null
    private set

  fun isReady(): Boolean = status == ModelRuntimeStatus.READY

  suspend fun ensureLoaded(): ModelAssetAuditReport = withContext(Dispatchers.IO) {
    mutex.withLock {
      val report = ModelAssetAudit.runAudit(context.assets)
      auditReport = report
      if (!report.isSlimSamReady) {
        status = ModelRuntimeStatus.NOT_READY
        lastError =
          "SlimSAM ONNX assets missing locally: " +
            listOf(
              ModelAssetAudit.SLIMSAM_ENCODER_PATH,
              ModelAssetAudit.SLIMSAM_DECODER_PATH,
              ModelAssetAudit.SLIMSAM_PROCESSOR_PATH,
            ).filter { !report.missingAssets.contains(it).not() }.joinToString(", ")
      } else {
        // Ready hook for local ONNX Runtime encoder/decoder session loading once weights are bundled
        status = ModelRuntimeStatus.NOT_READY
        lastError = "Local SlimSAM ONNX session loader awaiting verified model tensor metadata"
      }
      report
    }
  }

  /**
   * Executes SlimSAM segmentation for either automatic box-derived prompts or tap-to-refocus prompts.
   * Throws [ModelNotReadyException] if local SlimSAM ONNX weights are missing.
   */
  suspend fun segment(
    workBitmap: Bitmap,
    origWidth: Int,
    origHeight: Int,
    prompts: List<PromptPoint>,
  ): MaskResult = withContext(Dispatchers.Default) {
    ensureActive()
    val report = ensureLoaded()
    if (!report.isSlimSamReady || status != ModelRuntimeStatus.READY) {
      throw ModelNotReadyException(
        missingAssets = report.missingAssets,
        message = lastError ?: "SlimSAM local ONNX model assets are not ready.",
      )
    }

    // Real ONNX encoder + prompt/mask decoder execution path will run here once
    // `slimsam_vision_encoder.onnx` and `slimsam_prompt_mask_decoder.onnx` are bundled.
    throw ModelNotReadyException(
      missingAssets = report.missingAssets,
      message = "SlimSAM runtime status is NOT_READY.",
    )
  }

  fun close() {
    status = ModelRuntimeStatus.NOT_READY
  }
}
