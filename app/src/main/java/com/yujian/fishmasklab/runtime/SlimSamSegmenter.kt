package com.yujian.fishmasklab.runtime

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.yujian.fishmasklab.domain.FishDetection
import com.yujian.fishmasklab.domain.ImageCoordinateMapper
import com.yujian.fishmasklab.domain.MaskResult
import com.yujian.fishmasklab.domain.PromptPoint
import java.io.File
import java.security.MessageDigest
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

enum class SlimSamStage {
  PREPROCESSING,
  ENCODER,
  DECODER,
  POSTPROCESSING,
}

class ModelNotReadyException(
  val missingAssets: List<String>,
  val errorCode: SlimSamErrorCode? = null,
  message: String,
) : IllegalStateException(message)

data class CachedSamPhotoEmbedding(
  val photoKey: Long,
  val preprocessed: SamPreprocessedImage,
  val embedding: SamImageEmbedding,
)

data class SlimSamInferenceResult(
  val maskResult: MaskResult,
  val preprocessDurationMs: Long,
  val encoderDurationMs: Long,
  val decoderDurationMs: Long,
  val postprocessDurationMs: Long,
  val reusedCachedEmbedding: Boolean,
)

/**
 * SlimSAM-77 local ONNX Runtime CPU contract, session manager, and embedding cache.
 *
 * Uses official `Xenova/slimsam-77-uniform` Q8 quantized ONNX encoder + prompt/mask decoder
 * bundled inside APK `assets/models/slimsam/` with strict SHA-256 verification and zero network calls.
 */
class SlimSamSegmenter(
  private val context: Context,
  private val encoder: SlimSamVisionEncoder = SlimSamVisionEncoder(),
  private val decoder: SlimSamMaskDecoder = SlimSamMaskDecoder(),
) {
  companion object {
    private const val TAG = "SlimSamSegmenter"
    const val SAM_INPUT_SIZE = 1024
  }

  private val loadMutex = Mutex()
  private val cacheMutex = Mutex()

  @Volatile
  var status: ModelRuntimeStatus = ModelRuntimeStatus.NOT_READY
    private set

  @Volatile
  var modelState: SlimSamModelState = SlimSamModelState.NOT_READY
    private set

  @Volatile
  var lastErrorCode: SlimSamErrorCode? = null
    private set

  @Volatile
  var auditReport: ModelAssetAuditReport? = null
    private set

  @Volatile
  var lastError: String? = null
    private set

  @Volatile
  var processor: SlimSamImageProcessor? = null
    private set

  private var cachedPhotoContext: CachedSamPhotoEmbedding? = null

  fun isReady(): Boolean =
    status == ModelRuntimeStatus.READY &&
      modelState == SlimSamModelState.READY &&
      encoder.isLoaded() &&
      decoder.isLoaded() &&
      processor != null

  suspend fun hasCachedEmbeddingForPhoto(photoKey: Long): Boolean = cacheMutex.withLock {
    cachedPhotoContext?.photoKey == photoKey
  }

  suspend fun clearPhotoEmbeddingCache() = cacheMutex.withLock {
    cachedPhotoContext = null
  }

  suspend fun ensureLoaded(): ModelAssetAuditReport = withContext(Dispatchers.IO) {
    val existingAudit = auditReport
    if (isReady() && existingAudit != null) {
      return@withContext existingAudit
    }
    loadMutex.withLock {
      val cachedAudit = auditReport
      if (isReady() && cachedAudit != null) {
        return@withContext cachedAudit
      }

      modelState = SlimSamModelState.ASSET_VERIFYING
      status = ModelRuntimeStatus.WARMING_UP
      lastError = null
      lastErrorCode = null

      val report = ModelAssetAudit.runAudit(context.assets)
      auditReport = report

      if (!report.isSlimSamReady) {
        val code = if (report.shaMismatchedAssets.isNotEmpty()) {
          SlimSamErrorCode.ASSET_HASH_MISMATCH
        } else {
          SlimSamErrorCode.ASSET_MISSING
        }
        status = if (code == SlimSamErrorCode.ASSET_HASH_MISMATCH) {
          ModelRuntimeStatus.ERROR
        } else {
          ModelRuntimeStatus.NOT_READY
        }
        modelState = SlimSamModelState.ERROR
        lastErrorCode = code
        lastError = buildString {
          if (report.missingAssets.isNotEmpty()) {
            append("Missing SlimSAM assets: ${report.missingAssets.joinToString(", ")}. ")
          }
          if (report.shaMismatchedAssets.isNotEmpty()) {
            append("SHA-256 mismatch: ${report.shaMismatchedAssets.joinToString(", ")}.")
          }
        }.trim()
        return@withContext report
      }

      modelState = SlimSamModelState.ASSET_READY
      try {
        // Parse preprocessor_config.json strictly from local assets
        val procJson = context.assets.open(ModelAssetAudit.SLIMSAM_PROCESSOR_PATH)
          .bufferedReader()
          .use { it.readText() }
        processor = SlimSamImageProcessor(SamPreprocessorConfig.fromJson(procJson))

        // Install ONNX models atomically into app filesDir with SHA-256 validation
        val onnxDir = File(context.filesDir, "models/slimsam/onnx")
        val encoderFile = installAssetAtomically(
          assetPath = ModelAssetAudit.SLIMSAM_ENCODER_PATH,
          expectedSha256 = ModelAssetAudit.SLIMSAM_ENCODER_EXPECTED_SHA256,
          destFile = File(onnxDir, "vision_encoder_quantized.onnx"),
        )
        val decoderFile = installAssetAtomically(
          assetPath = ModelAssetAudit.SLIMSAM_DECODER_PATH,
          expectedSha256 = ModelAssetAudit.SLIMSAM_DECODER_EXPECTED_SHA256,
          destFile = File(onnxDir, "prompt_encoder_mask_decoder_quantized.onnx"),
        )

        modelState = SlimSamModelState.SESSION_LOADING
        encoder.ensureLoaded(encoderFile)

        modelState = SlimSamModelState.SESSION_VALIDATING
        decoder.ensureLoaded(decoderFile)

        modelState = SlimSamModelState.READY
        status = ModelRuntimeStatus.READY
        Log.i(
          TAG,
          "SlimSAM-77 Q8 CPU sessions READY | encoder=(${encoder.contractSummary}) | decoder=(${decoder.contractSummary})",
        )
      } catch (ce: CancellationException) {
        modelState = SlimSamModelState.NOT_READY
        status = ModelRuntimeStatus.NOT_LOADED
        throw ce
      } catch (sre: SlimSamRuntimeException) {
        modelState = SlimSamModelState.ERROR
        status = ModelRuntimeStatus.ERROR
        lastErrorCode = sre.errorCode
        lastError = sre.message
        Log.e(TAG, "SlimSAM runtime init failed [${sre.errorCode}]: ${sre.message}", sre)
      } catch (t: Throwable) {
        modelState = SlimSamModelState.ERROR
        status = ModelRuntimeStatus.ERROR
        lastErrorCode = SlimSamErrorCode.SESSION_CREATE_FAILED
        lastError = t.message ?: t.javaClass.simpleName
        Log.e(TAG, "SlimSAM init failed", t)
      }

      report
    }
  }

  suspend fun prepareOrReuseEmbedding(
    photoBitmap: Bitmap,
    photoKey: Long,
    forceReencode: Boolean = false,
    onStage: (SlimSamStage) -> Unit = {},
  ): Pair<CachedSamPhotoEmbedding, Boolean> = withContext(Dispatchers.Default) {
    ensureActive()
    val report = ensureLoaded()
    val activeProcessor = processor
    if (!report.isSlimSamReady || !isReady() || activeProcessor == null) {
      throw ModelNotReadyException(
        missingAssets = report.missingAssets,
        errorCode = lastErrorCode,
        message = lastError ?: "SlimSAM Q8 ONNX sessions are not ready.",
      )
    }

    if (!forceReencode) {
      val cached = cacheMutex.withLock { cachedPhotoContext?.takeIf { it.photoKey == photoKey } }
      if (cached != null) {
        return@withContext cached to true
      }
    }

    modelState = SlimSamModelState.INFERENCE_RUNNING
    try {
      onStage(SlimSamStage.PREPROCESSING)
      val pre = activeProcessor.preprocess(photoBitmap)
      ensureActive()

      onStage(SlimSamStage.ENCODER)
      val emb = encoder.encode(pre.pixelValues)
      ensureActive()

      val entry = CachedSamPhotoEmbedding(
        photoKey = photoKey,
        preprocessed = pre,
        embedding = emb,
      )
      cacheMutex.withLock {
        cachedPhotoContext = entry
      }
      modelState = SlimSamModelState.READY
      entry to false
    } catch (ce: CancellationException) {
      modelState = SlimSamModelState.READY
      throw ce
    } catch (sre: SlimSamRuntimeException) {
      modelState = SlimSamModelState.READY
      lastErrorCode = sre.errorCode
      lastError = sre.message
      throw sre
    }
  }

  suspend fun decodeAndPostProcess(
    cachedEmbedding: CachedSamPhotoEmbedding,
    mapper: ImageCoordinateMapper,
    prompts: List<PromptPoint>,
    detection: FishDetection?,
    reusedEmbedding: Boolean,
    preprocessDurationMs: Long = if (reusedEmbedding) 0L else cachedEmbedding.preprocessed.preprocessDurationMs,
    encoderDurationMs: Long = if (reusedEmbedding) 0L else cachedEmbedding.embedding.encoderDurationMs,
    onStage: (SlimSamStage) -> Unit = {},
  ): SlimSamInferenceResult = withContext(Dispatchers.Default) {
    ensureActive()
    if (!isReady()) {
      throw ModelNotReadyException(
        missingAssets = auditReport?.missingAssets.orEmpty(),
        errorCode = lastErrorCode,
        message = lastError ?: "SlimSAM Q8 ONNX sessions are not ready.",
      )
    }

    modelState = SlimSamModelState.INFERENCE_RUNNING
    val decodeStart = System.currentTimeMillis()
    try {
      val mappedPrompts = mapper.mapPromptsToSamSpace(
        prompts = prompts,
        resizedWidth = cachedEmbedding.preprocessed.resizedWidth,
        resizedHeight = cachedEmbedding.preprocessed.resizedHeight,
      )

      onStage(SlimSamStage.DECODER)
      val decoderOut = decoder.decode(
        embedding = cachedEmbedding.embedding,
        pointCoords = mappedPrompts.samPointCoords,
        pointLabels = mappedPrompts.samPointLabels,
        pointCount = mappedPrompts.pointCount,
      )
      ensureActive()

      onStage(SlimSamStage.POSTPROCESSING)
      val postStart = System.currentTimeMillis()
      val totalSamMs = (preprocessDurationMs + encoderDurationMs + (System.currentTimeMillis() - decodeStart))
        .coerceAtLeast(1L)
      val maskResult = try {
        SlimSamMaskPostProcessor.buildMaskResult(
          decoderOutput = decoderOut,
          resizedWidth = cachedEmbedding.preprocessed.resizedWidth,
          resizedHeight = cachedEmbedding.preprocessed.resizedHeight,
          originalWidth = cachedEmbedding.preprocessed.originalWidth,
          originalHeight = cachedEmbedding.preprocessed.originalHeight,
          detection = detection,
          totalSamInferenceMs = totalSamMs,
        )
      } catch (ce: CancellationException) {
        throw ce
      } catch (t: Throwable) {
        throw SlimSamRuntimeException(
          SlimSamErrorCode.POSTPROCESS_FAILED,
          "Mask post-processing failed: ${t.message ?: t.javaClass.simpleName}",
          t,
        )
      }
      val postMs = (System.currentTimeMillis() - postStart).coerceAtLeast(0L)

      modelState = SlimSamModelState.READY
      SlimSamInferenceResult(
        maskResult = maskResult,
        preprocessDurationMs = preprocessDurationMs,
        encoderDurationMs = encoderDurationMs,
        decoderDurationMs = decoderOut.decoderDurationMs,
        postprocessDurationMs = postMs,
        reusedCachedEmbedding = reusedEmbedding,
      )
    } catch (ce: CancellationException) {
      modelState = SlimSamModelState.READY
      throw ce
    } catch (sre: SlimSamRuntimeException) {
      modelState = SlimSamModelState.READY
      lastErrorCode = sre.errorCode
      lastError = sre.message
      throw sre
    }
  }

  /**
   * Runs or reuses the Vision Encoder embedding for [photoKey], then decodes [prompts]
   * and post-processes masks back to `[mapper.origWidth, mapper.origHeight]`.
   */
  suspend fun segmentPhoto(
    photoBitmap: Bitmap,
    photoKey: Long,
    mapper: ImageCoordinateMapper,
    prompts: List<PromptPoint>,
    detection: FishDetection?,
    forceReencode: Boolean = false,
  ): SlimSamInferenceResult {
    val (entry, reused) = prepareOrReuseEmbedding(
      photoBitmap = photoBitmap,
      photoKey = photoKey,
      forceReencode = forceReencode,
    )
    return decodeAndPostProcess(
      cachedEmbedding = entry,
      mapper = mapper,
      prompts = prompts,
      detection = detection,
      reusedEmbedding = reused,
    )
  }

  /**
   * Convenience wrapper preserving direct prompt-only segmentation signature.
   */
  suspend fun segment(
    workBitmap: Bitmap,
    origWidth: Int,
    origHeight: Int,
    prompts: List<PromptPoint>,
    photoKey: Long = System.identityHashCode(workBitmap).toLong(),
    detection: FishDetection? = null,
  ): MaskResult {
    val mapper = ImageCoordinateMapper.create(origWidth = origWidth, origHeight = origHeight)
    return segmentPhoto(
      photoBitmap = workBitmap,
      photoKey = photoKey,
      mapper = mapper,
      prompts = prompts,
      detection = detection,
    ).maskResult
  }

  private fun installAssetAtomically(
    assetPath: String,
    expectedSha256: String,
    destFile: File,
  ): File {
    destFile.parentFile?.mkdirs()
    if (destFile.exists() && destFile.length() > 0L) {
      val existingSha = sha256File(destFile)
      if (existingSha.equals(expectedSha256, ignoreCase = true)) {
        return destFile
      }
      destFile.delete()
    }

    val tmpFile = File(destFile.parentFile, "${destFile.name}.tmp.${System.nanoTime()}")
    try {
      val digest = MessageDigest.getInstance("SHA-256")
      val buffer = ByteArray(32 * 1024)
      context.assets.open(assetPath).use { input ->
        tmpFile.outputStream().use { output ->
          while (true) {
            val read = input.read(buffer)
            if (read <= 0) break
            digest.update(buffer, 0, read)
            output.write(buffer, 0, read)
          }
          output.fd.sync()
        }
      }
      val actualSha = digest.digest().joinToString("") { "%02x".format(it) }
      if (!actualSha.equals(expectedSha256, ignoreCase = true)) {
        throw SlimSamRuntimeException(
          SlimSamErrorCode.ASSET_HASH_MISMATCH,
          "Asset '$assetPath' SHA-256 mismatch: expected $expectedSha256, got $actualSha",
        )
      }
      if (destFile.exists()) destFile.delete()
      if (!tmpFile.renameTo(destFile)) {
        tmpFile.copyTo(destFile, overwrite = true)
        tmpFile.delete()
      }
      return destFile
    } catch (t: Throwable) {
      runCatching { tmpFile.delete() }
      throw t
    }
  }

  private fun sha256File(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    val buffer = ByteArray(32 * 1024)
    file.inputStream().use { input ->
      while (true) {
        val read = input.read(buffer)
        if (read <= 0) break
        digest.update(buffer, 0, read)
      }
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
  }

  fun close() {
    encoder.close()
    decoder.close()
    cachedPhotoContext = null
    status = ModelRuntimeStatus.NOT_LOADED
    modelState = SlimSamModelState.NOT_READY
  }
}
