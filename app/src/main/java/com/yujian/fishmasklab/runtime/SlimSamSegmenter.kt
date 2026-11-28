package com.yujian.fishmasklab.runtime

import android.content.Context
import android.graphics.Bitmap
import android.app.ActivityManager
import android.os.SystemClock
import android.system.Os
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
import kotlinx.coroutines.currentCoroutineContext
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
  val preprocessed: SamPreprocessedMetadata,
  val embedding: SamImageEmbedding,
)

data class SlimSamInferenceResult(
  val maskResult: MaskResult,
  val preprocessDurationMs: Long,
  val encoderDurationMs: Long,
  val decoderDurationMs: Long,
  val postprocessDurationMs: Long,
  val reusedCachedEmbedding: Boolean,
  internal val decoderOutput: SamDecoderOutput? = null,
  val promptMappingStartNs: Long = 0L,
  val promptMappingEndNs: Long = 0L,
  val postprocessStartNs: Long = 0L,
  val postprocessEndNs: Long = 0L,
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
    const val MIN_PARALLEL_INIT_AVAILABLE_BYTES = 1536L * 1024L * 1024L
    private const val MIN_ENCODER_WARMUP_AVAILABLE_BYTES = 512L * 1024L * 1024L
  }

  private val loadMutex = Mutex()
  private val auditMutex = Mutex()
  private val encoderWarmupMutex = Mutex()
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

  @Volatile
  var processorRuntimeState: RuntimeModuleState = RuntimeModuleState.NOT_STARTED
    private set

  @Volatile
  var encoderRuntimeState: RuntimeModuleState = RuntimeModuleState.NOT_STARTED
    private set

  @Volatile
  var decoderRuntimeState: RuntimeModuleState = RuntimeModuleState.NOT_STARTED
    private set

  @Volatile
  private var closed: Boolean = false

  @Volatile
  private var cachedPhotoContext: CachedSamPhotoEmbedding? = null

  @Volatile
  internal var lastInferenceResultForTest: SlimSamInferenceResult? = null
    private set

  internal fun getCachedPhotoContextForTest(): CachedSamPhotoEmbedding? = cachedPhotoContext

  fun isReady(): Boolean =
    !closed &&
      status == ModelRuntimeStatus.READY &&
      modelState == SlimSamModelState.READY &&
      encoder.isLoaded() &&
      decoder.isLoaded() &&
      processor != null

  fun isClosed(): Boolean = closed && encoder.isClosed() && decoder.isClosed()

  suspend fun hasCachedEmbeddingForPhoto(photoKey: Long): Boolean = cacheMutex.withLock {
    cachedPhotoContext?.photoKey == photoKey
  }

  suspend fun clearPhotoEmbeddingCache() = cacheMutex.withLock {
    cachedPhotoContext = null
    lastInferenceResultForTest = null
  }

  suspend fun auditAssets(
    recorder: RecognitionPerformanceRecorder? = null,
  ): ModelAssetAuditReport = withContext(Dispatchers.IO) {
    auditMutex.withLock {
      auditReport?.let { return@withLock it }
      val startNs = SystemClock.elapsedRealtimeNanos()
      recorder?.beginWarmupStage("model_asset_audit")
      processorRuntimeState = RuntimeModuleState.ASSET_VERIFYING
      recorder?.updateModuleState("PROCESSOR", RuntimeModuleState.ASSET_VERIFYING)
      val report = ModelAssetAudit.runAudit(context.assets)
      auditReport = report
      recorder?.recordWarmupStageDuration(
        "model_asset_audit",
        startNs,
        report.auditEndMonoNs.coerceAtLeast(startNs),
        metrics = mapOf(
          "durationMs" to report.auditDurationMs,
          "sha256DurationMs" to report.sha256TotalDurationMs,
          "verifiedAssetCount" to listOf(
            report.yoloxAsset,
            report.slimSamEncoderAsset,
            report.slimSamDecoderAsset,
            report.slimSamProcessorAsset,
            report.slimSamConfigAsset,
            report.slimSamQuantizeConfigAsset,
          ).count { it.shaVerified },
        ),
      )
      report
    }
  }

  suspend fun ensureLoaded(
    cachedAuditReport: ModelAssetAuditReport? = null,
    recorder: RecognitionPerformanceRecorder? = null,
  ): ModelAssetAuditReport = withContext(Dispatchers.IO) {
    if (closed) {
      throw CancellationException("SlimSamSegmenter is closed")
    }
    val existingAudit = auditReport
    if (isReady() && existingAudit != null) {
      return@withContext existingAudit
    }
    loadMutex.withLock {
      if (closed) {
        throw CancellationException("SlimSamSegmenter is closed")
      }
      val cachedAudit = auditReport
      if (isReady() && cachedAudit != null) {
        return@withContext cachedAudit
      }

      modelState = SlimSamModelState.ASSET_VERIFYING
      status = ModelRuntimeStatus.WARMING_UP
      lastError = null
      lastErrorCode = null
      processorRuntimeState = RuntimeModuleState.ASSET_VERIFYING
      encoderRuntimeState = RuntimeModuleState.SESSION_LOADING
      decoderRuntimeState = RuntimeModuleState.SESSION_LOADING
      recorder?.updateSlimSamLoadState(ModelLoadLifecycleState.LOADING)

      val report = cachedAuditReport ?: auditAssets(recorder)
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
        processorRuntimeState = RuntimeModuleState.ERROR
        encoderRuntimeState = RuntimeModuleState.ERROR
        decoderRuntimeState = RuntimeModuleState.ERROR
        lastErrorCode = code
        lastError = buildString {
          if (report.missingAssets.isNotEmpty()) {
            append("Missing SlimSAM assets: ${report.missingAssets.joinToString(", ")}. ")
          }
          if (report.shaMismatchedAssets.isNotEmpty()) {
            append("SHA-256 mismatch: ${report.shaMismatchedAssets.joinToString(", ")}.")
          }
        }.trim()
        recorder?.updateModuleState("PROCESSOR", RuntimeModuleState.ERROR, lastErrorCode?.name)
        recorder?.updateModuleState("ENCODER", RuntimeModuleState.ERROR, lastErrorCode?.name)
        recorder?.updateModuleState("DECODER", RuntimeModuleState.ERROR, lastErrorCode?.name)
        return@withContext report
      }

      modelState = SlimSamModelState.ASSET_READY
      try {
        // Parse preprocessor_config.json strictly from local assets
        val processorStartNs = SystemClock.elapsedRealtimeNanos()
        processorRuntimeState = RuntimeModuleState.ASSET_VERIFYING
        val procJson = context.assets.open(ModelAssetAudit.SLIMSAM_PROCESSOR_PATH)
          .bufferedReader()
          .use { it.readText() }
        val processorConfig = SamPreprocessorConfig.fromJson(procJson)
        processorConfig.validateSlimSam77Contract()
        processor = SlimSamImageProcessor(processorConfig)
        val processorEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(processorStartNs)
        processorRuntimeState = RuntimeModuleState.PROCESSOR_READY
        recorder?.recordWarmupStageDuration(
          "processor_config_init",
          processorStartNs,
          processorEndNs,
          metrics = mapOf(
            "longestEdge" to processorConfig.longestEdge,
            "padWidth" to processorConfig.padWidth,
            "padHeight" to processorConfig.padHeight,
            "channelOrder" to "RGB",
            "resample" to processorConfig.resample,
            "resampleName" to "BILINEAR",
            "roundingRule" to "ImageCoordinateMapper.computeSamResizedDims",
            "padding" to "RIGHT_BOTTOM_ZERO_AFTER_NORMALIZE",
            "layout" to "NCHW",
          ),
        )
        recorder?.updateModuleState("PROCESSOR", RuntimeModuleState.PROCESSOR_READY)

        // Install ONNX models atomically into app filesDir with SHA-256 validation
        val onnxDir = File(context.filesDir, "models/slimsam/onnx")
        val installStartNs = SystemClock.elapsedRealtimeNanos()
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
        val installEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(installStartNs)
        recorder?.recordWarmupStageDuration(
          "slimsam_model_install_and_verify",
          installStartNs,
          installEndNs,
          metrics = mapOf(
            "encoderBytes" to encoderFile.length(),
            "decoderBytes" to decoderFile.length(),
          ),
        )

        modelState = SlimSamModelState.SESSION_LOADING
        encoderRuntimeState = RuntimeModuleState.SESSION_LOADING
        recorder?.updateModuleState("ENCODER", RuntimeModuleState.SESSION_LOADING)
        val encoderStartNs = SystemClock.elapsedRealtimeNanos()
        encoder.ensureLoaded(encoderFile)
        val encoderEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(encoderStartNs)
        encoderRuntimeState = RuntimeModuleState.SESSION_READY
        recorder?.recordWarmupStageDuration(
          "encoder_session_create",
          encoderStartNs,
          encoderEndNs,
          metrics = mapOf(
            "contract" to encoder.contractSummary.orEmpty(),
            "intraOpThreads" to encoder.intraOpThreadCount,
          ),
        )
        recorder?.updateModuleState("ENCODER", RuntimeModuleState.SESSION_READY)

        modelState = SlimSamModelState.SESSION_VALIDATING
        decoderRuntimeState = RuntimeModuleState.SESSION_LOADING
        recorder?.updateModuleState("DECODER", RuntimeModuleState.SESSION_LOADING)
        val decoderStartNs = SystemClock.elapsedRealtimeNanos()
        decoder.ensureLoaded(decoderFile)
        val decoderEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(decoderStartNs)
        decoderRuntimeState = RuntimeModuleState.INFERENCE_READY
        recorder?.recordWarmupStageDuration(
          "decoder_session_and_smoke",
          decoderStartNs,
          decoderEndNs,
          metrics = mapOf(
            "contract" to decoder.contractSummary.orEmpty(),
            "smokeCheck" to "PASS",
            "intraOpThreads" to decoder.intraOpThreadCount,
          ),
        )
        recorder?.updateModuleState("DECODER", RuntimeModuleState.INFERENCE_READY)

        modelState = SlimSamModelState.READY
        status = ModelRuntimeStatus.READY
        recorder?.updateSlimSamLoadState(
          ModelLoadLifecycleState.READY,
          totalLoadMs = (decoderEndNs - processorStartNs).coerceAtLeast(0L) / 1_000_000.0,
        )
        Log.i(
          TAG,
          "SlimSAM-77 Q8 CPU sessions READY | encoder=(${encoder.contractSummary}) | decoder=(${decoder.contractSummary})",
        )
      } catch (ce: CancellationException) {
        modelState = SlimSamModelState.NOT_READY
        status = ModelRuntimeStatus.NOT_LOADED
        encoderRuntimeState = if (encoder.isLoaded()) RuntimeModuleState.SESSION_READY else RuntimeModuleState.NOT_STARTED
        decoderRuntimeState = if (decoder.isLoaded()) RuntimeModuleState.INFERENCE_READY else RuntimeModuleState.NOT_STARTED
        throw ce
      } catch (sre: SlimSamRuntimeException) {
        modelState = SlimSamModelState.ERROR
        status = ModelRuntimeStatus.ERROR
        lastErrorCode = sre.errorCode
        lastError = sre.message
        if (processorRuntimeState != RuntimeModuleState.PROCESSOR_READY) {
          processorRuntimeState = RuntimeModuleState.ERROR
          recorder?.updateModuleState("PROCESSOR", RuntimeModuleState.ERROR, sre.errorCode.name)
        }
        if (encoderRuntimeState != RuntimeModuleState.SESSION_READY) {
          encoderRuntimeState = RuntimeModuleState.ERROR
          recorder?.updateModuleState("ENCODER", RuntimeModuleState.ERROR, sre.errorCode.name)
        }
        if (decoderRuntimeState != RuntimeModuleState.INFERENCE_READY) {
          decoderRuntimeState = RuntimeModuleState.ERROR
          recorder?.updateModuleState("DECODER", RuntimeModuleState.ERROR, sre.errorCode.name)
        }
        recorder?.updateSlimSamLoadState(ModelLoadLifecycleState.ERROR, errorCode = sre.errorCode.name)
        Log.e(TAG, "SlimSAM runtime init failed [${sre.errorCode}]: ${sre.message}", sre)
      } catch (t: Throwable) {
        modelState = SlimSamModelState.ERROR
        status = ModelRuntimeStatus.ERROR
        lastErrorCode = SlimSamErrorCode.SESSION_CREATE_FAILED
        lastError = t.message ?: t.javaClass.simpleName
        if (processorRuntimeState != RuntimeModuleState.PROCESSOR_READY) {
          processorRuntimeState = RuntimeModuleState.ERROR
          recorder?.updateModuleState("PROCESSOR", RuntimeModuleState.ERROR, "PROCESSOR_INIT_FAILED")
        }
        if (encoderRuntimeState != RuntimeModuleState.SESSION_READY) {
          encoderRuntimeState = RuntimeModuleState.ERROR
          recorder?.updateModuleState("ENCODER", RuntimeModuleState.ERROR, "ENCODER_SESSION_FAILED")
        }
        if (decoderRuntimeState != RuntimeModuleState.INFERENCE_READY) {
          decoderRuntimeState = RuntimeModuleState.ERROR
          recorder?.updateModuleState("DECODER", RuntimeModuleState.ERROR, "DECODER_SESSION_FAILED")
        }
        recorder?.updateSlimSamLoadState(ModelLoadLifecycleState.ERROR, errorCode = lastErrorCode?.name)
        Log.e(TAG, "SlimSAM init failed", t)
      }

      report
    }
  }

  /** Runs one isolated 1024x1024 Encoder inference before a user photo is selected. */
  suspend fun warmupEncoder(recorder: RecognitionPerformanceRecorder? = null): Boolean =
    withContext(Dispatchers.Default) {
      encoderWarmupMutex.withLock {
        if (encoderRuntimeState == RuntimeModuleState.INFERENCE_READY) return@withLock true
        if (!isReady() || encoderRuntimeState != RuntimeModuleState.SESSION_READY) return@withLock false

        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager?.getMemoryInfo(memoryInfo)
        val skipReason = when {
          activityManager == null -> "ACTIVITY_MANAGER_UNAVAILABLE"
          activityManager.isLowRamDevice -> "LOW_RAM_DEVICE"
          memoryInfo.lowMemory -> "SYSTEM_LOW_MEMORY"
          memoryInfo.availMem < MIN_ENCODER_WARMUP_AVAILABLE_BYTES -> "INSUFFICIENT_AVAILABLE_MEMORY"
          else -> null
        }
        if (skipReason != null) {
          encoderRuntimeState = RuntimeModuleState.SKIPPED_RESOURCE
          recorder?.recordWarmupStageDuration(
            "encoder_inference_warmup",
            SystemClock.elapsedRealtimeNanos(),
            SystemClock.elapsedRealtimeNanos(),
            status = StageOutcome.CANCELLED,
            errorCode = "PREWARM_SKIPPED_RESOURCE",
            metrics = mapOf("reason" to skipReason, "availableMemoryBytes" to memoryInfo.availMem),
          )
          recorder?.updateModuleState("ENCODER", RuntimeModuleState.SKIPPED_RESOURCE, skipReason)
          return@withLock false
        }

        encoderRuntimeState = RuntimeModuleState.KERNEL_WARMING
        recorder?.updateModuleState("ENCODER", RuntimeModuleState.KERNEL_WARMING)
        val startNs = SystemClock.elapsedRealtimeNanos()
        recorder?.beginWarmupStage("encoder_inference_warmup")
        var inputValues: FloatArray? = null
        var output: SamImageEmbedding? = null
        try {
          val syntheticInput = FloatArray(3 * SAM_INPUT_SIZE * SAM_INPUT_SIZE)
          inputValues = syntheticInput
          val warmupOutput = encoder.encode(syntheticInput)
          output = warmupOutput
          check(warmupOutput.embeddingsShape.contentEquals(longArrayOf(1L, 256L, 64L, 64L))) {
            "Encoder warmup image embedding shape mismatch: ${warmupOutput.embeddingsShape.contentToString()}"
          }
          check(warmupOutput.positionalShape.contentEquals(longArrayOf(1L, 256L, 64L, 64L))) {
            "Encoder warmup positional embedding shape mismatch: ${warmupOutput.positionalShape.contentToString()}"
          }
          syntheticInput.fill(0f)
          warmupOutput.imageEmbeddings.fill(0f)
          warmupOutput.imagePositionalEmbeddings.fill(0f)
          val endNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(startNs)
          encoderRuntimeState = RuntimeModuleState.INFERENCE_READY
          recorder?.recordWarmupStageDuration(
            "encoder_inference_warmup",
            startNs,
            endNs,
            metrics = mapOf(
              "tensorShape" to "[1,3,1024,1024]",
              "outputShape" to "[1,256,64,64]",
              "intraOpThreads" to encoder.intraOpThreadCount,
              "durationMs" to ((endNs - startNs) / 1_000_000.0),
            ),
          )
          recorder?.updateModuleState("ENCODER", RuntimeModuleState.INFERENCE_READY)
          true
        } catch (ce: CancellationException) {
          inputValues?.fill(0f)
          output?.imageEmbeddings?.fill(0f)
          output?.imagePositionalEmbeddings?.fill(0f)
          encoderRuntimeState = RuntimeModuleState.SESSION_READY
          recorder?.completeWarmupStage("encoder_inference_warmup", StageOutcome.CANCELLED, "CANCELLED")
          recorder?.updateModuleState("ENCODER", RuntimeModuleState.SESSION_READY)
          throw ce
        } catch (t: Throwable) {
          inputValues?.fill(0f)
          output?.imageEmbeddings?.fill(0f)
          output?.imagePositionalEmbeddings?.fill(0f)
          encoderRuntimeState = RuntimeModuleState.ERROR
          recorder?.completeWarmupStage(
            "encoder_inference_warmup",
            StageOutcome.FAILED,
            "ENCODER_WARMUP_FAILED",
          )
          recorder?.updateModuleState("ENCODER", RuntimeModuleState.ERROR, "ENCODER_WARMUP_FAILED")
          Log.e(TAG, "SlimSAM Encoder inference warmup failed", t)
          false
        }
      }
    }

  fun canInitializeSessionsInParallel(): Boolean {
    val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager ?: return false
    val memoryInfo = ActivityManager.MemoryInfo()
    activityManager.getMemoryInfo(memoryInfo)
    return !activityManager.isLowRamDevice && !memoryInfo.lowMemory &&
      memoryInfo.availMem >= MIN_PARALLEL_INIT_AVAILABLE_BYTES
  }

  suspend fun prepareOrReuseEmbedding(
    photoBitmap: Bitmap,
    photoKey: Long,
    forceReencode: Boolean = false,
    onStage: (SlimSamStage) -> Unit = {},
    recorder: RecognitionPerformanceRecorder? = null,
  ): Pair<CachedSamPhotoEmbedding, Boolean> = withContext(Dispatchers.Default) {
    ensureActive()
    if (closed) {
      throw CancellationException("SlimSamSegmenter is closed")
    }
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

    if (photoBitmap.isRecycled) {
      throw CancellationException("Photo bitmap recycled before SlimSAM preprocessing")
    }

    modelState = SlimSamModelState.INFERENCE_RUNNING
    try {
      val (metadata, emb) = preprocessAndEncode(activeProcessor, photoBitmap, onStage)
      ensureActive()
      if (closed) {
        throw CancellationException("SlimSamSegmenter closed before caching Vision Encoder output")
      }
      encoderRuntimeState = RuntimeModuleState.INFERENCE_READY
      recorder?.updateModuleState("ENCODER", RuntimeModuleState.INFERENCE_READY)

      val entry = CachedSamPhotoEmbedding(
        photoKey = photoKey,
        preprocessed = metadata,
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

  /**
   * Keep the 12 MiB 1024x1024 RGB input local to this call. Returning only metadata ensures it is
   * no longer reachable from the photo embedding cache after Encoder execution completes.
   */
  private suspend fun preprocessAndEncode(
    activeProcessor: SlimSamImageProcessor,
    photoBitmap: Bitmap,
    onStage: (SlimSamStage) -> Unit,
  ): Pair<SamPreprocessedMetadata, SamImageEmbedding> {
    val coroutineContext = currentCoroutineContext()
    coroutineContext.ensureActive()
    onStage(SlimSamStage.PREPROCESSING)
    val preprocessed = activeProcessor.preprocess(photoBitmap)
    coroutineContext.ensureActive()
    if (closed) {
      throw CancellationException("SlimSamSegmenter closed before Vision Encoder")
    }
    val metadata = SamPreprocessedMetadata.from(preprocessed)
    onStage(SlimSamStage.ENCODER)
    val embedding = encoder.encode(preprocessed.pixelValues)
    coroutineContext.ensureActive()
    return metadata to embedding
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
    if (closed) {
      throw CancellationException("SlimSamSegmenter is closed")
    }
    if (!isReady()) {
      throw ModelNotReadyException(
        missingAssets = auditReport?.missingAssets.orEmpty(),
        errorCode = lastErrorCode,
        message = lastError ?: "SlimSAM Q8 ONNX sessions are not ready.",
      )
    }

    modelState = SlimSamModelState.INFERENCE_RUNNING
    val decodeStartNs = SystemClock.elapsedRealtimeNanos()
    try {
      val promptMappingStartNs = SystemClock.elapsedRealtimeNanos()
      val mappedPrompts = mapper.mapPromptsToSamSpace(
        prompts = prompts,
        resizedWidth = cachedEmbedding.preprocessed.resizedWidth,
        resizedHeight = cachedEmbedding.preprocessed.resizedHeight,
      )
      val promptMappingEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(promptMappingStartNs)

      onStage(SlimSamStage.DECODER)
      val decoderOut = decoder.decode(
        embedding = cachedEmbedding.embedding,
        pointCoords = mappedPrompts.samPointCoords,
        pointLabels = mappedPrompts.samPointLabels,
        pointCount = mappedPrompts.pointCount,
      )
      ensureActive()

      onStage(SlimSamStage.POSTPROCESSING)
      val decodeEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(decodeStartNs)
      val decodeDurationMs = (decodeEndNs - decodeStartNs) / 1_000_000L
      val postStartNs = SystemClock.elapsedRealtimeNanos()
      val totalSamMs = (preprocessDurationMs + encoderDurationMs + decodeDurationMs)
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
      val postEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(postStartNs)
      val postMs = ((postEndNs - postStartNs) / 1_000_000L).coerceAtLeast(0L)

      modelState = SlimSamModelState.READY
      SlimSamInferenceResult(
        maskResult = maskResult,
        preprocessDurationMs = preprocessDurationMs,
        encoderDurationMs = encoderDurationMs,
        decoderDurationMs = decoderOut.decoderDurationMs,
        postprocessDurationMs = postMs,
        reusedCachedEmbedding = reusedEmbedding,
        decoderOutput = decoderOut,
        promptMappingStartNs = promptMappingStartNs,
        promptMappingEndNs = promptMappingEndNs,
        postprocessStartNs = postStartNs,
        postprocessEndNs = postEndNs,
      ).also { lastInferenceResultForTest = it }
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
      // Same-directory rename is atomic on Android/Linux and replaces a corrupt old cache safely.
      Os.rename(tmpFile.absolutePath, destFile.absolutePath)
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

  fun closeEncoder() {
    closed = true
    encoder.close()
  }

  fun closeDecoder() {
    closed = true
    decoder.close()
  }

  fun close() {
    closed = true
    cachedPhotoContext = null
    lastInferenceResultForTest = null
    encoder.close()
    decoder.close()
    status = ModelRuntimeStatus.NOT_LOADED
    modelState = SlimSamModelState.NOT_READY
    processorRuntimeState = RuntimeModuleState.NOT_STARTED
    encoderRuntimeState = RuntimeModuleState.NOT_STARTED
    decoderRuntimeState = RuntimeModuleState.NOT_STARTED
  }
}
