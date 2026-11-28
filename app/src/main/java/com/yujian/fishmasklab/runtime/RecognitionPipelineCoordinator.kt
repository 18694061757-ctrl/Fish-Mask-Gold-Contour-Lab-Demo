package com.yujian.fishmasklab.runtime

import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.yujian.fishmasklab.domain.CandidateMask
import com.yujian.fishmasklab.domain.ContourData
import com.yujian.fishmasklab.domain.ContourGeometry
import com.yujian.fishmasklab.domain.ContourPoint
import com.yujian.fishmasklab.domain.FishDetection
import com.yujian.fishmasklab.domain.FrozenContourConfig
import com.yujian.fishmasklab.domain.ImageCoordinateMapper
import com.yujian.fishmasklab.domain.MaskQualityGate
import com.yujian.fishmasklab.domain.MaskResult
import com.yujian.fishmasklab.domain.PipelineState
import com.yujian.fishmasklab.domain.RecognitionErrorReason
import java.util.Collections
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

sealed class PipelineProgressEvent {
  data class StateChanged(
    val taskId: Long,
    val photoGenerationId: Long,
    val state: PipelineState,
    val detection: FishDetection? = null,
    val maskResult: MaskResult? = null,
    val contourData: ContourData? = null,
    val errorReason: RecognitionErrorReason? = null,
    val diagnosticMessage: String? = null,
    val telemetry: RuntimePerformanceSnapshot? = null,
    val reusedEncoderEmbedding: Boolean = false,
  ) : PipelineProgressEvent()
}

data class RuntimeStageSpan(
  val stageName: String,
  val startNs: Long,
  val endNs: Long,
  val metrics: Map<String, Any> = emptyMap(),
)

/**
 * Orchestrates background model warmup, YOLOX detection, SlimSAM Q8 preprocessing/encoding/decoding,
 * mask post-processing, 3-candidate ranking & quality gating, contour extraction, and strict taskId/generationId cancellation.
 */
class RecognitionPipelineCoordinator(
  private val yoloxDetector: YoloxFishDetector,
  private val slimSamSegmenter: SlimSamSegmenter,
  private val contourConfig: FrozenContourConfig = FrozenContourConfig(),
  internal val performanceRecorder: RecognitionPerformanceRecorder? = null,
) {
  companion object {
    private const val MODEL_WAIT_TIMEOUT_MS = 25_000L
    private const val INFERENCE_TIMEOUT_MS = 35_000L
  }

  private val scopeJob = SupervisorJob()
  private val scope = CoroutineScope(scopeJob + Dispatchers.Default)
  private val taskIdCounter = AtomicLong(0L)
  private val performanceSessions = ConcurrentHashMap<Long, String>()
  private val performanceSessionStartNs = ConcurrentHashMap<Long, Long>()
  private val animationStartNsByTask = ConcurrentHashMap<Long, Long>()
  private val taskLock = Any()
  private val modelLoadMutex = Mutex()
  private val modelInitializationLock = Any()
  private val warmupLock = Any()
  private val shutdownStarted = AtomicBoolean(false)
  private val nativeClosed = AtomicBoolean(false)
  private var activeJob: Job? = null

  @Volatile
  private var warmupJob: Job? = null

  /** Shared independently from the cancellable dummy warmup so a photo pick never restarts Session creation. */
  @Volatile
  private var modelInitializationJob: Deferred<ModelAssetAuditReport>? = null

  @Volatile
  private var shutdownCompleteFlag: Boolean = false

  @Volatile
  var currentShutdownPhase: ShutdownPhase = ShutdownPhase.RUNNING
    private set

  private val _lifecycleLog = Collections.synchronizedList(mutableListOf(ShutdownPhase.RUNNING.name))
  val lifecycleLog: List<String>
    get() = synchronized(_lifecycleLog) { _lifecycleLog.toList() }

  private val _auditReport = MutableStateFlow<ModelAssetAuditReport?>(null)
  val auditReport: StateFlow<ModelAssetAuditReport?> = _auditReport.asStateFlow()

  private val _events = MutableStateFlow<PipelineProgressEvent?>(null)
  val events: StateFlow<PipelineProgressEvent?> = _events.asStateFlow()

  fun isShutdownRequested(): Boolean = shutdownStarted.get()

  fun isShutdownComplete(): Boolean = shutdownCompleteFlag

  internal fun recordPhase(phase: ShutdownPhase) {
    currentShutdownPhase = phase
    synchronized(_lifecycleLog) {
      if (_lifecycleLog.lastOrNull() != phase.name) {
        _lifecycleLog.add(phase.name)
      }
    }
  }

  /**
   * Non-blocking startup warmup of local ONNX assets. Never blocks the EMPTY home screen.
   */
  fun warmupInBackground() {
    if (shutdownStarted.get()) return
    synchronized(warmupLock) {
      if (warmupJob?.isActive == true) return
      warmupJob = scope.launch {
        val warmupStartNs = SystemClock.elapsedRealtimeNanos()
        performanceRecorder?.markWarmupStarted()
        performanceRecorder?.beginWarmupStage("startup_warmup_total")
        try {
          performanceRecorder?.captureResourceSample("startup_warmup_start", 0L, 0L)
          val report = ensureModelsLoaded()
          if (shutdownStarted.get()) {
            performanceRecorder?.completeWarmupStage(
              "startup_warmup_total",
              StageOutcome.CANCELLED,
              "APP_SHUTDOWN_DURING_WARMUP",
            )
            return@launch
          }
          if (report.isYoloxReady && yoloxDetector.isReady()) {
            yoloxDetector.warmupInference(performanceRecorder)
          } else {
            performanceRecorder?.updateModuleState("YOLOX", RuntimeModuleState.ERROR, "YOLOX_SESSION_NOT_READY")
          }
          if (slimSamSegmenter.isReady()) {
            slimSamSegmenter.warmupEncoder(performanceRecorder)
          } else {
            performanceRecorder?.updateModuleState("ENCODER", RuntimeModuleState.ERROR, "SLIMSAM_SESSION_NOT_READY")
          }
          val endNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(warmupStartNs)
          performanceRecorder?.captureResourceSample("startup_warmup_end", 0L, 0L)
          performanceRecorder?.recordWarmupStageDuration(
            "startup_warmup_total",
            warmupStartNs,
            endNs,
            metrics = mapOf(
              "allModulesInferenceReady" to allModulesInferenceReady(),
              "yoloxState" to yoloxDetector.runtimeModuleState.name,
              "processorState" to slimSamSegmenter.processorRuntimeState.name,
              "encoderState" to slimSamSegmenter.encoderRuntimeState.name,
              "decoderState" to slimSamSegmenter.decoderRuntimeState.name,
            ),
          )
          RuntimeTelemetry.capture(taskId = 0L)
        } catch (ce: CancellationException) {
          performanceRecorder?.completeWarmupStage(
            "startup_warmup_total",
            StageOutcome.CANCELLED,
            if (shutdownStarted.get()) "APP_SHUTDOWN_DURING_WARMUP" else "PREEMPTED_BY_PHOTO_SELECTION",
          )
          throw ce
        } catch (t: Throwable) {
          performanceRecorder?.updateModuleState("STARTUP", RuntimeModuleState.ERROR, t.message ?: t.javaClass.simpleName)
          performanceRecorder?.completeWarmupStage("startup_warmup_total", StageOutcome.FAILED, "STARTUP_WARMUP_FAILED")
          Log.e("RecognitionPipelineCoordinator", "Startup model warmup failed", t)
        }
      }
    }
  }

  private suspend fun ensureModelsLoaded(): ModelAssetAuditReport {
    val initialization = synchronized(modelInitializationLock) {
      modelInitializationJob?.takeUnless { it.isCancelled }
        ?: scope.async { initializeModelsOnce() }.also { modelInitializationJob = it }
    }
    return try {
      val report = initialization.await()
      if (!report.isYoloxReady || !report.isSlimSamReady ||
        !yoloxDetector.isReady() || !slimSamSegmenter.isReady()
      ) {
        synchronized(modelInitializationLock) {
          if (modelInitializationJob === initialization) modelInitializationJob = null
        }
      }
      report
    } catch (t: Throwable) {
      synchronized(modelInitializationLock) {
        if (modelInitializationJob === initialization && initialization.isCancelled) {
          modelInitializationJob = null
        }
      }
      throw t
    }
  }

  private suspend fun initializeModelsOnce(): ModelAssetAuditReport = modelLoadMutex.withLock {
    val audit = slimSamSegmenter.auditAssets(performanceRecorder)
    _auditReport.value = audit
    val environmentBefore = DeviceEnvironmentInfo.capture()
    val memoryGatePassed = slimSamSegmenter.canInitializeSessionsInParallel()
    val parallelInitialization = audit.isSlimSamReady && audit.isYoloxReady &&
      memoryGatePassed
    val stageStartNs = SystemClock.elapsedRealtimeNanos()
    val initStartSample = performanceRecorder?.captureResourceSample("model_session_init_start", 0L, 0L)

    if (parallelInitialization) {
      coroutineScope {
        val samLoad = async { slimSamSegmenter.ensureLoaded(audit, performanceRecorder) }
        val yoloxLoad = async { yoloxDetector.ensureLoaded(audit, performanceRecorder) }
        samLoad.await()
        yoloxLoad.await()
      }
    } else {
      slimSamSegmenter.ensureLoaded(audit, performanceRecorder)
      yoloxDetector.ensureLoaded(audit, performanceRecorder)
    }

    val stageEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(stageStartNs)
    val initEndSample = performanceRecorder?.captureResourceSample("model_session_init_end", 0L, 0L)
    performanceRecorder?.recordWarmupStageDuration(
      "model_session_initialization",
      stageStartNs,
      stageEndNs,
      metrics = mapOf(
        "mode" to if (parallelInitialization) "PARALLEL" else "SEQUENTIAL",
        "parallelInitEligible" to parallelInitialization,
        "memoryGatePassed" to memoryGatePassed,
        "parallelInitMinimumAvailableMemoryBytes" to SlimSamSegmenter.MIN_PARALLEL_INIT_AVAILABLE_BYTES,
        "slimSamEncoderDecoderMode" to "SEQUENTIAL",
        "encoderDecoderConcurrencyBenchmark" to "NOT_RUN_WITHOUT_DEVICE_REPEATS",
        "availableMemoryBeforeKb" to (initStartSample?.availableMemoryKb ?: environmentBefore.availableMemoryKb),
        "pssBeforeKb" to (initStartSample?.pssTotalKb ?: environmentBefore.pssTotalKb),
        "pssAfterKb" to (initEndSample?.pssTotalKb ?: DeviceEnvironmentInfo.capture().pssTotalKb),
      ),
    )
    audit
  }

  private fun allModulesInferenceReady(): Boolean =
    yoloxDetector.runtimeModuleState == RuntimeModuleState.INFERENCE_READY &&
      slimSamSegmenter.processorRuntimeState == RuntimeModuleState.PROCESSOR_READY &&
      slimSamSegmenter.encoderRuntimeState == RuntimeModuleState.INFERENCE_READY &&
      slimSamSegmenter.decoderRuntimeState == RuntimeModuleState.INFERENCE_READY

  private fun areModulesInferenceReady(states: Map<String, RuntimeModuleState>): Boolean =
    states["YOLOX"] == RuntimeModuleState.INFERENCE_READY &&
      states["PROCESSOR"] == RuntimeModuleState.PROCESSOR_READY &&
      states["ENCODER"] == RuntimeModuleState.INFERENCE_READY &&
      states["DECODER"] == RuntimeModuleState.INFERENCE_READY

  private fun attachWarmupPickSnapshot(
    sessionId: String?,
    statesAtPick: Map<String, RuntimeModuleState>,
    loadStatesAtPick: Pair<ModelLoadLifecycleState, ModelLoadLifecycleState>,
    warmupAtPick: GlobalWarmupRecord?,
    photoLoadSpans: List<RuntimeStageSpan>,
    photoPickStartNs: Long,
    modelWaitMs: Double,
    modelWaitEndNs: Long,
  ) {
    val notReady = listOf(
      "YOLOX" to RuntimeModuleState.INFERENCE_READY,
      "PROCESSOR" to RuntimeModuleState.PROCESSOR_READY,
      "ENCODER" to RuntimeModuleState.INFERENCE_READY,
      "DECODER" to RuntimeModuleState.INFERENCE_READY,
    ).filter { (module, requiredState) -> statesAtPick[module] != requiredState }
      .map { (module, _) -> "$module:${statesAtPick[module]?.name ?: RuntimeModuleState.NOT_STARTED.name}" }

    val photoLoadStartNs = photoLoadSpans.minOfOrNull { it.startNs } ?: photoPickStartNs
    val photoLoadEndNs = photoLoadSpans.maxOfOrNull { it.endNs } ?: photoPickStartNs
    val warmupStartNs = warmupAtPick?.warmupStartNs ?: 0L
    val warmupEndNs = warmupAtPick?.allModelsReadyNs ?: modelWaitEndNs
    val overlapNs = if (warmupStartNs > 0L && photoLoadEndNs > photoLoadStartNs) {
      (minOf(warmupEndNs, photoLoadEndNs) - maxOf(warmupStartNs, photoLoadStartNs)).coerceAtLeast(0L)
    } else {
      0L
    }

    performanceRecorder?.attachWarmupSnapshot(
      sessionId,
      WarmupTimingSnapshot(
        yoloxStateAtPick = loadStatesAtPick.first,
        slimSamStateAtPick = loadStatesAtPick.second,
        allModelsReadyAtPick = areModulesInferenceReady(statesAtPick),
        notReadyModelsAtPick = notReady,
        modelWaitTimeMs = modelWaitMs,
        modelWaitParallelWithPhotoLoad = overlapNs > 0L,
        warmupOverlapWithPhotoLoadMs = RecognitionPerformanceRecorder.nanosToMs(overlapNs),
        warmupTotalMs = warmupAtPick?.warmupTotalMs,
        slimSamTotalLoadMs = warmupAtPick?.slimSamTotalLoadMs,
        yoloxTotalLoadMs = warmupAtPick?.yoloxTotalLoadMs,
        moduleStatesAtPick = statesAtPick,
      ),
    )
  }

  /**
   * Cancels the currently active recognition job (if any) and returns its [Job] handle
   * so callers can safely await completion before recycling shared Bitmaps.
   */
  fun cancelCurrentTask(): Job? = synchronized(taskLock) {
    taskIdCounter.incrementAndGet()
    val job = activeJob
    job?.cancel()
    job
  }

  fun startRecognition(
    photoGenerationId: Long,
    workBitmap: Bitmap,
    samBitmap: Bitmap = workBitmap,
    mapper: ImageCoordinateMapper,
    tapPoint: ContourPoint? = null,
    existingDetection: FishDetection? = null,
    photoPickStartNs: Long = SystemClock.elapsedRealtimeNanos(),
    photoLoadStageSpans: List<RuntimeStageSpan> = emptyList(),
  ): Long {
    if (shutdownStarted.get()) {
      return -1L
    }
    return synchronized(taskLock) {
      if (shutdownStarted.get()) {
        return@synchronized -1L
      }
      val taskId = taskIdCounter.incrementAndGet()
      val prevJob = activeJob
      prevJob?.cancel()
      val pendingWarmup = warmupJob
      val moduleStatesAtPick = performanceRecorder?.captureModuleStates().orEmpty()
      val loadStatesAtPick = performanceRecorder?.captureWarmupPickState()
        ?: (ModelLoadLifecycleState.NOT_STARTED to ModelLoadLifecycleState.NOT_STARTED)
      val warmupRecordAtPick = performanceRecorder?.currentWarmupRecord()
      pendingWarmup?.cancel()

      val newJob = scope.launch {
        runCatching { pendingWarmup?.cancelAndJoin() }
        try {
          runCatching { prevJob?.cancelAndJoin() }
          ensureActive()
          if (shutdownStarted.get() || taskIdCounter.get() != taskId) {
            return@launch
          }
          val performanceSessionId = performanceRecorder?.startSession(
            taskId = taskId,
            photoGenerationId = photoGenerationId,
            mode = if (tapPoint == null) "INITIAL_RECOGNITION" else "TAP_REFOCUS",
            startProfile = when {
              tapPoint != null -> "TAP_REFOCUS"
              areModulesInferenceReady(moduleStatesAtPick) -> "WARM_AFTER_READY"
              else -> "COLD_START"
            },
            startTimestampNs = photoPickStartNs,
            modelVersions = mapOf(
              "YOLOX" to ModelAssetAudit.YOLOX_EXPECTED_SHA256,
              "SlimSAM" to ModelAssetAudit.SLIMSAM_REVISION,
            ),
          )
          if (performanceSessionId != null) {
            performanceSessions[taskId] = performanceSessionId
            performanceSessionStartNs[taskId] = photoPickStartNs
            performanceRecorder?.recordStageSpan(
              performanceSessionId,
              "photo_selected",
              photoPickStartNs,
              photoPickStartNs,
              metrics = mapOf(
                "photoGenerationId" to photoGenerationId,
                "modelVersion" to "YOLOX:${ModelAssetAudit.YOLOX_EXPECTED_SHA256};SlimSAM:${ModelAssetAudit.SLIMSAM_REVISION}",
                "appVersion" to com.example.BuildConfig.VERSION_NAME,
                "deviceModel" to android.os.Build.MODEL.orEmpty(),
                "androidApiLevel" to android.os.Build.VERSION.SDK_INT,
              ),
            )
            photoLoadStageSpans.forEach { span ->
              performanceRecorder?.recordStageSpan(
                performanceSessionId,
                span.stageName,
                span.startNs,
                span.endNs,
                metrics = span.metrics,
              )
            }
          }
          emitIfCurrent(taskId, photoGenerationId, PipelineState.PHOTO_SELECTED)
          ensureActive()

        // Step 1: Ensure models are loaded, verified, and contract-checked
        emitIfCurrent(taskId, photoGenerationId, PipelineState.MODEL_LOADING)
        val modelWaitStartNs = SystemClock.elapsedRealtimeNanos()
        val report = withTimeout(MODEL_WAIT_TIMEOUT_MS) { ensureModelsLoaded() }
        val modelWaitEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(modelWaitStartNs)
        performanceRecorder?.recordStageSpan(
          performanceSessionId,
          "model_session_wait_after_photo_decode",
          modelWaitStartNs,
          modelWaitEndNs,
          metrics = mapOf(
            "yoloxSessionReady" to yoloxDetector.isReady(),
            "slimSamSessionsReady" to slimSamSegmenter.isReady(),
          ),
        )
        attachWarmupPickSnapshot(
          sessionId = performanceSessionId,
          statesAtPick = moduleStatesAtPick,
          loadStatesAtPick = loadStatesAtPick,
          warmupAtPick = warmupRecordAtPick,
          photoLoadSpans = photoLoadStageSpans,
          photoPickStartNs = photoPickStartNs,
          modelWaitMs = RecognitionPerformanceRecorder.nanosToMs(modelWaitEndNs - modelWaitStartNs),
          modelWaitEndNs = modelWaitEndNs,
        )
        ensureActive()

        if (!report.isYoloxReady) {
          emitError(
            taskId = taskId,
            photoGenerationId = photoGenerationId,
            reason = RecognitionErrorReason.MODEL_INIT_FAILED,
            detail = "YOLOX local ONNX asset missing or checksum mismatch: ${ModelAssetAudit.YOLOX_ASSET_PATH}",
          )
          performanceRecorder?.captureResourceSample("models_ready_after_photo_pick", taskId, photoGenerationId)
          return@launch
        }

        if (!report.isSlimSamReady || !slimSamSegmenter.isReady()) {
          val detail = slimSamSegmenter.lastError
            ?: "SlimSAM Q8 status is ${slimSamSegmenter.modelState} (${slimSamSegmenter.lastErrorCode})"
          emitError(
            taskId = taskId,
            photoGenerationId = photoGenerationId,
            reason = RecognitionErrorReason.MODEL_INIT_FAILED,
            detail = detail,
          )
          return@launch
        }

        // Step 2: YOLOX-Nano Fish Detection (skipped during tap-to-refocus when current detection & embedding exist)
        var detectorDurationMs: Long? = null
        val isTapRefocus = tapPoint != null &&
          existingDetection != null &&
          slimSamSegmenter.hasCachedEmbeddingForPhoto(photoGenerationId)

        val detection: FishDetection? = if (isTapRefocus) {
          val reusedAtNs = SystemClock.elapsedRealtimeNanos()
          performanceRecorder?.recordStageSpan(
            performanceSessionId,
            "yolox_skipped_embedding_reuse",
            reusedAtNs,
            reusedAtNs,
            metrics = mapOf("reusedDetection" to true),
          )
          existingDetection
        } else {
          val yoloxStartNs = SystemClock.elapsedRealtimeNanos()
          performanceRecorder?.recordStageSpan(
            performanceSessionId,
            "photo_selection_to_yolox_start",
            photoPickStartNs,
            yoloxStartNs,
            metrics = mapOf("modulesReadyAtPick" to areModulesInferenceReady(moduleStatesAtPick)),
          )
          performanceRecorder?.captureResourceSample("before_yolox_inference", taskId, photoGenerationId)
          emitIfCurrent(taskId, photoGenerationId, PipelineState.DETECTING)
          val outcome = withTimeout(INFERENCE_TIMEOUT_MS) {
            // Use the canonical display/original-coordinate bitmap for detection. The work bitmap is
            // pre-downscaled to 1024px and would introduce a second resize before YOLOX, while SAM,
            // the mapper, and the Web reference all use the canonical photo dimensions.
            yoloxDetector.detectDetailed(samBitmap, mapper)
          }
          val det = outcome.detection
          detectorDurationMs = outcome.totalDurationMs
          outcome.stages.forEach { stage ->
            performanceRecorder?.recordStageSpan(
              performanceSessionId,
              stage.stageName,
              stage.startNs,
              stage.endNs,
              metrics = mapOf(
                "rawCandidateCount" to outcome.rawCandidateCount,
                "userSelectionToInferenceStartMs" to RecognitionPerformanceRecorder.nanosToMs(stage.startNs - photoPickStartNs),
              ),
            )
          }
          performanceRecorder?.recordStageSpan(
            performanceSessionId,
            "yolox_detection_result",
            outcome.totalStartNs,
            outcome.totalEndNs,
            metrics = mapOf(
              "rawCandidateCount" to outcome.rawCandidateCount,
              "postNmsCandidateCount" to outcome.postNmsCandidateCount,
              "hasDetection" to (det != null),
              "confidence" to (det?.confidence ?: 0f),
              "bbox" to (det?.box?.let { listOf(it.x1, it.y1, it.x2, it.y2) } ?: emptyList<Int>()),
            ),
          )
          performanceRecorder?.captureResourceSample("after_yolox_inference", taskId, photoGenerationId)
          performanceRecorder?.updateModuleState("YOLOX", RuntimeModuleState.INFERENCE_READY)
          ensureActive()

          emitIfCurrent(
            taskId = taskId,
            photoGenerationId = photoGenerationId,
            state = PipelineState.YOLOX_RESULT_VALIDATION,
            detection = det,
          )
          det
        }
        ensureActive()

        if (detection == null) {
          emitError(
            taskId = taskId,
            photoGenerationId = photoGenerationId,
            reason = RecognitionErrorReason.NO_CLEAR_FISH,
            detail = "YOLOX-Nano found no fish candidate above threshold 0.20",
            telemetry = RuntimeTelemetry.capture(taskId, detectorDurationMs = detectorDurationMs),
          )
          return@launch
        }

        // Step 3: SlimSAM Preprocessing & Vision Encoder (reused on tap-to-refocus)
        val (cachedEmbedding, reusedEmbedding) = withTimeout(INFERENCE_TIMEOUT_MS) {
          slimSamSegmenter.prepareOrReuseEmbedding(
            photoBitmap = samBitmap,
            photoKey = photoGenerationId,
            forceReencode = false,
            recorder = performanceRecorder,
            onStage = { stage ->
              when (stage) {
                SlimSamStage.PREPROCESSING ->
                  emitIfCurrent(taskId, photoGenerationId, PipelineState.SAM_PREPROCESSING, detection = detection)
                SlimSamStage.ENCODER ->
                  emitIfCurrent(taskId, photoGenerationId, PipelineState.SAM_ENCODER, detection = detection)
                else -> Unit
              }
            },
          )
        }
        ensureActive()

        val timingEndNs = SystemClock.elapsedRealtimeNanos()
        val preprocessStartNs = if (reusedEmbedding) timingEndNs else cachedEmbedding.preprocessed.preprocessStartNs
        val preprocessEndNs = if (reusedEmbedding) timingEndNs else cachedEmbedding.preprocessed.preprocessEndNs
        performanceRecorder?.recordStageSpan(
          performanceSessionId,
          "slimsam_processor",
          preprocessStartNs,
          preprocessEndNs,
          metrics = mapOf(
            "reusedEmbedding" to reusedEmbedding,
            "inputTensorBytes" to if (reusedEmbedding) 0L else 3L * 1024L * 1024L * 4L,
            "tensorRetainedAfterEncoder" to false,
            "originalWidth" to cachedEmbedding.preprocessed.originalWidth,
            "originalHeight" to cachedEmbedding.preprocessed.originalHeight,
            "resizedWidth" to cachedEmbedding.preprocessed.resizedWidth,
            "resizedHeight" to cachedEmbedding.preprocessed.resizedHeight,
            "padWidth" to cachedEmbedding.preprocessed.padWidth,
            "padHeight" to cachedEmbedding.preprocessed.padHeight,
          ),
        )
        val encoderStartNs = if (reusedEmbedding) timingEndNs else cachedEmbedding.embedding.encoderStartNs
        val encoderEndNs = if (reusedEmbedding) timingEndNs else cachedEmbedding.embedding.encoderEndNs
        performanceRecorder?.recordStageSpan(
          performanceSessionId,
          "slimsam_encoder",
          encoderStartNs,
          encoderEndNs,
          metrics = mapOf(
            "reusedEmbedding" to reusedEmbedding,
            "inputTensorBytes" to if (reusedEmbedding) 0L else 3L * 1024L * 1024L * 4L,
            "embeddingCacheBytes" to 2L * SamImageEmbedding.ELEMENT_COUNT * 4L,
          ),
        )
        performanceRecorder?.recordStageSpan(
          performanceSessionId,
          "encoder_input_tensor_release",
          encoderEndNs,
          SystemClock.elapsedRealtimeNanos(),
          metrics = mapOf(
            "tensorBytes" to 3L * 1024L * 1024L * 4L,
            "retainedByPhotoCache" to false,
            "releaseMode" to "LOCAL_REFERENCE_ENDS_AFTER_ENCODER_CALL; JVM_HEAP_RECLAMATION_GC_MANAGED",
            "embeddingCacheBytes" to 2L * SamImageEmbedding.ELEMENT_COUNT * 4L,
          ),
        )
        performanceRecorder?.updateModuleState("PROCESSOR", slimSamSegmenter.processorRuntimeState)
        performanceRecorder?.updateModuleState("ENCODER", slimSamSegmenter.encoderRuntimeState)
        performanceRecorder?.captureResourceSample("after_slimsamm_encoder", taskId, photoGenerationId)

        // Step 4: Build closed-loop prompts in original image space -> SAM space
        emitIfCurrent(
          taskId = taskId,
          photoGenerationId = photoGenerationId,
          state = PipelineState.PROMPT_BUILDING,
          detection = detection,
        )
        val prompts = if (tapPoint != null) {
          MaskQualityGate.generatePromptsFromTap(
            tapX = tapPoint.x.toInt(),
            tapY = tapPoint.y.toInt(),
            imageWidth = mapper.origWidth,
            imageHeight = mapper.origHeight,
          )
        } else {
          MaskQualityGate.generatePromptsFromDetection(
            box = detection.box,
            imageWidth = mapper.origWidth,
            imageHeight = mapper.origHeight,
          )
        }

        // Step 5: SlimSAM Decoder + Mask Post-processing
        val samOutcome = withTimeout(INFERENCE_TIMEOUT_MS) {
          slimSamSegmenter.decodeAndPostProcess(
            cachedEmbedding = cachedEmbedding,
            mapper = mapper,
            prompts = prompts,
            detection = detection,
            reusedEmbedding = reusedEmbedding,
            onStage = { stage ->
              when (stage) {
                SlimSamStage.DECODER ->
                  emitIfCurrent(
                    taskId = taskId,
                    photoGenerationId = photoGenerationId,
                    state = PipelineState.SAM_DECODER,
                    detection = detection,
                    reusedEncoderEmbedding = reusedEmbedding,
                  )
                SlimSamStage.POSTPROCESSING ->
                  emitIfCurrent(
                    taskId = taskId,
                    photoGenerationId = photoGenerationId,
                    state = PipelineState.MASK_POSTPROCESSING,
                    detection = detection,
                    reusedEncoderEmbedding = reusedEmbedding,
                  )
                else -> Unit
              }
            },
          )
        }
        ensureActive()
        performanceRecorder?.recordStageSpan(
          performanceSessionId,
          "slimsam_prompt_mapping",
          samOutcome.promptMappingStartNs,
          samOutcome.promptMappingEndNs,
          metrics = mapOf("reusedEmbedding" to reusedEmbedding),
        )
        performanceRecorder?.captureResourceSample("after_slimsamm_decoder_postprocess", taskId, photoGenerationId)
        samOutcome.decoderOutput?.let { decoderOutput ->
          performanceRecorder?.recordStageSpan(
            performanceSessionId,
            "slimsam_decoder",
            decoderOutput.decoderStartNs,
            decoderOutput.decoderEndNs,
            metrics = mapOf("candidateCount" to SamDecoderOutput.NUM_CANDIDATES),
          )
        }
        performanceRecorder?.recordStageSpan(
          performanceSessionId,
          "mask_postprocess",
          samOutcome.postprocessStartNs,
          samOutcome.postprocessEndNs,
          metrics = mapOf("durationMs" to samOutcome.postprocessDurationMs),
        )

        // Step 6: Evaluate ALL 3 candidates (Score = 0.4*IoU + 0.3*Coverage + 0.3*Span) + Quality Gate
        val maskResult = samOutcome.maskResult
        emitIfCurrent(
          taskId = taskId,
          photoGenerationId = photoGenerationId,
          state = PipelineState.MASK_CANDIDATE_RANKING,
          detection = detection,
          maskResult = maskResult,
          reusedEncoderEmbedding = reusedEmbedding,
        )

        val sortedCandidates = maskResult.candidates.sortedByDescending { it.rankScore }
        emitIfCurrent(
          taskId = taskId,
          photoGenerationId = photoGenerationId,
          state = PipelineState.MASK_QUALITY_GATE,
          detection = detection,
          maskResult = maskResult,
          reusedEncoderEmbedding = reusedEmbedding,
        )

        var chosenCandidate: CandidateMask? = null
        var chosenContour: ContourData? = null
        val contourStartNs = System.nanoTime()
        val contourStageStartNs = SystemClock.elapsedRealtimeNanos()
        var contourExclusiveNs = 0L
        var qualityGateExclusiveNs = 0L

        for (candidate in sortedCandidates) {
          ensureActive()
          val geometryGateStartNs = SystemClock.elapsedRealtimeNanos()
          val geometryTrusted = MaskQualityGate.evaluateCandidateGeometryTrustworthiness(candidate, detection)
          val geometryGateEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(geometryGateStartNs)
          qualityGateExclusiveNs += geometryGateEndNs - geometryGateStartNs
          performanceRecorder?.recordStageSpan(
            performanceSessionId,
            "mask_quality_gate_geometry_${candidate.index}",
            geometryGateStartNs,
            geometryGateEndNs,
          )
          if (!geometryTrusted) {
            continue
          }
          val contourCandidateStartNs = SystemClock.elapsedRealtimeNanos()
          val contourCandidateStartNano = System.nanoTime()
          val candidateContour = ContourGeometry.extractContourFromMask(
            mask = candidate.data,
            width = maskResult.width,
            height = maskResult.height,
            simplificationTolerance = contourConfig.simplificationTolerance,
            smoothingPasses = contourConfig.smoothingPasses,
            glintOverlay = null,
            startTimeNs = contourCandidateStartNano,
          )
          val contourCandidateEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(contourCandidateStartNs)
          contourExclusiveNs += contourCandidateEndNs - contourCandidateStartNs
          performanceRecorder?.recordStageSpan(
            performanceSessionId,
            "contour_extract_candidate_${candidate.index}",
            contourCandidateStartNs,
            contourCandidateEndNs,
          )
          val trustGateStartNs = SystemClock.elapsedRealtimeNanos()
          val maskTrusted = MaskQualityGate.evaluateMaskTrustworthiness(candidate, candidateContour, detection)
          val trustGateEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(trustGateStartNs)
          qualityGateExclusiveNs += trustGateEndNs - trustGateStartNs
          performanceRecorder?.recordStageSpan(
            performanceSessionId,
            "mask_quality_gate_candidate_${candidate.index}",
            trustGateStartNs,
            trustGateEndNs,
          )
          if (maskTrusted) {
            chosenCandidate = candidate
            chosenContour = candidateContour
            break
          }
        }
        val contourStageEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(contourStageStartNs)
        performanceRecorder?.recordStageSpan(
          performanceSessionId,
          "contour_extraction_total",
          contourStageStartNs,
          contourStageEndNs,
          metrics = mapOf("exclusiveDurationMs" to RecognitionPerformanceRecorder.nanosToMs(contourExclusiveNs)),
        )
        performanceRecorder?.recordStageSpan(
          performanceSessionId,
          "mask_quality_gate_total",
          contourStageStartNs,
          contourStageEndNs,
          metrics = mapOf("exclusiveDurationMs" to RecognitionPerformanceRecorder.nanosToMs(qualityGateExclusiveNs)),
        )
        ensureActive()

        if (chosenCandidate == null || chosenContour == null) {
          emitError(
            taskId = taskId,
            photoGenerationId = photoGenerationId,
            reason = RecognitionErrorReason.SEGMENTATION_FAILED,
            detection = detection,
            detail = "All 3 SlimSAM candidates failed quality gate (covRatio/span/iou/contour)",
            telemetry = RuntimeTelemetry.capture(
              taskId = taskId,
              detectorDurationMs = detectorDurationMs,
              samDurationMs = maskResult.inferenceTimeMs,
              contourDurationMs = ((System.nanoTime() - contourStartNs) / 1_000_000L).coerceAtLeast(0L),
            ),
          )
          return@launch
        }

        val chosen = requireNotNull(chosenCandidate)
        val contour = requireNotNull(chosenContour)
        val candidateTargetArea = maxOf(1, detection.box.area).toFloat()
        performanceRecorder?.recordStageSpan(
          performanceSessionId,
          "mask_candidate_selected",
          contourStageStartNs,
          SystemClock.elapsedRealtimeNanos(),
          metrics = mapOf(
            "candidateCount" to maskResult.candidates.size,
            "selectedCandidateIndex" to chosen.index,
            "iouScore" to chosen.iouScore,
            "pixelCount" to chosen.pixelCount,
            "coverageRatio" to (chosen.pixelCount / candidateTargetArea),
            "rankScore" to chosen.rankScore,
            "areaRatio" to chosen.areaRatio,
            "contourClosed" to contour.traceClosed,
            "contourPointCount" to contour.rawPointCount,
          ),
        )

        // Step 7: Build final contour with natural specular glint overlay
        emitIfCurrent(
          taskId = taskId,
          photoGenerationId = photoGenerationId,
          state = PipelineState.BUILDING_CONTOUR,
          detection = detection,
          maskResult = maskResult,
          reusedEncoderEmbedding = reusedEmbedding,
        )
        val glint = InternalGlintExtractor.extract(
          sourceBitmap = samBitmap,
          cleanMask = chosen.data,
          maskWidth = maskResult.width,
          maskHeight = maskResult.height,
        )
        val finalContour = contour.copy(
          internalGlint = glint,
          extractionTimeMs = ((System.nanoTime() - contourStartNs) / 1_000_000L).coerceAtLeast(0L),
        )
        ensureActive()

        val contourReadyNs = SystemClock.elapsedRealtimeNanos()
        performanceRecorder?.recordStageSpan(
          performanceSessionId,
          "contour_ready",
          contourReadyNs,
          contourReadyNs,
          metrics = mapOf(
            "candidateIndex" to chosen.index,
            "rawPointCount" to finalContour.rawPointCount,
            "closed" to finalContour.traceClosed,
          ),
        )
        val resultVisibleNs = SystemClock.elapsedRealtimeNanos()
        val photoDecodeMs = photoLoadStageSpans
          .filter { it.stageName == "photo_load_total" }
          .sumOf { RecognitionPerformanceRecorder.nanosToMs(it.endNs - it.startNs) }
        val inferenceMs = (detectorDurationMs ?: 0L) + samOutcome.preprocessDurationMs +
          samOutcome.encoderDurationMs + samOutcome.decoderDurationMs + samOutcome.postprocessDurationMs
        performanceRecorder?.attachSummaryMetrics(
          performanceSessionId,
          RecognitionSummaryMetrics(
            modelWarmupTimeMs = warmupRecordAtPick?.warmupTotalMs,
            modelWaitTimeMs = RecognitionPerformanceRecorder.nanosToMs(modelWaitEndNs - modelWaitStartNs),
            photoDecodeTimeMs = photoDecodeMs,
            yoloxTimeMs = detectorDurationMs?.toDouble() ?: 0.0,
            samProcessorTimeMs = samOutcome.preprocessDurationMs.toDouble(),
            samEncoderTimeMs = samOutcome.encoderDurationMs.toDouble(),
            samDecoderTimeMs = samOutcome.decoderDurationMs.toDouble(),
            maskPostprocessTimeMs = samOutcome.postprocessDurationMs.toDouble(),
            inferenceTimeMs = inferenceMs.toDouble(),
            contourTimeMs = finalContour.extractionTimeMs.toDouble(),
            recognitionPipelineTimeMs = RecognitionPerformanceRecorder.nanosToMs(resultVisibleNs - modelWaitStartNs),
            firstResultTimeMs = RecognitionPerformanceRecorder.nanosToMs(resultVisibleNs - photoPickStartNs),
            animationTimeMs = null,
            fullExperienceTimeMs = null,
            reusedEncoderEmbedding = reusedEmbedding,
            shaderFps = null,
          ),
        )

        val telemetry = RuntimeTelemetry.capture(
          taskId = taskId,
          detectorDurationMs = detectorDurationMs,
          samDurationMs = maskResult.inferenceTimeMs,
          contourDurationMs = finalContour.extractionTimeMs,
        )
        emitIfCurrent(
          taskId = taskId,
          photoGenerationId = photoGenerationId,
          state = PipelineState.PLAYING,
          detection = detection,
          maskResult = maskResult.copy(
            selectedCandidateIndex = chosen.index,
            isMaskTrustworthy = true,
          ),
          contourData = finalContour,
          telemetry = telemetry,
          reusedEncoderEmbedding = reusedEmbedding,
        )
      } catch (timeout: TimeoutCancellationException) {
        emitError(
          taskId = taskId,
          photoGenerationId = photoGenerationId,
          reason = RecognitionErrorReason.RECOGNITION_INCOMPLETE,
          detail = "Recognition stage timed out after ${timeout.message.orEmpty()}",
        )
      } catch (ce: CancellationException) {
        performanceRecorder?.finishSession(
          performanceSessions.remove(taskId),
          StageOutcome.CANCELLED,
          "TASK_CANCELLED",
        )
        performanceSessionStartNs.remove(taskId)
        animationStartNsByTask.remove(taskId)
        throw ce
      } catch (mnr: ModelNotReadyException) {
        emitError(
          taskId = taskId,
          photoGenerationId = photoGenerationId,
          reason = RecognitionErrorReason.MODEL_INIT_FAILED,
          detail = mnr.message,
        )
      } catch (sre: SlimSamRuntimeException) {
        emitError(
          taskId = taskId,
          photoGenerationId = photoGenerationId,
          reason = when (sre.errorCode) {
            SlimSamErrorCode.ASSET_MISSING,
            SlimSamErrorCode.ASSET_HASH_MISMATCH,
            SlimSamErrorCode.SESSION_CREATE_FAILED,
            SlimSamErrorCode.TENSOR_CONTRACT_MISMATCH,
            SlimSamErrorCode.UNSUPPORTED_OPERATOR,
            -> RecognitionErrorReason.MODEL_INIT_FAILED
            SlimSamErrorCode.ENCODER_FAILED,
            SlimSamErrorCode.DECODER_FAILED,
            SlimSamErrorCode.POSTPROCESS_FAILED,
            -> RecognitionErrorReason.SEGMENTATION_FAILED
          },
          detail = "[${sre.errorCode}] ${sre.message}",
        )
      } catch (e: Exception) {
        emitError(
          taskId = taskId,
          photoGenerationId = photoGenerationId,
          reason = RecognitionErrorReason.RECOGNITION_INCOMPLETE,
          detail = e.message ?: e.javaClass.simpleName,
        )
      }
      }
      activeJob = newJob
      taskId
    }
  }

  private fun emitIfCurrent(
    taskId: Long,
    photoGenerationId: Long,
    state: PipelineState,
    detection: FishDetection? = null,
    maskResult: MaskResult? = null,
    contourData: ContourData? = null,
    telemetry: RuntimePerformanceSnapshot? = null,
    reusedEncoderEmbedding: Boolean = false,
  ) {
    if (shutdownStarted.get() || taskIdCounter.get() != taskId) return
    if (state == PipelineState.PLAYING) {
      val visibleNs = SystemClock.elapsedRealtimeNanos()
      val sessionId = performanceSessions[taskId]
      animationStartNsByTask[taskId] = visibleNs
      performanceRecorder?.recordStageSpan(
        sessionId,
        "result_visible",
        visibleNs,
        visibleNs,
        metrics = mapOf("photoGenerationId" to photoGenerationId),
      )
      performanceRecorder?.finishSession(sessionId, StageOutcome.SUCCESS, endTimestampNs = visibleNs)
      performanceRecorder?.captureResourceSample("result_visible", taskId, photoGenerationId)
    }
    _events.value = PipelineProgressEvent.StateChanged(
      taskId = taskId,
      photoGenerationId = photoGenerationId,
      state = state,
      detection = detection,
      maskResult = maskResult,
      contourData = contourData,
      telemetry = telemetry,
      reusedEncoderEmbedding = reusedEncoderEmbedding,
    )
  }

  private fun emitError(
    taskId: Long,
    photoGenerationId: Long,
    reason: RecognitionErrorReason,
    detection: FishDetection? = null,
    detail: String? = null,
    telemetry: RuntimePerformanceSnapshot? = null,
  ) {
    if (shutdownStarted.get() || taskIdCounter.get() != taskId) return
    performanceRecorder?.finishSession(
      performanceSessions.remove(taskId),
      StageOutcome.FAILED,
      errorCode = reason.name,
    )
    performanceSessionStartNs.remove(taskId)
    animationStartNsByTask.remove(taskId)
    _events.value = PipelineProgressEvent.StateChanged(
      taskId = taskId,
      photoGenerationId = photoGenerationId,
      state = PipelineState.ERROR,
      detection = detection,
      errorReason = reason,
      diagnosticMessage = detail,
      telemetry = telemetry,
    )
  }

  /** Completes animation timing after the first result animation window has actually finished. */
  fun recordAnimationCompletion(taskId: Long, shaderFps: Float? = null): Boolean {
    val sessionId = performanceSessions.remove(taskId) ?: return false
    val animationStart = animationStartNsByTask.remove(taskId) ?: return false
    val photoPickStart = performanceSessionStartNs.remove(taskId) ?: animationStart
    performanceRecorder?.updateAnimationCompletion(
      sessionId = sessionId,
      animationStartNs = animationStart,
      animationEndNs = SystemClock.elapsedRealtimeNanos(),
      shaderFps = shaderFps,
      photoPickStartNs = photoPickStart,
    )
    return true
  }

  /**
   * Idempotently initiates shutdown: stops accepting new tasks and cancels active recognition jobs
   * without prematurely closing native ONNX sessions while native calls may still be running.
   */
  fun beginShutdown() {
    if (shutdownStarted.compareAndSet(false, true)) {
      recordPhase(ShutdownPhase.SHUTDOWN_REQUESTED)
      recordPhase(ShutdownPhase.STOP_ACCEPTING_NEW_TASKS)
      cancelCurrentTask()
      recordPhase(ShutdownPhase.CANCEL_ACTIVE_RECOGNITION)
      scopeJob.cancel()
    }
  }

  /**
   * Suspends until all coroutines in the coordinator scope (including any in-flight `OrtSession.run`)
   * have completely exited.
   */
  suspend fun awaitAllNativeOperations() {
    beginShutdown()
    withContext(NonCancellable) {
      scopeJob.join()
    }
    recordPhase(ShutdownPhase.WAIT_ALL_NATIVE_OPERATIONS)
  }

  /**
   * Closes cached embeddings and native ONNX sessions in strict deterministic order after
   * [awaitAllNativeOperations] completes, then optionally recycles Bitmaps and marks completion.
   */
  suspend fun closeNativeResources(
    recycleBitmaps: (() -> Unit)? = null,
  ) {
    if (nativeClosed.compareAndSet(false, true)) {
      withContext(NonCancellable) {
        slimSamSegmenter.clearPhotoEmbeddingCache()
        recordPhase(ShutdownPhase.CLEAR_ENCODER_EMBEDDING_CACHE)

        yoloxDetector.close()
        recordPhase(ShutdownPhase.CLOSE_YOLOX_SESSION)

        slimSamSegmenter.closeEncoder()
        recordPhase(ShutdownPhase.CLOSE_SLIMSAM_ENCODER_SESSION)

        slimSamSegmenter.closeDecoder()
        recordPhase(ShutdownPhase.CLOSE_SLIMSAM_DECODER_SESSION)

        slimSamSegmenter.close()

        recycleBitmaps?.invoke()

        shutdownCompleteFlag = true
        recordPhase(ShutdownPhase.SHUTDOWN_COMPLETE)
      }
    }
  }

  suspend fun shutdownAndAwait(
    recycleBitmaps: (() -> Unit)? = null,
  ) {
    beginShutdown()
    awaitAllNativeOperations()
    closeNativeResources(recycleBitmaps = recycleBitmaps)
  }

  fun shutdown() {
    beginShutdown()
    runBlocking(Dispatchers.IO) {
      awaitAllNativeOperations()
      closeNativeResources()
    }
  }
}
