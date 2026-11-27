package com.yujian.fishmasklab.runtime

import android.graphics.Bitmap
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
import java.util.concurrent.atomic.AtomicLong
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeout

sealed class PipelineProgressEvent {
  data class StateChanged(
    val taskId: Long,
    val state: PipelineState,
    val detection: FishDetection? = null,
    val maskResult: MaskResult? = null,
    val contourData: ContourData? = null,
    val errorReason: RecognitionErrorReason? = null,
    val diagnosticMessage: String? = null,
    val telemetry: RuntimePerformanceSnapshot? = null,
  ) : PipelineProgressEvent()
}

/**
 * Orchestrates background model warmup, YOLOX detection, SlimSAM segmentation,
 * contour geometry extraction, strict taskId validation, and task cancellation.
 */
class RecognitionPipelineCoordinator(
  private val yoloxDetector: YoloxFishDetector,
  private val slimSamSegmenter: SlimSamSegmenter,
  private val contourConfig: FrozenContourConfig = FrozenContourConfig(),
) {
  companion object {
    private const val MODEL_WAIT_TIMEOUT_MS = 15_000L
    private const val INFERENCE_TIMEOUT_MS = 20_000L
  }

  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
  private val taskIdCounter = AtomicLong(0L)
  private var activeJob: Job? = null

  private val _auditReport = MutableStateFlow<ModelAssetAuditReport?>(null)
  val auditReport: StateFlow<ModelAssetAuditReport?> = _auditReport.asStateFlow()

  private val _events = MutableStateFlow<PipelineProgressEvent?>(null)
  val events: StateFlow<PipelineProgressEvent?> = _events.asStateFlow()

  /**
   * Non-blocking startup warmup of local ONNX assets. Never blocks the EMPTY home screen.
   */
  fun warmupInBackground() {
    scope.launch {
      val report = slimSamSegmenter.ensureLoaded()
      _auditReport.value = report
      if (report.isYoloxReady) {
        yoloxDetector.ensureLoaded()
      }
      RuntimeTelemetry.capture(taskId = 0L)
    }
  }

  fun cancelCurrentTask() {
    taskIdCounter.incrementAndGet()
    activeJob?.cancel()
    activeJob = null
  }

  fun startRecognition(
    workBitmap: Bitmap,
    mapper: ImageCoordinateMapper,
    tapPoint: ContourPoint? = null,
  ): Long {
    val taskId = taskIdCounter.incrementAndGet()
    activeJob?.cancel()

    activeJob = scope.launch {
      try {
        emitIfCurrent(taskId, PipelineState.PHOTO_SELECTED)
        ensureActive()

        // Step 1: Ensure models are loaded / audited
        emitIfCurrent(taskId, PipelineState.MODEL_LOADING)
        val report = withTimeout(MODEL_WAIT_TIMEOUT_MS) {
          val samAudit = slimSamSegmenter.ensureLoaded()
          _auditReport.value = samAudit
          if (samAudit.isYoloxReady && !yoloxDetector.isReady()) {
            yoloxDetector.ensureLoaded()
          }
          samAudit
        }
        ensureActive()

        if (!report.isYoloxReady) {
          emitError(
            taskId = taskId,
            reason = RecognitionErrorReason.MODEL_INIT_FAILED,
            detail = "YOLOX local ONNX asset missing or checksum mismatch: ${ModelAssetAudit.YOLOX_ASSET_PATH}",
          )
          return@launch
        }

        // Step 2: Run YOLOX-Nano fish detection (unless tap-to-refocus specifies focal coordinates directly)
        emitIfCurrent(taskId, PipelineState.DETECTING)
        var detectorDurationMs: Long? = null
        val detection: FishDetection? = if (tapPoint == null) {
          val (det, dur) = withTimeout(INFERENCE_TIMEOUT_MS) {
            yoloxDetector.detect(workBitmap, mapper)
          }
          detectorDurationMs = dur
          det
        } else {
          null
        }
        ensureActive()

        if (tapPoint == null && detection == null) {
          emitError(
            taskId = taskId,
            reason = RecognitionErrorReason.NO_CLEAR_FISH,
            detail = "YOLOX-Nano found no fish candidate above threshold 0.20",
            telemetry = RuntimeTelemetry.capture(taskId, detectorDurationMs = detectorDurationMs),
          )
          return@launch
        }

        // Step 3: Check SlimSAM readiness BEFORE segmenting (never fake a mask or use BBox as mask)
        if (!report.isSlimSamReady || !slimSamSegmenter.isReady()) {
          val missingList = report.missingAssets.joinToString(", ")
          emitError(
            taskId = taskId,
            reason = RecognitionErrorReason.MODEL_INIT_FAILED,
            detection = detection,
            detail = "SlimSAM status is NOT_READY. Missing local ONNX assets: $missingList",
            telemetry = RuntimeTelemetry.capture(taskId, detectorDurationMs = detectorDurationMs),
          )
          return@launch
        }

        // Step 4: SlimSAM segmentation
        emitIfCurrent(taskId, PipelineState.SEGMENTING, detection = detection)
        val prompts = if (tapPoint != null) {
          MaskQualityGate.generatePromptsFromTap(
            tapX = tapPoint.x.toInt(),
            tapY = tapPoint.y.toInt(),
            imageWidth = mapper.origWidth,
            imageHeight = mapper.origHeight,
          )
        } else {
          MaskQualityGate.generatePromptsFromDetection(
            box = detection!!.box,
            imageWidth = mapper.origWidth,
            imageHeight = mapper.origHeight,
          )
        }

        val maskResult = withTimeout(INFERENCE_TIMEOUT_MS) {
          slimSamSegmenter.segment(workBitmap, mapper.origWidth, mapper.origHeight, prompts)
        }
        ensureActive()

        // Step 5: Build smooth contour geometry & verify quality gate
        emitIfCurrent(taskId, PipelineState.BUILDING_CONTOUR, detection = detection, maskResult = maskResult)
        val bestIndex = MaskQualityGate.selectBestCandidateIndex(
          candidates = maskResult.candidates,
          detection = detection,
          imageWidth = mapper.origWidth,
          imageHeight = mapper.origHeight,
        )
        val candidate = maskResult.candidates.getOrNull(bestIndex)
        val contourStart = System.currentTimeMillis()
        val glint = candidate?.let {
          InternalGlintExtractor.extract(
            sourceBitmap = workBitmap,
            cleanMask = it.data,
            maskWidth = maskResult.width,
            maskHeight = maskResult.height,
          )
        }
        val contour = candidate?.let {
          ContourGeometry.extractContourFromMask(
            mask = it.data,
            width = maskResult.width,
            height = maskResult.height,
            simplificationTolerance = contourConfig.simplificationTolerance,
            smoothingPasses = contourConfig.smoothingPasses,
            glintOverlay = glint,
            startTimeMs = contourStart,
          )
        }
        ensureActive()

        val trustworthy = MaskQualityGate.evaluateMaskTrustworthiness(candidate, contour, detection)
        if (!trustworthy || contour == null) {
          emitError(
            taskId = taskId,
            reason = RecognitionErrorReason.SEGMENTATION_FAILED,
            detection = detection,
            detail = "Candidate mask failed quality gate",
            telemetry = RuntimeTelemetry.capture(
              taskId = taskId,
              detectorDurationMs = detectorDurationMs,
              samDurationMs = maskResult.inferenceTimeMs,
              contourDurationMs = contour?.extractionTimeMs,
            ),
          )
          return@launch
        }

        val telemetry = RuntimeTelemetry.capture(
          taskId = taskId,
          detectorDurationMs = detectorDurationMs,
          samDurationMs = maskResult.inferenceTimeMs,
          contourDurationMs = contour.extractionTimeMs,
        )
        emitIfCurrent(
          taskId = taskId,
          state = PipelineState.PLAYING,
          detection = detection,
          maskResult = maskResult.copy(selectedCandidateIndex = bestIndex, isMaskTrustworthy = true),
          contourData = contour,
          telemetry = telemetry,
        )
      } catch (ce: CancellationException) {
        // Cancelled cleanly by replacement or back action; do not emit stale error
        throw ce
      } catch (mnr: ModelNotReadyException) {
        emitError(
          taskId = taskId,
          reason = RecognitionErrorReason.MODEL_INIT_FAILED,
          detail = mnr.message,
        )
      } catch (e: Exception) {
        emitError(
          taskId = taskId,
          reason = RecognitionErrorReason.RECOGNITION_INCOMPLETE,
          detail = e.message ?: e.javaClass.simpleName,
        )
      }
    }

    return taskId
  }

  private fun emitIfCurrent(
    taskId: Long,
    state: PipelineState,
    detection: FishDetection? = null,
    maskResult: MaskResult? = null,
    contourData: ContourData? = null,
    telemetry: RuntimePerformanceSnapshot? = null,
  ) {
    if (taskIdCounter.get() != taskId) return
    _events.value = PipelineProgressEvent.StateChanged(
      taskId = taskId,
      state = state,
      detection = detection,
      maskResult = maskResult,
      contourData = contourData,
      telemetry = telemetry,
    )
  }

  private fun emitError(
    taskId: Long,
    reason: RecognitionErrorReason,
    detection: FishDetection? = null,
    detail: String? = null,
    telemetry: RuntimePerformanceSnapshot? = null,
  ) {
    if (taskIdCounter.get() != taskId) return
    _events.value = PipelineProgressEvent.StateChanged(
      taskId = taskId,
      state = PipelineState.ERROR,
      detection = detection,
      errorReason = reason,
      diagnosticMessage = detail,
      telemetry = telemetry,
    )
  }

  fun shutdown() {
    cancelCurrentTask()
    yoloxDetector.close()
    slimSamSegmenter.close()
  }
}
