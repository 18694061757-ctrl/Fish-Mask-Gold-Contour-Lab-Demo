package com.yujian.fishmasklab.runtime

import android.graphics.Bitmap
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

/**
 * Orchestrates background model warmup, YOLOX detection, SlimSAM Q8 preprocessing/encoding/decoding,
 * mask post-processing, 3-candidate ranking & quality gating, contour extraction, and strict taskId/generationId cancellation.
 */
class RecognitionPipelineCoordinator(
  private val yoloxDetector: YoloxFishDetector,
  private val slimSamSegmenter: SlimSamSegmenter,
  private val contourConfig: FrozenContourConfig = FrozenContourConfig(),
) {
  companion object {
    private const val MODEL_WAIT_TIMEOUT_MS = 25_000L
    private const val INFERENCE_TIMEOUT_MS = 35_000L
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

  /**
   * Cancels the currently active recognition job (if any) and returns its [Job] handle
   * so callers can safely await completion before recycling shared Bitmaps.
   */
  fun cancelCurrentTask(): Job? {
    taskIdCounter.incrementAndGet()
    val job = activeJob
    job?.cancel()
    activeJob = null
    return job
  }

  fun startRecognition(
    photoGenerationId: Long,
    workBitmap: Bitmap,
    samBitmap: Bitmap = workBitmap,
    mapper: ImageCoordinateMapper,
    tapPoint: ContourPoint? = null,
    existingDetection: FishDetection? = null,
  ): Long {
    val taskId = taskIdCounter.incrementAndGet()
    activeJob?.cancel()

    activeJob = scope.launch {
      try {
        emitIfCurrent(taskId, photoGenerationId, PipelineState.PHOTO_SELECTED)
        ensureActive()

        // Step 1: Ensure models are loaded, verified, and contract-checked
        emitIfCurrent(taskId, photoGenerationId, PipelineState.MODEL_LOADING)
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
            photoGenerationId = photoGenerationId,
            reason = RecognitionErrorReason.MODEL_INIT_FAILED,
            detail = "YOLOX local ONNX asset missing or checksum mismatch: ${ModelAssetAudit.YOLOX_ASSET_PATH}",
          )
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
          existingDetection
        } else {
          emitIfCurrent(taskId, photoGenerationId, PipelineState.DETECTING)
          val (det, dur) = withTimeout(INFERENCE_TIMEOUT_MS) {
            yoloxDetector.detect(workBitmap, mapper)
          }
          detectorDurationMs = dur
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
        val contourStart = System.currentTimeMillis()

        for (candidate in sortedCandidates) {
          ensureActive()
          if (!MaskQualityGate.evaluateCandidateGeometryTrustworthiness(candidate, detection)) {
            continue
          }
          val candidateContour = ContourGeometry.extractContourFromMask(
            mask = candidate.data,
            width = maskResult.width,
            height = maskResult.height,
            simplificationTolerance = contourConfig.simplificationTolerance,
            smoothingPasses = contourConfig.smoothingPasses,
            glintOverlay = null,
            startTimeMs = contourStart,
          )
          if (MaskQualityGate.evaluateMaskTrustworthiness(candidate, candidateContour, detection)) {
            chosenCandidate = candidate
            chosenContour = candidateContour
            break
          }
        }
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
              contourDurationMs = (System.currentTimeMillis() - contourStart).coerceAtLeast(0L),
            ),
          )
          return@launch
        }

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
          cleanMask = chosenCandidate.data,
          maskWidth = maskResult.width,
          maskHeight = maskResult.height,
        )
        val finalContour = chosenContour.copy(
          internalGlint = glint,
          extractionTimeMs = (System.currentTimeMillis() - contourStart).coerceAtLeast(0L),
        )
        ensureActive()

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
            selectedCandidateIndex = chosenCandidate.index,
            isMaskTrustworthy = true,
          ),
          contourData = finalContour,
          telemetry = telemetry,
          reusedEncoderEmbedding = reusedEmbedding,
        )
      } catch (ce: CancellationException) {
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

    return taskId
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
    if (taskIdCounter.get() != taskId) return
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
    if (taskIdCounter.get() != taskId) return
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

  fun shutdown() {
    cancelCurrentTask()
    yoloxDetector.close()
    slimSamSegmenter.close()
  }
}
