package com.yujian.fishmasklab.app

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.yujian.fishmasklab.domain.ContourData
import com.yujian.fishmasklab.domain.ContourPoint
import com.yujian.fishmasklab.domain.FishDetection
import com.yujian.fishmasklab.domain.FrozenContourConfig
import com.yujian.fishmasklab.domain.FrozenEffect02Config
import com.yujian.fishmasklab.domain.ImageCoordinateMapper
import com.yujian.fishmasklab.domain.MaskQualityGate
import com.yujian.fishmasklab.domain.PipelineState
import com.yujian.fishmasklab.domain.RecognitionErrorReason
import com.yujian.fishmasklab.runtime.PipelineProgressEvent
import com.yujian.fishmasklab.runtime.RecognitionPipelineCoordinator
import com.yujian.fishmasklab.runtime.RuntimeTelemetry
import com.yujian.fishmasklab.runtime.SlimSamSegmenter
import com.yujian.fishmasklab.runtime.YoloxFishDetector
import java.io.File
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class FishMaskUiState(
  val pipelineState: PipelineState = PipelineState.EMPTY,
  val photoGenerationId: Long = 0L,
  val displayBitmap: Bitmap? = null,
  val coordinateMapper: ImageCoordinateMapper? = null,
  val contourData: ContourData? = null,
  val errorReason: RecognitionErrorReason? = null,
  val diagnosticDetail: String? = null,
  val replayToken: Int = 0,
  val isCameraOpen: Boolean = false,
  val isSourceSheetVisible: Boolean = false,
  val cameraPermissionDenied: Boolean = false,
  val contourConfig: FrozenContourConfig = FrozenContourConfig(),
  val effect02Config: FrozenEffect02Config = FrozenEffect02Config(),
) {
  val isProcessing: Boolean
    get() = pipelineState == PipelineState.PHOTO_SELECTED ||
      pipelineState == PipelineState.MODEL_LOADING ||
      pipelineState == PipelineState.DETECTING ||
      pipelineState == PipelineState.YOLOX_RESULT_VALIDATION ||
      pipelineState == PipelineState.SAM_PREPROCESSING ||
      pipelineState == PipelineState.SAM_ENCODER ||
      pipelineState == PipelineState.PROMPT_BUILDING ||
      pipelineState == PipelineState.SAM_DECODER ||
      pipelineState == PipelineState.MASK_POSTPROCESSING ||
      pipelineState == PipelineState.MASK_CANDIDATE_RANKING ||
      pipelineState == PipelineState.MASK_QUALITY_GATE ||
      pipelineState == PipelineState.BUILDING_CONTOUR

  val isResultVisible: Boolean
    get() = pipelineState == PipelineState.PLAYING || pipelineState == PipelineState.RESULT
}

class FishMaskViewModel(application: Application) : AndroidViewModel(application) {

  private val yoloxDetector = YoloxFishDetector(application.applicationContext)
  private val slimSamSegmenter = SlimSamSegmenter(application.applicationContext)
  private val coordinator = RecognitionPipelineCoordinator(yoloxDetector, slimSamSegmenter)

  private val _uiState = MutableStateFlow(FishMaskUiState())
  val uiState: StateFlow<FishMaskUiState> = _uiState.asStateFlow()

  private val photoGenerationCounter = AtomicLong(0L)
  private var workBitmap: Bitmap? = null
  private var currentDetection: FishDetection? = null
  private var activeTaskId: Long = 0L
  private var loadJob: Job? = null

  init {
    // Non-blocking background model warmup & asset audit
    coordinator.warmupInBackground()

    viewModelScope.launch {
      coordinator.events.collect { event ->
        if (
          event is PipelineProgressEvent.StateChanged &&
          event.taskId == activeTaskId &&
          event.photoGenerationId == _uiState.value.photoGenerationId
        ) {
          if (event.detection != null) {
            currentDetection = event.detection
          }
          _uiState.update { curr ->
            curr.copy(
              pipelineState = event.state,
              contourData = if (event.state == PipelineState.ERROR) null else event.contourData ?: curr.contourData,
              errorReason = event.errorReason,
              diagnosticDetail = event.diagnosticMessage,
            )
          }
        }
      }
    }
  }

  fun openSourceSheet() {
    _uiState.update { it.copy(isSourceSheetVisible = true) }
  }

  fun dismissSourceSheet() {
    _uiState.update { it.copy(isSourceSheetVisible = false) }
  }

  fun openCamera() {
    _uiState.update {
      it.copy(
        isCameraOpen = true,
        isSourceSheetVisible = false,
        cameraPermissionDenied = false,
      )
    }
  }

  fun closeCamera() {
    _uiState.update { it.copy(isCameraOpen = false) }
  }

  fun onCameraPermissionDenied() {
    _uiState.update { it.copy(cameraPermissionDenied = true, isCameraOpen = false) }
  }

  fun dismissPermissionNotice() {
    _uiState.update { it.copy(cameraPermissionDenied = false) }
  }

  fun onPhotoPicked(uri: Uri) {
    loadAndRecognize { ExifPhotoLoader.loadFromUri(getApplication(), uri) }
  }

  fun onPhotoCaptured(file: File) {
    _uiState.update { it.copy(isCameraOpen = false) }
    loadAndRecognize { ExifPhotoLoader.loadFromFileAndDeleteTemp(file) }
  }

  private fun loadAndRecognize(loader: suspend () -> LoadedFishPhoto) {
    val generationId = photoGenerationCounter.incrementAndGet()
    val prevJob = coordinator.cancelCurrentTask()
    loadJob?.cancel()
    currentDetection = null

    val oldDisplay = _uiState.value.displayBitmap
    val oldWork = workBitmap
    workBitmap = null

    _uiState.update {
      it.copy(
        pipelineState = PipelineState.PHOTO_SELECTED,
        photoGenerationId = generationId,
        displayBitmap = null,
        coordinateMapper = null,
        contourData = null,
        errorReason = null,
        diagnosticDetail = null,
        isSourceSheetVisible = false,
        cameraPermissionDenied = false,
      )
    }

    loadJob = viewModelScope.launch {
      // Wait for any cancelled native ORT job to finish before releasing cached embeddings & Bitmaps
      runCatching { prevJob?.join() }
      slimSamSegmenter.clearPhotoEmbeddingCache()
      recycleSafely(oldDisplay, oldWork)

      try {
        val loaded = loader()
        if (photoGenerationCounter.get() != generationId) {
          recycleSafely(loaded.displayBitmap, loaded.workBitmap)
          return@launch
        }
        workBitmap = loaded.workBitmap
        _uiState.update {
          it.copy(
            pipelineState = PipelineState.PHOTO_SELECTED,
            photoGenerationId = generationId,
            displayBitmap = loaded.displayBitmap,
            coordinateMapper = loaded.mapper,
            contourData = null,
            errorReason = null,
            diagnosticDetail = null,
            replayToken = it.replayToken + 1,
          )
        }
        activeTaskId = coordinator.startRecognition(
          photoGenerationId = generationId,
          workBitmap = loaded.workBitmap,
          samBitmap = loaded.displayBitmap,
          mapper = loaded.mapper,
          tapPoint = null,
          existingDetection = null,
        )
      } catch (e: Exception) {
        if (photoGenerationCounter.get() == generationId) {
          _uiState.update {
            it.copy(
              pipelineState = PipelineState.ERROR,
              errorReason = RecognitionErrorReason.RECOGNITION_INCOMPLETE,
              diagnosticDetail = e.message,
            )
          }
        }
      }
    }
  }

  /**
   * Handles user tap on the photo in RESULT/PLAYING state.
   * Ignores taps outside the fitted photo rect (blank margins) or outside the fish focus region,
   * and reuses the cached Encoder embedding + current FishDetection without re-running YOLOX/Encoder.
   */
  fun onTapViewport(tapX: Float, tapY: Float, viewportWidth: Float, viewportHeight: Float) {
    val state = _uiState.value
    if (!state.isResultVisible) return
    val mapper = state.coordinateMapper ?: return
    val display = state.displayBitmap ?: return
    val work = workBitmap ?: return
    if (display.isRecycled || work.isRecycled) return

    val origPoint: ContourPoint =
      mapper.viewportTapToOriginalPixel(tapX, tapY, viewportWidth, viewportHeight) ?: return

    val detection = currentDetection
    if (!MaskQualityGate.isTapWithinFishFocusRegion(origPoint, detection, mapper.origWidth, mapper.origHeight)) {
      return
    }

    val generationId = state.photoGenerationId
    coordinator.cancelCurrentTask()
    _uiState.update {
      it.copy(
        pipelineState = PipelineState.PROMPT_BUILDING,
        contourData = null,
        errorReason = null,
        diagnosticDetail = null,
        replayToken = it.replayToken + 1,
      )
    }
    activeTaskId = coordinator.startRecognition(
      photoGenerationId = generationId,
      workBitmap = work,
      samBitmap = display,
      mapper = mapper,
      tapPoint = origPoint,
      existingDetection = detection,
    )
  }

  /**
   * Replays only the 0-2600ms+ visual animation timeline without re-running models or contour math.
   */
  fun onReplayClicked() {
    val state = _uiState.value
    if (!state.isResultVisible || state.contourData == null) return
    _uiState.update {
      it.copy(
        pipelineState = PipelineState.PLAYING,
        replayToken = it.replayToken + 1,
      )
    }
  }

  fun onTimelineInitialWindowCompleted() {
    _uiState.update { curr ->
      if (curr.pipelineState == PipelineState.PLAYING) {
        curr.copy(pipelineState = PipelineState.RESULT)
      } else {
        curr
      }
    }
  }

  fun onFpsSampled(fps: Float) {
    RuntimeTelemetry.capture(taskId = activeTaskId, shaderFps = fps)
  }

  fun cancelAndResetToEmpty() {
    photoGenerationCounter.incrementAndGet()
    val prevJob = coordinator.cancelCurrentTask()
    loadJob?.cancel()
    currentDetection = null
    val oldDisplay = _uiState.value.displayBitmap
    val oldWork = workBitmap
    workBitmap = null

    _uiState.update {
      FishMaskUiState(
        contourConfig = it.contourConfig,
        effect02Config = it.effect02Config,
      )
    }

    viewModelScope.launch {
      runCatching { prevJob?.join() }
      slimSamSegmenter.clearPhotoEmbeddingCache()
      recycleSafely(oldDisplay, oldWork)
    }
  }

  private fun recycleSafely(vararg bitmaps: Bitmap?) {
    for (bmp in bitmaps) {
      if (bmp != null && !bmp.isRecycled) {
        bmp.recycle()
      }
    }
  }

  override fun onCleared() {
    super.onCleared()
    loadJob?.cancel()
    coordinator.shutdown()
    val oldDisplay = _uiState.value.displayBitmap
    val oldWork = workBitmap
    workBitmap = null
    recycleSafely(oldDisplay, oldWork)
  }
}
