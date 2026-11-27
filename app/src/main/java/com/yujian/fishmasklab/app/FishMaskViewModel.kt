package com.yujian.fishmasklab.app

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.yujian.fishmasklab.domain.ContourData
import com.yujian.fishmasklab.domain.ContourPoint
import com.yujian.fishmasklab.domain.FrozenContourConfig
import com.yujian.fishmasklab.domain.FrozenEffect02Config
import com.yujian.fishmasklab.domain.ImageCoordinateMapper
import com.yujian.fishmasklab.domain.PipelineState
import com.yujian.fishmasklab.domain.RecognitionErrorReason
import com.yujian.fishmasklab.runtime.PipelineProgressEvent
import com.yujian.fishmasklab.runtime.RecognitionPipelineCoordinator
import com.yujian.fishmasklab.runtime.RuntimeTelemetry
import com.yujian.fishmasklab.runtime.SlimSamSegmenter
import com.yujian.fishmasklab.runtime.YoloxFishDetector
import java.io.File
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class FishMaskUiState(
  val pipelineState: PipelineState = PipelineState.EMPTY,
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
      pipelineState == PipelineState.SEGMENTING ||
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

  private var workBitmap: Bitmap? = null
  private var activeTaskId: Long = 0L
  private var loadJob: Job? = null

  init {
    // Non-blocking background model warmup & asset audit
    coordinator.warmupInBackground()

    viewModelScope.launch {
      coordinator.events.collect { event ->
        if (event is PipelineProgressEvent.StateChanged && event.taskId == activeTaskId) {
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
    // Immediately cancel previous recognition & clean up old mask/contour/animation state
    coordinator.cancelCurrentTask()
    loadJob?.cancel()

    val oldDisplay = _uiState.value.displayBitmap
    val oldWork = workBitmap
    workBitmap = null

    _uiState.update {
      it.copy(
        pipelineState = PipelineState.PHOTO_SELECTED,
        displayBitmap = null,
        coordinateMapper = null,
        contourData = null,
        errorReason = null,
        diagnosticDetail = null,
        isSourceSheetVisible = false,
        cameraPermissionDenied = false,
      )
    }

    if (oldDisplay != null && !oldDisplay.isRecycled) oldDisplay.recycle()
    if (oldWork != null && !oldWork.isRecycled) oldWork.recycle()

    loadJob = viewModelScope.launch {
      try {
        val loaded = loader()
        workBitmap = loaded.workBitmap
        _uiState.update {
          it.copy(
            pipelineState = PipelineState.PHOTO_SELECTED,
            displayBitmap = loaded.displayBitmap,
            coordinateMapper = loaded.mapper,
            contourData = null,
            errorReason = null,
            diagnosticDetail = null,
            replayToken = it.replayToken + 1,
          )
        }
        activeTaskId = coordinator.startRecognition(
          workBitmap = loaded.workBitmap,
          mapper = loaded.mapper,
          tapPoint = null,
        )
      } catch (e: Exception) {
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

  /**
   * Handles user tap on the photo in RESULT/PLAYING state.
   * Ignores taps outside the fitted photo rect (blank margins).
   */
  fun onTapViewport(tapX: Float, tapY: Float, viewportWidth: Float, viewportHeight: Float) {
    val state = _uiState.value
    if (!state.isResultVisible) return
    val mapper = state.coordinateMapper ?: return
    val work = workBitmap ?: return
    if (work.isRecycled) return

    val origPoint: ContourPoint =
      mapper.viewportTapToOriginalPixel(tapX, tapY, viewportWidth, viewportHeight) ?: return

    coordinator.cancelCurrentTask()
    _uiState.update {
      it.copy(
        pipelineState = PipelineState.SEGMENTING,
        contourData = null,
        errorReason = null,
        diagnosticDetail = null,
      )
    }
    activeTaskId = coordinator.startRecognition(
      workBitmap = work,
      mapper = mapper,
      tapPoint = origPoint,
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
    coordinator.cancelCurrentTask()
    loadJob?.cancel()
    val oldDisplay = _uiState.value.displayBitmap
    val oldWork = workBitmap
    workBitmap = null

    _uiState.update {
      FishMaskUiState(
        contourConfig = it.contourConfig,
        effect02Config = it.effect02Config,
      )
    }

    if (oldDisplay != null && !oldDisplay.isRecycled) oldDisplay.recycle()
    if (oldWork != null && !oldWork.isRecycled) oldWork.recycle()
  }

  override fun onCleared() {
    super.onCleared()
    loadJob?.cancel()
    coordinator.shutdown()
    val oldDisplay = _uiState.value.displayBitmap
    val oldWork = workBitmap
    workBitmap = null
    if (oldDisplay != null && !oldDisplay.isRecycled) oldDisplay.recycle()
    if (oldWork != null && !oldWork.isRecycled) oldWork.recycle()
  }
}
