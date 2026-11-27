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
import com.yujian.fishmasklab.runtime.ShutdownPhase
import com.yujian.fishmasklab.runtime.SlimSamSegmenter
import com.yujian.fishmasklab.runtime.YoloxFishDetector
import java.io.File
import java.util.Collections
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
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

class FishMaskViewModel internal constructor(
  application: Application,
  internal val yoloxDetector: YoloxFishDetector,
  internal val slimSamSegmenter: SlimSamSegmenter,
  internal val coordinator: RecognitionPipelineCoordinator = RecognitionPipelineCoordinator(yoloxDetector, slimSamSegmenter),
) : AndroidViewModel(application) {

  constructor(application: Application) : this(
    application = application,
    yoloxDetector = YoloxFishDetector(application.applicationContext),
    slimSamSegmenter = SlimSamSegmenter(application.applicationContext),
  )

  private val _uiState = MutableStateFlow(FishMaskUiState())
  val uiState: StateFlow<FishMaskUiState> = _uiState.asStateFlow()

  private val cleanupScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
  private val pendingCleanupJobs = Collections.synchronizedList(mutableListOf<Job>())
  private val shutdownLock = Any()
  private var shutdownJob: Job? = null

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

  internal fun loadPhotoWithLoaderForTest(loader: suspend () -> LoadedFishPhoto) {
    loadAndRecognize(loader)
  }

  private fun loadAndRecognize(loader: suspend () -> LoadedFishPhoto) {
    if (coordinator.isShutdownRequested()) return
    val generationId = photoGenerationCounter.incrementAndGet()
    val prevJob = coordinator.cancelCurrentTask()
    val prevLoadJob = loadJob
    prevLoadJob?.cancel()
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
      // Wait for any cancelled load or native ORT job to finish before releasing cached embeddings & Bitmaps
      runCatching { prevLoadJob?.join() }
      runCatching { prevJob?.join() }
      slimSamSegmenter.clearPhotoEmbeddingCache()
      recycleSafely(oldDisplay, oldWork)

      ensureActive()
      if (photoGenerationCounter.get() != generationId || coordinator.isShutdownRequested()) {
        return@launch
      }

      try {
        val loaded = loader()
        if (!isActive || photoGenerationCounter.get() != generationId || coordinator.isShutdownRequested()) {
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
        val startedTaskId = coordinator.startRecognition(
          photoGenerationId = generationId,
          workBitmap = loaded.workBitmap,
          samBitmap = loaded.displayBitmap,
          mapper = loaded.mapper,
          tapPoint = null,
          existingDetection = null,
        )
        if (startedTaskId < 0L) {
          workBitmap = null
          _uiState.update { curr -> curr.copy(displayBitmap = null, coordinateMapper = null) }
          recycleSafely(loaded.displayBitmap, loaded.workBitmap)
          return@launch
        }
        activeTaskId = startedTaskId
      } catch (e: Exception) {
        if (isActive && photoGenerationCounter.get() == generationId && !coordinator.isShutdownRequested()) {
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
    if (coordinator.isShutdownRequested()) return
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
    val startedTaskId = coordinator.startRecognition(
      photoGenerationId = generationId,
      workBitmap = work,
      samBitmap = display,
      mapper = mapper,
      tapPoint = origPoint,
      existingDetection = detection,
    )
    if (startedTaskId < 0L) return
    activeTaskId = startedTaskId
    _uiState.update {
      it.copy(
        pipelineState = PipelineState.PROMPT_BUILDING,
        contourData = null,
        errorReason = null,
        diagnosticDetail = null,
        replayToken = it.replayToken + 1,
      )
    }
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
    val prevLoadJob = loadJob
    prevLoadJob?.cancel()
    loadJob = null
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

    if (coordinator.isShutdownRequested()) {
      return
    }

    val job = cleanupScope.launch {
      runCatching { prevLoadJob?.join() }
      runCatching { prevJob?.join() }
      slimSamSegmenter.clearPhotoEmbeddingCache()
      recycleSafely(oldDisplay, oldWork)
    }
    pendingCleanupJobs.add(job)
    job.invokeOnCompletion { pendingCleanupJobs.remove(job) }
  }

  private val bitmapRecycleLock = Any()

  private fun recycleSafely(vararg bitmaps: Bitmap?) {
    synchronized(bitmapRecycleLock) {
      for (bmp in bitmaps) {
        if (bmp != null && !bmp.isRecycled) {
          bmp.recycle()
        }
      }
    }
  }

  internal fun shutdownAsync(): Job {
    synchronized(shutdownLock) {
      shutdownJob?.let { return it }
      photoGenerationCounter.incrementAndGet()
      coordinator.beginShutdown()
      val pendingLoadJob = loadJob
      pendingLoadJob?.cancel()
      loadJob = null
      currentDetection = null
      val oldDisplay = _uiState.value.displayBitmap
      val oldWork = workBitmap
      workBitmap = null
      _uiState.update {
        it.copy(
          displayBitmap = null,
          coordinateMapper = null,
          contourData = null,
        )
      }

      val job = cleanupScope.launch {
        try {
          coordinator.awaitAllNativeOperations()
          val resetCleanups = synchronized(pendingCleanupJobs) {
            pendingCleanupJobs.toList().also { pendingCleanupJobs.clear() }
          }
          resetCleanups.forEach { runCatching { it.join() } }
          runCatching { pendingLoadJob?.join() }
          coordinator.recordPhase(ShutdownPhase.WAIT_VIEWMODEL_LOAD_JOB)
          coordinator.closeNativeResources(
            recycleBitmaps = {
              recycleSafely(oldDisplay)
              coordinator.recordPhase(ShutdownPhase.RECYCLE_DISPLAY_BITMAP)
              recycleSafely(oldWork)
              coordinator.recordPhase(ShutdownPhase.RECYCLE_WORK_BITMAP)
            },
          )
        } finally {
          cleanupScope.cancel()
        }
      }
      shutdownJob = job
      return job
    }
  }

  internal suspend fun clearForTestAndAwait() {
    onCleared()
    shutdownAsync().join()
  }

  override fun onCleared() {
    super.onCleared()
    shutdownAsync()
  }
}
