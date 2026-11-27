package com.yujian.fishmasklab.runtime

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import android.system.Os
import android.util.Log
import com.yujian.fishmasklab.domain.FishDetection
import com.yujian.fishmasklab.domain.ImageCoordinateMapper
import com.yujian.fishmasklab.domain.PixelBox
import java.io.File
import java.nio.FloatBuffer
import java.security.MessageDigest
import kotlin.coroutines.cancellation.CancellationException
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToLong
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

data class YoloxLoadMetrics(
  val onnxFileCopyStartNs: Long,
  val onnxFileCopyEndNs: Long,
  val sha256VerifyStartNs: Long,
  val sha256VerifyEndNs: Long,
  val reusedAuditResult: Boolean,
  val ortEnvInitStartNs: Long,
  val ortEnvInitEndNs: Long,
  val sessionCreateStartNs: Long,
  val sessionCreateEndNs: Long,
  val sessionReadyNs: Long,
  val totalStartNs: Long,
  val totalEndNs: Long,
  val executionProvider: String,
)

data class YoloxStageSpan(
  val stageName: String,
  val startNs: Long,
  val endNs: Long,
) {
  val durationMs: Double
    get() = (endNs - startNs).coerceAtLeast(0L) / 1_000_000.0
}

data class YoloxDetectionOutcome(
  val detection: FishDetection?,
  val totalDurationMs: Long,
  val exactDurationMs: Double,
  val totalStartNs: Long,
  val totalEndNs: Long,
  val rawCandidateCount: Int,
  val postNmsCandidateCount: Int,
  val stages: List<YoloxStageSpan>,
)

/**
 * Native ONNX Runtime YOLOX-Nano fish detector.
 * Faithfully preserves `src/services/detectorService.ts`:
 * - Input: [1, 3, 416, 416] float32 BGR letterbox with fill value 114.0
 * - Output: [1, numRows, 6] -> (cx, cy, w, h, obj, cls), conf = obj * cls >= 0.20
 * - Min box size > 25px, NMS IoU threshold = 0.45
 */
class YoloxFishDetector(
  private val context: Context,
  private val enableNnapi: Boolean = false,
) {
  companion object {
    private const val TAG = "YoloxFishDetector"
    private const val INPUT_SIZE = 416
    private const val LETTERBOX_FILL = 114.0f
    private const val CONFIDENCE_THRESHOLD = 0.20f
    private const val NMS_IOU_THRESHOLD = 0.45f
    private const val MIN_BOX_DIM_PX = 25
    private const val MIN_WARMUP_AVAILABLE_BYTES = 256L * 1024L * 1024L
  }

  private val loadMutex = Mutex()
  private val warmupMutex = Mutex()
  private val gate = NativeSessionGate("YoloxFishDetector")
  val intraOpThreadCount: Int = Runtime.getRuntime().availableProcessors().coerceIn(2, 4)

  @Volatile
  internal var nativeRunObserver: ((String) -> Unit)? = null

  @Volatile
  internal var sessionClosedObserver: ((String) -> Unit)? = null

  @Volatile
  var status: ModelRuntimeStatus = ModelRuntimeStatus.NOT_LOADED
    private set

  @Volatile
  var loadState: ModelLoadLifecycleState = ModelLoadLifecycleState.NOT_STARTED
    private set

  @Volatile
  var runtimeModuleState: RuntimeModuleState = RuntimeModuleState.NOT_STARTED
    private set

  @Volatile
  var inferenceWarmupDurationMs: Double? = null
    private set

  @Volatile
  var contractSummary: String? = null
    private set

  @Volatile
  var loadDurationMs: Long = 0L
    private set

  @Volatile
  var exactLoadDurationMs: Double = 0.0
    private set

  @Volatile
  var lastLoadMetrics: YoloxLoadMetrics? = null
    private set

  @Volatile
  var sessionCreateCount: Int = 0
    private set

  @Volatile
  var executionProviderName: String = "CPU"
    private set

  @Volatile
  var lastError: String? = null
    private set

  private var ortEnv: OrtEnvironment? = null
  private var session: OrtSession? = null

  fun isReady(): Boolean = !gate.isClosingOrClosed() && status == ModelRuntimeStatus.READY && session != null

  fun isClosed(): Boolean = gate.isClosed()

  fun activeOperationCount(): Int = gate.activeOperationCount()

  suspend fun ensureLoaded(
    cachedAuditReport: ModelAssetAuditReport? = null,
    recorder: RecognitionPerformanceRecorder? = null,
  ): Boolean = withContext(Dispatchers.IO) {
    if (gate.isClosingOrClosed()) return@withContext false
    if (isReady()) {
      recorder?.incrementDuplicateEnsureLoadedGuard()
      return@withContext true
    }
    loadMutex.withLock {
      if (gate.isClosingOrClosed()) return@withContext false
      if (isReady()) {
        recorder?.incrementDuplicateEnsureLoadedGuard()
        return@withContext true
      }

      status = ModelRuntimeStatus.WARMING_UP
      loadState = ModelLoadLifecycleState.LOADING
      runtimeModuleState = RuntimeModuleState.SESSION_LOADING
      recorder?.updateYoloxLoadState(ModelLoadLifecycleState.LOADING)
      lastError = null
      val totalStartNs = SystemClock.elapsedRealtimeNanos()
      var unpublishedSession: OrtSession? = null

      try {
        // 1. Copy ONNX model to cache (only when missing or empty)
        val copyStartNs = SystemClock.elapsedRealtimeNanos()
        val cachedModelFile = File(context.cacheDir, "ort_yolox_nano_v0_1.onnx")
        val copiedFresh = installYoloxAssetAtomically(cachedModelFile)
        val copyEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(copyStartNs)

        // 2. SHA-256 verification (reuse prior audit when asset already verified and size matches cached file)
        val shaStartNs = SystemClock.elapsedRealtimeNanos()
        val canReuseAudit = !copiedFresh &&
          cachedAuditReport != null &&
          cachedAuditReport.isYoloxReady &&
          cachedAuditReport.yoloxAsset.sizeBytes == cachedModelFile.length()
        if (!canReuseAudit && !copiedFresh) {
          val fileSha = sha256File(cachedModelFile)
          check(fileSha.equals(ModelAssetAudit.YOLOX_EXPECTED_SHA256, ignoreCase = true)) {
            "YOLOX cached file SHA-256 mismatch: expected=${ModelAssetAudit.YOLOX_EXPECTED_SHA256}, actual=$fileSha"
          }
        }
        val shaEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(shaStartNs)

        // 3. OrtEnvironment initialization
        val envStartNs = SystemClock.elapsedRealtimeNanos()
        val env = OrtEnvironment.getEnvironment()
        val envEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(envStartNs)

        // 4. YOLOX OrtSession creation
        var activeEp = "CPU"
        val sessStartNs = SystemClock.elapsedRealtimeNanos()
        val options = OrtSession.SessionOptions().apply {
          setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
          setIntraOpNumThreads(intraOpThreadCount)
          if (enableNnapi) {
            runCatching { addNnapi() }.onSuccess {
              activeEp = "NNAPI+CPU"
            }.onFailure {
              Log.w(TAG, "NNAPI unavailable, keeping CPU execution provider", it)
            }
          }
        }
        val created = options.use { opts ->
          env.createSession(cachedModelFile.absolutePath, opts)
        }
        unpublishedSession = created
        val contractSummary = validateContract(created)
        val sessEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(sessStartNs)

        // 5. Session READY publication
        val readyStartNs = SystemClock.elapsedRealtimeNanos()
        val published = gate.publishIfOpen {
          ortEnv = env
          session = created
          this.contractSummary = contractSummary
          sessionCreateCount += 1
          executionProviderName = activeEp
          val readyNowNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(totalStartNs)
          exactLoadDurationMs = (readyNowNs - totalStartNs).coerceAtLeast(0L) / 1_000_000.0
          loadDurationMs = exactLoadDurationMs.roundToLong()
          status = ModelRuntimeStatus.READY
          loadState = ModelLoadLifecycleState.READY
          runtimeModuleState = RuntimeModuleState.SESSION_READY
        }
        if (!published) {
          runCatching { created.close() }
          unpublishedSession = null
          status = ModelRuntimeStatus.NOT_LOADED
          loadState = ModelLoadLifecycleState.NOT_STARTED
          return@withContext false
        }
        unpublishedSession = null
        val totalEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(readyStartNs)
        val metrics = YoloxLoadMetrics(
          onnxFileCopyStartNs = copyStartNs,
          onnxFileCopyEndNs = copyEndNs,
          sha256VerifyStartNs = shaStartNs,
          sha256VerifyEndNs = shaEndNs,
          reusedAuditResult = canReuseAudit,
          ortEnvInitStartNs = envStartNs,
          ortEnvInitEndNs = envEndNs,
          sessionCreateStartNs = sessStartNs,
          sessionCreateEndNs = sessEndNs,
          sessionReadyNs = totalEndNs,
          totalStartNs = totalStartNs,
          totalEndNs = totalEndNs,
          executionProvider = activeEp,
        )
        lastLoadMetrics = metrics

        recorder?.apply {
          updateExecutionProvider(activeEp)
          recordWarmupStageDuration(
            "yolox_onnx_file_copy",
            copyStartNs,
            copyEndNs,
            metrics = mapOf("copiedFresh" to copiedFresh, "sizeBytes" to cachedModelFile.length()),
          )
          recordWarmupStageDuration(
            "yolox_sha256_verify",
            shaStartNs,
            shaEndNs,
            metrics = mapOf(
              "reusedAuditResult" to canReuseAudit,
              "verifiedDuringAtomicCopy" to copiedFresh,
            ),
          )
          recordWarmupStageDuration("yolox_ort_env_init", envStartNs, envEndNs)
          recordWarmupStageDuration(
            "yolox_session_create",
            sessStartNs,
            sessEndNs,
            metrics = mapOf(
              "executionProvider" to activeEp,
              "contract" to contractSummary,
              "intraOpThreads" to intraOpThreadCount,
            ),
          )
          recordWarmupStageDuration("yolox_session_ready", readyStartNs, totalEndNs)
          recordWarmupStageDuration(
            "yolox_total_load",
            totalStartNs,
            totalEndNs,
            metrics = mapOf("reusedAuditResult" to canReuseAudit, "sessionCreateCount" to sessionCreateCount),
          )
          updateYoloxLoadState(
            state = ModelLoadLifecycleState.READY,
            totalLoadMs = (totalEndNs - totalStartNs).coerceAtLeast(0L) / 1_000_000.0,
            reusedAudit = canReuseAudit,
          )
        }

        Log.i(TAG, "YOLOX-Nano loaded on $activeEp in ${"%.2f".format(exactLoadDurationMs)}ms")
        true
      } catch (ce: CancellationException) {
        runCatching { unpublishedSession?.close() }
        status = ModelRuntimeStatus.NOT_LOADED
        loadState = ModelLoadLifecycleState.NOT_STARTED
        runtimeModuleState = RuntimeModuleState.NOT_STARTED
        throw ce
      } catch (t: Throwable) {
        runCatching { unpublishedSession?.close() }
        status = ModelRuntimeStatus.ERROR
        loadState = ModelLoadLifecycleState.ERROR
        runtimeModuleState = RuntimeModuleState.ERROR
        lastError = t.message ?: t.javaClass.simpleName
        recorder?.updateYoloxLoadState(
          state = ModelLoadLifecycleState.ERROR,
          errorCode = lastError,
        )
        Log.e(TAG, "Failed loading YOLOX-Nano ONNX", t)
        false
      }
    }
  }

  private fun validateContract(activeSession: OrtSession): String {
    val input = activeSession.inputInfo.entries.singleOrNull()
      ?: throw IllegalStateException("YOLOX must expose exactly one input; found ${activeSession.inputNames}")
    val inputInfo = input.value.info as? TensorInfo
      ?: throw IllegalStateException("YOLOX input '${input.key}' is not a tensor")
    val expectedInputShape = longArrayOf(1L, 3L, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
    check(inputInfo.type == OnnxJavaType.FLOAT && inputInfo.shape.contentEquals(expectedInputShape)) {
      "Invalid YOLOX input contract: type=${inputInfo.type}, shape=${inputInfo.shape.contentToString()}"
    }

    val output = activeSession.outputInfo.entries.singleOrNull()
      ?: throw IllegalStateException("YOLOX must expose exactly one output; found ${activeSession.outputNames}")
    val outputInfo = output.value.info as? TensorInfo
      ?: throw IllegalStateException("YOLOX output '${output.key}' is not a tensor")
    check(
      outputInfo.type == OnnxJavaType.FLOAT && outputInfo.shape.size == 3 &&
        outputInfo.shape[0] == 1L && outputInfo.shape[1] == 3549L && outputInfo.shape[2] == 6L,
    ) {
      "Invalid YOLOX output contract: type=${outputInfo.type}, shape=${outputInfo.shape.contentToString()}"
    }
    return "in[${input.key}:${inputInfo.shape.contentToString()}] -> " +
      "out[${output.key}:${outputInfo.shape.contentToString()}]"
  }

  /** Verifies and atomically installs the bundled model; an interrupted copy never replaces the cache. */
  private fun installYoloxAssetAtomically(destination: File): Boolean {
    destination.parentFile?.mkdirs()
    if (destination.exists() && destination.length() > 0L &&
      sha256File(destination).equals(ModelAssetAudit.YOLOX_EXPECTED_SHA256, ignoreCase = true)
    ) {
      return false
    }

    val tempFile = File(destination.parentFile, "${destination.name}.tmp.${System.nanoTime()}")
    try {
      val digest = MessageDigest.getInstance("SHA-256")
      val buffer = ByteArray(32 * 1024)
      context.assets.open(ModelAssetAudit.YOLOX_ASSET_PATH).use { input ->
        tempFile.outputStream().use { output ->
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
      check(actualSha.equals(ModelAssetAudit.YOLOX_EXPECTED_SHA256, ignoreCase = true)) {
        "YOLOX asset SHA-256 mismatch: expected=${ModelAssetAudit.YOLOX_EXPECTED_SHA256}, actual=$actualSha"
      }
      Os.rename(tempFile.absolutePath, destination.absolutePath)
      return true
    } finally {
      runCatching { if (tempFile.exists()) tempFile.delete() }
    }
  }

  /** Executes one isolated real-shape tensor Run without interpreting detections or touching UI state. */
  suspend fun warmupInference(recorder: RecognitionPerformanceRecorder? = null): Boolean =
    withContext(Dispatchers.Default) {
      warmupMutex.withLock {
        if (runtimeModuleState == RuntimeModuleState.INFERENCE_READY) return@withLock true
        if (!isReady()) return@withLock false

        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager?.getMemoryInfo(memoryInfo)
        val skipReason = when {
          activityManager == null -> "ACTIVITY_MANAGER_UNAVAILABLE"
          activityManager.isLowRamDevice -> "LOW_RAM_DEVICE"
          memoryInfo.lowMemory -> "SYSTEM_LOW_MEMORY"
          memoryInfo.availMem < MIN_WARMUP_AVAILABLE_BYTES -> "INSUFFICIENT_AVAILABLE_MEMORY"
          else -> null
        }
        if (skipReason != null) {
          runtimeModuleState = RuntimeModuleState.SKIPPED_RESOURCE
          recorder?.recordWarmupStageDuration(
            "yolox_inference_warmup",
            SystemClock.elapsedRealtimeNanos(),
            SystemClock.elapsedRealtimeNanos(),
            status = StageOutcome.CANCELLED,
            errorCode = "PREWARM_SKIPPED_RESOURCE",
            metrics = mapOf("reason" to skipReason, "availableMemoryBytes" to memoryInfo.availMem),
          )
          recorder?.updateModuleState("YOLOX", RuntimeModuleState.SKIPPED_RESOURCE, skipReason)
          return@withLock false
        }

        runtimeModuleState = RuntimeModuleState.KERNEL_WARMING
        recorder?.updateModuleState("YOLOX", RuntimeModuleState.KERNEL_WARMING)
        val stageStartNs = SystemClock.elapsedRealtimeNanos()
        recorder?.beginWarmupStage("yolox_inference_warmup")
        try {
          val inputValues = FloatArray(3 * INPUT_SIZE * INPUT_SIZE) { LETTERBOX_FILL }
          val shape = longArrayOf(1L, 3L, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
          gate.withNativeOperation {
            val activeEnv = ortEnv ?: throw IllegalStateException("YOLOX OrtEnvironment is unavailable")
            val activeSession = session ?: throw IllegalStateException("YOLOX session is unavailable")
            val inputName = activeSession.inputInfo.keys.single()
            val outputName = activeSession.outputInfo.keys.single()
            OnnxTensor.createTensor(activeEnv, FloatBuffer.wrap(inputValues), shape).use { inputTensor ->
              activeSession.run(mapOf(inputName to inputTensor)).use { results ->
                val output = results[outputName].orElse(null) as? OnnxTensor
                  ?: throw IllegalStateException("YOLOX warmup returned no '$outputName' tensor")
                check(output.info.shape.contentEquals(longArrayOf(1L, 3549L, 6L))) {
                  "YOLOX warmup output shape mismatch: ${output.info.shape.contentToString()}"
                }
                check(output.floatBuffer.remaining() == 3549 * 6) {
                  "YOLOX warmup output element count mismatch"
                }
              }
            }
          }
          val endNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(stageStartNs)
          inferenceWarmupDurationMs = (endNs - stageStartNs) / 1_000_000.0
          runtimeModuleState = RuntimeModuleState.INFERENCE_READY
          recorder?.recordWarmupStageDuration(
            "yolox_inference_warmup",
            stageStartNs,
            endNs,
            metrics = mapOf("tensorShape" to "[1,3,416,416]", "intraOpThreads" to intraOpThreadCount),
          )
          recorder?.updateModuleState("YOLOX", RuntimeModuleState.INFERENCE_READY)
          true
        } catch (ce: CancellationException) {
          runtimeModuleState = RuntimeModuleState.SESSION_READY
          recorder?.completeWarmupStage("yolox_inference_warmup", StageOutcome.CANCELLED, "CANCELLED")
          recorder?.updateModuleState("YOLOX", RuntimeModuleState.SESSION_READY)
          throw ce
        } catch (t: Throwable) {
          runtimeModuleState = RuntimeModuleState.ERROR
          lastError = "YOLOX_WARMUP_FAILED: ${t.message ?: t.javaClass.simpleName}"
          recorder?.completeWarmupStage(
            "yolox_inference_warmup",
            StageOutcome.FAILED,
            "YOLOX_WARMUP_FAILED",
          )
          recorder?.updateModuleState("YOLOX", RuntimeModuleState.ERROR, "YOLOX_WARMUP_FAILED")
          Log.e(TAG, "YOLOX inference warmup failed", t)
          false
        }
      }
    }

  suspend fun detect(
    sourceBitmap: Bitmap,
    mapper: ImageCoordinateMapper,
  ): Pair<FishDetection?, Long> {
    val outcome = detectDetailed(sourceBitmap, mapper)
    return outcome.detection to outcome.totalDurationMs
  }

  suspend fun detectDetailed(
    sourceBitmap: Bitmap,
    mapper: ImageCoordinateMapper,
  ): YoloxDetectionOutcome = withContext(Dispatchers.Default) {
    ensureActive()
    if (gate.isClosingOrClosed()) {
      throw CancellationException("YoloxFishDetector is closing or closed")
    }
    if (sourceBitmap.isRecycled) {
      throw CancellationException("Source bitmap recycled before YOLOX detection")
    }
    if (!ensureLoaded()) {
      if (gate.isClosingOrClosed()) {
        throw CancellationException("YoloxFishDetector closed before detection")
      }
      throw IllegalStateException(lastError ?: "YOLOX-Nano failed to initialize")
    }

    val totalStartNs = SystemClock.elapsedRealtimeNanos()
    val stages = ArrayList<YoloxStageSpan>(8)
    val sourceW = mapper.origWidth
    val sourceH = mapper.origHeight
    val scale = min(INPUT_SIZE.toFloat() / sourceW, INPUT_SIZE.toFloat() / sourceH)
    val drawW = max(1, floor(sourceW * scale).toInt())
    val drawH = max(1, floor(sourceH * scale).toInt())
    val planeSize = INPUT_SIZE * INPUT_SIZE
    val rawDetections = ArrayList<FishDetection>(16)

    var numRows = 0
    var stride = 6
    var flatOutput = FloatArray(0)

    gate.withNativeOperation {
      if (sourceBitmap.isRecycled) {
        throw CancellationException("Source bitmap recycled during YOLOX detection")
      }
      val activeSession = session ?: throw IllegalStateException("YOLOX session is null")
      val env = ortEnv ?: throw IllegalStateException("OrtEnvironment is null")

      // 1. Bitmap Resize
      val resizeStartNs = SystemClock.elapsedRealtimeNanos()
      val resized = Bitmap.createScaledBitmap(sourceBitmap, drawW, drawH, true)
      val resizeEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(resizeStartNs)
      stages.add(YoloxStageSpan("yolox_bitmap_resize", resizeStartNs, resizeEndNs))

      // 2. BGR Tensor Build
      val bgrStartNs = SystemClock.elapsedRealtimeNanos()
      val tensorData = FloatArray(3 * planeSize) { LETTERBOX_FILL }
      val pixels = IntArray(drawW * drawH)
      try {
        resized.getPixels(pixels, 0, drawW, 0, 0, drawW, drawH)
        for (y in 0 until drawH) {
          ensureActive()
          val srcRow = y * drawW
          val dstRow = y * INPUT_SIZE
          for (x in 0 until drawW) {
            val argb = pixels[srcRow + x]
            val r = Color.red(argb).toFloat()
            val g = Color.green(argb).toFloat()
            val b = Color.blue(argb).toFloat()
            val dstIdx = dstRow + x
            // BGR channel order matching detectorService.ts
            tensorData[dstIdx] = b
            tensorData[planeSize + dstIdx] = g
            tensorData[2 * planeSize + dstIdx] = r
          }
        }
      } finally {
        if (resized !== sourceBitmap && !resized.isRecycled) {
          resized.recycle()
        }
      }
      val bgrEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(bgrStartNs)
      stages.add(YoloxStageSpan("yolox_bgr_tensor_build", bgrStartNs, bgrEndNs))

      ensureActive()

      val inputName = activeSession.inputNames.firstOrNull() ?: "images"
      val outputName = activeSession.outputNames.firstOrNull() ?: "output"
      val shape = longArrayOf(1L, 3L, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())

      // 3. ONNX Tensor Create
      val tensorCreateStartNs = SystemClock.elapsedRealtimeNanos()
      OnnxTensor.createTensor(env, FloatBuffer.wrap(tensorData), shape).use { inputTensor ->
        val tensorCreateEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(tensorCreateStartNs)
        stages.add(YoloxStageSpan("yolox_onnx_tensor_create", tensorCreateStartNs, tensorCreateEndNs))

        ensureActive()
        nativeRunObserver?.invoke("YOLOX_RUN")

        // 4. YOLOX Session.run()
        val runStartNs = SystemClock.elapsedRealtimeNanos()
        activeSession.run(mapOf(inputName to inputTensor)).use { results ->
          val runEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(runStartNs)
          stages.add(YoloxStageSpan("yolox_session_run", runStartNs, runEndNs))

          // 5. Output Tensor Read
          val readStartNs = SystemClock.elapsedRealtimeNanos()
          val outValue = results[outputName].orElse(null)
            ?: throw IllegalStateException("Missing YOLOX output tensor '$outputName'")
          val outTensor = outValue as? OnnxTensor
            ?: throw IllegalStateException("Unexpected YOLOX output type")
          val outShape = outTensor.info.shape
          numRows = if (outShape.size >= 2) outShape[1].toInt() else 0
          stride = if (outShape.size >= 3) outShape[2].toInt() else 6
          val fb = outTensor.floatBuffer
          flatOutput = FloatArray(numRows * stride)
          fb.get(flatOutput)
          val readEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(readStartNs)
          stages.add(YoloxStageSpan("yolox_output_tensor_read", readStartNs, readEndNs))
        }
      }
    }

    // 6. Detection Decode
    val decodeStartNs = SystemClock.elapsedRealtimeNanos()
    for (i in 0 until numRows) {
      val off = i * stride
      val cx = flatOutput[off]
      val cy = flatOutput[off + 1]
      val w = flatOutput[off + 2]
      val h = flatOutput[off + 3]
      val obj = flatOutput[off + 4]
      val cls = flatOutput[off + 5]
      val conf = obj * cls

      if (conf >= CONFIDENCE_THRESHOLD) {
        val box = mapper.detectorBoxToOriginalBox(cx, cy, w, h, scale)
        if (box.width > MIN_BOX_DIM_PX && box.height > MIN_BOX_DIM_PX) {
          val areaRatio = box.area.toFloat() / max(1, sourceW * sourceH).toFloat()
          rawDetections.add(FishDetection(box = box, confidence = conf, areaRatio = areaRatio))
        }
      }
    }
    val decodeEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(decodeStartNs)
    stages.add(YoloxStageSpan("yolox_detection_decode", decodeStartNs, decodeEndNs))

    // 7. NMS
    val nmsStartNs = SystemClock.elapsedRealtimeNanos()
    val selected = ArrayList<FishDetection>(rawDetections.size)
    if (rawDetections.isNotEmpty()) {
      rawDetections.sortByDescending { it.confidence }
      for (det in rawDetections) {
        var keep = true
        for (kept in selected) {
          if (computeIoU(det.box, kept.box) > NMS_IOU_THRESHOLD) {
            keep = false
            break
          }
        }
        if (keep) selected.add(det)
      }
    }
    val nmsEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(nmsStartNs)
    stages.add(YoloxStageSpan("yolox_nms", nmsStartNs, nmsEndNs))

    // 8. Final Detection Return
    val totalEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(totalStartNs)
    stages.add(YoloxStageSpan("yolox_total_detect", totalStartNs, totalEndNs))
    val exactMs = (totalEndNs - totalStartNs).coerceAtLeast(0L) / 1_000_000.0

    YoloxDetectionOutcome(
      detection = selected.firstOrNull(),
      totalDurationMs = exactMs.roundToLong(),
      exactDurationMs = exactMs,
      totalStartNs = totalStartNs,
      totalEndNs = totalEndNs,
      rawCandidateCount = rawDetections.size,
      postNmsCandidateCount = selected.size,
      stages = stages,
    ).also {
      runtimeModuleState = RuntimeModuleState.INFERENCE_READY
    }
  }

  private fun computeIoU(a: PixelBox, b: PixelBox): Float {
    val xA = max(a.x1, b.x1)
    val yA = max(a.y1, b.y1)
    val xB = min(a.x2, b.x2)
    val yB = min(a.y2, b.y2)
    val inter = max(0, xB - xA) * max(0, yB - yA)
    val union = max(1, a.area + b.area - inter)
    return inter.toFloat() / union.toFloat()
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
    gate.closeIdempotent {
      runCatching { session?.close() }
      session = null
      contractSummary = null
      ortEnv = null
      status = ModelRuntimeStatus.NOT_LOADED
      loadState = ModelLoadLifecycleState.NOT_STARTED
      runtimeModuleState = RuntimeModuleState.NOT_STARTED
      inferenceWarmupDurationMs = null
      sessionClosedObserver?.invoke("CLOSE_YOLOX_SESSION")
    }
  }
}
