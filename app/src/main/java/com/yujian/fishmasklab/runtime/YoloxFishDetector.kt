package com.yujian.fishmasklab.runtime

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.util.Log
import com.yujian.fishmasklab.domain.FishDetection
import com.yujian.fishmasklab.domain.ImageCoordinateMapper
import com.yujian.fishmasklab.domain.PixelBox
import java.io.File
import java.nio.FloatBuffer
import kotlin.coroutines.cancellation.CancellationException
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

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
  }

  private val loadMutex = Mutex()

  @Volatile
  var status: ModelRuntimeStatus = ModelRuntimeStatus.NOT_LOADED
    private set

  @Volatile
  var loadDurationMs: Long = 0L
    private set

  @Volatile
  var lastError: String? = null
    private set

  private var ortEnv: OrtEnvironment? = null
  private var session: OrtSession? = null

  fun isReady(): Boolean = status == ModelRuntimeStatus.READY && session != null

  suspend fun ensureLoaded(): Boolean = withContext(Dispatchers.IO) {
    if (isReady()) return@withContext true
    loadMutex.withLock {
      if (isReady()) return@withContext true
      status = ModelRuntimeStatus.WARMING_UP
      lastError = null
      val start = System.currentTimeMillis()
      try {
        val env = OrtEnvironment.getEnvironment()
        ortEnv = env

        // Copy to app cache once to allow file-backed OrtSession creation without holding full byte[] in Java heap
        val cachedModelFile = File(context.cacheDir, "ort_yolox_nano_v0_1.onnx")
        if (!cachedModelFile.exists() || cachedModelFile.length() == 0L) {
          context.assets.open(ModelAssetAudit.YOLOX_ASSET_PATH).use { input ->
            cachedModelFile.outputStream().use { output ->
              input.copyTo(output)
            }
          }
        }

        val options = OrtSession.SessionOptions().apply {
          setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
          setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(2, 4))
          if (enableNnapi) {
            runCatching { addNnapi() }.onFailure {
              Log.w(TAG, "NNAPI unavailable, keeping CPU execution provider", it)
            }
          }
        }

        session = options.use { opts ->
          env.createSession(cachedModelFile.absolutePath, opts)
        }
        loadDurationMs = System.currentTimeMillis() - start
        status = ModelRuntimeStatus.READY
        Log.i(TAG, "YOLOX-Nano loaded on CPU in ${loadDurationMs}ms")
        true
      } catch (ce: CancellationException) {
        status = ModelRuntimeStatus.NOT_LOADED
        throw ce
      } catch (t: Throwable) {
        status = ModelRuntimeStatus.ERROR
        lastError = t.message ?: t.javaClass.simpleName
        Log.e(TAG, "Failed loading YOLOX-Nano ONNX", t)
        false
      }
    }
  }

  suspend fun detect(
    sourceBitmap: Bitmap,
    mapper: ImageCoordinateMapper,
  ): Pair<FishDetection?, Long> = withContext(Dispatchers.Default) {
    ensureActive()
    if (!ensureLoaded()) {
      throw IllegalStateException(lastError ?: "YOLOX-Nano failed to initialize")
    }
    val activeSession = session ?: throw IllegalStateException("YOLOX session is null")
    val env = ortEnv ?: throw IllegalStateException("OrtEnvironment is null")

    val start = System.currentTimeMillis()
    val sourceW = mapper.origWidth
    val sourceH = mapper.origHeight
    val scale = min(INPUT_SIZE.toFloat() / sourceW, INPUT_SIZE.toFloat() / sourceH)
    val drawW = max(1, floor(sourceW * scale).toInt())
    val drawH = max(1, floor(sourceH * scale).toInt())

    // Build resized bitmap proportionally for BGR extraction
    val resized = Bitmap.createScaledBitmap(sourceBitmap, drawW, drawH, true)
    val planeSize = INPUT_SIZE * INPUT_SIZE
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

    ensureActive()

    val inputName = activeSession.inputNames.firstOrNull() ?: "images"
    val outputName = activeSession.outputNames.firstOrNull() ?: "output"
    val shape = longArrayOf(1L, 3L, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())

    val rawDetections = ArrayList<FishDetection>(16)

    OnnxTensor.createTensor(env, FloatBuffer.wrap(tensorData), shape).use { inputTensor ->
      ensureActive()
      activeSession.run(mapOf(inputName to inputTensor)).use { results ->
        val outValue = results[outputName].orElse(null)
          ?: throw IllegalStateException("Missing YOLOX output tensor '$outputName'")
        val outTensor = outValue as? OnnxTensor
          ?: throw IllegalStateException("Unexpected YOLOX output type")
        val outShape = outTensor.info.shape
        val numRows = if (outShape.size >= 2) outShape[1].toInt() else 0
        val stride = if (outShape.size >= 3) outShape[2].toInt() else 6
        val fb = outTensor.floatBuffer

        for (i in 0 until numRows) {
          val off = i * stride
          val cx = fb.get(off)
          val cy = fb.get(off + 1)
          val w = fb.get(off + 2)
          val h = fb.get(off + 3)
          val obj = fb.get(off + 4)
          val cls = fb.get(off + 5)
          val conf = obj * cls

          if (conf >= CONFIDENCE_THRESHOLD) {
            val box = mapper.detectorBoxToOriginalBox(cx, cy, w, h, scale)
            if (box.width > MIN_BOX_DIM_PX && box.height > MIN_BOX_DIM_PX) {
              val areaRatio = box.area.toFloat() / max(1, sourceW * sourceH).toFloat()
              rawDetections.add(FishDetection(box = box, confidence = conf, areaRatio = areaRatio))
            }
          }
        }
      }
    }

    if (rawDetections.isEmpty()) {
      return@withContext null to (System.currentTimeMillis() - start)
    }

    rawDetections.sortByDescending { it.confidence }
    val selected = ArrayList<FishDetection>(rawDetections.size)
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

    val duration = System.currentTimeMillis() - start
    selected.firstOrNull() to duration
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

  fun close() {
    runCatching { session?.close() }
    session = null
    status = ModelRuntimeStatus.NOT_LOADED
  }
}
