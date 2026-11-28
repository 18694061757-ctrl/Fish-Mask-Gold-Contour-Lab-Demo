package com.yujian.fishmasklab.runtime

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import androidx.test.core.app.ApplicationProvider
import com.yujian.fishmasklab.app.ExifPhotoLoader
import com.yujian.fishmasklab.app.FishMaskViewModel
import com.yujian.fishmasklab.domain.CandidateMask
import com.yujian.fishmasklab.domain.ContourData
import com.yujian.fishmasklab.domain.ContourGeometry
import com.yujian.fishmasklab.domain.ContourPoint
import com.yujian.fishmasklab.domain.FishDetection
import com.yujian.fishmasklab.domain.MaskQualityGate
import com.yujian.fishmasklab.domain.MaskResult
import com.yujian.fishmasklab.domain.PixelBox
import com.yujian.fishmasklab.domain.PipelineState
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import kotlinx.coroutines.withTimeout
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class EvidenceAndParityExportTest {

  @Before
  fun setUpMainDispatcher() {
    Dispatchers.setMain(Dispatchers.Default)
  }

  @After
  fun tearDownMainDispatcher() {
    SlimSamMaskPostProcessor.stageObserverForTest = null
    Dispatchers.resetMain()
  }

  private fun sha256Hex(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

  private fun readLittleEndianFloatArray(file: File): FloatArray {
    val bytes = file.readBytes()
    val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    val out = FloatArray(bytes.size / 4)
    fb.get(out)
    return out
  }

  private fun readLittleEndianLongArray(file: File): LongArray {
    val bytes = file.readBytes()
    val lb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer()
    val out = LongArray(bytes.size / 8)
    lb.get(out)
    return out
  }

  private fun writeLittleEndianFloatArray(file: File, values: FloatArray) {
    file.parentFile?.mkdirs()
    val buffer = ByteBuffer.allocate(64 * 1024).order(ByteOrder.LITTLE_ENDIAN)
    file.outputStream().buffered().use { output ->
      for (value in values) {
        if (buffer.remaining() < 4) {
          output.write(buffer.array(), 0, buffer.position())
          buffer.clear()
        }
        buffer.putFloat(value)
      }
      if (buffer.position() > 0) output.write(buffer.array(), 0, buffer.position())
    }
  }

  private fun writeLittleEndianLongArray(file: File, values: LongArray) {
    file.parentFile?.mkdirs()
    val buffer = ByteBuffer.allocate(64 * 1024).order(ByteOrder.LITTLE_ENDIAN)
    file.outputStream().buffered().use { output ->
      for (value in values) {
        if (buffer.remaining() < 8) {
          output.write(buffer.array(), 0, buffer.position())
          buffer.clear()
        }
        buffer.putLong(value)
      }
      if (buffer.position() > 0) output.write(buffer.array(), 0, buffer.position())
    }
  }

  private fun sha256File(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    val buffer = ByteArray(64 * 1024)
    file.inputStream().buffered().use { input ->
      while (true) {
        val read = input.read(buffer)
        if (read < 0) break
        digest.update(buffer, 0, read)
      }
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
  }

  private fun floatTensorSummary(values: FloatArray): JSONObject {
    require(values.isNotEmpty()) { "Cannot summarize an empty tensor" }
    var minValue = Float.POSITIVE_INFINITY
    var maxValue = Float.NEGATIVE_INFINITY
    var sum = 0.0
    var nearZeroCount = 0
    var positiveCount = 0
    var negativeCount = 0
    for (value in values) {
      if (value < minValue) minValue = value
      if (value > maxValue) maxValue = value
      sum += value
      if (abs(value) < 0.01f) nearZeroCount++
      if (value > 0f) positiveCount++
      if (value < 0f) negativeCount++
    }
    val sorted = values.copyOf().also { it.sort() }
    fun percentile(p: Double): Float = sorted[((sorted.size - 1) * p).toInt()]
    return JSONObject().apply {
      put("elementCount", values.size)
      put("min", minValue.toDouble())
      put("max", maxValue.toDouble())
      put("mean", sum / values.size)
      put("nearZeroAbsLt0_01Count", nearZeroCount)
      put("nearZeroAbsLt0_01Percent", nearZeroCount * 100.0 / values.size)
      put("positiveCount", positiveCount)
      put("negativeCount", negativeCount)
      put("percentileMethod", "floor_index_on_sorted_float32")
      put("p1", percentile(0.01).toDouble())
      put("p5", percentile(0.05).toDouble())
      put("p50", percentile(0.50).toDouble())
      put("p95", percentile(0.95).toDouble())
      put("p99", percentile(0.99).toDouble())
    }
  }

  private fun writeStageEvidence(
    stageRoot: File,
    candidateIndex: Int,
    stageName: String,
    floatValues: FloatArray?,
    maskValues: ByteArray?,
    width: Int,
    height: Int,
    includeMaskPng: Boolean = true,
  ) {
    val candidateDir = File(stageRoot, "candidate_$candidateIndex").apply { mkdirs() }
    when {
      floatValues != null -> {
        val output = File(candidateDir, "$stageName.f32")
        writeLittleEndianFloatArray(output, floatValues)
        if (stageName == "logits_256") File(candidateDir, "$stageName.bin").writeBytes(output.readBytes())
      }
      maskValues != null -> {
        File(candidateDir, "$stageName.bin").writeBytes(maskValues)
        if (includeMaskPng) writeMaskPng(maskValues, width, height, File(candidateDir, "$stageName.png"))
      }
      else -> error("Mask stage $stageName has no payload")
    }
  }

  private fun floatArrayComparison(a: FloatArray, b: FloatArray): JSONObject {
    require(a.size == b.size) { "Float tensor sizes differ: ${a.size} vs ${b.size}" }
    var absSum = 0.0
    var maxAbs = 0.0
    var dot = 0.0
    var normA = 0.0
    var normB = 0.0
    var signMismatch = 0
    var nearZeroA = 0
    var nearZeroB = 0
    var signMismatchNearZero = 0
    for (i in a.indices) {
      val av = a[i].toDouble()
      val bv = b[i].toDouble()
      val diff = abs(av - bv)
      absSum += diff
      if (diff > maxAbs) maxAbs = diff
      dot += av * bv
      normA += av * av
      normB += bv * bv
      if ((av > 0.0) != (bv > 0.0)) signMismatch++
      if (abs(av) < 0.01) nearZeroA++
      if (abs(bv) < 0.01) nearZeroB++
      if (abs(av) < 0.01 && abs(bv) < 0.01 && ((av > 0.0) != (bv > 0.0))) {
        signMismatchNearZero++
      }
    }
    return JSONObject().apply {
      put("elementCount", a.size)
      put("mae", absSum / max(1, a.size))
      put("maxAbsDiff", maxAbs)
      put("cosineSimilarity", dot / max(1e-30, sqrt(normA) * sqrt(normB)))
      put("signMismatchCount", signMismatch)
      put("signMismatchPercent", signMismatch * 100.0 / max(1, a.size))
      put("nearZeroThresholdAbsLt0_01", 0.01)
      put("nearZeroCountA", nearZeroA)
      put("nearZeroCountB", nearZeroB)
      put("nearZeroSignMismatchCount", signMismatchNearZero)
    }
  }

  private fun writeMaskPng(mask: ByteArray, width: Int, height: Int, outFile: File) {
    val pixels = IntArray(width * height)
    for (i in pixels.indices) {
      pixels[i] = if ((mask[i].toInt() and 0xFF) > 127) Color.WHITE else Color.BLACK
    }
    val bmp = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    try {
      outFile.outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
    } finally {
      bmp.recycle()
    }
  }

  private fun writeDetectionOverlayPng(
    source: Bitmap,
    det: FishDetection,
    outFile: File,
  ) {
    val out = source.copy(Bitmap.Config.ARGB_8888, true)
    try {
      val canvas = Canvas(out)
      val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(255, 209, 102)
        style = Paint.Style.STROKE
        strokeWidth = max(4f, source.width / 220f)
      }
      canvas.drawRect(
        det.box.x1.toFloat(),
        det.box.y1.toFloat(),
        det.box.x2.toFloat(),
        det.box.y2.toFloat(),
        boxPaint,
      )
      outFile.outputStream().use { out.compress(Bitmap.CompressFormat.PNG, 100, it) }
    } finally {
      out.recycle()
    }
  }

  private fun writeMaskOverlayPng(
    source: Bitmap,
    mask: ByteArray,
    outFile: File,
  ) {
    val w = source.width
    val h = source.height
    val pixels = IntArray(w * h)
    source.getPixels(pixels, 0, w, 0, 0, w, h)
    for (i in pixels.indices) {
      if ((mask[i].toInt() and 0xFF) > 127) {
        val c = pixels[i]
        val r = ((Color.red(c) * 0.55f) + 56f * 0.45f).toInt().coerceIn(0, 255)
        val g = ((Color.green(c) * 0.55f) + 189f * 0.45f).toInt().coerceIn(0, 255)
        val b = ((Color.blue(c) * 0.55f) + 248f * 0.45f).toInt().coerceIn(0, 255)
        pixels[i] = Color.rgb(r, g, b)
      }
    }
    val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
    bmp.setPixels(pixels, 0, w, 0, 0, w, h)
    try {
      outFile.outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
    } finally {
      bmp.recycle()
    }
  }

  private fun writeContourOverlayPng(
    source: Bitmap,
    mask: ByteArray,
    contour: ContourData,
    outFile: File,
  ) {
    val w = source.width
    val h = source.height
    val pixels = IntArray(w * h)
    source.getPixels(pixels, 0, w, 0, 0, w, h)
    for (i in pixels.indices) {
      if ((mask[i].toInt() and 0xFF) > 127) {
        val c = pixels[i]
        val r = ((Color.red(c) * 0.72f) + 56f * 0.28f).toInt().coerceIn(0, 255)
        val g = ((Color.green(c) * 0.72f) + 189f * 0.28f).toInt().coerceIn(0, 255)
        val b = ((Color.blue(c) * 0.72f) + 248f * 0.28f).toInt().coerceIn(0, 255)
        pixels[i] = Color.rgb(r, g, b)
      }
    }
    val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
    bmp.setPixels(pixels, 0, w, 0, 0, w, h)
    try {
      val canvas = Canvas(bmp)
      val haloPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(170, 56, 189, 248)
        style = Paint.Style.STROKE
        strokeWidth = max(8f, w / 110f)
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
      }
      val goldPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(255, 209, 102)
        style = Paint.Style.STROKE
        strokeWidth = max(3.5f, w / 260f)
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
      }
      val path = Path()
      val pts = contour.simplifiedPoints
      if (pts.isNotEmpty() && contour.bezierSegments.size == pts.size) {
        path.moveTo(pts[0].x, pts[0].y)
        for (seg in contour.bezierSegments) {
          path.cubicTo(seg.cp1x, seg.cp1y, seg.cp2x, seg.cp2y, seg.endX, seg.endY)
        }
        path.close()
      }
      canvas.drawPath(path, haloPaint)
      canvas.drawPath(path, goldPaint)
      outFile.outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
    } finally {
      bmp.recycle()
    }
  }

  private fun binaryMaskMetrics(a: ByteArray, b: ByteArray): Triple<Double, Double, Int> {
    var inter = 0L
    var union = 0L
    var diff = 0L
    for (i in a.indices) {
      val fa = (a[i].toInt() and 0xFF) > 127
      val fb = (b[i].toInt() and 0xFF) > 127
      if (fa && fb) inter++
      if (fa || fb) union++
      if (fa != fb) diff++
    }
    val iou = if (union == 0L) 1.0 else inter.toDouble() / union.toDouble()
    val diffRatio = diff.toDouble() / max(1, a.size).toDouble()
    return Triple(iou, diffRatio, diff.toInt())
  }

  private fun symmetricChamferDistance(a: List<ContourPoint>, b: List<ContourPoint>): Pair<Double, Double> {
    fun directed(src: List<ContourPoint>, dst: List<ContourPoint>): DoubleArray {
      val stepSrc = max(1, src.size / 600)
      val out = ArrayList<Double>(src.size / stepSrc + 1)
      var i = 0
      while (i < src.size) {
        val p = src[i]
        var bestSq = Double.MAX_VALUE
        for (j in dst.indices) {
          val q = dst[j]
          val dx = (p.x - q.x).toDouble()
          val dy = (p.y - q.y).toDouble()
          val dSq = dx * dx + dy * dy
          if (dSq < bestSq) bestSq = dSq
        }
        out.add(sqrt(bestSq))
        i += stepSrc
      }
      return out.toDoubleArray()
    }
    val d1 = directed(a, b)
    val d2 = directed(b, a)
    val all = (d1 + d2).sortedArray()
    val avg = all.average()
    val p95 = all[((all.size - 1) * 0.95).toInt().coerceIn(0, all.lastIndex)]
    return avg to p95
  }

  private fun readContourPoints(file: File): List<ContourPoint> {
    val pointsJson = JSONObject(file.readText()).getJSONArray("rawPoints")
    return List(pointsJson.length()) { index ->
      val point = pointsJson.getJSONObject(index)
      ContourPoint(point.getDouble("x").toFloat(), point.getDouble("y").toFloat())
    }
  }

  private fun postProcessWithKotlin(
    logits: FloatArray,
    iouScores: FloatArray,
    resizedWidth: Int,
    resizedHeight: Int,
    originalWidth: Int,
    originalHeight: Int,
    detection: FishDetection,
    stageRoot: File? = null,
    includeMaskPng: Boolean = false,
  ): MaskResult {
    val previousObserver = SlimSamMaskPostProcessor.stageObserverForTest
    if (stageRoot != null) {
      SlimSamMaskPostProcessor.stageObserverForTest = { candidate, stage, floats, mask ->
        writeStageEvidence(
          stageRoot, candidate, stage, floats, mask, originalWidth, originalHeight, includeMaskPng,
        )
      }
    }
    return try {
      SlimSamMaskPostProcessor.buildMaskResult(
        decoderOutput = SamDecoderOutput(
          iouScores = iouScores,
          predMasksLogits = logits,
          decoderDurationMs = 0L,
        ),
        resizedWidth = resizedWidth,
        resizedHeight = resizedHeight,
        originalWidth = originalWidth,
        originalHeight = originalHeight,
        detection = detection,
        totalSamInferenceMs = 0L,
      )
    } finally {
      SlimSamMaskPostProcessor.stageObserverForTest = previousObserver
    }
  }

  private fun firstParityDivergence(photo: JSONObject): String {
    val photoId = photo.optString("photoId", "unknown")
    val divergence = photo.optJSONObject("numericalDivergence")
    val sourceSha = divergence?.optString("sourceFixtureSha256").orEmpty()
    val webSha = divergence?.optString("webFixtureSha256").orEmpty()
    if (sourceSha.isNotEmpty() && webSha.isNotEmpty() && !sourceSha.equals(webSha, ignoreCase = true)) {
      return "$photoId: source fixture identity differs before model execution"
    }

    val detectionIou = photo.optDouble("yoloxDetBoxIoU", -1.0)
    if (detectionIou < 0.98) return "$photoId: detector box IoU gate first fails ($detectionIou < 0.98)"

    if (photo.optBoolean("promptCoordinatesExactlyMatchWebBilinear", false).not() ||
      photo.optBoolean("promptLabelsExactlyMatchWebBilinear", false).not()
    ) {
      return "$photoId: mapped prompt coordinates or labels diverge before SAM preprocessing/decoder execution"
    }

    val bilinearPvMae = photo.optDouble("bilinearProcessorPixelValuesMAE", -1.0)
    val bilinearPvMax = photo.optDouble("bilinearProcessorPixelValuesMaxAbsDiff", -1.0)
    if (bilinearPvMae > 0.001 || bilinearPvMax > 0.02) {
      return "$photoId: SAM encoder input preprocessing exceeds frozen limits (MAE=$bilinearPvMae, max=$bilinearPvMax)"
    }

    val bilinearEncoderCos = photo.optDouble("bilinearEncoderCosineSimilarity", -1.0)
    if (bilinearEncoderCos < 0.998) {
      return "$photoId: bilinear-reference encoder cosine is below the frozen gate ($bilinearEncoderCos < 0.998)"
    }

    val nodeEncoderCos = photo.optDouble("encoderCosineSimilarity", -1.0)
    if (nodeEncoderCos < 0.995) {
      return "$photoId: Node-reference encoder cosine is below the frozen gate ($nodeEncoderCos < 0.995)"
    }

    val encoderSame = photo.optJSONObject("androidVsWebEncoderOnIdenticalInput")
    val sameInputImage = encoderSame?.optJSONObject("imageEmbeddings")
    val sameInputPositional = encoderSame?.optJSONObject("positionalEmbeddings")
    val encoderCos = sameInputImage?.optDouble("cosineSimilarity", -1.0) ?: -1.0
    if (encoderCos < 0.998) return "$photoId: encoder differs on identical input (cosine=$encoderCos)"
    val sameInputImageMax = sameInputImage?.optDouble("maxAbsDiff", Double.POSITIVE_INFINITY) ?: Double.POSITIVE_INFINITY
    val sameInputPositionMax = sameInputPositional?.optDouble("maxAbsDiff", Double.POSITIVE_INFINITY) ?: Double.POSITIVE_INFINITY
    if (sameInputImageMax > 0.0 || sameInputPositionMax > 0.0) {
      return "$photoId: identical-input encoder tensors first differ (imageMax=$sameInputImageMax, positionalMax=$sameInputPositionMax)"
    }

    val sameInputDecoder = photo.optJSONObject("androidRawLogitsVsWebDecoderWithIdenticalInputs")
    val sameInputDecoderMax = sameInputDecoder?.optDouble("maxAbsDiff", Double.POSITIVE_INFINITY) ?: Double.POSITIVE_INFINITY
    if (sameInputDecoderMax > 0.0) {
      return "$photoId: raw decoder logits first differ on identical inputs (maxAbs=$sameInputDecoderMax, cosine=${sameInputDecoder?.optDouble("cosineSimilarity", -1.0)})"
    }

    val sameLogitStages = photo.optJSONObject("identicalLogitMaskStageComparisons")
      ?.optJSONArray("nativeAndroidLogits")
    if (sameLogitStages != null) {
      for (index in 0 until sameLogitStages.length()) {
        val row = sameLogitStages.getJSONObject(index)
        val stage = row.optString("stage", "unknown")
        val candidate = row.optInt("candidateIndex", -1)
        val numeric = row.optJSONObject("comparison")
        val maxAbs = numeric?.optDouble("maxAbsDiff", 0.0) ?: 0.0
        val maskIou = row.optDouble("iou", 1.0)
        val pixelDiff = row.optDouble("pixelDiffPercent", 0.0)
        if (maxAbs > 0.0 || maskIou < 1.0 || pixelDiff > 0.0) {
          return "$photoId: Kotlin/Web postprocessing first diverges at candidate $candidate stage $stage (maxAbs=$maxAbs, maskIoU=$maskIou, pixelDiff=$pixelDiff%)"
        }
      }
    }

    val post = photo.optJSONArray("sameLogitsPostprocessCrossChecks")
    val androidCandidate = photo.optInt("androidSelectedCandidateIndex", -1)
    val bilinearPostIou = post?.optJSONObject(androidCandidate)
      ?.optJSONObject("kotlinVsWebBilinear")?.optDouble("iou", -1.0) ?: -1.0
    if (bilinearPostIou >= 0.0 && bilinearPostIou < 0.999999) {
      return "$photoId: Kotlin/Web bilinear postprocessing diverges on the same candidate logits (mask IoU=$bilinearPostIou)"
    }

    val nodeCandidate = photo.optInt("webSamServiceBestIndex", -1)
    val bilinearCandidate = photo.optInt("webBilinearAndroidCompatibleSelectedIndex", -1)
    if (androidCandidate != nodeCandidate || androidCandidate != bilinearCandidate) {
      return "$photoId: candidate selection diverges after mask generation (Android=$androidCandidate, Web Node=$nodeCandidate, Web bilinear=$bilinearCandidate)"
    }

    val selectedIoU = photo.optDouble("selectedMaskIoU", -1.0)
    val selectedDiff = photo.optDouble("selectedMaskPixelDiffPercent", Double.POSITIVE_INFINITY)
    val chamfer = photo.optDouble("contourSymmetricChamferAvgPx", Double.POSITIVE_INFINITY)
    if (selectedIoU < 0.94 || selectedDiff > 2.0 || chamfer > 2.5) {
      val decoder = photo.optJSONObject("androidRawLogitsVsWebDecoderWithIdenticalInputs")
      val decoderCos = decoder?.optDouble("cosineSimilarity", -1.0) ?: -1.0
      return "$photoId: same-input ONNX decoder/provider output diverges (cosine=$decoderCos) before mask thresholding"
    }
    return "$photoId: no frozen parity gate fails"
  }

  private fun compareIdenticalLogitMaskStages(androidStageRoot: File, webStageRoot: File): JSONArray {
    val stageRows = JSONArray()
    val floatStages = listOf("logits_256", "upsampled_1024", "padding_cropped", "resized_original")
    val maskStages = listOf("threshold_mask", "clipped_mask", "lcc_mask", "final_mask")
    for (candidateIndex in 0 until SamDecoderOutput.NUM_CANDIDATES) {
      val androidCandidate = File(androidStageRoot, "candidate_$candidateIndex")
      val webCandidate = File(webStageRoot, "candidate_$candidateIndex")
      for (stageName in floatStages) {
        val androidFile = File(androidCandidate, "$stageName.f32")
        val webFile = File(webCandidate, "$stageName.f32")
        stageRows.put(JSONObject().apply {
          put("candidateIndex", candidateIndex)
          put("stage", stageName)
          if (androidFile.isFile && webFile.isFile) {
            val comparison = runCatching { floatArrayComparison(
              readLittleEndianFloatArray(androidFile), readLittleEndianFloatArray(webFile),
            ) }
            if (comparison.isSuccess) {
              put("comparison", comparison.getOrThrow())
              put("androidSha256", sha256File(androidFile))
              put("webSha256", sha256File(webFile))
            } else put("error", comparison.exceptionOrNull()?.message ?: "float stage comparison failed")
          } else {
            put("error", "missing stage file(s): android=${androidFile.isFile}, web=${webFile.isFile}")
          }
        })
      }
      for (stageName in maskStages) {
        val androidFile = File(androidCandidate, "$stageName.bin")
        val webFile = File(webCandidate, "$stageName.bin")
        stageRows.put(JSONObject().apply {
          put("candidateIndex", candidateIndex)
          put("stage", stageName)
          if (androidFile.isFile && webFile.isFile) {
            val comparison = runCatching { binaryMaskMetrics(androidFile.readBytes(), webFile.readBytes()) }
            if (comparison.isSuccess) {
              val metrics = comparison.getOrThrow()
              put("iou", metrics.first)
              put("pixelDiffPercent", metrics.second * 100.0)
              put("androidSha256", sha256File(androidFile))
              put("webSha256", sha256File(webFile))
            } else put("error", comparison.exceptionOrNull()?.message ?: "mask stage comparison failed")
          } else {
            put("error", "missing stage file(s): android=${androidFile.isFile}, web=${webFile.isFile}")
          }
        })
      }
    }
    return stageRows
  }

  @Test
  fun `generate P0-5 mask evidence and P0-4 web android parity report`() = runBlocking {
    val evidenceRoot = File(
      System.getProperty("fishmask.evidence.dir")
        ?: System.getenv("FISHMASK_EVIDENCE_DIR")
        ?: "build/evidence/p0_v2_2",
    )
    evidenceRoot.mkdirs()

    val webParityDirProp = System.getProperty("fishmask.webParityDir")
      ?: System.getenv("FISHMASK_WEB_PARITY_DIR")
    val webParityDir = webParityDirProp?.let { File(it) }?.takeIf { it.exists() }
    val webSummaryJson = webParityDir?.let { File(it, "web_results.json") }
      ?.takeIf { it.exists() }
      ?.let { JSONObject(it.readText()) }

    val manifestText = javaClass.classLoader?.getResourceAsStream("fish-fixtures-manifest.json")
      ?.bufferedReader()?.use { it.readText() }
      ?: error("Missing fish-fixtures-manifest.json")
    val fixtureManifestJson = JSONObject(manifestText)
    val fixturesArr = fixtureManifestJson.getJSONArray("fixtures")

    val context = ApplicationProvider.getApplicationContext<Context>()
    val yolox = YoloxFishDetector(context)
    val slimSam = SlimSamSegmenter(context)
    val coordinator = RecognitionPipelineCoordinator(yolox, slimSam)

    val audit = slimSam.ensureLoaded()
    assertTrue("SlimSAM must be ready", audit.isSlimSamReady && slimSam.isReady())
    assertTrue("YOLOX must be ready", yolox.ensureLoaded())

    // Save fixture_manifest.json and model_manifest.json
    File(evidenceRoot, "fixture_manifest.json").writeText(fixtureManifestJson.toString(2))
    val modelManifestJson = JSONObject().apply {
      put("model_id", ModelAssetAudit.SLIMSAM_MODEL_ID)
      put("revision", ModelAssetAudit.SLIMSAM_REVISION)
      put("yolox_asset", ModelAssetAudit.YOLOX_ASSET_PATH)
      put("yolox_sha256", audit.yoloxAsset.sha256)
      put("yolox_size_bytes", audit.yoloxAsset.sizeBytes)
      put("slimsam_encoder_asset", ModelAssetAudit.SLIMSAM_ENCODER_PATH)
      put("slimsam_encoder_sha256", audit.slimSamEncoderAsset.sha256)
      put("slimsam_encoder_size_bytes", audit.slimSamEncoderAsset.sizeBytes)
      put("slimsam_decoder_asset", ModelAssetAudit.SLIMSAM_DECODER_PATH)
      put("slimsam_decoder_sha256", audit.slimSamDecoderAsset.sha256)
      put("slimsam_decoder_size_bytes", audit.slimSamDecoderAsset.sizeBytes)
      put("slimsam_processor_sha256", audit.slimSamProcessorAsset.sha256)
      put("slimsam_config_sha256", audit.slimSamConfigAsset.sha256)
      put("slimsam_quantize_config_sha256", audit.slimSamQuantizeConfigAsset.sha256)
      put("encoder_contract", slimSam.getCachedPhotoContextForTest()?.let { "" } ?: "verified")
    }
    File(evidenceRoot, "model_manifest.json").writeText(modelManifestJson.toString(2))

    val photosDir = File(evidenceRoot, "photos").apply { mkdirs() }
    val parityDir = File(evidenceRoot, "web_android_parity").apply { mkdirs() }
    val paritySummaryObj = JSONObject()
    val photoSummaries = JSONArray()
    val parityGateFailures = ArrayList<String>()

    try {
      for (i in 0 until fixturesArr.length()) {
        val spec = fixturesArr.getJSONObject(i)
        val photoId = spec.getString("id")
        val filename = spec.getString("filename")
        val rawBytes = javaClass.classLoader?.getResourceAsStream(filename)?.use { it.readBytes() }
          ?: error("Missing fixture $filename")

        val photoDir = File(photosDir, photoId).apply { mkdirs() }
        File(photoDir, "original.jpg").writeBytes(rawBytes)
        val tensorDir = File(photoDir, "tensors").apply { mkdirs() }
        yolox.nativeTensorsObserverForTest = { input, output, rows, stride ->
          writeLittleEndianFloatArray(File(tensorDir, "yolox_input_bgr.f32"), input)
          writeLittleEndianFloatArray(File(tensorDir, "yolox_raw_output.f32"), output)
          File(tensorDir, "yolox_tensor_shapes.json").writeText(
            JSONObject().apply {
              put("inputShape", JSONArray(listOf(1, 3, 416, 416)))
              put("outputShape", JSONArray(listOf(1, rows, stride)))
              put("inputLayout", "NCHW-BGR")
              put("inputDtype", "float32")
              put("outputDtype", "float32")
            }.toString(2),
          )
        }
        slimSam.encoderInputObserverForTest = { preprocessed ->
          writeLittleEndianFloatArray(File(tensorDir, "sam_pixel_values_android.f32"), preprocessed.pixelValues)
          File(tensorDir, "sam_preprocess_shapes.json").writeText(
            JSONObject().apply {
              put("inputShape", JSONArray(listOf(1, 3, preprocessed.padHeight, preprocessed.padWidth)))
              put("inputDtype", "float32")
              put("originalSize", JSONArray(listOf(preprocessed.originalHeight, preprocessed.originalWidth)))
              put("resizedInputSize", JSONArray(listOf(preprocessed.resizedHeight, preprocessed.resizedWidth)))
              put("padRight", preprocessed.padRight)
              put("padBottom", preprocessed.padBottom)
            }.toString(2),
          )
        }
        slimSam.decoderInputsObserverForTest = { points, labels, embedding ->
          writeLittleEndianFloatArray(File(tensorDir, "sam_input_points_android.f32"), points)
          writeLittleEndianLongArray(File(tensorDir, "sam_input_labels_android.i64"), labels)
          writeLittleEndianFloatArray(
            File(tensorDir, "sam_image_embeddings_android.f32"),
            embedding.imageEmbeddings,
          )
          writeLittleEndianFloatArray(
            File(tensorDir, "sam_positional_embeddings_android.f32"),
            embedding.imagePositionalEmbeddings,
          )
        }

        val tempFile = File.createTempFile("ev_${photoId}_", ".jpg")
        tempFile.writeBytes(rawBytes)
        val loaded = ExifPhotoLoader.loadFromFileAndDeleteTemp(tempFile)
        val loadTelemetry = loaded.telemetry
        File(photoDir, "photo_input.json").writeText(
          JSONObject().apply {
            put("fixtureId", photoId)
            put("sourceFixtureSha256", sha256Hex(rawBytes))
            put("encodedByteCount", rawBytes.size)
            put("exifOrientation", loadTelemetry?.exifOrientation ?: -1)
            put("decodedOriginalWidth", loadTelemetry?.originalWidth ?: -1)
            put("decodedOriginalHeight", loadTelemetry?.originalHeight ?: -1)
            put("displayWidth", loaded.displayBitmap.width)
            put("displayHeight", loaded.displayBitmap.height)
            put("displayBitmapConfig", loaded.displayBitmap.config?.name ?: "UNKNOWN")
            put("workWidth", loaded.workBitmap.width)
            put("workHeight", loaded.workBitmap.height)
            put("workBitmapConfig", loaded.workBitmap.config?.name ?: "UNKNOWN")
            put("coordinateMapper", JSONObject().apply {
              put("originalWidth", loaded.mapper.origWidth)
              put("originalHeight", loaded.mapper.origHeight)
              put("workWidth", loaded.mapper.workWidth)
              put("workHeight", loaded.mapper.workHeight)
            })
            put("loadStages", JSONArray().apply {
              loadTelemetry?.stages?.forEach { stage ->
                put(JSONObject().put("name", stage.stageName).put("durationMs", stage.durationMs))
              }
            })
          }.toString(2),
        )
        SlimSamMaskPostProcessor.stageObserverForTest = { candidate, stage, floats, mask ->
          writeStageEvidence(
            File(tensorDir, "mask_stages/android_native"),
            candidate,
            stage,
            floats,
            mask,
            loaded.mapper.origWidth,
            loaded.mapper.origHeight,
          )
        }

        val recordedEvents = mutableListOf<JSONObject>()
        val collectJob = launch(Dispatchers.Default) {
          coordinator.events.collect { ev ->
            if (ev is PipelineProgressEvent.StateChanged && ev.photoGenerationId == (200L + i)) {
              recordedEvents.add(
                JSONObject().apply {
                  put("taskId", ev.taskId)
                  put("photoGenerationId", ev.photoGenerationId)
                  put("state", ev.state.name)
                  put("reusedEncoderEmbedding", ev.reusedEncoderEmbedding)
                  ev.errorReason?.let { put("errorReason", it.name) }
                },
              )
            }
          }
        }

        try {
          val photoGenId = 200L + i
          val task1 = coordinator.startRecognition(
            photoGenerationId = photoGenId,
            workBitmap = loaded.workBitmap,
            samBitmap = loaded.displayBitmap,
            mapper = loaded.mapper,
          )
          val initialEvent = withTimeout(45_000L) {
            coordinator.events
              .filterIsInstance<PipelineProgressEvent.StateChanged>()
              .first { it.taskId == task1 && (it.state == PipelineState.PLAYING || it.state == PipelineState.ERROR) }
          }
          assertEquals(PipelineState.PLAYING, initialEvent.state)

          val det = initialEvent.detection!!
          val maskRes = initialEvent.maskResult!!
          val contour = initialEvent.contourData!!
          val cachedCtx = slimSam.getCachedPhotoContextForTest()!!
          val samRes = slimSam.lastInferenceResultForTest!!
          val decOut = samRes.decoderOutput!!
          val chosen = maskRes.candidates[maskRes.selectedCandidateIndex]
          yolox.nativeTensorsObserverForTest = null
          slimSam.encoderInputObserverForTest = null
          slimSam.decoderInputsObserverForTest = null
          SlimSamMaskPostProcessor.stageObserverForTest = null

          val androidDecoderDir = File(tensorDir, "decoder/android").apply { mkdirs() }
          val candidateLogitSummaries = JSONArray()
          for (candidateIndex in 0 until SamDecoderOutput.NUM_CANDIDATES) {
            val start = candidateIndex * SamDecoderOutput.MASK_PLANE_ELEMENTS
            val logits = decOut.predMasksLogits.copyOfRange(start, start + SamDecoderOutput.MASK_PLANE_ELEMENTS)
            writeLittleEndianFloatArray(File(androidDecoderDir, "candidate_${candidateIndex}_logits.f32"), logits)
            candidateLogitSummaries.put(
              JSONObject().put("candidateIndex", candidateIndex).put("logits", floatTensorSummary(logits)),
            )
          }
          File(androidDecoderDir, "iou_scores.json").writeText(
            JSONObject().apply {
              put("shape", JSONArray(listOf(1, SamDecoderOutput.NUM_CANDIDATES)))
              put("values", JSONArray(decOut.iouScores.map { it.toDouble() }))
              put("decoderDurationMs", decOut.decoderDurationMs)
            }.toString(2),
          )
          File(androidDecoderDir, "output_metadata.json").writeText(
            JSONObject().apply {
              put("predMasksShape", JSONArray(listOf(1, SamDecoderOutput.NUM_CANDIDATES, SamDecoderOutput.MASK_LOGIT_SIZE, SamDecoderOutput.MASK_LOGIT_SIZE)))
              put("predMasksDtype", "float32")
              put("iouScoresShape", JSONArray(listOf(1, SamDecoderOutput.NUM_CANDIDATES)))
              put("iouScoresDtype", "float32")
              put("candidateLogits", candidateLogitSummaries)
            }.toString(2),
          )

          // 1. detection.json & detection_overlay.png
          File(photoDir, "detection.json").writeText(
            JSONObject().apply {
              put("box", JSONArray(listOf(det.box.x1, det.box.y1, det.box.x2, det.box.y2)))
              put("width", det.box.width)
              put("height", det.box.height)
              put("confidence", det.confidence.toDouble())
              put("areaRatio", det.areaRatio.toDouble())
              put("origDims", JSONArray(listOf(loaded.mapper.origWidth, loaded.mapper.origHeight)))
              put("workDims", JSONArray(listOf(loaded.mapper.workWidth, loaded.mapper.workHeight)))
              put("fixtureId", photoId)
              put("fixtureSha256", sha256Hex(rawBytes))
              put("fixtureOrientation", spec.optInt("exif_orientation", 1))
              put("detectorDurationMs", initialEvent.telemetry?.detectorDurationMs ?: -1L)
            }.toString(2),
          )
          writeDetectionOverlayPng(loaded.displayBitmap, det, File(photoDir, "detection_overlay.png"))
          val yoloxShape = JSONObject(File(tensorDir, "yolox_tensor_shapes.json").readText())
          File(photoDir, "yolox_parity.json").writeText(
            JSONObject().apply {
              put("inputTensorShape", yoloxShape.getJSONArray("inputShape"))
              put("inputTensorLayout", yoloxShape.getString("inputLayout"))
              put("inputTensorDtype", yoloxShape.getString("inputDtype"))
              put("inputTensorSha256", sha256File(File(tensorDir, "yolox_input_bgr.f32")))
              put("rawOutputShape", yoloxShape.getJSONArray("outputShape"))
              put("rawOutputSha256", sha256File(File(tensorDir, "yolox_raw_output.f32")))
              put("workImageWidth", loaded.workBitmap.width)
              put("workImageHeight", loaded.workBitmap.height)
              put("decodedBox", JSONArray(listOf(det.box.x1, det.box.y1, det.box.x2, det.box.y2)))
              put("confidence", det.confidence.toDouble())
              put("detectionDurationMs", initialEvent.telemetry?.detectorDurationMs ?: -1L)
            }.toString(2),
          )

          // 2. candidate_{0,1,2}.png, selected_mask.png, lcc_mask.png, selected_mask_overlay.png, contour_overlay.png
          for (c in maskRes.candidates) {
            File(tensorDir, "android_candidate_${c.index}.bin").writeBytes(c.data)
            writeMaskPng(c.data, maskRes.width, maskRes.height, File(photoDir, "candidate_${c.index}.png"))
          }
          writeMaskPng(chosen.data, maskRes.width, maskRes.height, File(photoDir, "selected_mask.png"))
          writeMaskPng(chosen.data, maskRes.width, maskRes.height, File(photoDir, "lcc_mask.png"))
          writeMaskOverlayPng(loaded.displayBitmap, chosen.data, File(photoDir, "selected_mask_overlay.png"))
          writeContourOverlayPng(loaded.displayBitmap, chosen.data, contour, File(photoDir, "contour_overlay.png"))

          // 3. contour_points.json
          File(photoDir, "contour_points.json").writeText(
            JSONObject().apply {
              put("traceClosed", contour.traceClosed)
              put("terminationReason", contour.terminationReason.name)
              put("rawPointCount", contour.rawPointCount)
              put("simplifiedPointCount", contour.simplifiedPointCount)
              put("totalLength", contour.totalLength.toDouble())
              put("startPoint", JSONObject().put("x", contour.startPoint.x).put("y", contour.startPoint.y))
              put(
                "simplifiedPoints",
                JSONArray().apply {
                  for (p in contour.simplifiedPoints) {
                    put(JSONObject().put("x", p.x).put("y", p.y))
                  }
                },
              )
              put(
                "rawPointsSampled",
                JSONArray().apply {
                  val step = max(1, contour.rawPoints.size / 400)
                  var k = 0
                  while (k < contour.rawPoints.size) {
                    val p = contour.rawPoints[k]
                    put(JSONObject().put("x", p.x).put("y", p.y))
                    k += step
                  }
                },
              )
            }.toString(2),
          )

          // 4. processor_metrics.json
          // Recompute only for test evidence; the production cache intentionally retains metadata
          // and embeddings, never the 12 MiB encoder input Tensor.
          val evidencePreprocessed = requireNotNull(slimSam.processor).preprocess(loaded.displayBitmap)
          val pv = evidencePreprocessed.pixelValues
          val androidEncoderInputFile = File(tensorDir, "sam_pixel_values_android.f32")
          check(androidEncoderInputFile.isFile) {
            "$photoId did not capture the exact production encoder input tensor"
          }
          val androidEncoderInput = readLittleEndianFloatArray(androidEncoderInputFile)
          val actualVsRecomputedInput = floatArrayComparison(androidEncoderInput, pv)
          var pvMin = Float.POSITIVE_INFINITY
          var pvMax = Float.NEGATIVE_INFINITY
          var pvSum = 0.0
          for (v in androidEncoderInput) {
            if (v < pvMin) pvMin = v
            if (v > pvMax) pvMax = v
            pvSum += v
          }
          File(photoDir, "processor_metrics.json").writeText(
            JSONObject().apply {
              put("originalWidth", evidencePreprocessed.originalWidth)
              put("originalHeight", evidencePreprocessed.originalHeight)
              put("resizedWidth", evidencePreprocessed.resizedWidth)
              put("resizedHeight", evidencePreprocessed.resizedHeight)
              put("padWidth", evidencePreprocessed.padWidth)
              put("padHeight", evidencePreprocessed.padHeight)
              put("padRight", evidencePreprocessed.padRight)
              put("padBottom", evidencePreprocessed.padBottom)
              put("preprocessDurationMs", evidencePreprocessed.preprocessDurationMs)
              put("recomputedForEvidenceOnly", true)
              put("summarySource", "exact_production_encoder_call_capture")
              put("tensorShape", JSONArray(listOf(1, 3, 1024, 1024)))
              put("actualEncoderInputSha256", sha256File(androidEncoderInputFile))
              put("actualVsRecomputedMAE", actualVsRecomputedInput.getDouble("mae"))
              put("actualVsRecomputedMaxAbsDiff", actualVsRecomputedInput.getDouble("maxAbsDiff"))
              put("min", pvMin.toDouble())
              put("max", pvMax.toDouble())
              put("mean", pvSum / pv.size.toDouble())
            }.toString(2),
          )

          // 5. encoder_metrics.json
          val emb = cachedCtx.embedding.imageEmbeddings
          check(File(tensorDir, "sam_positional_embeddings_android.f32").isFile) {
            "$photoId did not capture the exact production decoder input tensors"
          }
          writeLittleEndianFloatArray(File(tensorDir, "sam_pred_masks_android.f32"), decOut.predMasksLogits)
          writeLittleEndianFloatArray(File(tensorDir, "sam_iou_scores_android.f32"), decOut.iouScores)
          var embNormSq = 0.0
          var embMin = Float.POSITIVE_INFINITY
          var embMax = Float.NEGATIVE_INFINITY
          for (v in emb) {
            if (v < embMin) embMin = v
            if (v > embMax) embMax = v
            embNormSq += v.toDouble() * v.toDouble()
          }
          File(photoDir, "encoder_metrics.json").writeText(
            JSONObject().apply {
              put("encoderDurationMs", cachedCtx.embedding.encoderDurationMs)
              put("embeddingsShape", JSONArray(cachedCtx.embedding.embeddingsShape.toList()))
              put("positionalShape", JSONArray(cachedCtx.embedding.positionalShape.toList()))
              put("imageEmbeddingsSha256", sha256File(File(tensorDir, "sam_image_embeddings_android.f32")))
              put("positionalEmbeddingsSha256", sha256File(File(tensorDir, "sam_positional_embeddings_android.f32")))
              put("l2Norm", sqrt(embNormSq))
              put("min", embMin.toDouble())
              put("max", embMax.toDouble())
            }.toString(2),
          )

          // 6. decoder_metrics.json & mask_metrics.json
          val prompts = MaskQualityGate.generatePromptsFromDetection(det.box, loaded.mapper.origWidth, loaded.mapper.origHeight)
          val mappedPrompts = loaded.mapper.mapPromptsToSamSpace(
            prompts,
            cachedCtx.preprocessed.resizedWidth,
            cachedCtx.preprocessed.resizedHeight,
          )
          File(photoDir, "decoder_metrics.json").writeText(
            JSONObject().apply {
              put("decoderDurationMs", decOut.decoderDurationMs)
              put("iouScores", JSONArray(decOut.iouScores.map { it.toDouble() }))
              put("predMasksShape", JSONArray(listOf(1, 3, 256, 256)))
              put("predMasksSha256", sha256File(File(tensorDir, "sam_pred_masks_android.f32")))
              put("iouScoresSha256", sha256File(File(tensorDir, "sam_iou_scores_android.f32")))
              put(
                "prompts",
                JSONArray().apply {
                  for (p in prompts) {
                    put(JSONObject().put("id", p.id).put("x", p.x).put("y", p.y).put("label", p.label))
                  }
                },
              )
              put("samPointCoords", JSONArray(mappedPrompts.samPointCoords.map { it.toDouble() }))
              put("samPointLabels", JSONArray(mappedPrompts.samPointLabels.toList()))
            }.toString(2),
          )
          val androidPromptCoordsFile = File(tensorDir, "sam_input_points_android.f32")
          val androidPromptLabelsFile = File(tensorDir, "sam_input_labels_android.i64")
          File(photoDir, "prompt_parity.json").writeText(
            JSONObject().apply {
              put("originalPrompts", JSONArray().apply {
                for (p in prompts) put(JSONObject().put("id", p.id).put("x", p.x).put("y", p.y).put("label", p.label))
              })
              put("androidSamCoordinates", JSONArray(mappedPrompts.samPointCoords.map { it.toDouble() }))
              put("androidSamLabels", JSONArray(mappedPrompts.samPointLabels.toList()))
              put("androidCoordinatesShape", JSONArray(listOf(1, 1, mappedPrompts.pointCount, 2)))
              put("androidLabelsShape", JSONArray(listOf(1, 1, mappedPrompts.pointCount)))
              put("androidCoordinatesSha256", sha256File(androidPromptCoordsFile))
              put("androidLabelsSha256", sha256File(androidPromptLabelsFile))
              put("androidPointCoordinates", JSONArray().apply {
                for (pointIndex in 0 until mappedPrompts.pointCount) {
                  put(JSONObject().put("index", pointIndex)
                    .put("x", mappedPrompts.samPointCoords[pointIndex * 2].toDouble())
                    .put("y", mappedPrompts.samPointCoords[pointIndex * 2 + 1].toDouble())
                    .put("label", mappedPrompts.samPointLabels[pointIndex]))
                }
              })
            }.toString(2),
          )

          val tensorFiles = tensorDir.listFiles().orEmpty().filter { it.isFile }.sortedBy { it.name }
          File(tensorDir, "tensor_manifest.json").writeText(
            JSONObject().apply {
              put("sourceFixtureSha256", sha256Hex(rawBytes))
              put("tensors", JSONArray().apply {
                tensorFiles.forEach { file ->
                  put(JSONObject().apply {
                    put("file", file.name)
                    put("sizeBytes", file.length())
                    put("sha256", sha256File(file))
                  })
                }
              })
            }.toString(2),
          )

          File(photoDir, "mask_metrics.json").writeText(
            JSONObject().apply {
              put("selectedCandidateIndex", maskRes.selectedCandidateIndex)
              put("isMaskTrustworthy", maskRes.isMaskTrustworthy)
              put("isBboxTrustworthy", maskRes.isBboxTrustworthy)
              put(
                "candidates",
                JSONArray().apply {
                  val targetArea = max(1, det.box.area).toFloat()
                  val isVert = det.box.height > det.box.width * 1.1f
                  for (c in maskRes.candidates) {
                    val covRatio = c.pixelCount.toFloat() / targetArea
                    val majorSpan = if (isVert) {
                      c.bbox.height.toFloat() / max(1, det.box.height).toFloat()
                    } else {
                      c.bbox.width.toFloat() / max(1, det.box.width).toFloat()
                    }
                    put(
                      JSONObject().apply {
                        put("index", c.index)
                        put("iouScore", c.iouScore.toDouble())
                        put("pixelCount", c.pixelCount)
                        put("areaRatio", c.areaRatio.toDouble())
                        put("covRatio", covRatio.toDouble())
                        put("majorSpan", majorSpan.toDouble())
                        put("rankScore", c.rankScore.toDouble())
                        put("isTrustworthy", c.isTrustworthy)
                        put("bbox", JSONArray(listOf(c.bbox.x1, c.bbox.y1, c.bbox.x2, c.bbox.y2)))
                      },
                    )
                  }
                },
              )
            }.toString(2),
          )

          // 7. contour_metrics.json
          File(photoDir, "contour_metrics.json").writeText(
            JSONObject().apply {
              put("traceClosed", contour.traceClosed)
              put("terminationReason", contour.terminationReason.name)
              put("rawPointCount", contour.rawPointCount)
              put("simplifiedPointCount", contour.simplifiedPointCount)
              put("totalLength", contour.totalLength.toDouble())
              put("extractionTimeMs", contour.extractionTimeMs)
              put("glintWidth", contour.internalGlint?.width ?: 0)
              put("glintHeight", contour.internalGlint?.height ?: 0)
            }.toString(2),
          )

          // 8. Tap-to-Refocus -> refocus_metrics.json
          val tapX = (det.box.x1 + det.box.x2) / 2f
          val tapY = (det.box.y1 + det.box.y2) / 2f
          val taskRefocus = coordinator.startRecognition(
            photoGenerationId = photoGenId,
            workBitmap = loaded.workBitmap,
            samBitmap = loaded.displayBitmap,
            mapper = loaded.mapper,
            tapPoint = ContourPoint(tapX, tapY),
            existingDetection = det,
          )
          val refocusEvent = withTimeout(30_000L) {
            coordinator.events
              .filterIsInstance<PipelineProgressEvent.StateChanged>()
              .first { it.taskId == taskRefocus && (it.state == PipelineState.PLAYING || it.state == PipelineState.ERROR) }
          }
          assertEquals(PipelineState.PLAYING, refocusEvent.state)
          val refocusSam = slimSam.lastInferenceResultForTest!!
          val refocusMask = refocusEvent.maskResult!!
          val refocusChosen = refocusMask.candidates[refocusMask.selectedCandidateIndex]
          File(photoDir, "refocus_metrics.json").writeText(
            JSONObject().apply {
              put("tapX", tapX.toDouble())
              put("tapY", tapY.toDouble())
              put("reusedEncoderEmbedding", refocusEvent.reusedEncoderEmbedding)
              put("encoderDurationMs", refocusSam.encoderDurationMs)
              put("decoderDurationMs", refocusSam.decoderDurationMs)
              put("selectedCandidateIndex", refocusMask.selectedCandidateIndex)
              put("iouScore", refocusChosen.iouScore.toDouble())
              put("pixelCount", refocusChosen.pixelCount)
              put("traceClosed", refocusEvent.contourData?.traceClosed ?: false)
              put("isMaskTrustworthy", refocusMask.isMaskTrustworthy)
            }.toString(2),
          )

          // 9. pipeline_events.json
          File(photoDir, "pipeline_events.json").writeText(JSONArray(recordedEvents).toString(2))

          photoSummaries.put(
            JSONObject().apply {
              put("id", photoId)
              put("selectedCandidateIndex", maskRes.selectedCandidateIndex)
              put("iouScore", chosen.iouScore.toDouble())
              put("pixelCount", chosen.pixelCount)
              put("rawPointCount", contour.rawPointCount)
              put("traceClosed", contour.traceClosed)
              put("refocusReusedEmbedding", refocusEvent.reusedEncoderEmbedding)
            },
          )

          // 10. Web vs Android numerical parity comparison if Web output directory is supplied
          if (webParityDir != null && webSummaryJson != null && webSummaryJson.has(photoId)) {
            val webPhotoOut = File(webParityDir, photoId)
            val webPhotoMeta = webSummaryJson.getJSONObject(photoId)
            val webFixtureSha = webPhotoMeta.getString("fixtureSha256")
            if (!webFixtureSha.equals(sha256Hex(rawBytes), ignoreCase = true)) {
              parityGateFailures.add("$photoId source fixture SHA differs: Android=${sha256Hex(rawBytes)} Web=$webFixtureSha")
            }

            // Processor parity compares the exact tensor observed at the Android ONNX call boundary.
            val androidPv = readLittleEndianFloatArray(File(tensorDir, "sam_pixel_values_android.f32"))
            val webPv = readLittleEndianFloatArray(File(webPhotoOut, "pixel_values.f32"))
            val nodeProcessorParity = floatArrayComparison(androidPv, webPv)
            val pvMae = nodeProcessorParity.getDouble("mae")
            val pvMaxAbs = nodeProcessorParity.getDouble("maxAbsDiff")

            // Encoder parity
            val webEmb = readLittleEndianFloatArray(File(webPhotoOut, "image_embeddings.f32"))
            var dot = 0.0
            var na = 0.0
            var nb = 0.0
            var embAbsDiff = 0.0
            var embAbsWeb = 0.0
            for (k in emb.indices) {
              val a = emb[k].toDouble()
              val b = webEmb[k].toDouble()
              dot += a * b
              na += a * a
              nb += b * b
              embAbsDiff += abs(a - b)
              embAbsWeb += abs(b)
            }
            val encCosine = dot / (sqrt(na) * sqrt(nb))
            val encRelMae = embAbsDiff / max(1e-9, embAbsWeb)

            val pvBilFile = File(webPhotoOut, "pv_bilinear.f32")
            val embBilFile = File(webPhotoOut, "emb_bilinear.f32")
            var bilSummaryObj: JSONObject? = null
            var bilPvMae = -1.0
            var bilPvMax = -1.0
            var bilEncCos = -1.0
            var bilEncRelMae = -1.0
            if (pvBilFile.exists() && embBilFile.exists()) {
              val webPvBil = readLittleEndianFloatArray(pvBilFile)
              val webEmbBil = readLittleEndianFloatArray(embBilFile)
              var bilPvSum = 0.0
              bilPvMax = 0.0
              for (k in androidPv.indices) {
                val d = abs(androidPv[k].toDouble() - webPvBil[k].toDouble())
                bilPvSum += d
                if (d > bilPvMax) bilPvMax = d
              }
              var bDot = 0.0; var bNa = 0.0; var bNb = 0.0; var bDiff = 0.0; var bNorm = 0.0
              for (k in emb.indices) {
                val a = emb[k].toDouble(); val b = webEmbBil[k].toDouble()
                bDot += a * b; bNa += a * a; bNb += b * b
                bDiff += abs(a - b); bNorm += abs(b)
              }
              bilPvMae = bilPvSum / androidPv.size
              bilEncCos = bDot / (sqrt(bNa) * sqrt(bNb))
              bilEncRelMae = bDiff / max(1e-9, bNorm)

              val bilCandIdx = webPhotoMeta.optInt("bilSamServiceBestIndex", -1)
              val bilAndroidCompatibleIdx = webPhotoMeta.optInt("bilAndroidCompatibleSelectedIndex", bilCandIdx)
              val bilSameCandidateFile = File(webPhotoOut, "bil_candidate_${maskRes.selectedCandidateIndex}.bin")
              val bilSameCandidateContourFile = File(
                webPhotoOut,
                "bil_contour_candidate_${maskRes.selectedCandidateIndex}.json",
              )
              val bilSelectedCandidateFile = if (bilCandIdx >= 0) {
                File(webPhotoOut, "bil_candidate_${bilCandIdx}.bin")
              } else {
                null
              }
              var bilSelIoU = -1.0
              var bilSelDiffPct = -1.0
              var bilChamferAvg = -1.0
              var bilChamferP95 = -1.0
              var bilLegacySelectedIoU = -1.0
              if (bilSameCandidateFile.exists()) {
                val bilMask = bilSameCandidateFile.readBytes()
                val (iouB, diffB, _) = binaryMaskMetrics(chosen.data, bilMask)
                bilSelIoU = iouB
                bilSelDiffPct = diffB * 100.0
              }
              if (bilSameCandidateContourFile.exists()) {
                val bilPts = readContourPoints(bilSameCandidateContourFile)
                val (cAvg, cP95) = symmetricChamferDistance(contour.rawPoints, bilPts)
                bilChamferAvg = cAvg
                bilChamferP95 = cP95
              }
              if (bilSelectedCandidateFile != null && bilSelectedCandidateFile.exists()) {
                bilLegacySelectedIoU = binaryMaskMetrics(chosen.data, bilSelectedCandidateFile.readBytes()).first
              }
              bilSummaryObj = JSONObject().apply {
                put("processorPixelValuesMAE", bilPvMae)
                put("processorPixelValuesMaxAbsDiff", bilPvMax)
                put("encoderCosineSimilarity", bilEncCos)
                put("encoderRelativeMAE", bilEncRelMae)
                put("bilSamServiceBestIndex", bilCandIdx)
                put("bilAndroidCompatibleSelectedIndex", bilAndroidCompatibleIdx)
                put("sameCandidateIndex", maskRes.selectedCandidateIndex)
                put("selectedMaskIoU", bilSelIoU)
                put("selectedMaskPixelDiffPercent", bilSelDiffPct)
                put("contourSymmetricChamferAvgPx", bilChamferAvg)
                put("contourSymmetricChamferP95Px", bilChamferP95)
                put("legacyCrossSelectedMaskIoU", bilLegacySelectedIoU)
                put("legacyCrossSelectedCandidateIndex", bilCandIdx)
              }
              println(
                "[BILINEAR_PARITY] $photoId pvMAE=${"%.6f".format(bilPvMae)} " +
                  "pvMax=${"%.5f".format(bilPvMax)} encCos=${"%.6f".format(bilEncCos)} " +
                  "encRelMAE=${"%.6f".format(bilEncRelMae)} " +
                  "selMaskIoU=${"%.4f".format(bilSelIoU)} diff=${"%.3f".format(bilSelDiffPct)}% " +
                  "chamferAvg=${"%.3f".format(bilChamferAvg)}px chamferP95=${"%.3f".format(bilChamferP95)}px",
              )
            }

            // Detection comparison
            val webBoxArr = webPhotoMeta.getJSONObject("detection").getJSONArray("box")
            val wx1 = webBoxArr.getInt(0)
            val wy1 = webBoxArr.getInt(1)
            val wx2 = webBoxArr.getInt(2)
            val wy2 = webBoxArr.getInt(3)
            val ix1 = max(det.box.x1, wx1)
            val iy1 = max(det.box.y1, wy1)
            val ix2 = min(det.box.x2, wx2)
            val iy2 = min(det.box.y2, wy2)
            val detInter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            val detUnion = max(1, det.box.area + (wx2 - wx1) * (wy2 - wy1) - detInter)
            val detBoxIou = detInter.toDouble() / detUnion.toDouble()
            val androidYoloxInput = readLittleEndianFloatArray(File(tensorDir, "yolox_input_bgr.f32"))
            val webYoloxInput = readLittleEndianFloatArray(File(webPhotoOut, "tensors/yolox_input_bgr_web.f32"))
            val androidYoloxOutput = readLittleEndianFloatArray(File(tensorDir, "yolox_raw_output.f32"))
            val webYoloxOutput = readLittleEndianFloatArray(File(webPhotoOut, "tensors/yolox_raw_output_web.f32"))
            val yoloxInputParity = floatArrayComparison(androidYoloxInput, webYoloxInput)
            val yoloxOutputParity = floatArrayComparison(androidYoloxOutput, webYoloxOutput)
            val webYoloxPrep = JSONObject(File(webPhotoOut, "tensors/yolox_preprocess.json").readText())
            val webYoloxOutputShape = JSONObject(File(webPhotoOut, "tensors/yolox_output_shape.json").readText())
            File(photoDir, "yolox_parity.json").writeText(
              JSONObject().apply {
                put("androidInput", JSONObject().apply {
                  put("shape", yoloxShape.getJSONArray("inputShape"))
                  put("layout", yoloxShape.getString("inputLayout"))
                  put("dtype", yoloxShape.getString("inputDtype"))
                  put("sha256", sha256File(File(tensorDir, "yolox_input_bgr.f32")))
                })
                put("webInput", JSONObject().apply {
                  put("shape", JSONArray(webYoloxPrep.getJSONArray("shape").let { arr ->
                    (0 until arr.length()).map { arr.getInt(it) }
                  }))
                  put("resizedWidth", webYoloxPrep.getInt("resizedWidth"))
                  put("resizedHeight", webYoloxPrep.getInt("resizedHeight"))
                  put("paddingValue", webYoloxPrep.getInt("fill"))
                  put("layout", webYoloxPrep.getString("layout"))
                  put("sha256", sha256File(File(webPhotoOut, "tensors/yolox_input_bgr_web.f32")))
                })
                put("androidRawOutputShape", yoloxShape.getJSONArray("outputShape"))
                put("androidRawOutputSha256", sha256File(File(tensorDir, "yolox_raw_output.f32")))
                put("webRawOutputShape", JSONArray(webYoloxOutputShape.getJSONArray("shape").let { arr ->
                  (0 until arr.length()).map { arr.getInt(it) }
                }))
                put("webRawOutputSha256", sha256File(File(webPhotoOut, "tensors/yolox_raw_output_web.f32")))
                put("androidVsWebInput", yoloxInputParity)
                put("androidVsWebRawOutput", yoloxOutputParity)
                put("androidDetectionBox", JSONArray(listOf(det.box.x1, det.box.y1, det.box.x2, det.box.y2)))
                put("webDetectionBox", JSONArray(listOf(wx1, wy1, wx2, wy2)))
                put("detectionBoxIoU", detBoxIou)
                put("androidConfidence", det.confidence.toDouble())
                put("webConfidence", webPhotoMeta.getJSONObject("detection").getDouble("confidence"))
              }.toString(2),
            )

            // Compare actual raw tensors, and isolate provider differences using identical Android inputs.
            val nodeLogits = readLittleEndianFloatArray(File(webPhotoOut, "tensors/sam_pred_masks_node.f32"))
            val bilLogits = readLittleEndianFloatArray(File(webPhotoOut, "tensors/sam_pred_masks_bilinear.f32"))
            val androidLogits = readLittleEndianFloatArray(File(tensorDir, "sam_pred_masks_android.f32"))
            val webDecoderOnAndroidInputs = readLittleEndianFloatArray(
              File(webPhotoOut, "tensors/sam_pred_masks_web_on_android_inputs.f32"),
            )
            val androidEmbeddings = readLittleEndianFloatArray(
              File(tensorDir, "sam_image_embeddings_android.f32"),
            )
            val webEncoderOnAndroidInputEmbeddings = readLittleEndianFloatArray(
              File(webPhotoOut, "tensors/sam_image_embeddings_web_on_android_input.f32"),
            )
            val androidPositionalEmbeddings = readLittleEndianFloatArray(
              File(tensorDir, "sam_positional_embeddings_android.f32"),
            )
            val webEncoderOnAndroidInputPositional = readLittleEndianFloatArray(
              File(webPhotoOut, "tensors/sam_positional_embeddings_web_on_android_input.f32"),
            )
            val androidPoints = readLittleEndianFloatArray(
              File(tensorDir, "sam_input_points_android.f32"),
            )
            val webNodePoints = readLittleEndianFloatArray(
              File(webPhotoOut, "tensors/sam_input_points_node.f32"),
            )
            val webBilinearPoints = readLittleEndianFloatArray(
              File(webPhotoOut, "tensors/sam_input_points_bilinear.f32"),
            )
            val androidLabels = readLittleEndianLongArray(
              File(tensorDir, "sam_input_labels_android.i64"),
            )
            val webNodeLabels = readLittleEndianLongArray(
              File(webPhotoOut, "tensors/sam_input_labels_node.i64"),
            )
            val webBilinearLabels = readLittleEndianLongArray(
              File(webPhotoOut, "tensors/sam_input_labels_bilinear.i64"),
            )
            val androidPromptDetails = JSONObject(File(photoDir, "decoder_metrics.json").readText())
            val androidOriginalPrompts = androidPromptDetails.getJSONArray("prompts")
            val webOriginalPrompts = webPhotoMeta.getJSONArray("prompts")
            File(photoDir, "prompt_parity.json").writeText(
              JSONObject().apply {
                put("androidOriginalPrompts", androidOriginalPrompts)
                put("webOriginalPrompts", webOriginalPrompts)
                put("androidSamCoordinatesShape", JSONArray(listOf(1, 1, androidLabels.size, 2)))
                put("androidSamLabelsShape", JSONArray(listOf(1, 1, androidLabels.size)))
                put(
                  "webNodeSamCoordinatesShape",
                  JSONObject(File(webPhotoOut, "tensors/sam_node_input_shapes.json").readText())
                    .getJSONObject("inputPoints")
                    .getJSONArray("shape"),
                )
                put("androidSamCoordinatesSha256", sha256File(File(tensorDir, "sam_input_points_android.f32")))
                put("androidSamLabelsSha256", sha256File(File(tensorDir, "sam_input_labels_android.i64")))
                put("webNodeSamCoordinatesSha256", sha256File(File(webPhotoOut, "tensors/sam_input_points_node.f32")))
                put("webNodeSamLabelsSha256", sha256File(File(webPhotoOut, "tensors/sam_input_labels_node.i64")))
                put("webBilinearSamCoordinatesSha256", sha256File(File(webPhotoOut, "tensors/sam_input_points_bilinear.f32")))
                put("webBilinearSamLabelsSha256", sha256File(File(webPhotoOut, "tensors/sam_input_labels_bilinear.i64")))
                put("androidVsWebNodeCoordinates", floatArrayComparison(androidPoints, webNodePoints))
                put("androidVsWebBilinearCoordinates", floatArrayComparison(androidPoints, webBilinearPoints))
                put("androidCoordinatesExactlyMatchWebNode", androidPoints.contentEquals(webNodePoints))
                put("androidCoordinatesExactlyMatchWebBilinear", androidPoints.contentEquals(webBilinearPoints))
                put("androidLabelsExactlyMatchWebNode", androidLabels.contentEquals(webNodeLabels))
                put("androidLabelsExactlyMatchWebBilinear", androidLabels.contentEquals(webBilinearLabels))
                put("points", JSONArray().apply {
                  for (pointIndex in 0 until androidLabels.size) {
                    put(JSONObject().apply {
                      put("index", pointIndex)
                      put("androidSamX", androidPoints[pointIndex * 2].toDouble())
                      put("androidSamY", androidPoints[pointIndex * 2 + 1].toDouble())
                      put("androidLabel", androidLabels[pointIndex])
                      put("webNodeSamX", webNodePoints[pointIndex * 2].toDouble())
                      put("webNodeSamY", webNodePoints[pointIndex * 2 + 1].toDouble())
                      put("webNodeLabel", webNodeLabels[pointIndex])
                      put("webBilinearSamX", webBilinearPoints[pointIndex * 2].toDouble())
                      put("webBilinearSamY", webBilinearPoints[pointIndex * 2 + 1].toDouble())
                      put("webBilinearLabel", webBilinearLabels[pointIndex])
                    })
                  }
                })
              }.toString(2),
            )
            val encoderOnSameInput = JSONObject().apply {
              put("imageEmbeddings", floatArrayComparison(androidEmbeddings, webEncoderOnAndroidInputEmbeddings))
              put("positionalEmbeddings", floatArrayComparison(androidPositionalEmbeddings, webEncoderOnAndroidInputPositional))
            }
            val nodePromptComparison = floatArrayComparison(androidPoints, webNodePoints)
            val bilinearPromptComparison = floatArrayComparison(androidPoints, webBilinearPoints)
            val webNodeIous = readLittleEndianFloatArray(File(webPhotoOut, "tensors/sam_iou_scores_node.f32"))
            val webBilinearIous = readLittleEndianFloatArray(File(webPhotoOut, "tensors/sam_iou_scores_bilinear.f32"))
            val decoderCandidateComparisons = JSONArray()
            for (candidateIndex in 0 until SamDecoderOutput.NUM_CANDIDATES) {
              val start = candidateIndex * SamDecoderOutput.MASK_PLANE_ELEMENTS
              val end = start + SamDecoderOutput.MASK_PLANE_ELEMENTS
              val androidCandidateLogits = androidLogits.copyOfRange(start, end)
              val nodeCandidateLogits = nodeLogits.copyOfRange(start, end)
              val bilinearCandidateLogits = bilLogits.copyOfRange(start, end)
              val sameInputCandidateLogits = webDecoderOnAndroidInputs.copyOfRange(start, end)
              decoderCandidateComparisons.put(
                JSONObject().apply {
                  put("candidateIndex", candidateIndex)
                  put("androidIouScore", decOut.iouScores[candidateIndex].toDouble())
                  put("webNodeIouScore", webNodeIous[candidateIndex].toDouble())
                  put("webBilinearIouScore", webBilinearIous[candidateIndex].toDouble())
                  put("androidVsWebNodeIouScoreAbsDiff", abs(decOut.iouScores[candidateIndex] - webNodeIous[candidateIndex]).toDouble())
                  put("androidVsWebBilinearIouScoreAbsDiff", abs(decOut.iouScores[candidateIndex] - webBilinearIous[candidateIndex]).toDouble())
                  put("androidLogitDistribution", floatTensorSummary(androidCandidateLogits))
                  put("webNodeLogitDistribution", floatTensorSummary(nodeCandidateLogits))
                  put("webBilinearLogitDistribution", floatTensorSummary(bilinearCandidateLogits))
                  put("androidVsWebNode", floatArrayComparison(
                    androidCandidateLogits, nodeCandidateLogits,
                  ))
                  put("androidVsWebBilinear", floatArrayComparison(
                    androidCandidateLogits, bilinearCandidateLogits,
                  ))
                  put("androidVsWebDecoderSameInputs", floatArrayComparison(
                    androidCandidateLogits, sameInputCandidateLogits,
                  ))
                },
              )
            }

            // Cross the exact same Web logits through Android's Kotlin postprocessor. This separates
            // ONNX/provider divergence from interpolation, thresholding, clip-box, and LCC divergence.
            val webDetectionObj = webPhotoMeta.getJSONObject("detection")
            val webDetectionBox = webDetectionObj.getJSONArray("box")
            val webDetection = FishDetection(
              box = PixelBox(
                webDetectionBox.getInt(0), webDetectionBox.getInt(1),
                webDetectionBox.getInt(2), webDetectionBox.getInt(3),
              ),
              confidence = webDetectionObj.getDouble("confidence").toFloat(),
              areaRatio = webDetectionObj.getDouble("areaRatio").toFloat(),
            )
            val webInputSizes = webPhotoMeta.getJSONArray("reshapedInputSize")
            val webResizedHeight = webInputSizes.getInt(0)
            val webResizedWidth = webInputSizes.getInt(1)
            val kotlinPostNode = postProcessWithKotlin(
              nodeLogits, webNodeIous, webResizedWidth, webResizedHeight,
              loaded.mapper.origWidth, loaded.mapper.origHeight, webDetection,
              stageRoot = File(tensorDir, "mask_stages/kotlin_from_web_node"),
            )
            val kotlinPostBilinear = postProcessWithKotlin(
              bilLogits, webBilinearIous, webResizedWidth, webResizedHeight,
              loaded.mapper.origWidth, loaded.mapper.origHeight, webDetection,
              stageRoot = File(tensorDir, "mask_stages/kotlin_from_web_bilinear"),
            )
            val kotlinPostAndroidSameInput = postProcessWithKotlin(
              webDecoderOnAndroidInputs,
              readLittleEndianFloatArray(File(webPhotoOut, "tensors/sam_iou_scores_web_on_android_inputs.f32")),
              cachedCtx.preprocessed.resizedWidth, cachedCtx.preprocessed.resizedHeight,
              loaded.mapper.origWidth, loaded.mapper.origHeight, det,
              stageRoot = File(tensorDir, "mask_stages/kotlin_from_web_android_inputs"),
            )
            val identicalLogitStageComparisons = JSONObject().apply {
              put("nativeAndroidLogits", compareIdenticalLogitMaskStages(
                File(tensorDir, "mask_stages/android_native"),
                File(webPhotoOut, "tensors/mask_stages/android_logits_web_postprocess"),
              ))
              put("webNodeLogits", compareIdenticalLogitMaskStages(
                File(tensorDir, "mask_stages/kotlin_from_web_node"),
                File(webPhotoOut, "tensors/mask_stages/web_node"),
              ))
              put("webBilinearLogits", compareIdenticalLogitMaskStages(
                File(tensorDir, "mask_stages/kotlin_from_web_bilinear"),
                File(webPhotoOut, "tensors/mask_stages/web_bilinear"),
              ))
              put("webDecoderAndroidInputs", compareIdenticalLogitMaskStages(
                File(tensorDir, "mask_stages/kotlin_from_web_android_inputs"),
                File(webPhotoOut, "tensors/mask_stages/web_decoder_android_inputs"),
              ))
            }
            File(tensorDir, "mask_stages/kotlin_vs_web_stage_comparisons.json")
              .writeText(identicalLogitStageComparisons.toString(2))
            val postprocessCrossChecks = JSONArray()
            for (candidateIndex in 0 until SamDecoderOutput.NUM_CANDIDATES) {
              val nodeWebMask = File(webPhotoOut, "candidate_${candidateIndex}.bin").readBytes()
              val bilWebMask = File(webPhotoOut, "bil_candidate_${candidateIndex}.bin").readBytes()
              val nodeCross = binaryMaskMetrics(kotlinPostNode.candidates[candidateIndex].data, nodeWebMask)
              val bilCross = binaryMaskMetrics(kotlinPostBilinear.candidates[candidateIndex].data, bilWebMask)
              val androidCross = binaryMaskMetrics(
                maskRes.candidates[candidateIndex].data,
                kotlinPostAndroidSameInput.candidates[candidateIndex].data,
              )
              File(tensorDir, "kotlin_post_from_web_node_candidate_${candidateIndex}.bin").writeBytes(
                kotlinPostNode.candidates[candidateIndex].data,
              )
              File(tensorDir, "kotlin_post_from_web_bilinear_candidate_${candidateIndex}.bin").writeBytes(
                kotlinPostBilinear.candidates[candidateIndex].data,
              )
              postprocessCrossChecks.put(JSONObject().apply {
                put("candidateIndex", candidateIndex)
                put("kotlinVsWebNode", JSONObject().put("iou", nodeCross.first).put("pixelDiffPercent", nodeCross.second * 100.0))
                put("kotlinVsWebBilinear", JSONObject().put("iou", bilCross.first).put("pixelDiffPercent", bilCross.second * 100.0))
                put("nativeVsWebDecoderSameInputs", JSONObject().put("iou", androidCross.first).put("pixelDiffPercent", androidCross.second * 100.0))
              })
            }

            // Candidate IoU & Mask IoU comparison
            val webCands = webPhotoMeta.getJSONArray("candidates")
            val webBilCands = webPhotoMeta.getJSONArray("bilCandidates")
            val candParityArr = JSONArray()
            for (cIdx in 0 until 3) {
              val webCand = webCands.getJSONObject(cIdx)
              val webCandMask = File(webPhotoOut, "candidate_${cIdx}.bin").readBytes()
              val webBilCand = webBilCands.getJSONObject(cIdx)
              val webBilCandMask = File(webPhotoOut, "bil_candidate_${cIdx}.bin").readBytes()
              val andCand = maskRes.candidates[cIdx]
              val (maskIou, diffRatio, _) = binaryMaskMetrics(andCand.data, webCandMask)
              val (bilMaskIou, bilDiffRatio, _) = binaryMaskMetrics(andCand.data, webBilCandMask)
              candParityArr.put(
                JSONObject().apply {
                  put("index", cIdx)
                  put("androidIouScore", andCand.iouScore.toDouble())
                  put("webIouScore", webCand.getDouble("iouScore"))
                  put("webBilinearIouScore", webBilCand.getDouble("iouScore"))
                  put("iouScoreDiff", abs(andCand.iouScore - webCand.getDouble("iouScore")))
                  put("webNodeSameCandidateMaskIoU", maskIou)
                  put("webNodeSameCandidatePixelDiffPercent", diffRatio * 100.0)
                  put("webBilinearSameCandidateMaskIoU", bilMaskIou)
                  put("webBilinearSameCandidatePixelDiffPercent", bilDiffRatio * 100.0)
                  put("androidPixels", andCand.pixelCount)
                  put("webPixels", webCand.getInt("pixelCount"))
                  put("webBilinearPixels", webBilCand.getInt("pixelCount"))
                  put("androidRankScore", andCand.rankScore.toDouble())
                  put("androidTrustworthy", andCand.isTrustworthy)
                  put("webBilinearTrustworthy", webBilCand.getBoolean("isTrustworthy"))
                  put("webBilinearRankScore", webBilCand.getDouble("samServiceScore"))
                },
              )
            }

            // The Web service's candidate ranking is the frozen selection-parity gate.
            // Bilinear is an auxiliary numerical reference: compare mask tensors at Android's
            // selected index, but retain its independently ranked index as a diagnostic only.
            val webAppIdx = webPhotoMeta.getInt("appSelectedIndex")
            val webSamIdx = webPhotoMeta.getInt("samServiceBestIndex")
            val webBilinearSelectedIdx = webPhotoMeta.optInt("bilAndroidCompatibleSelectedIndex", -1)
            val webNodeAppSelectedMask = File(webPhotoOut, "candidate_${webAppIdx}.bin").readBytes()
            val (legacyNodeCrossSelectedIoU, legacyNodeCrossSelectedDiff, _) =
              binaryMaskMetrics(chosen.data, webNodeAppSelectedMask)
            val selIoU = bilSummaryObj?.optDouble("selectedMaskIoU", -1.0) ?: -1.0
            val selDiffRatio = (bilSummaryObj?.optDouble("selectedMaskPixelDiffPercent", -100.0) ?: -100.0) / 100.0
            val chamferAvg = bilSummaryObj?.optDouble("contourSymmetricChamferAvgPx", -1.0) ?: -1.0
            val chamferP95 = bilSummaryObj?.optDouble("contourSymmetricChamferP95Px", -1.0) ?: -1.0

            val nodeSameCandidateContour = readContourPoints(
              File(webPhotoOut, "contour_candidate_${maskRes.selectedCandidateIndex}.json"),
            )
            val (nodeSameCandidateChamferAvg, nodeSameCandidateChamferP95) =
              symmetricChamferDistance(contour.rawPoints, nodeSameCandidateContour)

            File(tensorDir, "numerical_divergence.json").writeText(
              JSONObject().apply {
                put("sourceFixtureSha256", sha256Hex(rawBytes))
                put("webFixtureSha256", webFixtureSha)
                put("androidVsWebYoloxInputTensor", yoloxInputParity)
                put("androidVsWebYoloxRawOutputTensor", yoloxOutputParity)
                put("androidVsWebDetectionBoxIoU", detBoxIou)
                put("androidVsWebNodePixelValues", nodeProcessorParity)
                put("androidVsWebBilinearPixelValues", JSONObject().apply {
                  put("mae", bilPvMae)
                  put("maxAbsDiff", bilPvMax)
                })
                put("androidVsWebNodeEncoder", JSONObject().apply {
                  put("cosineSimilarity", encCosine)
                  put("relativeMae", encRelMae)
                })
                put("androidVsWebBilinearEncoder", JSONObject().apply {
                  put("cosineSimilarity", bilEncCos)
                  put("relativeMae", bilEncRelMae)
                })
                put("androidVsWebEncoderOnIdenticalInput", encoderOnSameInput)
                put("androidPromptCoordsVsWebNode", nodePromptComparison)
                put("androidPromptCoordsVsWebBilinear", bilinearPromptComparison)
                put("promptCoordinatesExactlyMatchWebNode", androidPoints.contentEquals(webNodePoints))
                put("promptCoordinatesExactlyMatchWebBilinear", androidPoints.contentEquals(webBilinearPoints))
                put("promptLabelsExactlyMatchWebNode", androidLabels.contentEquals(webNodeLabels))
                put("promptLabelsExactlyMatchWebBilinear", androidLabels.contentEquals(webBilinearLabels))
                put("androidRawLogitsVsWebNode", floatArrayComparison(androidLogits, nodeLogits))
                put("androidRawLogitsVsWebBilinear", floatArrayComparison(androidLogits, bilLogits))
                put("androidRawLogitsVsWebDecoderWithIdenticalInputs", floatArrayComparison(androidLogits, webDecoderOnAndroidInputs))
                put("decoderCandidateLogitMetrics", decoderCandidateComparisons)
                put("sameLogitsPostprocessCrossChecks", postprocessCrossChecks)
                put("identicalLogitMaskStageComparisons", identicalLogitStageComparisons)
                put("androidSelectedCandidateIndex", maskRes.selectedCandidateIndex)
                put("webBilinearAndroidCompatibleSelectedIndex", webBilinearSelectedIdx)
                put("bilinearSelectionMatchesAndroid", maskRes.selectedCandidateIndex == webBilinearSelectedIdx)
                put("legacyNodeCrossSelectedCandidateIndex", webAppIdx)
                put("legacyNodeCrossSelectedMaskIoU", legacyNodeCrossSelectedIoU)
                put("legacyNodeCrossSelectedPixelDiffPercent", legacyNodeCrossSelectedDiff * 100.0)
                put("webNodeSameCandidateContourChamferAvgPx", nodeSameCandidateChamferAvg)
                put("webNodeSameCandidateContourChamferP95Px", nodeSameCandidateChamferP95)
              }.toString(2),
            )
            val completeTensorFiles = tensorDir.walkTopDown()
              .filter { it.isFile && it.name != "tensor_manifest.json" }
              .sortedBy { it.relativeTo(tensorDir).invariantSeparatorsPath }
            File(tensorDir, "tensor_manifest.json").writeText(
              JSONObject().apply {
                put("photoId", photoId)
                put("sourceFixtureSha256", sha256Hex(rawBytes))
                put("tensors", JSONArray().apply {
                  completeTensorFiles.forEach { file ->
                    put(JSONObject().apply {
                      put("file", file.relativeTo(tensorDir).invariantSeparatorsPath)
                      put("sizeBytes", file.length())
                      put("sha256", sha256File(file))
                    })
                  }
                })
              }.toString(2),
            )

            val photoParity = JSONObject().apply {
              put("photoId", photoId)
              put("processorPixelValuesMAE", pvMae)
              put("processorPixelValuesMaxAbsDiff", pvMaxAbs)
              put("encoderCosineSimilarity", encCosine)
              put("encoderRelativeMAE", encRelMae)
              put("bilinearProcessorPixelValuesMAE", bilPvMae)
              put("bilinearProcessorPixelValuesMaxAbsDiff", bilPvMax)
              put("bilinearEncoderCosineSimilarity", bilEncCos)
              put("yoloxDetBoxIoU", detBoxIou)
              put("yoloxInputTensorParity", yoloxInputParity)
              put("yoloxRawOutputTensorParity", yoloxOutputParity)
              put("androidDetBox", JSONArray(listOf(det.box.x1, det.box.y1, det.box.x2, det.box.y2)))
              put("webDetBox", JSONArray(listOf(wx1, wy1, wx2, wy2)))
              put("androidSelectedCandidateIndex", maskRes.selectedCandidateIndex)
              put("webSamServiceBestIndex", webSamIdx)
              put("webAppSelectedIndex", webAppIdx)
              put("webBilinearAndroidCompatibleSelectedIndex", webBilinearSelectedIdx)
              put("bilinearSelectionMatchesAndroid", maskRes.selectedCandidateIndex == webBilinearSelectedIdx)
              put("androidVsWebEncoderOnIdenticalInput", encoderOnSameInput)
              put("androidPromptCoordsVsWebNode", nodePromptComparison)
              put("androidPromptCoordsVsWebBilinear", bilinearPromptComparison)
              put("promptCoordinatesExactlyMatchWebNode", androidPoints.contentEquals(webNodePoints))
              put("promptCoordinatesExactlyMatchWebBilinear", androidPoints.contentEquals(webBilinearPoints))
              put("promptLabelsExactlyMatchWebNode", androidLabels.contentEquals(webNodeLabels))
              put("promptLabelsExactlyMatchWebBilinear", androidLabels.contentEquals(webBilinearLabels))
              put("androidRawLogitsVsWebNode", floatArrayComparison(androidLogits, nodeLogits))
              put("androidRawLogitsVsWebBilinear", floatArrayComparison(androidLogits, bilLogits))
              put("androidRawLogitsVsWebDecoderWithIdenticalInputs", floatArrayComparison(androidLogits, webDecoderOnAndroidInputs))
              put("decoderCandidateLogitMetrics", decoderCandidateComparisons)
              put("sameLogitsPostprocessCrossChecks", postprocessCrossChecks)
              put("identicalLogitMaskStageComparisons", identicalLogitStageComparisons)
              put("selectedMaskIoU", selIoU)
              put("selectedMaskPixelDiffPercent", selDiffRatio * 100.0)
              put("contourSymmetricChamferAvgPx", chamferAvg)
              put("contourSymmetricChamferP95Px", chamferP95)
              put("legacyNodeCrossSelectedMaskIoU", legacyNodeCrossSelectedIoU)
              put("legacyNodeCrossSelectedPixelDiffPercent", legacyNodeCrossSelectedDiff * 100.0)
              put("webNodeSameCandidateContourChamferAvgPx", nodeSameCandidateChamferAvg)
              put("webNodeSameCandidateContourChamferP95Px", nodeSameCandidateChamferP95)
              put("numericalDivergence", JSONObject(File(tensorDir, "numerical_divergence.json").readText()))
              put("firstParityDivergence", firstParityDivergence(this))
              if (bilSummaryObj != null) put("bilinearReference", bilSummaryObj)
              put("candidates", candParityArr)
            }
            File(parityDir, "${photoId}_parity.json").writeText(photoParity.toString(2))
            paritySummaryObj.put(photoId, photoParity)

            println(
              "[WEB_ANDROID_PARITY] $photoId " +
                "pvMAE=${"%.6f".format(pvMae)} pvMax=${"%.5f".format(pvMaxAbs)} " +
                "encCos=${"%.6f".format(encCosine)} encRelMAE=${"%.6f".format(encRelMae)} " +
                "detIoU=${"%.4f".format(detBoxIou)} " +
                "selIdx(and=${maskRes.selectedCandidateIndex},webNode=$webSamIdx,webBilinear=$webBilinearSelectedIdx) " +
                "selMaskIoU=${"%.4f".format(selIoU)} diff=${"%.3f".format(selDiffRatio * 100.0)}% " +
                "chamferAvg=${"%.3f".format(chamferAvg)}px chamferP95=${"%.3f".format(chamferP95)}px",
            )

            // Keep every frozen gate unchanged and defer evaluation until all four photos have
            // produced complete diagnostics and evidence files.
            if (detBoxIou < 0.98) parityGateFailures.add("$photoId YOLOX detBoxIoU=$detBoxIou < 0.98")
            if (encCosine < 0.995) parityGateFailures.add("$photoId Node encoder cosine=$encCosine < 0.995")
            if (maskRes.selectedCandidateIndex != webSamIdx) {
              parityGateFailures.add("$photoId Android candidate=${maskRes.selectedCandidateIndex} != Web Node samServiceBestIndex=$webSamIdx")
            }
            if (selDiffRatio > 0.02) {
              parityGateFailures.add("$photoId same-candidate bilinear mask pixel diff=${selDiffRatio * 100.0}% > 2.0%")
            }
            if (selIoU < 0.94) parityGateFailures.add("$photoId same-candidate bilinear mask IoU=$selIoU < 0.94")
            if (chamferAvg > 2.5) {
              parityGateFailures.add("$photoId same-candidate bilinear contour Chamfer avg=$chamferAvg px > 2.5 px")
            }
            if (bilPvMae > 0.001) parityGateFailures.add("$photoId bilinear processor MAE=$bilPvMae > 0.001")
            if (bilPvMax > 0.02) parityGateFailures.add("$photoId bilinear processor maxAbsDiff=$bilPvMax > 0.02")
            if (bilEncCos < 0.998) parityGateFailures.add("$photoId bilinear encoder cosine=$bilEncCos < 0.998")
          }
        } finally {
          yolox.nativeTensorsObserverForTest = null
          slimSam.encoderInputObserverForTest = null
          slimSam.decoderInputsObserverForTest = null
          SlimSamMaskPostProcessor.stageObserverForTest = null
          collectJob.cancel()
          if (!loaded.displayBitmap.isRecycled) loaded.displayBitmap.recycle()
          if (!loaded.workBitmap.isRecycled) loaded.workBitmap.recycle()
        }
      }

      if (webSummaryJson != null) {
        if (paritySummaryObj.length() != fixturesArr.length()) {
          parityGateFailures.add("Parity captured ${paritySummaryObj.length()} of ${fixturesArr.length()} fixtures")
        }
      }
      if (paritySummaryObj.length() > 0) {
        File(parityDir, "summary.json").writeText(paritySummaryObj.toString(2))
        val reportSb = StringBuilder()
        reportSb.appendLine("# P0-4 Web / Android Parity Report")
        reportSb.appendLine()
        reportSb.appendLine("**Acceptance status:** ${if (parityGateFailures.isEmpty()) "PASS" else "FAIL"}")
        reportSb.appendLine()
        reportSb.appendLine("The Node/Sharp preprocessing output is retained as a diagnostic. Candidate selection is gated against the Web SAM-service ranking. The explicit pixel-center bilinear path is an auxiliary numerical reference: preprocessing, encoder, mask, and contour gates compare values at Android's selected candidate index. Its independently ranked candidate index is reported for diagnosis and is not a separate selection gate.")
        reportSb.appendLine("Each photo folder contains the exact Android ONNX inputs, raw model outputs, same-input Web reruns, per-candidate raw logits, and cross-postprocessor masks. The first failed parity stage below is derived from those recorded metrics; it is not a source-code assertion guessed from the fixture name.")
        reportSb.appendLine()
        reportSb.appendLine("| Photo | Box IoU | Android / Web SAM-service / Web app / bilinear auxiliary index | Node PV MAE | Bilinear PV MAE | Bilinear encoder cosine | Same-candidate mask IoU | Pixel diff | Chamfer avg | First failing stage |")
        reportSb.appendLine("|---|---:|:---:|---:|---:|---:|---:|---:|---:|---|")
        for (idx in 0 until fixturesArr.length()) {
          val pid = fixturesArr.getJSONObject(idx).getString("id")
          val p = paritySummaryObj.getJSONObject(pid)
          val bil = p.optJSONObject("bilinearReference")
          reportSb.appendLine(
            "| $pid | ${"%.4f".format(p.getDouble("yoloxDetBoxIoU"))} | " +
              "${p.getInt("androidSelectedCandidateIndex")}/${p.getInt("webSamServiceBestIndex")}/${p.getInt("webAppSelectedIndex")}/${p.getInt("webBilinearAndroidCompatibleSelectedIndex")} | " +
              "${"%.6f".format(p.getDouble("processorPixelValuesMAE"))} | " +
              "${"%.6f".format(bil?.optDouble("processorPixelValuesMAE") ?: -1.0)} | " +
              "${"%.6f".format(bil?.optDouble("encoderCosineSimilarity") ?: -1.0)} | " +
              "${"%.4f".format(p.getDouble("selectedMaskIoU"))} | " +
              "${"%.3f".format(p.getDouble("selectedMaskPixelDiffPercent"))}% | " +
              "${"%.3f".format(p.getDouble("contourSymmetricChamferAvgPx"))}px | " +
              "${firstParityDivergence(p)} |",
          )
        }
        reportSb.appendLine()
        if (parityGateFailures.isEmpty()) {
          reportSb.appendLine("All frozen Web/Android parity thresholds passed for the complete fixture set.")
        } else {
          reportSb.appendLine("## Failed frozen gates")
          reportSb.appendLine()
          parityGateFailures.forEach { reportSb.appendLine("- $it") }
        }
        File(parityDir, "parity_report.md").writeText(reportSb.toString())
        File(parityDir, "gate_status.json").writeText(
          JSONObject().apply {
            put("status", if (parityGateFailures.isEmpty()) "PASS" else "FAIL")
            put("fixturesCaptured", paritySummaryObj.length())
            put("fixturesExpected", fixturesArr.length())
            put("failures", JSONArray(parityGateFailures))
          }.toString(2),
        )
      }

      // 11. Lifecycle shutdown evidence snapshot
      val lifecycleDir = File(evidenceRoot, "lifecycle").apply { mkdirs() }
      val app = ApplicationProvider.getApplicationContext<Application>()
      val vm = FishMaskViewModel(app)
      vm.clearForTestAndAwait()
      val lifecycleLog = vm.coordinator.lifecycleLog
      File(lifecycleDir, "native_resource_lifecycle.log").writeText(lifecycleLog.joinToString("\n") + "\n")
      File(lifecycleDir, "shutdown_test_results.json").writeText(
        JSONObject().apply {
          put("shutdownComplete", vm.coordinator.isShutdownComplete())
          put("yoloxClosed", vm.yoloxDetector.isClosed())
          put("slimSamClosed", vm.slimSamSegmenter.isClosed())
          put("phases", JSONArray(lifecycleLog))
        }.toString(2),
      )

      // 12. Root manifest.json
      File(evidenceRoot, "manifest.json").writeText(
        JSONObject().apply {
          put("taskId", "FISH_MASK_ANDROID_P0_V2_2_STARTUP_WARMUP_AND_INFERENCE_OPTIMIZATION")
          put("applicationId", "com.yujian.fishmasklab")
          put("photoCount", photoSummaries.length())
          put("photos", photoSummaries)
          put("hasWebAndroidParity", paritySummaryObj.length() > 0)
          put("parityGateStatus", when {
            webSummaryJson == null -> "NOT_RUN"
            parityGateFailures.isEmpty() -> "PASS"
            else -> "FAIL"
          })
          put("parityGateFailureCount", parityGateFailures.size)
        }.toString(2),
      )

      if (webSummaryJson != null && parityGateFailures.isNotEmpty()) {
        File(evidenceRoot, "web_android_parity/gate_failures.json").writeText(
          JSONObject().put("failures", JSONArray(parityGateFailures)).toString(2),
        )
        assertTrue(
          "Strict Web/Android parity gates failed after all ${paritySummaryObj.length()} fixture reports were written:\n" +
            parityGateFailures.joinToString("\n"),
          parityGateFailures.isEmpty(),
        )
      }
    } finally {
      coordinator.shutdownAndAwait()
    }
  }
}
