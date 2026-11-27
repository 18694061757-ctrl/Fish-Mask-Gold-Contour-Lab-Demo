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
@Config(sdk = [36])
class EvidenceAndParityExportTest {

  @Before
  fun setUpMainDispatcher() {
    Dispatchers.setMain(Dispatchers.Default)
  }

  @After
  fun tearDownMainDispatcher() {
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
      val stepDst = max(1, dst.size / 600)
      val out = ArrayList<Double>(src.size / stepSrc + 1)
      var i = 0
      while (i < src.size) {
        val p = src[i]
        var best = Double.MAX_VALUE
        var j = 0
        while (j < dst.size) {
          val q = dst[j]
          val d = hypot((p.x - q.x).toDouble(), (p.y - q.y).toDouble())
          if (d < best) best = d
          j += stepDst
        }
        out.add(best)
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

  @Test
  fun `generate P0-5 mask evidence and P0-4 web android parity report`() = runBlocking {
    val evidenceRoot = File(
      System.getProperty("fishmask.evidence.dir")
        ?: System.getenv("FISHMASK_EVIDENCE_DIR")
        ?: "build/evidence/p0_v2_1",
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

    try {
      for (i in 0 until fixturesArr.length()) {
        val spec = fixturesArr.getJSONObject(i)
        val photoId = spec.getString("id")
        val filename = spec.getString("filename")
        val rawBytes = javaClass.classLoader?.getResourceAsStream(filename)?.use { it.readBytes() }
          ?: error("Missing fixture $filename")

        val photoDir = File(photosDir, photoId).apply { mkdirs() }
        File(photoDir, "original.jpg").writeBytes(rawBytes)

        val tempFile = File.createTempFile("ev_${photoId}_", ".jpg")
        tempFile.writeBytes(rawBytes)
        val loaded = ExifPhotoLoader.loadFromFileAndDeleteTemp(tempFile)

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
              put("detectorDurationMs", initialEvent.telemetry?.detectorDurationMs ?: -1L)
            }.toString(2),
          )
          writeDetectionOverlayPng(loaded.displayBitmap, det, File(photoDir, "detection_overlay.png"))

          // 2. candidate_{0,1,2}.png, selected_mask.png, lcc_mask.png, selected_mask_overlay.png, contour_overlay.png
          for (c in maskRes.candidates) {
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
          val pv = cachedCtx.preprocessed.pixelValues
          var pvMin = Float.POSITIVE_INFINITY
          var pvMax = Float.NEGATIVE_INFINITY
          var pvSum = 0.0
          for (v in pv) {
            if (v < pvMin) pvMin = v
            if (v > pvMax) pvMax = v
            pvSum += v
          }
          File(photoDir, "processor_metrics.json").writeText(
            JSONObject().apply {
              put("originalWidth", cachedCtx.preprocessed.originalWidth)
              put("originalHeight", cachedCtx.preprocessed.originalHeight)
              put("resizedWidth", cachedCtx.preprocessed.resizedWidth)
              put("resizedHeight", cachedCtx.preprocessed.resizedHeight)
              put("padWidth", cachedCtx.preprocessed.padWidth)
              put("padHeight", cachedCtx.preprocessed.padHeight)
              put("padRight", cachedCtx.preprocessed.padRight)
              put("padBottom", cachedCtx.preprocessed.padBottom)
              put("preprocessDurationMs", cachedCtx.preprocessed.preprocessDurationMs)
              put("tensorShape", JSONArray(listOf(1, 3, 1024, 1024)))
              put("min", pvMin.toDouble())
              put("max", pvMax.toDouble())
              put("mean", pvSum / pv.size.toDouble())
            }.toString(2),
          )

          // 5. encoder_metrics.json
          val emb = cachedCtx.embedding.imageEmbeddings
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
              put(
                "prompts",
                JSONArray().apply {
                  for (p in prompts) {
                    put(JSONObject().put("id", p.id).put("x", p.x).put("y", p.y).put("label", p.label))
                  }
                },
              )
              put("samPointCoords", JSONArray(mappedPrompts.samPointCoords.map { it.toDouble() }))
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

            // Processor parity
            val webPv = readLittleEndianFloatArray(File(webPhotoOut, "pixel_values.f32"))
            var pvAbsSum = 0.0
            var pvMaxAbs = 0.0
            for (k in pv.indices) {
              val d = abs(pv[k].toDouble() - webPv[k].toDouble())
              pvAbsSum += d
              if (d > pvMaxAbs) pvMaxAbs = d
            }
            val pvMae = pvAbsSum / pv.size.toDouble()

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
            if (pvBilFile.exists() && embBilFile.exists()) {
              val webPvBil = readLittleEndianFloatArray(pvBilFile)
              val webEmbBil = readLittleEndianFloatArray(embBilFile)
              var bilPvSum = 0.0
              var bilPvMax = 0.0
              for (k in pv.indices) {
                val d = abs(pv[k].toDouble() - webPvBil[k].toDouble())
                bilPvSum += d
                if (d > bilPvMax) bilPvMax = d
              }
              var bDot = 0.0; var bNa = 0.0; var bNb = 0.0; var bDiff = 0.0; var bNorm = 0.0
              for (k in emb.indices) {
                val a = emb[k].toDouble(); val b = webEmbBil[k].toDouble()
                bDot += a * b; bNa += a * a; bNb += b * b
                bDiff += abs(a - b); bNorm += abs(b)
              }
              println(
                "[BILINEAR_PARITY] $photoId pvMAE=${"%.6f".format(bilPvSum / pv.size)} " +
                  "pvMax=${"%.5f".format(bilPvMax)} encCos=${"%.6f".format(bDot / (sqrt(bNa) * sqrt(bNb)))} " +
                  "encRelMAE=${"%.6f".format(bDiff / max(1e-9, bNorm))}",
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

            // Candidate IoU & Mask IoU comparison
            val webCands = webPhotoMeta.getJSONArray("candidates")
            val candParityArr = JSONArray()
            for (cIdx in 0 until 3) {
              val webCand = webCands.getJSONObject(cIdx)
              val webCandMask = File(webPhotoOut, "candidate_${cIdx}.bin").readBytes()
              val andCand = maskRes.candidates[cIdx]
              val (maskIou, diffRatio, _) = binaryMaskMetrics(andCand.data, webCandMask)
              candParityArr.put(
                JSONObject().apply {
                  put("index", cIdx)
                  put("androidIouScore", andCand.iouScore.toDouble())
                  put("webIouScore", webCand.getDouble("iouScore"))
                  put("iouScoreDiff", abs(andCand.iouScore - webCand.getDouble("iouScore")))
                  put("maskIoU", maskIou)
                  put("pixelDiffPercent", diffRatio * 100.0)
                  put("androidPixels", andCand.pixelCount)
                  put("webPixels", webCand.getInt("pixelCount"))
                },
              )
            }

            // Selected mask comparison against Web appSelectedIndex & samServiceBestIndex
            val webAppIdx = webPhotoMeta.getInt("appSelectedIndex")
            val webSamIdx = webPhotoMeta.getInt("samServiceBestIndex")
            val webAppMask = File(webPhotoOut, "candidate_${webAppIdx}.bin").readBytes()
            val (selIoU, selDiffRatio, _) = binaryMaskMetrics(chosen.data, webAppMask)

            // Contour Chamfer distance comparison
            val webContourObj = JSONObject(File(webPhotoOut, "contour_app.json").readText())
            val webRawPtsJson = webContourObj.getJSONArray("rawPoints")
            val webRawPts = ArrayList<ContourPoint>(webRawPtsJson.length())
            for (pIdx in 0 until webRawPtsJson.length()) {
              val pt = webRawPtsJson.getJSONObject(pIdx)
              webRawPts.add(ContourPoint(pt.getDouble("x").toFloat(), pt.getDouble("y").toFloat()))
            }
            val (chamferAvg, chamferP95) = symmetricChamferDistance(contour.rawPoints, webRawPts)

            val photoParity = JSONObject().apply {
              put("photoId", photoId)
              put("processorPixelValuesMAE", pvMae)
              put("processorPixelValuesMaxAbsDiff", pvMaxAbs)
              put("encoderCosineSimilarity", encCosine)
              put("encoderRelativeMAE", encRelMae)
              put("yoloxDetBoxIoU", detBoxIou)
              put("androidDetBox", JSONArray(listOf(det.box.x1, det.box.y1, det.box.x2, det.box.y2)))
              put("webDetBox", JSONArray(listOf(wx1, wy1, wx2, wy2)))
              put("androidSelectedCandidateIndex", maskRes.selectedCandidateIndex)
              put("webSamServiceBestIndex", webSamIdx)
              put("webAppSelectedIndex", webAppIdx)
              put("selectedMaskIoU", selIoU)
              put("selectedMaskPixelDiffPercent", selDiffRatio * 100.0)
              put("contourSymmetricChamferAvgPx", chamferAvg)
              put("contourSymmetricChamferP95Px", chamferP95)
              put("candidates", candParityArr)
            }
            File(parityDir, "${photoId}_parity.json").writeText(photoParity.toString(2))
            paritySummaryObj.put(photoId, photoParity)

            println(
              "[WEB_ANDROID_PARITY] $photoId " +
                "pvMAE=${"%.6f".format(pvMae)} pvMax=${"%.5f".format(pvMaxAbs)} " +
                "encCos=${"%.6f".format(encCosine)} encRelMAE=${"%.6f".format(encRelMae)} " +
                "detIoU=${"%.4f".format(detBoxIou)} " +
                "selIdx(and=${maskRes.selectedCandidateIndex},webSam=$webSamIdx,webApp=$webAppIdx) " +
                "selMaskIoU=${"%.4f".format(selIoU)} diff=${"%.3f".format(selDiffRatio * 100.0)}% " +
                "chamferAvg=${"%.3f".format(chamferAvg)}px chamferP95=${"%.3f".format(chamferP95)}px",
            )
          }
        } finally {
          collectJob.cancel()
          if (!loaded.displayBitmap.isRecycled) loaded.displayBitmap.recycle()
          if (!loaded.workBitmap.isRecycled) loaded.workBitmap.recycle()
        }
      }

      if (paritySummaryObj.length() > 0) {
        File(parityDir, "summary.json").writeText(paritySummaryObj.toString(2))
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
          put("taskId", "FISH_MASK_ANDROID_P0_V2_1_CI_NATIVE_PARITY_EVIDENCE")
          put("applicationId", "com.yujian.fishmasklab")
          put("photoCount", photoSummaries.length())
          put("photos", photoSummaries)
          put("hasWebAndroidParity", paritySummaryObj.length() > 0)
        }.toString(2),
      )
    } finally {
      coordinator.shutdownAndAwait()
    }
  }
}
