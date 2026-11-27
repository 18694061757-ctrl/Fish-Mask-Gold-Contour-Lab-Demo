package com.yujian.fishmasklab.runtime

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.core.app.ApplicationProvider
import com.yujian.fishmasklab.domain.ContourGeometry
import com.yujian.fishmasklab.domain.ContourPoint
import com.yujian.fishmasklab.domain.ImageCoordinateMapper
import com.yujian.fishmasklab.domain.MaskQualityGate
import java.io.File
import java.io.InputStream
import javax.imageio.ImageIO
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class SlimSamOnnxRuntimeTest {

  private fun decodeRealJpegToBitmap(stream: InputStream): Bitmap {
    val buffered = ImageIO.read(stream) ?: error("Failed to decode JPEG with ImageIO")
    val w = buffered.width
    val h = buffered.height
    val pixels = IntArray(w * h)
    buffered.getRGB(0, 0, w, h, pixels, 0, w)
    return Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888)
  }

  @Test
  fun `slimsam q8 sessions load validate real tensor contracts and execute end-to-end on real fish photo and tap refocus`() =
    runBlocking {
      val context = ApplicationProvider.getApplicationContext<Context>()
      val yolox = YoloxFishDetector(context)
      val slimSam = SlimSamSegmenter(context)
      try {
        val audit = slimSam.ensureLoaded()
        assertTrue("SlimSAM assets must be SHA-verified: ${audit.missingAssets} ${audit.shaMismatchedAssets}", audit.isSlimSamReady)
        assertTrue("SlimSAM must reach READY state (error=${slimSam.lastError})", slimSam.isReady())
        assertEquals(SlimSamModelState.READY, slimSam.modelState)
        assertTrue(yolox.ensureLoaded())

        val testStream = javaClass.classLoader!!.getResourceAsStream("preset_fish.jpg")
          ?: error("Missing test resource preset_fish.jpg")
        val photoBitmap = testStream.use { decodeRealJpegToBitmap(it) }
        val mapper = ImageCoordinateMapper.create(photoBitmap.width, photoBitmap.height)

        // 1. Real YOLOX detection
        val (detection, detMs) = yolox.detect(photoBitmap, mapper)
        assertNotNull("YOLOX must detect fish on preset_fish.jpg", detection)
        assertTrue("YOLOX detection confidence=${detection!!.confidence}", detection.confidence >= 0.35f)
        assertTrue("YOLOX duration=${detMs}ms", detMs >= 0L)

        // 2. Real SlimSAM Q8 Encoder + Decoder + PostProcess + QualityGate
        val prompts = MaskQualityGate.generatePromptsFromDetection(
          box = detection.box,
          imageWidth = mapper.origWidth,
          imageHeight = mapper.origHeight,
        )
        val samRun1 = slimSam.segmentPhoto(
          photoBitmap = photoBitmap,
          photoKey = 101L,
          mapper = mapper,
          prompts = prompts,
          detection = detection,
        )
        assertFalse(samRun1.reusedCachedEmbedding)
        assertTrue("Encoder duration should be measured", samRun1.encoderDurationMs >= 0L)
        assertEquals(3, samRun1.maskResult.candidates.size)
        assertEquals(mapper.origWidth, samRun1.maskResult.width)
        assertEquals(mapper.origHeight, samRun1.maskResult.height)

        val chosen = samRun1.maskResult.candidates[samRun1.maskResult.selectedCandidateIndex]
        assertTrue("IoU should exceed 0.60, got ${chosen.iouScore}", chosen.iouScore >= 0.60f)
        assertTrue("Mask pixelCount should be substantial, got ${chosen.pixelCount}", chosen.pixelCount > 10_000)

        val contour = ContourGeometry.extractContourFromMask(
          mask = chosen.data,
          width = samRun1.maskResult.width,
          height = samRun1.maskResult.height,
        )
        assertNotNull("Contour extraction must succeed on real SlimSAM mask", contour)
        assertTrue("Raw contour points=${contour!!.rawPointCount}", contour.rawPointCount >= 100)
        assertTrue("Mask must pass strict quality gate", MaskQualityGate.evaluateMaskTrustworthiness(chosen, contour, detection))
        println(
          "[REAL_FISH_PASS] preset_fish.jpg detConf=${detection.confidence} detMs=$detMs " +
            "encMs=${samRun1.encoderDurationMs} decMs=${samRun1.decoderDurationMs} " +
            "candIdx=${chosen.index} iou=${chosen.iouScore} px=${chosen.pixelCount} rawPts=${contour.rawPointCount}",
        )

        // 3. Tap-to-Refocus on fish center must REUSE cached encoder embedding (encoderDurationMs == 0)
        val tapX = (detection.box.x1 + detection.box.x2) / 2
        val tapY = (detection.box.y1 + detection.box.y2) / 2
        val tapPrompts = MaskQualityGate.generatePromptsFromTap(tapX, tapY, mapper.origWidth, mapper.origHeight)
        val samRefocus = slimSam.segmentPhoto(
          photoBitmap = photoBitmap,
          photoKey = 101L,
          mapper = mapper,
          prompts = tapPrompts,
          detection = detection,
        )
        assertTrue("Tap refocus must reuse cached encoder embedding", samRefocus.reusedCachedEmbedding)
        assertEquals(0L, samRefocus.encoderDurationMs)
        assertTrue(samRefocus.maskResult.isMaskTrustworthy)
        println("[REAL_FISH_REFOCUS_PASS] reused=${samRefocus.reusedCachedEmbedding} decMs=${samRefocus.decoderDurationMs}")

        // 4. Negative non-fish blank/gradient image must be rejected by YOLOX
        val nonFish = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
        nonFish.eraseColor(Color.rgb(24, 48, 60))
        val nonFishMapper = ImageCoordinateMapper.create(640, 480)
        val (noDet, _) = yolox.detect(nonFish, nonFishMapper)
        assertNull("Uniform non-fish image must yield null detection", noDet)

        // 5. Additional local reference fish photos when present on developer/CI workspace
        val extraPhotos = listOf(
          "/tmp/ref_repo/public/preset_catfish.jpg",
          "/tmp/ref_repo/public/preset_snakehead.jpg",
          "/tmp/ref_repo/public/fish_crop.jpg",
        )
        for ((idx, path) in extraPhotos.withIndex()) {
          val f = File(path)
          if (!f.exists()) continue
          val bmp = f.inputStream().use { decodeRealJpegToBitmap(it) }
          val m = ImageCoordinateMapper.create(bmp.width, bmp.height)
          val (det, dMs) = yolox.detect(bmp, m)
          assertNotNull("YOLOX must detect fish in $path", det)
          val res = slimSam.segmentPhoto(
            photoBitmap = bmp,
            photoKey = 200L + idx,
            mapper = m,
            prompts = MaskQualityGate.generatePromptsFromDetection(det!!.box, m.origWidth, m.origHeight),
            detection = det,
          )
          val best = res.maskResult.candidates[res.maskResult.selectedCandidateIndex]
          val cnt = ContourGeometry.extractContourFromMask(best.data, res.maskResult.width, res.maskResult.height)
          assertNotNull("Contour must exist for $path", cnt)
          assertTrue(
            "Quality gate must pass for $path (iou=${best.iouScore}, px=${best.pixelCount})",
            MaskQualityGate.evaluateMaskTrustworthiness(best, cnt, det),
          )
          println(
            "[REAL_FISH_PASS] ${f.name} detConf=${det.confidence} detMs=$dMs " +
              "encMs=${res.encoderDurationMs} decMs=${res.decoderDurationMs} " +
              "candIdx=${best.index} iou=${best.iouScore} px=${best.pixelCount} rawPts=${cnt!!.rawPointCount}",
          )
        }
      } finally {
        yolox.close()
        slimSam.close()
      }
    }
}
