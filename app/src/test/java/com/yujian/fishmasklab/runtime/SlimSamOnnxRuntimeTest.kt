package com.yujian.fishmasklab.runtime

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.core.app.ApplicationProvider
import com.yujian.fishmasklab.app.ExifPhotoLoader
import com.yujian.fishmasklab.domain.BoundaryTerminationReason
import com.yujian.fishmasklab.domain.ContourPoint
import com.yujian.fishmasklab.domain.ImageCoordinateMapper
import com.yujian.fishmasklab.domain.MaskQualityGate
import com.yujian.fishmasklab.domain.PipelineState
import com.yujian.fishmasklab.domain.RecognitionErrorReason
import java.io.File
import java.security.MessageDigest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import kotlinx.coroutines.withTimeout
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class SlimSamOnnxRuntimeTest {

  data class FixtureSpec(
    val id: String,
    val filename: String,
    val sizeBytes: Long,
    val sha256: String,
    val width: Int,
    val height: Int,
  )

  @Before
  fun setUpMainDispatcher() {
    Dispatchers.setMain(Dispatchers.Default)
  }

  @After
  fun tearDownMainDispatcher() {
    Dispatchers.resetMain()
  }

  private fun loadFixtureSpecs(): List<FixtureSpec> {
    val manifestStream = javaClass.classLoader?.getResourceAsStream("fish-fixtures-manifest.json")
      ?: error("Missing required test resource: fish-fixtures-manifest.json")
    val manifestText = manifestStream.bufferedReader().use { it.readText() }
    val root = JSONObject(manifestText)
    val fixturesArr = root.getJSONArray("fixtures")
    val list = ArrayList<FixtureSpec>(fixturesArr.length())
    for (i in 0 until fixturesArr.length()) {
      val item = fixturesArr.getJSONObject(i)
      list.add(
        FixtureSpec(
          id = item.getString("id"),
          filename = item.getString("filename"),
          sizeBytes = item.getLong("size_bytes"),
          sha256 = item.getString("sha256"),
          width = item.getInt("width"),
          height = item.getInt("height"),
        ),
      )
    }
    return list
  }

  private fun sha256Hex(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

  @Test
  fun `all 4 bundled fish fixtures run reproducibly through production ExifPhotoLoader Coordinator SlimSam Q8 and tap refocus`() =
    runBlocking {
      val specs = loadFixtureSpecs()
      assertEquals("Expected exactly 4 bundled P0 test fixtures", 4, specs.size)
      assertEquals(
        listOf("preset_fish", "preset_catfish", "preset_snakehead", "fish_crop"),
        specs.map { it.id },
      )

      val context = ApplicationProvider.getApplicationContext<Context>()
      val yolox = YoloxFishDetector(context)
      val slimSam = SlimSamSegmenter(context)
      val coordinator = RecognitionPipelineCoordinator(yolox, slimSam)

      try {
        val audit = slimSam.ensureLoaded()
        assertTrue(
          "SlimSAM assets must be SHA-verified: ${audit.missingAssets} ${audit.shaMismatchedAssets}",
          audit.isSlimSamReady,
        )
        assertTrue("SlimSAM must reach READY state (error=${slimSam.lastError})", slimSam.isReady())
        assertEquals(SlimSamModelState.READY, slimSam.modelState)
        assertTrue("YOLOX must load", yolox.ensureLoaded())

        for ((idx, spec) in specs.withIndex()) {
          val rawBytes = javaClass.classLoader?.getResourceAsStream(spec.filename)?.use { it.readBytes() }
            ?: error("Missing required fixture JPEG resource: ${spec.filename}")
          assertEquals("Fixture ${spec.filename} size mismatch", spec.sizeBytes, rawBytes.size.toLong())
          assertEquals("Fixture ${spec.filename} SHA-256 mismatch", spec.sha256, sha256Hex(rawBytes))

          val tempFile = File.createTempFile("p0_fixture_${spec.id}_", ".jpg")
          tempFile.writeBytes(rawBytes)
          val loaded = ExifPhotoLoader.loadFromFileAndDeleteTemp(tempFile)
          assertFalse("Production ExifPhotoLoader must delete temp capture file", tempFile.exists())
          assertEquals("${spec.id} width mismatch", spec.width, loaded.displayBitmap.width)
          assertEquals("${spec.id} height mismatch", spec.height, loaded.displayBitmap.height)

          try {
            val photoGenId = 100L + idx
            val task1 = coordinator.startRecognition(
              photoGenerationId = photoGenId,
              workBitmap = loaded.workBitmap,
              samBitmap = loaded.displayBitmap,
              mapper = loaded.mapper,
            )
            assertTrue("startRecognition must return valid taskId", task1 > 0L)

            val initialEvent = withTimeout(45_000L) {
              coordinator.events
                .filterIsInstance<PipelineProgressEvent.StateChanged>()
                .first { it.taskId == task1 && (it.state == PipelineState.PLAYING || it.state == PipelineState.ERROR) }
            }
            assertEquals(
              "Photo ${spec.id} must reach PLAYING (error=${initialEvent.errorReason}: ${initialEvent.diagnosticMessage})",
              PipelineState.PLAYING,
              initialEvent.state,
            )
            assertFalse("Initial recognition must run encoder fresh", initialEvent.reusedEncoderEmbedding)

            val det = initialEvent.detection
            val maskResult = initialEvent.maskResult
            val contour = initialEvent.contourData
            val samRes = slimSam.lastInferenceResultForTest

            assertNotNull("${spec.id} must have YOLOX detection", det)
            assertNotNull("${spec.id} must have MaskResult", maskResult)
            assertNotNull("${spec.id} must have ContourData", contour)
            assertNotNull("${spec.id} must have SlimSamInferenceResult", samRes)

            assertTrue("${spec.id} YOLOX bbox trustworthy", MaskQualityGate.isBboxTrustworthy(det))
            assertEquals("${spec.id} must produce 3 decoder candidates", 3, maskResult!!.candidates.size)
            assertTrue("${spec.id} mask must pass trustworthiness", maskResult.isMaskTrustworthy)
            assertTrue("${spec.id} contour trace must be closed", contour!!.traceClosed)
            assertEquals(BoundaryTerminationReason.RETURNED_TO_START, contour.terminationReason)
            assertTrue("${spec.id} rawPoints=${contour.rawPointCount} must be >= 100", contour.rawPointCount >= 100)

            val chosen = maskResult.candidates[maskResult.selectedCandidateIndex]
            assertTrue(
              "${spec.id} chosen candidate must pass strict MaskQualityGate",
              MaskQualityGate.evaluateMaskTrustworthiness(chosen, contour, det),
            )
            assertTrue(
              "${spec.id} encoder embedding must be cached for photoGenId=$photoGenId",
              slimSam.hasCachedEmbeddingForPhoto(photoGenId),
            )

            println(
              "[REAL_FISH_PASS] ${spec.id} (${spec.width}x${spec.height}) " +
                "detBox=[${det!!.box.x1},${det.box.y1},${det.box.x2},${det.box.y2}] detConf=${det.confidence} " +
                "encMs=${samRes!!.encoderDurationMs} decMs=${samRes.decoderDurationMs} " +
                "candIdx=${chosen.index} iou=${chosen.iouScore} px=${chosen.pixelCount} " +
                "rawPts=${contour.rawPointCount} simpPts=${contour.simplifiedPointCount} len=${contour.totalLength}",
            )

            // Tap-to-Refocus on fish box center must REUSE cached encoder embedding (encoderDurationMs == 0)
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
            assertTrue("Tap refocus startRecognition must return valid taskId", taskRefocus > 0L)

            val refocusEvent = withTimeout(30_000L) {
              coordinator.events
                .filterIsInstance<PipelineProgressEvent.StateChanged>()
                .first { it.taskId == taskRefocus && (it.state == PipelineState.PLAYING || it.state == PipelineState.ERROR) }
            }
            assertEquals(
              "Photo ${spec.id} tap refocus must reach PLAYING (error=${refocusEvent.errorReason}: ${refocusEvent.diagnosticMessage})",
              PipelineState.PLAYING,
              refocusEvent.state,
            )
            assertTrue("${spec.id} tap refocus must reuse cached encoder embedding", refocusEvent.reusedEncoderEmbedding)
            assertEquals(
              "${spec.id} tap refocus encoderDurationMs must be 0",
              0L,
              slimSam.lastInferenceResultForTest?.encoderDurationMs,
            )
            assertTrue("${spec.id} refocus contour must stay closed", refocusEvent.contourData?.traceClosed == true)
            assertTrue("${spec.id} refocus mask must be trustworthy", refocusEvent.maskResult?.isMaskTrustworthy == true)

            println(
              "[REAL_FISH_REFOCUS_PASS] ${spec.id} reused=${refocusEvent.reusedEncoderEmbedding} " +
                "decMs=${slimSam.lastInferenceResultForTest?.decoderDurationMs} " +
                "candIdx=${refocusEvent.maskResult?.selectedCandidateIndex}",
            )
          } finally {
            if (!loaded.displayBitmap.isRecycled) loaded.displayBitmap.recycle()
            if (!loaded.workBitmap.isRecycled) loaded.workBitmap.recycle()
          }
        }

        // Negative non-fish blank/gradient image must be rejected with NO_CLEAR_FISH
        val nonFish = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
        try {
          nonFish.eraseColor(Color.rgb(24, 48, 60))
          val nonFishMapper = ImageCoordinateMapper.create(640, 480)
          val (noDet, _) = yolox.detect(nonFish, nonFishMapper)
          assertNull("Uniform non-fish image must yield null detection", noDet)

          val negTask = coordinator.startRecognition(
            photoGenerationId = 999L,
            workBitmap = nonFish,
            samBitmap = nonFish,
            mapper = nonFishMapper,
          )
          val negEvent = withTimeout(25_000L) {
            coordinator.events
              .filterIsInstance<PipelineProgressEvent.StateChanged>()
              .first { it.taskId == negTask && (it.state == PipelineState.PLAYING || it.state == PipelineState.ERROR) }
          }
          assertEquals(PipelineState.ERROR, negEvent.state)
          assertEquals(RecognitionErrorReason.NO_CLEAR_FISH, negEvent.errorReason)
        } finally {
          if (!nonFish.isRecycled) nonFish.recycle()
        }
      } finally {
        coordinator.shutdownAndAwait()
        assertTrue(yolox.isClosed())
        assertTrue(slimSam.isClosed())
      }
    }
}
