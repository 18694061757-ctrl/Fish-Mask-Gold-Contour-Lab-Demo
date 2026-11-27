package com.yujian.fishmasklab.runtime

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.core.app.ApplicationProvider
import com.yujian.fishmasklab.domain.ContourPoint
import com.yujian.fishmasklab.domain.ImageCoordinateMapper
import com.yujian.fishmasklab.domain.PipelineState
import com.yujian.fishmasklab.domain.RecognitionErrorReason
import java.io.InputStream
import javax.imageio.ImageIO
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class RecognitionLifecycleTest {

  private fun decodeRealJpegToBitmap(stream: InputStream): Bitmap {
    val buffered = ImageIO.read(stream) ?: error("Failed to decode JPEG")
    val pixels = IntArray(buffered.width * buffered.height)
    buffered.getRGB(0, 0, buffered.width, buffered.height, pixels, 0, buffered.width)
    return Bitmap.createBitmap(pixels, buffered.width, buffered.height, Bitmap.Config.ARGB_8888)
  }

  @Test
  fun `coordinator cancels stale task when switching photos and finishes negative image with NO_CLEAR_FISH`() =
    runBlocking {
      val context = ApplicationProvider.getApplicationContext<Context>()
      val yolox = YoloxFishDetector(context)
      val slimSam = SlimSamSegmenter(context)
      val coordinator = RecognitionPipelineCoordinator(yolox, slimSam)

      try {
        val fishBmp = javaClass.classLoader!!.getResourceAsStream("preset_fish.jpg")!!
          .use { decodeRealJpegToBitmap(it) }
        val fishMapper = ImageCoordinateMapper.create(fishBmp.width, fishBmp.height)

        val noFishBmp = Bitmap.createBitmap(480, 480, Bitmap.Config.ARGB_8888)
        noFishBmp.eraseColor(Color.rgb(20, 40, 50))
        val noFishMapper = ImageCoordinateMapper.create(480, 480)

        // Start task 1 on fish photo, then immediately replace with task 2 on no-fish photo
        val t1 = coordinator.startRecognition(
          photoGenerationId = 1L,
          workBitmap = fishBmp,
          samBitmap = fishBmp,
          mapper = fishMapper,
        )
        val t2 = coordinator.startRecognition(
          photoGenerationId = 2L,
          workBitmap = noFishBmp,
          samBitmap = noFishBmp,
          mapper = noFishMapper,
        )
        assertTrue(t2 > t1)

        val terminal = withTimeout(30_000L) {
          coordinator.events
            .filterIsInstance<PipelineProgressEvent.StateChanged>()
            .first { it.taskId == t2 && it.state == PipelineState.ERROR }
        }
        assertEquals(2L, terminal.photoGenerationId)
        assertEquals(RecognitionErrorReason.NO_CLEAR_FISH, terminal.errorReason)
        assertNull(terminal.contourData)
      } finally {
        coordinator.shutdown()
      }
    }

  @Test
  fun `coordinator executes full pipeline to PLAYING and tap refocus reuses encoder embedding`() =
    runBlocking {
      val context = ApplicationProvider.getApplicationContext<Context>()
      val yolox = YoloxFishDetector(context)
      val slimSam = SlimSamSegmenter(context)
      val coordinator = RecognitionPipelineCoordinator(yolox, slimSam)

      try {
        val fishBmp = javaClass.classLoader!!.getResourceAsStream("preset_fish.jpg")!!
          .use { decodeRealJpegToBitmap(it) }
        val mapper = ImageCoordinateMapper.create(fishBmp.width, fishBmp.height)

        val t1 = coordinator.startRecognition(
          photoGenerationId = 10L,
          workBitmap = fishBmp,
          samBitmap = fishBmp,
          mapper = mapper,
        )
        val playing1 = withTimeout(45_000L) {
          coordinator.events
            .filterIsInstance<PipelineProgressEvent.StateChanged>()
            .first { it.taskId == t1 && (it.state == PipelineState.PLAYING || it.state == PipelineState.ERROR) }
        }
        assertEquals("Expected PLAYING, got error=${playing1.diagnosticMessage}", PipelineState.PLAYING, playing1.state)
        assertNotNull(playing1.detection)
        assertNotNull(playing1.contourData)
        assertTrue(slimSam.hasCachedEmbeddingForPhoto(10L))

        val det = playing1.detection!!
        val tapCenter = ContourPoint(
          (det.box.x1 + det.box.x2) / 2f,
          (det.box.y1 + det.box.y2) / 2f,
        )
        val t2 = coordinator.startRecognition(
          photoGenerationId = 10L,
          workBitmap = fishBmp,
          samBitmap = fishBmp,
          mapper = mapper,
          tapPoint = tapCenter,
          existingDetection = det,
        )
        val playing2 = withTimeout(20_000L) {
          coordinator.events
            .filterIsInstance<PipelineProgressEvent.StateChanged>()
            .first { it.taskId == t2 && (it.state == PipelineState.PLAYING || it.state == PipelineState.ERROR) }
        }
        assertEquals(PipelineState.PLAYING, playing2.state)
        assertTrue("Tap refocus must reuse cached Encoder embedding", playing2.reusedEncoderEmbedding)
        assertNotNull(playing2.contourData)

        // Cancelling clears job cleanly without throwing
        val job = coordinator.cancelCurrentTask()
        job?.join()
        delay(20L)
      } finally {
        coordinator.shutdown()
      }
    }
}
