package com.yujian.fishmasklab.runtime

import kotlin.math.abs
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class SlimSamImageProcessorTest {

  @Test
  fun `preprocessor_config json parses official slimsam parameters`() {
    val json = """
      {
        "do_convert_rgb": true,
        "do_normalize": true,
        "do_pad": true,
        "do_rescale": true,
        "do_resize": true,
        "image_mean": [0.485, 0.456, 0.406],
        "image_processor_type": "SamImageProcessor",
        "image_std": [0.229, 0.224, 0.225],
        "pad_size": {"height": 1024, "width": 1024},
        "resample": 2,
        "rescale_factor": 0.00392156862745098,
        "size": {"longest_edge": 1024}
      }
    """.trimIndent()
    val cfg = SamPreprocessorConfig.fromJson(json)
    assertEquals(1024, cfg.longestEdge)
    assertEquals(1024, cfg.padWidth)
    assertEquals(1024, cfg.padHeight)
    assertEquals(1f / 255f, cfg.rescaleFactor, 1e-6f)
    assertEquals(2, cfg.resample)
    assertEquals(0.485f, cfg.imageMean[0], 1e-5f)
    assertEquals(0.229f, cfg.imageStd[0], 1e-5f)
    cfg.validateSlimSam77Contract()
    assertTrue("Non-bilinear settings must be rejected", runCatching { cfg.copy(resample = 0).validateSlimSam77Contract() }.isFailure)
  }

  @Test
  fun `preprocessArgb resizes longest edge to 1024 preserves RGB order normalizes and zero-pads bottom and right`() {
    val origW = 1440
    val origH = 1920
    // Pure red pixel across entire image: ARGB = 0xFFFF0000 -> R=255, G=0, B=0
    val pixels = IntArray(origW * origH) { 0xFFFF0000.toInt() }
    val processor = SlimSamImageProcessor()

    val out = processor.preprocessArgb(pixels, origW, origH)

    assertEquals(1440, out.originalWidth)
    assertEquals(1920, out.originalHeight)
    assertEquals(768, out.resizedWidth)
    assertEquals(1024, out.resizedHeight)
    assertEquals(256, out.padRight)
    assertEquals(0, out.padBottom)
    assertEquals(3 * 1024 * 1024, out.pixelValues.size)

    val plane = 1024 * 1024
    val expectedR = (1.0f - 0.485f) / 0.229f
    val expectedG = (0.0f - 0.456f) / 0.224f
    val expectedB = (0.0f - 0.406f) / 0.225f

    // Inside resized region (x=100, y=500)
    val validIdx = 500 * 1024 + 100
    assertTrue(abs(out.pixelValues[validIdx] - expectedR) < 1e-3f)
    assertTrue(abs(out.pixelValues[plane + validIdx] - expectedG) < 1e-3f)
    assertTrue(abs(out.pixelValues[2 * plane + validIdx] - expectedB) < 1e-3f)

    // Inside right zero-padded region (x=900 >= 768) must be EXACT 0.0f after normalization
    val padIdx = 500 * 1024 + 900
    assertEquals(0.0f, out.pixelValues[padIdx], 0.0f)
    assertEquals(0.0f, out.pixelValues[plane + padIdx], 0.0f)
    assertEquals(0.0f, out.pixelValues[2 * plane + padIdx], 0.0f)
  }

  @Test
  fun `preprocessArgb landscape photo pads bottom with exact zero`() {
    val origW = 1920
    val origH = 1080
    val pixels = IntArray(origW * origH) { 0xFF00FF00.toInt() } // Pure green
    val out = SlimSamImageProcessor().preprocessArgb(pixels, origW, origH)

    assertEquals(1024, out.resizedWidth)
    assertEquals(576, out.resizedHeight)
    assertEquals(0, out.padRight)
    assertEquals(448, out.padBottom)

    val plane = 1024 * 1024
    // Bottom padding y=800 >= 576
    val bottomPadIdx = 800 * 1024 + 400
    assertEquals(0.0f, out.pixelValues[bottomPadIdx], 0.0f)
    assertEquals(0.0f, out.pixelValues[plane + bottomPadIdx], 0.0f)
    assertEquals(0.0f, out.pixelValues[2 * plane + bottomPadIdx], 0.0f)
  }

  @Test
  fun `preprocessArgb 1279x1706 matches SamImageProcessor rounding 767x1024`() {
    val origW = 1279
    val origH = 1706
    val pixels = IntArray(origW * origH) { 0xFF112233.toInt() }
    val out = SlimSamImageProcessor().preprocessArgb(pixels, origW, origH)

    assertEquals(767, out.resizedWidth)
    assertEquals(1024, out.resizedHeight)
    assertEquals(257, out.padRight)
    assertEquals(0, out.padBottom)

    // Column 767 must be exact zero-padding (0.0f) matching HuggingFace SamImageProcessor
    val plane = 1024 * 1024
    val col767Idx = 500 * 1024 + 767
    assertEquals(0.0f, out.pixelValues[col767Idx], 0.0f)
    assertEquals(0.0f, out.pixelValues[plane + col767Idx], 0.0f)
    assertEquals(0.0f, out.pixelValues[2 * plane + col767Idx], 0.0f)
  }
}
