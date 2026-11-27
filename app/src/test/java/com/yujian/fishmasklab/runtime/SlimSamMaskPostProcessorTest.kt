package com.yujian.fishmasklab.runtime

import com.yujian.fishmasklab.domain.FishDetection
import com.yujian.fishmasklab.domain.PixelBox
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SlimSamMaskPostProcessorTest {

  @Test
  fun `postProcessCandidateLogits crops right and bottom padding restores original size and thresholds gt zero`() {
    val origW = 800
    val origH = 1600
    val resizedW = 512
    val resizedH = 1024

    // Create 256x256 logits: positive (+5.0) ONLY inside left unpadded fish core [x in 32..96, y in 64..192],
    // AND put strong positive (+10.0) in right padded area [x >= 160] which MUST be cropped away!
    val logits = FloatArray(256 * 256) { -5.0f }
    for (y in 0 until 256) {
      for (x in 0 until 256) {
        if (x in 32..96 && y in 64..192) {
          logits[y * 256 + x] = 5.0f
        } else if (x >= 160) {
          logits[y * 256 + x] = 10.0f // lies in padded region [512..1024] -> must be cropped out
        }
      }
    }

    val mask = SlimSamMaskPostProcessor.postProcessCandidateLogits(
      logits256 = logits,
      logitOffset = 0,
      resizedWidth = resizedW,
      resizedHeight = resizedH,
      originalWidth = origW,
      originalHeight = origH,
    )

    assertEquals(origW * origH, mask.size)

    var fgCount = 0
    var maxFgX = 0
    for (y in 0 until origH) {
      for (x in 0 until origW) {
        if ((mask[y * origW + x].toInt() and 0xFF) > 127) {
          fgCount++
          if (x > maxFgX) maxFgX = x
        }
      }
    }
    assertTrue("Foreground fish mask should exist", fgCount > 10_000)
    assertTrue("Right padding logits must be cropped out (maxFgX=$maxFgX)", maxFgX < 680)
  }

  @Test
  fun `buildMaskResult clips to 45 percent expanded box removes noise via LCC and ranks all 3 candidates`() {
    val origW = 400
    val origH = 600
    val resizedW = 683
    val resizedH = 1024
    val detection = FishDetection(
      box = PixelBox(100, 150, 300, 450), // 200x300
      confidence = 0.85f,
      areaRatio = 0.25f,
    )

    val predMasks = FloatArray(3 * 256 * 256) { -4.0f }
    // Candidate 0: tiny fragment (high IoU but too small)
    for (y in 100..115) for (x in 70..85) predMasks[y * 256 + x] = 4.0f
    // Candidate 1: good whole-fish torso inside box + distant noise blob outside 45% expanded box
    val c1Off = 256 * 256
    for (y in 70..185) for (x in 55..115) predMasks[c1Off + y * 256 + x] = 4.0f
    for (y in 5..15) for (x in 5..15) predMasks[c1Off + y * 256 + x] = 6.0f // outside expanded box
    // Candidate 2: empty
    val ious = floatArrayOf(0.92f, 0.86f, 0.40f)

    val result = SlimSamMaskPostProcessor.buildMaskResult(
      decoderOutput = SamDecoderOutput(ious, predMasks, 40L),
      resizedWidth = resizedW,
      resizedHeight = resizedH,
      originalWidth = origW,
      originalHeight = origH,
      detection = detection,
      totalSamInferenceMs = 120L,
    )

    assertEquals(3, result.candidates.size)
    // Candidate 1 must win over Candidate 0 tiny fragment
    assertEquals(1, result.selectedCandidateIndex)
    assertTrue(result.isMaskTrustworthy)
    // Top-left noise blob (around pixel 10,10) must be clipped out
    assertEquals(0, result.candidates[1].data[10 * origW + 10].toInt() and 0xFF)
  }
}
