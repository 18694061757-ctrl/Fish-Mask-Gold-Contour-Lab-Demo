package com.yujian.fishmasklab.domain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ContourClosureTest {

  @Test
  fun `ellipse and 3x3 square produce closed contour returning to start directed edge`() {
    val sqW = 5
    val sqH = 5
    val sqMask = ByteArray(sqW * sqH)
    for (y in 1..3) {
      for (x in 1..3) {
        sqMask[y * sqW + x] = 0xFF.toByte()
      }
    }
    val sqTrace = ContourGeometry.extractBoundaryTrace(sqMask, sqW, sqH, 1, 1)
    assertTrue(sqTrace.traceClosed)
    assertEquals(BoundaryTerminationReason.RETURNED_TO_START, sqTrace.terminationReason)
    assertEquals(8, sqTrace.points.size)

    val w = 240
    val h = 160
    val ellipse = ByteArray(w * h)
    for (y in 0 until h) {
      for (x in 0 until w) {
        val dx = (x - 120) / 65.0
        val dy = (y - 80) / 34.0
        if (dx * dx + dy * dy <= 1.0) {
          ellipse[y * w + x] = 0xFF.toByte()
        }
      }
    }
    val lcc = ContourGeometry.getLargestConnectedComponent(ellipse, w, h)
    val trace = ContourGeometry.extractBoundaryTrace(lcc.cleanMask, w, h, lcc.startX, lcc.startY)
    assertTrue(trace.traceClosed)
    assertEquals(BoundaryTerminationReason.RETURNED_TO_START, trace.terminationReason)
    assertTrue(trace.points.size > 200)

    val contour = ContourGeometry.extractContourFromMask(ellipse, w, h)
    assertNotNull(contour)
    assertTrue(contour!!.traceClosed)
    assertEquals(BoundaryTerminationReason.RETURNED_TO_START, contour.terminationReason)
  }

  @Test
  fun `figure-8 pinch touching start pixel traces both lobes before closing`() {
    // Two 3x3 blocks touching diagonally at (3,3) where top-left start is (1,1) -> wait:
    // Start at top-left (3,1) of upper triangle/block + lower-left lobe sharing (3,3):
    // Upper block: x in 3..5, y in 1..3
    // Lower block: x in 1..3, y in 3..5
    val w = 7
    val h = 7
    val mask = ByteArray(w * h)
    for (y in 1..3) for (x in 3..5) mask[y * w + x] = 0xFF.toByte()
    for (y in 3..5) for (x in 1..3) mask[y * w + x] = 0xFF.toByte()

    val trace = ContourGeometry.extractBoundaryTrace(mask, w, h, 3, 1)
    assertTrue(trace.traceClosed)
    assertEquals(BoundaryTerminationReason.RETURNED_TO_START, trace.terminationReason)
    // Must include pixels from both lobes: e.g. x=5 (upper-right) and x=1 (lower-left)
    assertTrue(trace.points.any { it.x == 5f })
    assertTrue(trace.points.any { it.x == 1f })
  }

  @Test
  fun `isolated pixel and 2-pixel segment fail closure cleanly`() {
    val w = 8
    val h = 8
    val single = ByteArray(w * h)
    single[3 * w + 3] = 0xFF.toByte()
    val singleTrace = ContourGeometry.extractBoundaryTrace(single, w, h, 3, 3)
    assertFalse(singleTrace.traceClosed)
    assertEquals(BoundaryTerminationReason.ISOLATED_PIXEL, singleTrace.terminationReason)
    assertNull(ContourGeometry.extractContourFromMask(single, w, h))

    val pair = ByteArray(w * h)
    pair[3 * w + 3] = 0xFF.toByte()
    pair[3 * w + 4] = 0xFF.toByte()
    val pairTrace = ContourGeometry.extractBoundaryTrace(pair, w, h, 3, 3)
    assertFalse(pairTrace.traceClosed)
    assertEquals(BoundaryTerminationReason.REPEATED_STATE, pairTrace.terminationReason)
  }

  @Test
  fun `step budget exhaustion reports MAX_STEPS and traceClosed false`() {
    val w = 40
    val h = 40
    val rect = ByteArray(w * h)
    for (y in 5..30) for (x in 5..30) rect[y * w + x] = 0xFF.toByte()

    val truncated = ContourGeometry.extractBoundaryTrace(rect, w, h, 5, 5, maxStepsOverride = 12)
    assertFalse(truncated.traceClosed)
    assertEquals(BoundaryTerminationReason.MAX_STEPS, truncated.terminationReason)
    assertEquals(12, truncated.steps)
  }

  @Test
  fun `quality gate rejects unclosed contour even when point count and geometry look valid`() {
    val detection = FishDetection(
      box = PixelBox(100, 100, 500, 300),
      confidence = 0.90f,
      areaRatio = 0.25f,
    )
    val candidate = CandidateMask(
      index = 0,
      iouScore = 0.88f,
      data = ByteArray(0),
      pixelCount = 32_000,
      areaRatio = 0.05f,
      bbox = PixelBox(110, 120, 480, 280),
      isTrustworthy = true,
    )
    val unclosedContour = ContourData(
      rawPoints = List(80) { ContourPoint(it.toFloat(), 150f) },
      simplifiedPoints = List(16) { ContourPoint(it * 5f, 150f) },
      startPoint = ContourPoint(0f, 150f),
      bezierSegments = emptyList(),
      totalLength = 400f,
      rawPointCount = 80,
      simplifiedPointCount = 16,
      extractionTimeMs = 2L,
      traceClosed = false,
      terminationReason = BoundaryTerminationReason.MAX_STEPS,
    )

    assertFalse(MaskQualityGate.evaluateMaskTrustworthiness(candidate, unclosedContour, detection))
    assertTrue(
      MaskQualityGate.evaluateMaskTrustworthiness(
        candidate,
        unclosedContour.copy(traceClosed = true, terminationReason = BoundaryTerminationReason.RETURNED_TO_START),
        detection,
      ),
    )
  }
}
