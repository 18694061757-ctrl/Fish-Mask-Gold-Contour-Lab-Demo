package com.yujian.fishmasklab.domain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MaskQualityGateTest {

  private fun dummyContour(rawCount: Int = 120) = ContourData(
    rawPoints = List(rawCount) { ContourPoint(it.toFloat(), it.toFloat()) },
    simplifiedPoints = List(20) { ContourPoint(it.toFloat(), it.toFloat()) },
    startPoint = ContourPoint(0f, 0f),
    bezierSegments = emptyList(),
    totalLength = 500f,
    rawPointCount = rawCount,
    simplifiedPointCount = 20,
    extractionTimeMs = 4L,
  )

  @Test
  fun `selectBestCandidateIndex prefers trustworthy whole-fish candidate over higher-iou fragment`() {
    val detection = FishDetection(
      box = PixelBox(100, 100, 300, 500), // 200x400 = 80,000 area
      confidence = 0.88f,
      areaRatio = 0.25f,
    )

    val fragment = CandidateMask(
      index = 0,
      iouScore = 0.94f,
      data = ByteArray(0),
      pixelCount = 6_000, // covRatio = 0.075 < 0.18 -> NOT trustworthy
      areaRatio = 0.01f,
      bbox = PixelBox(150, 200, 250, 280),
      isTrustworthy = false,
    )
    val fullFish = CandidateMask(
      index = 1,
      iouScore = 0.84f,
      data = ByteArray(0),
      pixelCount = 34_000, // covRatio = 0.425 in 0.18..0.85
      areaRatio = 0.06f,
      bbox = PixelBox(115, 130, 285, 480), // spanH = 350/400 = 0.875 >= 0.48
      isTrustworthy = true,
    )
    val backgroundLeak = CandidateMask(
      index = 2,
      iouScore = 0.55f,
      data = ByteArray(0),
      pixelCount = 78_000, // covRatio = 0.975 > 0.85 -> NOT trustworthy
      areaRatio = 0.12f,
      bbox = PixelBox(80, 80, 320, 520),
      isTrustworthy = false,
    )

    val chosen = MaskQualityGate.selectBestCandidateIndex(
      candidates = listOf(fragment, fullFish, backgroundLeak),
      detection = detection,
      imageWidth = 800,
      imageHeight = 800,
    )
    assertEquals(1, chosen)
    assertTrue(MaskQualityGate.evaluateMaskTrustworthiness(fullFish, dummyContour(), detection))
    assertFalse(MaskQualityGate.evaluateMaskTrustworthiness(fragment, dummyContour(), detection))
    assertFalse(MaskQualityGate.evaluateMaskTrustworthiness(backgroundLeak, dummyContour(), detection))
  }

  @Test
  fun `quality gate rejects low iou short span and too few contour points`() {
    val detection = FishDetection(
      box = PixelBox(100, 100, 500, 300), // 400x200 = 80,000 horizontal
      confidence = 0.8f,
      areaRatio = 0.2f,
    )
    val validGeom = CandidateMask(
      index = 0,
      iouScore = 0.72f,
      data = ByteArray(0),
      pixelCount = 30_000,
      areaRatio = 0.05f,
      bbox = PixelBox(120, 120, 440, 280), // width span 320/400 = 0.80
      isTrustworthy = true,
    )

    assertTrue(MaskQualityGate.evaluateMaskTrustworthiness(validGeom, dummyContour(50), detection))
    assertFalse(MaskQualityGate.evaluateMaskTrustworthiness(validGeom.copy(iouScore = 0.52f), dummyContour(50), detection))
    assertFalse(MaskQualityGate.evaluateMaskTrustworthiness(validGeom.copy(bbox = PixelBox(120, 120, 240, 280)), dummyContour(50), detection))
    assertFalse(MaskQualityGate.evaluateMaskTrustworthiness(validGeom, dummyContour(12), detection))
  }
}
