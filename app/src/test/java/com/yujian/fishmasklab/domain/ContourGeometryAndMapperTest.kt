package com.yujian.fishmasklab.domain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ContourGeometryAndMapperTest {

  @Test
  fun `viewportTapToOriginalPixel rejects letterbox blank margins and maps photo area accurately`() {
    val mapper = ImageCoordinateMapper.create(origWidth = 1000, origHeight = 500, maxWorkDimension = 1024)
    val rect = mapper.computeDisplayRect(viewportWidth = 1000f, viewportHeight = 1000f)

    // Fitted 2:1 image inside 1000x1000 viewport -> left=0, top=250, width=1000, height=500
    assertEquals(0f, rect.left, 0.01f)
    assertEquals(250f, rect.top, 0.01f)
    assertEquals(1000f, rect.width, 0.01f)
    assertEquals(500f, rect.height, 0.01f)

    // Tap in top letterbox blank region must return null
    assertNull(mapper.viewportTapToOriginalPixel(500f, 100f, 1000f, 1000f))
    // Tap in bottom letterbox blank region must return null
    assertNull(mapper.viewportTapToOriginalPixel(500f, 900f, 1000f, 1000f))

    // Tap at center of viewport (500, 500) maps to (500, 250) in original photo
    val mapped = mapper.viewportTapToOriginalPixel(500f, 500f, 1000f, 1000f)
    assertNotNull(mapped)
    assertEquals(500f, mapped!!.x, 0.5f)
    assertEquals(250f, mapped.y, 0.5f)
  }

  @Test
  fun `extractContourFromMask isolates largest connected component and produces closed bezier curve`() {
    val w = 240
    val h = 160
    val mask = ByteArray(w * h)

    // Main fish body ellipse centered at (120, 80) with rx=70, ry=36
    var ellipsePixels = 0
    for (y in 0 until h) {
      for (x in 0 until w) {
        val dx = (x - 120) / 70.0
        val dy = (y - 80) / 36.0
        if (dx * dx + dy * dy <= 1.0) {
          mask[y * w + x] = 0xFF.toByte()
          ellipsePixels++
        }
      }
    }
    // Isolated noise / gripper fragment (4..10 x 4..10)
    for (y in 4..10) {
      for (x in 4..10) {
        mask[y * w + x] = 0xFF.toByte()
      }
    }

    val lcc = ContourGeometry.getLargestConnectedComponent(mask, w, h)
    assertEquals(ellipsePixels, lcc.pixelCount)
    assertEquals(50, lcc.bbox.x1)
    assertEquals(44, lcc.bbox.y1)
    assertEquals(190, lcc.bbox.x2)
    assertEquals(116, lcc.bbox.y2)

    val contour = ContourGeometry.extractContourFromMask(
      mask = mask,
      width = w,
      height = h,
      simplificationTolerance = 2.2f,
      smoothingPasses = 3,
    )
    assertNotNull(contour)
    assertTrue(contour!!.rawPointCount >= 200)
    assertTrue(contour.simplifiedPointCount >= 8)
    assertEquals(contour.simplifiedPointCount, contour.bezierSegments.size)
    assertTrue(contour.totalLength > 200f)
  }

  @Test
  fun `mask quality gate rejects tiny or partial fragments and accepts valid torso coverage`() {
    val detection = FishDetection(
      box = PixelBox(100, 100, 300, 500),
      confidence = 0.85f,
      areaRatio = 0.25f,
    )
    val prompts = MaskQualityGate.generatePromptsFromDetection(detection.box, 800, 800)
    // Vertical fish box generates 7 positive torso points + 1 negative top gripper point
    assertEquals(8, prompts.size)
    assertEquals(0, prompts.last().label)

    val goodCandidate = CandidateMask(
      index = 0,
      iouScore = 0.82f,
      data = ByteArray(0),
      pixelCount = 35_000, // 35000 / 80000 = 0.4375 (within 0.18..0.85)
      areaRatio = 0.05f,
      bbox = PixelBox(110, 140, 290, 480), // height span 340 / 400 = 0.85 (>= 0.48)
    )
    val dummyContour = ContourData(
      rawPoints = List(30) { ContourPoint(0f, 0f) },
      simplifiedPoints = List(10) { ContourPoint(0f, 0f) },
      startPoint = ContourPoint(0f, 0f),
      bezierSegments = emptyList(),
      totalLength = 500f,
      rawPointCount = 30,
      simplifiedPointCount = 10,
      extractionTimeMs = 5L,
    )
    assertTrue(MaskQualityGate.evaluateMaskTrustworthiness(goodCandidate, dummyContour, detection))
  }
}
