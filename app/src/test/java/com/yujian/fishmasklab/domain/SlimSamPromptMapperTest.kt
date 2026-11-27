package com.yujian.fishmasklab.domain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SlimSamPromptMapperTest {

  @Test
  fun `originalPixelToSamPrompt maps original coordinates into unpadded resized SAM space`() {
    val mapper = ImageCoordinateMapper.create(origWidth = 1440, origHeight = 1920)
    assertEquals(768, mapper.samResizedWidth)
    assertEquals(1024, mapper.samResizedHeight)

    val (sxMid, syMid) = mapper.originalPixelToSamPrompt(720f, 960f)
    assertEquals(384f, sxMid, 0.01f)
    assertEquals(512f, syMid, 0.01f)

    val (sxMax, syMax) = mapper.originalPixelToSamPrompt(1440f, 1920f)
    assertEquals(768f, sxMax, 0.01f)
    assertEquals(1024f, syMax, 0.01f)
  }

  @Test
  fun `vertical detection prompt generates 7 pos + 1 neg and maps into SAM tensors`() {
    val mapper = ImageCoordinateMapper.create(origWidth = 1152, origHeight = 1536)
    val vertBox = PixelBox(232, 419, 490, 1262)
    val vertPrompts = MaskQualityGate.generatePromptsFromDetection(vertBox, 1152, 1536)

    assertEquals(8, vertPrompts.size)
    assertEquals(7, vertPrompts.count { it.label == 1 })
    assertEquals(1, vertPrompts.count { it.label == 0 })

    val mapped = mapper.mapPromptsToSamSpace(vertPrompts)
    assertEquals(8, mapped.pointCount)
    assertEquals(16, mapped.samPointCoords.size)
    assertEquals(8, mapped.samPointLabels.size)
    assertEquals(0L, mapped.samPointLabels.last())

    for (i in 0 until mapped.pointCount) {
      val sx = mapped.samPointCoords[i * 2]
      val sy = mapped.samPointCoords[i * 2 + 1]
      assertTrue("sx=$sx must be in 0..768", sx in 0f..768f)
      assertTrue("sy=$sy must be in 0..1024", sy in 0f..1024f)
    }

    val horizBox = PixelBox(100, 300, 900, 600) // 800 x 300 -> horizontal
    val horizPrompts = MaskQualityGate.generatePromptsFromDetection(horizBox, 1152, 1536)
    assertEquals(7, horizPrompts.size)
    assertTrue(horizPrompts.all { it.label == 1 })
  }

  @Test
  fun `tap refocus prompt builds 5-point cross and validates fish focus boundary`() {
    val detection = FishDetection(
      box = PixelBox(300, 200, 700, 900),
      confidence = 0.85f,
      areaRatio = 0.28f,
    )
    val prompts = MaskQualityGate.generatePromptsFromTap(500, 550, 1000, 1200)
    assertEquals(5, prompts.size)
    assertTrue(prompts.all { it.label == 1 })
    // 0.08 * 1000 = 80px horizontal offset
    assertEquals(420, prompts[1].x)
    assertEquals(580, prompts[2].x)

    assertTrue(MaskQualityGate.isTapWithinFishFocusRegion(ContourPoint(500f, 550f), detection, 1000, 1200))
    assertFalse(MaskQualityGate.isTapWithinFishFocusRegion(ContourPoint(20f, 20f), detection, 1000, 1200))
  }
}
