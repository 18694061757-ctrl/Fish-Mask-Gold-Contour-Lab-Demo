package com.yujian.fishmasklab.domain

import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Port of prompt generation, candidate scoring, and strict mask trustworthiness gating
 * from `src/services/samService.ts` and `src/App.tsx`.
 */
object MaskQualityGate {

  fun isBboxTrustworthy(detection: FishDetection?): Boolean {
    if (detection == null) return false
    return detection.confidence >= 0.45f &&
      detection.box.width >= 35 &&
      detection.box.height >= 35
  }

  /**
   * Generates SlimSAM prompt points from a YOLOX fish bounding box, distinguishing vertical
   * lip-gripper holds (with negative top point) from horizontal holds.
   */
  fun generatePromptsFromDetection(
    box: PixelBox,
    imageWidth: Int,
    imageHeight: Int,
  ): List<PromptPoint> {
    val bx1 = box.x1
    val by1 = box.y1
    val bx2 = box.x2
    val by2 = box.y2
    val bw = max(20, box.width)
    val bh = max(20, box.height)
    val bcx = ((bx1 + bx2) / 2f).roundToInt()
    val bcy = ((by1 + by2) / 2f).roundToInt()

    val isVertical = bh > bw * 1.15f
    return if (isVertical) {
      listOf(
        PromptPoint("box-center", bcx, (by1 + bh * 0.56f).roundToInt(), 1),
        PromptPoint("box-upper", bcx, (by1 + bh * 0.28f).roundToInt(), 1),
        PromptPoint("box-mid-up", bcx, (by1 + bh * 0.44f).roundToInt(), 1),
        PromptPoint("box-mid-down", bcx, (by1 + bh * 0.72f).roundToInt(), 1),
        PromptPoint("box-lower", bcx, (by1 + bh * 0.88f).roundToInt(), 1),
        PromptPoint("box-lat-left", max(bx1 + 8, (bcx - bw * 0.22f).roundToInt()), (by1 + bh * 0.55f).roundToInt(), 1),
        PromptPoint("box-lat-right", min(bx2 - 8, (bcx + bw * 0.22f).roundToInt()), (by1 + bh * 0.55f).roundToInt(), 1),
        PromptPoint("box-gripper-neg", bcx, max(0, by1 - 10), 0),
      )
    } else {
      listOf(
        PromptPoint("box-center", bcx, bcy, 1),
        PromptPoint("box-head", (bx1 + bw * 0.20f).roundToInt(), bcy, 1),
        PromptPoint("box-mid-left", (bx1 + bw * 0.38f).roundToInt(), bcy, 1),
        PromptPoint("box-mid-right", (bx1 + bw * 0.62f).roundToInt(), bcy, 1),
        PromptPoint("box-tail", (bx1 + bw * 0.80f).roundToInt(), bcy, 1),
        PromptPoint("box-dorsal", bcx, max(by1 + 5, (bcy - bh * 0.20f).roundToInt()), 1),
        PromptPoint("box-ventral", bcx, min(by2 - 5, (bcy + bh * 0.20f).roundToInt()), 1),
      )
    }.map { pt ->
      pt.copy(
        x = pt.x.coerceIn(0, (imageWidth - 1).coerceAtLeast(0)),
        y = pt.y.coerceIn(0, (imageHeight - 1).coerceAtLeast(0)),
      )
    }
  }

  /**
   * Generates a 5-point positive cross prompt around the user's tap-to-refocus point.
   */
  fun generatePromptsFromTap(
    tapX: Int,
    tapY: Int,
    imageWidth: Int,
    imageHeight: Int,
  ): List<PromptPoint> {
    val maxW = (imageWidth - 1).coerceAtLeast(0)
    val maxH = (imageHeight - 1).coerceAtLeast(0)
    val fx = tapX.coerceIn(0, maxW)
    val fy = tapY.coerceIn(0, maxH)
    val dx = (imageWidth * 0.08f).roundToInt()
    val dy = (imageHeight * 0.08f).roundToInt()

    return listOf(
      PromptPoint("focus-main", fx, fy, 1),
      PromptPoint("focus-left", (fx - dx).coerceIn(10.coerceAtMost(maxW), maxW), fy, 1),
      PromptPoint("focus-right", (fx + dx).coerceIn(0, (imageWidth - 10).coerceAtLeast(0)), fy, 1),
      PromptPoint("focus-top", fx, (fy - dy).coerceIn(10.coerceAtMost(maxH), maxH), 1),
      PromptPoint("focus-bot", fx, (fy + dy).coerceIn(0, (imageHeight - 10).coerceAtLeast(0)), 1),
    )
  }

  fun selectBestCandidateIndex(
    candidates: List<CandidateMask>,
    detection: FishDetection?,
    imageWidth: Int,
    imageHeight: Int,
  ): Int {
    if (candidates.isEmpty()) return 0
    val targetArea = detection?.box?.area?.toFloat()
      ?: (imageWidth.toFloat() * imageHeight.toFloat() * 0.15f)

    var bestIdx = 0
    var bestScore = -1f
    candidates.forEachIndexed { i, c ->
      if (c.pixelCount <= 0) return@forEachIndexed
      val covRatio = c.pixelCount.toFloat() / max(1f, targetArea)
      val sizeScore = min(covRatio / 0.35f, 1.0f) * min(1.8f / max(0.01f, covRatio), 1.0f)
      val score = c.iouScore * 0.5f + sizeScore * 0.5f
      if (score > bestScore) {
        bestScore = score
        bestIdx = i
      }
    }
    return bestIdx
  }

  fun evaluateMaskTrustworthiness(
    candidate: CandidateMask?,
    contour: ContourData?,
    detection: FishDetection?,
  ): Boolean {
    if (candidate == null || contour == null || contour.rawPointCount < 20 || candidate.pixelCount < 60) {
      return false
    }
    return if (detection != null) {
      val targetArea = max(1, detection.box.area).toFloat()
      val covRatio = candidate.pixelCount.toFloat() / targetArea
      val isVert = detection.box.height > detection.box.width * 1.1f
      val majorSpan = if (isVert) {
        candidate.bbox.height.toFloat() / max(1, detection.box.height).toFloat()
      } else {
        candidate.bbox.width.toFloat() / max(1, detection.box.width).toFloat()
      }
      covRatio in 0.18f..0.85f && majorSpan >= 0.48f && candidate.iouScore >= 0.60f
    } else {
      candidate.pixelCount > 1000 && candidate.iouScore >= 0.65f
    }
  }
}
