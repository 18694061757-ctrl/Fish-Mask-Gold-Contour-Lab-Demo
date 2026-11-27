package com.yujian.fishmasklab.domain

import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Port of prompt generation, 45% expanded detection box clipping, candidate scoring,
 * and strict mask trustworthiness gating from `src/services/samService.ts` and `src/App.tsx`.
 */
object MaskQualityGate {

  const val DETECTION_CLIP_PADDING_RATIO = 0.45f

  fun isBboxTrustworthy(detection: FishDetection?): Boolean {
    if (detection == null) return false
    return detection.confidence >= 0.45f &&
      detection.box.width >= 35 &&
      detection.box.height >= 35
  }

  /**
   * Computes the 45% expanded safety clipping box around a YOLOX fish detection box
   * so long fins and tails stay intact while far background regions are excluded.
   */
  fun computeExpandedDetectionBox(
    box: PixelBox,
    imageWidth: Int,
    imageHeight: Int,
    paddingRatio: Float = DETECTION_CLIP_PADDING_RATIO,
  ): PixelBox {
    val padX = (box.width * paddingRatio).roundToInt()
    val padY = (box.height * paddingRatio).roundToInt()
    return PixelBox(
      x1 = max(0, box.x1 - padX),
      y1 = max(0, box.y1 - padY),
      x2 = min(imageWidth, box.x2 + padX),
      y2 = min(imageHeight, box.y2 + padY),
    )
  }

  /**
   * Checks whether a user tap-to-refocus point falls within the fish detection box (plus
   * reasonable fin/tail padding) so tapping distant background does not target non-fish objects.
   */
  fun isTapWithinFishFocusRegion(
    tapPoint: ContourPoint,
    detection: FishDetection?,
    imageWidth: Int,
    imageHeight: Int,
  ): Boolean {
    if (tapPoint.x < 0f || tapPoint.y < 0f || tapPoint.x >= imageWidth || tapPoint.y >= imageHeight) {
      return false
    }
    if (detection == null) return true
    val expanded = computeExpandedDetectionBox(detection.box, imageWidth, imageHeight, 0.30f)
    return tapPoint.x >= expanded.x1 &&
      tapPoint.x <= expanded.x2 &&
      tapPoint.y >= expanded.y1 &&
      tapPoint.y <= expanded.y2
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

  /**
   * Applies 45% expanded detection box clipping in-place on [mask] (values 0 or 0xFF).
   */
  fun clipMaskWithExpandedDetectionBox(
    mask: ByteArray,
    width: Int,
    height: Int,
    detectionBox: PixelBox?,
  ) {
    if (detectionBox == null || width <= 0 || height <= 0 || mask.size < width * height) return
    val clip = computeExpandedDetectionBox(detectionBox, width, height)
    for (y in 0 until height) {
      val rowOff = y * width
      if (y < clip.y1 || y > clip.y2) {
        for (x in 0 until width) {
          mask[rowOff + x] = 0
        }
      } else {
        for (x in 0 until clip.x1) {
          mask[rowOff + x] = 0
        }
        for (x in (clip.x2 + 1) until width) {
          mask[rowOff + x] = 0
        }
      }
    }
  }

  /**
   * Computes candidate ranking score matching `samService.ts`:
   *   Score = 0.4 * IoU + 0.3 * Coverage + 0.3 * Span
   */
  fun computeCandidateScore(
    iouScore: Float,
    pixelCount: Int,
    maskBbox: PixelBox,
    detection: FishDetection?,
  ): Float {
    if (pixelCount <= 0) return -1f
    if (detection == null) return iouScore
    val targetW = max(1, detection.box.width).toFloat()
    val targetH = max(1, detection.box.height).toFloat()
    val targetArea = max(1f, targetW * targetH)
    val coverage = min(pixelCount.toFloat() / (targetArea * 0.35f), 1.0f)
    val spanH = min(maskBbox.height.toFloat() / max(1f, targetH * 0.7f), 1.0f)
    val spanW = min(maskBbox.width.toFloat() / max(1f, targetW * 0.7f), 1.0f)
    val span = spanH * spanW
    return iouScore * 0.4f + coverage * 0.3f + span * 0.3f
  }

  /**
   * Ranks candidates and returns the best index among trustworthy candidates (or best overall
   * score if none are trustworthy).
   */
  fun selectBestCandidateIndex(
    candidates: List<CandidateMask>,
    detection: FishDetection?,
    imageWidth: Int,
    imageHeight: Int,
  ): Int {
    if (candidates.isEmpty()) return 0

    var bestTrustworthyIdx = -1
    var bestTrustworthyScore = -Float.MAX_VALUE
    var bestOverallIdx = 0
    var bestOverallScore = -Float.MAX_VALUE

    candidates.forEachIndexed { i, c ->
      if (c.pixelCount <= 0) return@forEachIndexed
      val score = computeCandidateScore(c.iouScore, c.pixelCount, c.bbox, detection)
      if (score > bestOverallScore) {
        bestOverallScore = score
        bestOverallIdx = i
      }
      if (c.isTrustworthy && score > bestTrustworthyScore) {
        bestTrustworthyScore = score
        bestTrustworthyIdx = i
      }
    }

    return if (bestTrustworthyIdx >= 0) bestTrustworthyIdx else bestOverallIdx
  }

  /**
   * Evaluates candidate geometry trustworthiness without requiring a pre-extracted ContourData.
   */
  fun evaluateCandidateGeometryTrustworthiness(
    candidate: CandidateMask?,
    detection: FishDetection?,
  ): Boolean {
    if (candidate == null || candidate.pixelCount < 60) {
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

  fun evaluateMaskTrustworthiness(
    candidate: CandidateMask?,
    contour: ContourData?,
    detection: FishDetection?,
  ): Boolean {
    if (candidate == null || contour == null || !contour.traceClosed || contour.rawPointCount < 20) {
      return false
    }
    return evaluateCandidateGeometryTrustworthiness(candidate, detection)
  }
}
