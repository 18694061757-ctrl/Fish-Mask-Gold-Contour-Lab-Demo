package com.yujian.fishmasklab.domain

import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Immutable coordinate mapping across:
 * 1) Original EXIF-corrected photo pixels (`origWidth` x `origHeight`)
 * 2) Inference work image (`workWidth` x `workHeight`) with proportional letterbox scale
 * 3) Display viewport (`viewportWidth` x `viewportHeight`) with ContentScale.Fit rect
 */
data class ImageCoordinateMapper(
  val origWidth: Int,
  val origHeight: Int,
  val workWidth: Int,
  val workHeight: Int,
  val workScale: Float,
) {
  data class DisplayPhotoRect(
    val left: Float,
    val top: Float,
    val width: Float,
    val height: Float,
    val scale: Float,
  ) {
    val right: Float get() = left + width
    val bottom: Float get() = top + height
  }

  /**
   * Computes the exact fitted photo rectangle inside a viewport of [viewportWidth] x [viewportHeight].
   */
  fun computeDisplayRect(viewportWidth: Float, viewportHeight: Float): DisplayPhotoRect {
    if (origWidth <= 0 || origHeight <= 0 || viewportWidth <= 0f || viewportHeight <= 0f) {
      return DisplayPhotoRect(0f, 0f, 0f, 0f, 1f)
    }
    val scale = min(viewportWidth / origWidth.toFloat(), viewportHeight / origHeight.toFloat())
    val drawnW = origWidth * scale
    val drawnH = origHeight * scale
    val left = (viewportWidth - drawnW) * 0.5f
    val top = (viewportHeight - drawnH) * 0.5f
    return DisplayPhotoRect(left = left, top = top, width = drawnW, height = drawnH, scale = scale)
  }

  /**
   * Converts a touch position in viewport space into original image pixel coordinates.
   * Returns null if the tap falls outside the rendered photo area (letterbox margins).
   */
  fun viewportTapToOriginalPixel(
    tapX: Float,
    tapY: Float,
    viewportWidth: Float,
    viewportHeight: Float,
  ): ContourPoint? {
    val rect = computeDisplayRect(viewportWidth, viewportHeight)
    if (rect.width <= 0f || rect.height <= 0f || rect.scale <= 0f) return null
    if (tapX < rect.left || tapX > rect.right || tapY < rect.top || tapY > rect.bottom) {
      return null
    }
    val origX = ((tapX - rect.left) / rect.scale).coerceIn(0f, (origWidth - 1).coerceAtLeast(0).toFloat())
    val origY = ((tapY - rect.top) / rect.scale).coerceIn(0f, (origHeight - 1).coerceAtLeast(0).toFloat())
    return ContourPoint(origX, origY)
  }

  /**
   * Maps a detector box in [inputSize]x[inputSize] top-left letterboxed coordinates back to
   * original image pixel coordinates.
   */
  fun detectorBoxToOriginalBox(
    cx: Float,
    cy: Float,
    w: Float,
    h: Float,
    detectorScale: Float,
  ): PixelBox {
    val safeScale = max(1e-5f, detectorScale)
    val x1 = ((cx - w * 0.5f) / safeScale).roundToInt().coerceIn(0, origWidth)
    val y1 = ((cy - h * 0.5f) / safeScale).roundToInt().coerceIn(0, origHeight)
    val x2 = ((cx + w * 0.5f) / safeScale).roundToInt().coerceIn(0, origWidth)
    val y2 = ((cy + h * 0.5f) / safeScale).roundToInt().coerceIn(0, origHeight)
    return PixelBox(x1, y1, x2, y2)
  }

  /**
   * Converts original pixel coordinates into SlimSAM reshaped input coordinates (longest side = [targetMaxSide]).
   */
  fun originalPixelToSamPrompt(origX: Float, origY: Float, targetMaxSide: Int = 1024): Pair<Float, Float> {
    val longest = max(origWidth, origHeight).coerceAtLeast(1)
    val samScale = targetMaxSide.toFloat() / longest.toFloat()
    return (origX * samScale) to (origY * samScale)
  }

  companion object {
    fun create(origWidth: Int, origHeight: Int, maxWorkDimension: Int = 1024): ImageCoordinateMapper {
      val safeW = origWidth.coerceAtLeast(1)
      val safeH = origHeight.coerceAtLeast(1)
      val scale = min(1f, maxWorkDimension.toFloat() / max(safeW, safeH).toFloat())
      val workW = max(1, floor(safeW * scale).toInt())
      val workH = max(1, floor(safeH * scale).toInt())
      return ImageCoordinateMapper(
        origWidth = safeW,
        origHeight = safeH,
        workWidth = workW,
        workHeight = workH,
        workScale = scale,
      )
    }
  }
}
