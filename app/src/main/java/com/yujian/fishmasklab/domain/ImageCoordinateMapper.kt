package com.yujian.fishmasklab.domain

import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class MappedSamPrompts(
  val originalPrompts: List<PromptPoint>,
  val samPointCoords: FloatArray, // [1, 1, N, 2] flattened (x_sam, y_sam)
  val samPointLabels: LongArray,  // [1, 1, N] (1L or 0L)
  val pointCount: Int,
  val originalWidth: Int,
  val originalHeight: Int,
  val samResizedWidth: Int,
  val samResizedHeight: Int,
)

/**
 * Immutable coordinate mapping across:
 * 1) Original EXIF-corrected photo pixels (`origWidth` x `origHeight`)
 * 2) Inference work image (`workWidth` x `workHeight`) with proportional scale
 * 3) SlimSAM resized image (`samResizedWidth` x `samResizedHeight`) before 1024x1024 zero-padding
 * 4) Display viewport (`viewportWidth` x `viewportHeight`) with ContentScale.Fit rect
 */
data class ImageCoordinateMapper(
  val origWidth: Int,
  val origHeight: Int,
  val workWidth: Int,
  val workHeight: Int,
  val workScale: Float,
  val samResizedWidth: Int = computeSamResizedDims(origWidth, origHeight).first,
  val samResizedHeight: Int = computeSamResizedDims(origWidth, origHeight).second,
  val samScale: Double = 1024.0 / max(1, max(origWidth, origHeight)).toDouble(),
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
   * Converts original pixel coordinates into SlimSAM reshaped input coordinates:
   *   xSam = xOriginal * resizedWidth / originalWidth
   *   ySam = yOriginal * resizedHeight / originalHeight
   */
  fun originalPixelToSamPrompt(
    origX: Float,
    origY: Float,
    resizedWidth: Int = samResizedWidth,
    resizedHeight: Int = samResizedHeight,
  ): Pair<Float, Float> {
    val safeOrigW = max(1, origWidth).toFloat()
    val safeOrigH = max(1, origHeight).toFloat()
    val xSam = (origX * resizedWidth.toFloat() / safeOrigW).coerceIn(0f, resizedWidth.toFloat())
    val ySam = (origY * resizedHeight.toFloat() / safeOrigH).coerceIn(0f, resizedHeight.toFloat())
    return xSam to ySam
  }

  /**
   * Maps prompt points defined in original image pixel coordinates into SlimSAM decoder
   * `input_points` ([1, 1, N, 2] Float32) and `input_labels` ([1, 1, N] Int64).
   */
  fun mapPromptsToSamSpace(
    prompts: List<PromptPoint>,
    resizedWidth: Int = samResizedWidth,
    resizedHeight: Int = samResizedHeight,
  ): MappedSamPrompts {
    val coords = FloatArray(prompts.size * 2)
    val labels = LongArray(prompts.size)
    for (i in prompts.indices) {
      val pt = prompts[i]
      val (xSam, ySam) = originalPixelToSamPrompt(
        origX = pt.x.toFloat(),
        origY = pt.y.toFloat(),
        resizedWidth = resizedWidth,
        resizedHeight = resizedHeight,
      )
      coords[i * 2] = xSam
      coords[i * 2 + 1] = ySam
      labels[i] = pt.label.toLong()
    }
    return MappedSamPrompts(
      originalPrompts = prompts,
      samPointCoords = coords,
      samPointLabels = labels,
      pointCount = prompts.size,
      originalWidth = origWidth,
      originalHeight = origHeight,
      samResizedWidth = resizedWidth,
      samResizedHeight = resizedHeight,
    )
  }

  companion object {
    const val SAM_TARGET_MAX_SIDE = 1024

    fun computeSamResizedDims(
      origWidth: Int,
      origHeight: Int,
      targetMaxSide: Int = SAM_TARGET_MAX_SIDE,
    ): Pair<Int, Int> {
      val safeW = origWidth.coerceAtLeast(1)
      val safeH = origHeight.coerceAtLeast(1)
      val scale = targetMaxSide.toDouble() / max(safeW, safeH).toDouble()
      val resizedW = floor(safeW * scale + 0.5).toInt().coerceIn(1, targetMaxSide)
      val resizedH = floor(safeH * scale + 0.5).toInt().coerceIn(1, targetMaxSide)
      return resizedW to resizedH
    }

    fun create(origWidth: Int, origHeight: Int, maxWorkDimension: Int = 1024): ImageCoordinateMapper {
      val safeW = origWidth.coerceAtLeast(1)
      val safeH = origHeight.coerceAtLeast(1)
      val scale = min(1f, maxWorkDimension.toFloat() / max(safeW, safeH).toFloat())
      val workW = max(1, floor(safeW * scale).toInt())
      val workH = max(1, floor(safeH * scale).toInt())
      val (samW, samH) = computeSamResizedDims(safeW, safeH, SAM_TARGET_MAX_SIDE)
      val samScale = SAM_TARGET_MAX_SIDE.toDouble() / max(safeW, safeH).toDouble()
      return ImageCoordinateMapper(
        origWidth = safeW,
        origHeight = safeH,
        workWidth = workW,
        workHeight = workH,
        workScale = scale,
        samResizedWidth = samW,
        samResizedHeight = samH,
        samScale = samScale,
      )
    }
  }
}
