package com.yujian.fishmasklab.runtime

import android.graphics.Bitmap
import android.graphics.Color
import com.yujian.fishmasklab.domain.GlintOverlayData
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Extracts natural specular highlights inside a verified fish mask.
 * Port of `extractInternalGlintMask` from `src/utils/contourExtraction.ts`:
 * - Samples within mask up to maxDim = 800
 * - Rec.601 luminance histogram, selects top ~8% brightest pixels clamped >= 110
 * - Colors pixels #F0DCA5 (240, 220, 165) with proportional alpha
 */
object InternalGlintExtractor {

  private const val MAX_SAMPLE_DIM = 800
  private const val MIN_MASK_PIXELS = 50

  fun extract(
    sourceBitmap: Bitmap,
    cleanMask: ByteArray,
    maskWidth: Int,
    maskHeight: Int,
    targetCoverage: Float = 0.08f,
  ): GlintOverlayData? {
    if (maskWidth <= 0 || maskHeight <= 0 || cleanMask.size < maskWidth * maskHeight) {
      return null
    }

    val scale = min(1.0f, MAX_SAMPLE_DIM.toFloat() / max(maskWidth, maskHeight).toFloat())
    val sw = max(1, (maskWidth * scale).roundToInt())
    val sh = max(1, (maskHeight * scale).roundToInt())

    val sampled = Bitmap.createScaledBitmap(sourceBitmap, sw, sh, true)
    val rgb = IntArray(sw * sh)
    try {
      sampled.getPixels(rgb, 0, sw, 0, 0, sw, sh)
    } finally {
      if (sampled !== sourceBitmap && !sampled.isRecycled) {
        sampled.recycle()
      }
    }

    val maskSample = BooleanArray(sw * sh)
    val lumHistogram = IntArray(256)
    val luminances = IntArray(sw * sh)
    var maskCount = 0

    for (sy in 0 until sh) {
      val origY = min(maskHeight - 1, floor(sy / scale).toInt())
      val rowOff = sy * sw
      val origRowOff = origY * maskWidth
      for (sx in 0 until sw) {
        val origX = min(maskWidth - 1, floor(sx / scale).toInt())
        if ((cleanMask[origRowOff + origX].toInt() and 0xFF) > 127) {
          val sIdx = rowOff + sx
          maskSample[sIdx] = true
          maskCount++
          val c = rgb[sIdx]
          val lum = (0.299f * Color.red(c) + 0.587f * Color.green(c) + 0.114f * Color.blue(c))
            .roundToInt()
            .coerceIn(0, 255)
          luminances[sIdx] = lum
          lumHistogram[lum]++
        }
      }
    }

    if (maskCount < MIN_MASK_PIXELS) return null

    val clampedCov = targetCoverage.coerceIn(0.05f, 0.10f)
    val targetCount = (maskCount * clampedCov).roundToInt()
    var accumulated = 0
    var threshold = 210
    for (l in 255 downTo 0) {
      accumulated += lumHistogram[l]
      if (accumulated >= targetCount) {
        threshold = l
        break
      }
    }
    threshold = max(110, threshold)

    val outPixels = IntArray(sw * sh)
    val range = max(1, 255 - threshold).toFloat()
    for (i in 0 until sw * sh) {
      if (maskSample[i] && luminances[i] >= threshold) {
        val strength = (luminances[i] - threshold) / range
        val alpha = (strength * 255f).roundToInt().coerceIn(0, 255)
        outPixels[i] = Color.argb(alpha, 240, 220, 165) // #F0DCA5
      }
    }

    return GlintOverlayData(width = sw, height = sh, argbPixels = outPixels)
  }
}
