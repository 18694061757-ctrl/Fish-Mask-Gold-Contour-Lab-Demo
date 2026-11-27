package com.yujian.fishmasklab.runtime

import android.graphics.Bitmap
import com.yujian.fishmasklab.domain.ImageCoordinateMapper
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import org.json.JSONObject

data class SamPreprocessorConfig(
  val longestEdge: Int = 1024,
  val padWidth: Int = 1024,
  val padHeight: Int = 1024,
  val rescaleFactor: Float = 1.0f / 255.0f,
  val imageMean: FloatArray = floatArrayOf(0.485f, 0.456f, 0.406f),
  val imageStd: FloatArray = floatArrayOf(0.229f, 0.224f, 0.225f),
  val doResize: Boolean = true,
  val doRescale: Boolean = true,
  val doNormalize: Boolean = true,
  val doPad: Boolean = true,
) {
  companion object {
    fun fromJson(json: String): SamPreprocessorConfig {
      val root = JSONObject(json)
      val sizeObj = root.optJSONObject("size")
      val padObj = root.optJSONObject("pad_size")
      val meanArr = root.optJSONArray("image_mean")
      val stdArr = root.optJSONArray("image_std")
      val mean = if (meanArr != null && meanArr.length() == 3) {
        floatArrayOf(
          meanArr.getDouble(0).toFloat(),
          meanArr.getDouble(1).toFloat(),
          meanArr.getDouble(2).toFloat(),
        )
      } else {
        floatArrayOf(0.485f, 0.456f, 0.406f)
      }
      val std = if (stdArr != null && stdArr.length() == 3) {
        floatArrayOf(
          stdArr.getDouble(0).toFloat(),
          stdArr.getDouble(1).toFloat(),
          stdArr.getDouble(2).toFloat(),
        )
      } else {
        floatArrayOf(0.229f, 0.224f, 0.225f)
      }
      return SamPreprocessorConfig(
        longestEdge = sizeObj?.optInt("longest_edge", 1024) ?: 1024,
        padWidth = padObj?.optInt("width", 1024) ?: 1024,
        padHeight = padObj?.optInt("height", 1024) ?: 1024,
        rescaleFactor = root.optDouble("rescale_factor", 1.0 / 255.0).toFloat(),
        imageMean = mean,
        imageStd = std,
        doResize = root.optBoolean("do_resize", true),
        doRescale = root.optBoolean("do_rescale", true),
        doNormalize = root.optBoolean("do_normalize", true),
        doPad = root.optBoolean("do_pad", true),
      )
    }
  }
}

data class SamPreprocessedImage(
  val pixelValues: FloatArray, // NCHW [1, 3, padHeight, padWidth]
  val originalWidth: Int,
  val originalHeight: Int,
  val resizedWidth: Int,
  val resizedHeight: Int,
  val padWidth: Int,
  val padHeight: Int,
  val padRight: Int,
  val padBottom: Int,
  val preprocessDurationMs: Long,
)

/**
 * Full parity port of HuggingFace `SamImageProcessor` (`preprocessor_config.json`):
 * - Convert ARGB -> RGB
 * - Bilinear resize (`align_corners=false` pixel-center mapping) so longest edge == 1024
 * - Rescale by `1/255`
 * - Normalize with ImageNet RGB mean `[0.485, 0.456, 0.406]` & std `[0.229, 0.224, 0.225]`
 * - Zero-pad bottom & right AFTER normalization to `[1, 3, 1024, 1024]` FLOAT32 NCHW
 */
class SlimSamImageProcessor(
  val config: SamPreprocessorConfig = SamPreprocessorConfig(),
) {

  fun preprocess(bitmap: Bitmap): SamPreprocessedImage {
    check(!bitmap.isRecycled) { "Cannot preprocess recycled Bitmap" }
    val w = bitmap.width
    val h = bitmap.height
    val pixels = IntArray(w * h)
    bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
    return preprocessArgb(pixels, w, h)
  }

  fun preprocessArgb(argbPixels: IntArray, originalWidth: Int, originalHeight: Int): SamPreprocessedImage {
    require(originalWidth > 0 && originalHeight > 0) {
      "Invalid image dimensions: ${originalWidth}x$originalHeight"
    }
    require(argbPixels.size >= originalWidth * originalHeight) {
      "Pixel buffer size ${argbPixels.size} smaller than ${originalWidth * originalHeight}"
    }

    val start = System.currentTimeMillis()
    val (resizedW, resizedH) = if (config.doResize) {
      ImageCoordinateMapper.computeSamResizedDims(originalWidth, originalHeight, config.longestEdge)
    } else {
      originalWidth.coerceAtMost(config.padWidth) to originalHeight.coerceAtMost(config.padHeight)
    }

    val padW = if (config.doPad) config.padWidth else resizedW
    val padH = if (config.doPad) config.padHeight else resizedH
    val planeSize = padW * padH
    val tensorData = FloatArray(3 * planeSize) // zero-filled -> bottom/right zero padding after norm

    val meanR = config.imageMean[0]
    val meanG = config.imageMean[1]
    val meanB = config.imageMean[2]
    val invStdR = 1f / config.imageStd[0]
    val invStdG = 1f / config.imageStd[1]
    val invStdB = 1f / config.imageStd[2]
    val rescale = if (config.doRescale) config.rescaleFactor else 1f

    // Precompute bilinear X sampling indices & weights (`align_corners=false`)
    val x0Arr = IntArray(resizedW)
    val x1Arr = IntArray(resizedW)
    val wxArr = FloatArray(resizedW)
    val xScale = originalWidth.toFloat() / resizedW.toFloat()
    for (x in 0 until resizedW) {
      val srcX = ((x + 0.5f) * xScale - 0.5f).coerceIn(0f, (originalWidth - 1).toFloat())
      val x0 = floor(srcX).toInt()
      val x1 = min(originalWidth - 1, x0 + 1)
      x0Arr[x] = x0
      x1Arr[x] = x1
      wxArr[x] = srcX - x0
    }

    val yScale = originalHeight.toFloat() / resizedH.toFloat()
    val maxOrigY = originalHeight - 1
    val planeG = planeSize
    val planeB = 2 * planeSize

    for (y in 0 until resizedH) {
      val srcY = ((y + 0.5f) * yScale - 0.5f).coerceIn(0f, maxOrigY.toFloat())
      val y0 = floor(srcY).toInt()
      val y1 = min(maxOrigY, y0 + 1)
      val wy = srcY - y0
      val invWy = 1f - wy

      val srcRow0 = y0 * originalWidth
      val srcRow1 = y1 * originalWidth
      val dstRow = y * padW

      for (x in 0 until resizedW) {
        val x0 = x0Arr[x]
        val x1 = x1Arr[x]
        val wx = wxArr[x]
        val invWx = 1f - wx

        val c00 = argbPixels[srcRow0 + x0]
        val c10 = argbPixels[srcRow0 + x1]
        val c01 = argbPixels[srcRow1 + x0]
        val c11 = argbPixels[srcRow1 + x1]

        val w00 = invWx * invWy
        val w10 = wx * invWy
        val w01 = invWx * wy
        val w11 = wx * wy

        val r = (((c00 shr 16) and 0xFF) * w00 +
          ((c10 shr 16) and 0xFF) * w10 +
          ((c01 shr 16) and 0xFF) * w01 +
          ((c11 shr 16) and 0xFF) * w11) * rescale

        val g = (((c00 shr 8) and 0xFF) * w00 +
          ((c10 shr 8) and 0xFF) * w10 +
          ((c01 shr 8) and 0xFF) * w01 +
          ((c11 shr 8) and 0xFF) * w11) * rescale

        val b = ((c00 and 0xFF) * w00 +
          (c10 and 0xFF) * w10 +
          (c01 and 0xFF) * w01 +
          (c11 and 0xFF) * w11) * rescale

        val dstIdx = dstRow + x
        if (config.doNormalize) {
          tensorData[dstIdx] = (r - meanR) * invStdR
          tensorData[planeG + dstIdx] = (g - meanG) * invStdG
          tensorData[planeB + dstIdx] = (b - meanB) * invStdB
        } else {
          tensorData[dstIdx] = r
          tensorData[planeG + dstIdx] = g
          tensorData[planeB + dstIdx] = b
        }
      }
    }

    return SamPreprocessedImage(
      pixelValues = tensorData,
      originalWidth = originalWidth,
      originalHeight = originalHeight,
      resizedWidth = resizedW,
      resizedHeight = resizedH,
      padWidth = padW,
      padHeight = padH,
      padRight = (padW - resizedW).coerceAtLeast(0),
      padBottom = (padH - resizedH).coerceAtLeast(0),
      preprocessDurationMs = (System.currentTimeMillis() - start).coerceAtLeast(0L),
    )
  }
}
