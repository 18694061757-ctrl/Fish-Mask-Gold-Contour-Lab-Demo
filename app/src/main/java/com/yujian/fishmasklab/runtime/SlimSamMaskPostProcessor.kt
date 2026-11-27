package com.yujian.fishmasklab.runtime

import com.yujian.fishmasklab.domain.CandidateMask
import com.yujian.fishmasklab.domain.ContourGeometry
import com.yujian.fishmasklab.domain.FishDetection
import com.yujian.fishmasklab.domain.MaskQualityGate
import com.yujian.fishmasklab.domain.MaskResult
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Full parity port of HuggingFace `SamImageProcessor.post_process_masks`:
 * 1) Bilinear upsample each 256x256 logit candidate (`align_corners=false`) to 1024x1024.
 * 2) Crop out right/bottom padding down to `[resizedHeight, resizedWidth]`.
 * 3) Bilinear resize (`align_corners=false`) back to `[originalHeight, originalWidth]`.
 * 4) Threshold `logit > 0.0f` to binary mask (0 / 0xFF).
 * 5) Clip outside 45% expanded YOLOX box, isolate LCC, score & evaluate trustworthiness.
 */
object SlimSamMaskPostProcessor {

  private const val SRC_SIZE = SamDecoderOutput.MASK_LOGIT_SIZE // 256
  private const val PAD_SIZE = SlimSamVisionEncoder.INPUT_SIZE   // 1024

  fun postProcessCandidateLogits(
    logits256: FloatArray,
    logitOffset: Int,
    resizedWidth: Int,
    resizedHeight: Int,
    originalWidth: Int,
    originalHeight: Int,
    upsampled1024Scratch: FloatArray = FloatArray(PAD_SIZE * PAD_SIZE),
  ): ByteArray {
    require(logits256.size >= logitOffset + SRC_SIZE * SRC_SIZE) {
      "Insufficient logits buffer size ${logits256.size} for offset $logitOffset"
    }
    require(resizedWidth in 1..PAD_SIZE && resizedHeight in 1..PAD_SIZE) {
      "Invalid resized dims ${resizedWidth}x$resizedHeight"
    }
    require(originalWidth > 0 && originalHeight > 0) {
      "Invalid original dims ${originalWidth}x$originalHeight"
    }

    // Step 1: Bilinear upsample 256x256 -> 1024x1024 (align_corners=false)
    // Only need rows 0 until resizedHeight and cols 0 until resizedWidth since padding is cropped
    bilinearUpsampleCroppedRegion(
      src = logits256,
      srcOffset = logitOffset,
      srcW = SRC_SIZE,
      srcH = SRC_SIZE,
      fullDstW = PAD_SIZE,
      fullDstH = PAD_SIZE,
      cropW = resizedWidth,
      cropH = resizedHeight,
      dst = upsampled1024Scratch,
    )

    // Step 2 & 3: Crop [resizedHeight, resizedWidth] and bilinear resize -> [originalHeight, originalWidth]
    // Threshold logit > 0.0f into binary mask ByteArray
    val outMask = ByteArray(originalWidth * originalHeight)
    val x0Arr = IntArray(originalWidth)
    val x1Arr = IntArray(originalWidth)
    val wxArr = FloatArray(originalWidth)
    val xScale = resizedWidth.toFloat() / originalWidth.toFloat()
    val maxCropX = (resizedWidth - 1).coerceAtLeast(0)
    for (x in 0 until originalWidth) {
      val sx = ((x + 0.5f) * xScale - 0.5f).coerceIn(0f, maxCropX.toFloat())
      val x0 = floor(sx).toInt()
      val x1 = min(maxCropX, x0 + 1)
      x0Arr[x] = x0
      x1Arr[x] = x1
      wxArr[x] = sx - x0
    }

    val yScale = resizedHeight.toFloat() / originalHeight.toFloat()
    val maxCropY = (resizedHeight - 1).coerceAtLeast(0)
    for (y in 0 until originalHeight) {
      val sy = ((y + 0.5f) * yScale - 0.5f).coerceIn(0f, maxCropY.toFloat())
      val y0 = floor(sy).toInt()
      val y1 = min(maxCropY, y0 + 1)
      val wy = sy - y0
      val invWy = 1f - wy

      val row0 = y0 * resizedWidth
      val row1 = y1 * resizedWidth
      val dstRow = y * originalWidth

      for (x in 0 until originalWidth) {
        val x0 = x0Arr[x]
        val x1 = x1Arr[x]
        val wx = wxArr[x]
        val invWx = 1f - wx

        val v00 = upsampled1024Scratch[row0 + x0]
        val v10 = upsampled1024Scratch[row0 + x1]
        val v01 = upsampled1024Scratch[row1 + x0]
        val v11 = upsampled1024Scratch[row1 + x1]

        val logit = (v00 * invWx + v10 * wx) * invWy + (v01 * invWx + v11 * wx) * wy
        if (logit > 0.0f) {
          outMask[dstRow + x] = 0xFF.toByte()
        }
      }
    }

    return outMask
  }

  /**
   * Bilinear interpolation (`align_corners=false`) from [srcW]x[srcH] into target grid of size
   * [fullDstW]x[fullDstH], storing only the unpadded cropped `[cropH, cropW]` region densely
   * at stride `cropW`.
   */
  internal fun bilinearUpsampleCroppedRegion(
    src: FloatArray,
    srcOffset: Int,
    srcW: Int,
    srcH: Int,
    fullDstW: Int,
    fullDstH: Int,
    cropW: Int,
    cropH: Int,
    dst: FloatArray,
  ) {
    val x0Arr = IntArray(cropW)
    val x1Arr = IntArray(cropW)
    val wxArr = FloatArray(cropW)
    val xScale = srcW.toFloat() / fullDstW.toFloat()
    val maxSrcX = srcW - 1
    for (x in 0 until cropW) {
      val sx = ((x + 0.5f) * xScale - 0.5f).coerceIn(0f, maxSrcX.toFloat())
      val x0 = floor(sx).toInt()
      val x1 = min(maxSrcX, x0 + 1)
      x0Arr[x] = x0
      x1Arr[x] = x1
      wxArr[x] = sx - x0
    }

    val yScale = srcH.toFloat() / fullDstH.toFloat()
    val maxSrcY = srcH - 1
    for (y in 0 until cropH) {
      val sy = ((y + 0.5f) * yScale - 0.5f).coerceIn(0f, maxSrcY.toFloat())
      val y0 = floor(sy).toInt()
      val y1 = min(maxSrcY, y0 + 1)
      val wy = sy - y0
      val invWy = 1f - wy

      val row0 = srcOffset + y0 * srcW
      val row1 = srcOffset + y1 * srcW
      val dstRow = y * cropW

      for (x in 0 until cropW) {
        val x0 = x0Arr[x]
        val x1 = x1Arr[x]
        val wx = wxArr[x]
        val invWx = 1f - wx
        val v00 = src[row0 + x0]
        val v10 = src[row0 + x1]
        val v01 = src[row1 + x0]
        val v11 = src[row1 + x1]
        dst[dstRow + x] = (v00 * invWx + v10 * wx) * invWy + (v01 * invWx + v11 * wx) * wy
      }
    }
  }

  /**
   * Converts decoder logits & IoU scores into 3 cleaned, clipped, LCC-filtered, scored
   * [CandidateMask] entries and picks the best candidate.
   */
  fun buildMaskResult(
    decoderOutput: SamDecoderOutput,
    resizedWidth: Int,
    resizedHeight: Int,
    originalWidth: Int,
    originalHeight: Int,
    detection: FishDetection?,
    totalSamInferenceMs: Long,
  ): MaskResult {
    val totalPixels = max(1, originalWidth * originalHeight)
    val scratch = FloatArray(PAD_SIZE * PAD_SIZE)
    val candidates = ArrayList<CandidateMask>(SamDecoderOutput.NUM_CANDIDATES)

    for (c in 0 until SamDecoderOutput.NUM_CANDIDATES) {
      val binaryMask = postProcessCandidateLogits(
        logits256 = decoderOutput.predMasksLogits,
        logitOffset = c * SamDecoderOutput.MASK_PLANE_ELEMENTS,
        resizedWidth = resizedWidth,
        resizedHeight = resizedHeight,
        originalWidth = originalWidth,
        originalHeight = originalHeight,
        upsampled1024Scratch = scratch,
      )
      MaskQualityGate.clipMaskWithExpandedDetectionBox(
        mask = binaryMask,
        width = originalWidth,
        height = originalHeight,
        detectionBox = detection?.box,
      )
      val lcc = ContourGeometry.getLargestConnectedComponent(binaryMask, originalWidth, originalHeight)
      val iou = decoderOutput.iouScores[c]
      val areaRatio = lcc.pixelCount.toFloat() / totalPixels.toFloat()
      val draft = CandidateMask(
        index = c,
        iouScore = iou,
        data = lcc.cleanMask,
        pixelCount = lcc.pixelCount,
        areaRatio = areaRatio,
        bbox = lcc.bbox,
      )
      val geoTrustworthy = MaskQualityGate.evaluateCandidateGeometryTrustworthiness(draft, detection)
      val score = MaskQualityGate.computeCandidateScore(iou, lcc.pixelCount, lcc.bbox, detection)
      candidates.add(draft.copy(isTrustworthy = geoTrustworthy, rankScore = score))
    }

    val bestIdx = MaskQualityGate.selectBestCandidateIndex(
      candidates = candidates,
      detection = detection,
      imageWidth = originalWidth,
      imageHeight = originalHeight,
    )

    return MaskResult(
      width = originalWidth,
      height = originalHeight,
      selectedCandidateIndex = bestIdx,
      candidates = candidates,
      inferenceTimeMs = totalSamInferenceMs,
      isMaskTrustworthy = candidates.getOrNull(bestIdx)?.isTrustworthy == true,
      isBboxTrustworthy = MaskQualityGate.isBboxTrustworthy(detection),
    )
  }
}
