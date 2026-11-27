package com.yujian.fishmasklab.renderer

import android.graphics.Bitmap
import android.graphics.BlurMaskFilter
import android.graphics.DashPathEffect
import android.graphics.LinearGradient
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PathMeasure
import android.graphics.RectF
import android.graphics.Shader
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalDensity
import com.yujian.fishmasklab.domain.ContourData
import com.yujian.fishmasklab.domain.FrozenContourConfig
import com.yujian.fishmasklab.domain.ImageCoordinateMapper
import kotlin.math.PI
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

private class CachedContourRenderResources(
  val contourPath: Path,
  val pathLength: Float,
  val pathMeasure: PathMeasure,
  val segmentPath: Path,
  val wavePath: Path,
  val glintBitmap: Bitmap?,
  val blueGoldShader: LinearGradient,
  val sunlitGoldShader: LinearGradient,
  val haloPaint: Paint,
  val softGlowPaint: Paint,
  val goldGlowPaint: Paint,
  val underStrokePaint: Paint,
  val softEdgePaint: Paint,
  val corePaint: Paint,
  val highlightPaint: Paint,
  val wavePaint: Paint,
  val glintPaint: Paint,
)

private fun smoothstep(edge0: Float, edge1: Float, x: Float): Float {
  if (edge1 <= edge0) return if (x >= edge1) 1f else 0f
  val t = ((x - edge0) / (edge1 - edge0)).coerceIn(0f, 1f)
  return t * t * (3f - 2f * t)
}

/**
 * Renders Fish Contour Glow V1.3 (Blue-Gold 2-cycle gradient, L0/L1/L2/L3 optical layers,
 * Halo, Soft Glow, and mask-clipped Internal Glint) over the fitted photo rect.
 *
 * All Paths, Shaders, Paints, and Glint Bitmaps are cached per [contourData] and [displayRect]
 * so replaying the 0-2600ms+ timeline allocates zero new GPU/Bitmap objects.
 */
@Composable
fun FishContourGlowCanvas(
  contourData: ContourData,
  elapsedMs: Float,
  displayRect: ImageCoordinateMapper.DisplayPhotoRect,
  config: FrozenContourConfig = FrozenContourConfig(),
  modifier: Modifier = Modifier,
) {
  val density = LocalDensity.current.density

  val resources = remember(contourData, displayRect, config, density) {
    val path = Path().apply {
      moveTo(contourData.startPoint.x, contourData.startPoint.y)
      for (seg in contourData.bezierSegments) {
        cubicTo(seg.cp1x, seg.cp1y, seg.cp2x, seg.cp2y, seg.endX, seg.endY)
      }
      close()
    }

    // Transform path from original photo pixels to viewport fitted display coordinates
    val matrix = Matrix().apply {
      postScale(displayRect.scale, displayRect.scale)
      postTranslate(displayRect.left, displayRect.top)
    }
    path.transform(matrix)

    val measure = PathMeasure(path, true)
    val measuredLen = max(1f, measure.length)

    val bounds = RectF()
    path.computeBounds(bounds, true)
    val x0 = bounds.left
    val y0 = bounds.top
    val x1 = max(bounds.left + 1f, bounds.right)
    val y1 = max(bounds.top + 1f, bounds.bottom)

    // 2-cycle Blue-Gold-Blue-Gold gradient matching `fcg-blue-gold-grad`
    val blueGoldShader = LinearGradient(
      x0,
      y0,
      x1,
      y1,
      intArrayOf(
        0xFF0284C7.toInt(),
        0xFF38BDF8.toInt(),
        0xFFFDE68A.toInt(),
        0xFFF59E0B.toInt(),
        0xFF0284C7.toInt(),
        0xFF38BDF8.toInt(),
        0xFFFDE68A.toInt(),
        0xFFF59E0B.toInt(),
        0xFF0284C7.toInt(),
      ),
      floatArrayOf(0f, 0.14f, 0.28f, 0.44f, 0.52f, 0.66f, 0.80f, 0.94f, 1.0f),
      Shader.TileMode.CLAMP,
    )

    val sunlitGoldShader = LinearGradient(
      x0,
      y0,
      x1,
      y1,
      intArrayOf(
        0x1AFFD166,
        0xE6FFE494.toInt(),
        0xFFFFF4D0.toInt(),
        0xE6FFD166.toInt(),
        0x1AE5A62D,
      ),
      floatArrayOf(0f, 0.30f, 0.50f, 0.70f, 1.0f),
      Shader.TileMode.CLAMP,
    )

    val corePx = max(0.5f, config.coreLineWidthDp) * density
    val underPx = max(1.0f, config.underStrokeWidthDp) * density
    val highlightPx = max(0.6f, config.coreLineWidthDp * 0.46f) * density
    val softGlowPx = max(1f, config.softGlowRadiusDp * 0.45f) * density
    val goldGlowPx = max(1f, config.softGlowRadiusDp * 0.30f) * density
    val haloPx = max(2f, config.haloRadiusDp * 0.45f) * density

    fun baseStrokePaint(strokePx: Float, colorArgb: Long): Paint =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = strokePx
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        color = colorArgb.toInt()
      }

    val haloPaint = baseStrokePaint(corePx * 2.8f, config.haloColorArgb).apply {
      maskFilter = BlurMaskFilter(haloPx, BlurMaskFilter.Blur.NORMAL)
    }
    val softGlowPaint = baseStrokePaint(corePx * 1.9f, config.softGlowColorArgb).apply {
      maskFilter = BlurMaskFilter(softGlowPx, BlurMaskFilter.Blur.NORMAL)
    }
    val goldGlowPaint = baseStrokePaint(corePx * 1.5f, config.goldGlowColorArgb).apply {
      maskFilter = BlurMaskFilter(goldGlowPx, BlurMaskFilter.Blur.NORMAL)
    }
    val underStrokePaint = baseStrokePaint(underPx, config.underStrokeColorArgb)
    val softEdgePaint = baseStrokePaint(corePx * 1.3f, config.coreColorArgb).apply {
      shader = blueGoldShader
      maskFilter = BlurMaskFilter(max(0.8f, 0.9f * density), BlurMaskFilter.Blur.NORMAL)
    }
    val corePaint = baseStrokePaint(corePx, config.coreColorArgb).apply {
      shader = blueGoldShader
    }
    val highlightPaint = baseStrokePaint(highlightPx, config.highlightColorArgb)
    val wavePaint = baseStrokePaint(max(1.4f * density, corePx * 0.95f), config.goldAccentColorArgb).apply {
      shader = sunlitGoldShader
      maskFilter = BlurMaskFilter(max(1.5f * density, corePx * 0.8f), BlurMaskFilter.Blur.NORMAL)
    }

    val glintBmp = contourData.internalGlint?.let { g ->
      Bitmap.createBitmap(g.argbPixels, g.width, g.height, Bitmap.Config.ARGB_8888)
    }
    val glintPaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)

    CachedContourRenderResources(
      contourPath = path,
      pathLength = measuredLen,
      pathMeasure = measure,
      segmentPath = Path(),
      wavePath = Path(),
      glintBitmap = glintBmp,
      blueGoldShader = blueGoldShader,
      sunlitGoldShader = sunlitGoldShader,
      haloPaint = haloPaint,
      softGlowPaint = softGlowPaint,
      goldGlowPaint = goldGlowPaint,
      underStrokePaint = underStrokePaint,
      softEdgePaint = softEdgePaint,
      corePaint = corePaint,
      highlightPaint = highlightPaint,
      wavePaint = wavePaint,
      glintPaint = glintPaint,
    )
  }

  DisposableEffect(resources) {
    onDispose {
      resources.glintBitmap?.let { bmp ->
        if (!bmp.isRecycled) bmp.recycle()
      }
    }
  }

  val t = max(0f, elapsedMs)
  val strokeDuration = max(200f, config.strokeDurationMs.toFloat())
  val growthProgress = smoothstep(0f, strokeDuration, t)

  val fadeStart = strokeDuration + 150f
  val fadeEnd = strokeDuration + 400f
  val fadeFactor = smoothstep(fadeStart, fadeEnd, t)

  val cycleMs = max(500f, config.motionWaveSpeedMs.toFloat())
  val loopModulation = if (t < fadeEnd) {
    0f
  } else {
    sin(((t - fadeEnd) / cycleMs) * 2f * PI.toFloat()) * 0.03f
  }

  val confirmStart = fadeEnd + 200f
  val confirmPeak = confirmStart + 80f
  val confirmDecay = confirmPeak + 80f
  val confirmEnd = confirmDecay + 90f
  val confirmationFactor = when {
    t < confirmStart || t > confirmEnd -> 0f
    t <= confirmPeak -> smoothstep(confirmStart, confirmPeak, t)
    t <= confirmDecay -> 1.0f - 0.5f * smoothstep(confirmPeak, confirmDecay, t)
    else -> 0.5f * (1.0f - smoothstep(confirmDecay, confirmEnd, t))
  }

  val glowMultiplier = max(0.1f, config.glowIntensity)
  val coreAlpha = if (t <= 0f) 0f else min(1f, config.goldBrightness * (1f + 0.12f * confirmationFactor))
  val softGlowAlpha =
    if (fadeFactor <= 0f) 0f
    else min(0.95f, config.softGlowIntensity * (1f + loopModulation) * glowMultiplier * (1f + 0.20f * confirmationFactor) * fadeFactor)
  val haloAlpha =
    if (fadeFactor <= 0f) 0f
    else min(0.60f, 0.20f * (1f + loopModulation) * glowMultiplier * (1f + 0.15f * confirmationFactor) * fadeFactor)
  val glintAlpha =
    if (fadeFactor <= 0f) 0f
    else min(0.22f, 0.12f * glowMultiplier * (1f + 0.25f * confirmationFactor) * fadeFactor)

  Canvas(modifier = modifier) {
    if (growthProgress <= 0f || displayRect.width <= 1f || displayRect.height <= 1f) return@Canvas

    resources.segmentPath.reset()
    resources.pathMeasure.getSegment(0f, resources.pathLength * growthProgress, resources.segmentPath, true)

    drawIntoCanvas { canvas ->
      val nc = canvas.nativeCanvas

      // Halo wide diffuse field (1150ms+)
      if (haloAlpha > 0.005f) {
        resources.haloPaint.alpha = (haloAlpha * 255f).toInt().coerceIn(0, 255)
        nc.drawPath(resources.contourPath, resources.haloPaint)
      }

      // Soft Glow & Warm Gold Accent Glow (1150ms+)
      if (softGlowAlpha > 0.005f) {
        resources.softGlowPaint.alpha = (softGlowAlpha * 255f).toInt().coerceIn(0, 255)
        nc.drawPath(resources.contourPath, resources.softGlowPaint)

        resources.goldGlowPaint.alpha = (softGlowAlpha * 0.55f * 255f).toInt().coerceIn(0, 255)
        nc.drawPath(resources.contourPath, resources.goldGlowPaint)
      }

      // Internal Specular Glint clipped strictly to real fish contour path
      val gBmp = resources.glintBitmap
      if (gBmp != null && !gBmp.isRecycled && glintAlpha > 0.005f) {
        nc.save()
        nc.clipPath(resources.contourPath)
        resources.glintPaint.alpha = (glintAlpha * 255f).toInt().coerceIn(0, 255)
        val dstRect = RectF(displayRect.left, displayRect.top, displayRect.right, displayRect.bottom)
        nc.drawBitmap(gBmp, null, dstRect, resources.glintPaint)
        nc.restore()
      }

      // L0 Deep Teal Contrast Under-Stroke
      resources.underStrokePaint.alpha = (config.underStrokeOpacity * 255f).toInt().coerceIn(0, 255)
      nc.drawPath(resources.segmentPath, resources.underStrokePaint)

      // Anti-Aliasing Soft Edge Cushion
      if (config.antiAliasingSoftEdge) {
        resources.softEdgePaint.alpha = (coreAlpha * 0.40f * 255f).toInt().coerceIn(0, 255)
        nc.drawPath(resources.segmentPath, resources.softEdgePaint)
      }

      // L1 Blue-Gold 2-Cycle Gradient Core Stroke
      resources.corePaint.alpha = (coreAlpha * 255f).toInt().coerceIn(0, 255)
      nc.drawPath(resources.segmentPath, resources.corePaint)

      // L2 Ice Cyan White Fine Highlight
      if (config.highlightOpacity > 0f) {
        resources.highlightPaint.alpha = (config.highlightOpacity * coreAlpha * 255f).toInt().coerceIn(0, 255)
        nc.drawPath(resources.segmentPath, resources.highlightPaint)
      }

      // L3 Traveling Sunlit Gold Glint (1400ms+, 20% arc length cruising entire perimeter)
      if (t >= 1400f && config.motionWaveIntensity > 0f) {
        val elapsed = t - 1400f
        val phase = (elapsed % cycleMs) / cycleMs
        val waveLen = 0.20f * resources.pathLength
        val waveStart = phase * resources.pathLength
        val waveAlpha = min(
          0.95f,
          config.motionWaveIntensity * (1f + 0.35f * confirmationFactor) * fadeFactor,
        )

        resources.wavePath.reset()
        val firstEnd = min(resources.pathLength, waveStart + waveLen)
        resources.pathMeasure.getSegment(waveStart, firstEnd, resources.wavePath, true)
        if (waveStart + waveLen > resources.pathLength) {
          val wrapLen = (waveStart + waveLen) - resources.pathLength
          resources.pathMeasure.getSegment(0f, wrapLen, resources.wavePath, true)
        }
        resources.wavePaint.alpha = (waveAlpha * 255f).toInt().coerceIn(0, 255)
        nc.drawPath(resources.wavePath, resources.wavePaint)
      }
    }
  }
}
