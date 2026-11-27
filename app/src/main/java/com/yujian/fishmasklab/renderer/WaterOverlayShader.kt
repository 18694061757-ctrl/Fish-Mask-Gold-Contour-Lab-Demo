package com.yujian.fishmasklab.renderer

import android.graphics.Color as AndroidColor
import android.graphics.RuntimeShader
import android.os.Build
import androidx.annotation.RequiresApi
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.ShaderBrush
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.unit.dp
import com.yujian.fishmasklab.domain.FrozenEffect02Config
import com.yujian.fishmasklab.domain.ImageCoordinateMapper
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * AGSL port of `FISH_CATCH_FRAG_SRC` (Effect 02) from `src/components/FishCatchWaterOverlay.ts`.
 * Converts AGSL top-left fragCoord to bottom-left origin via `u_resolution.y - fragCoord.y`
 * so upward spread reveal and bottom-up caustics match the WebGL reference exactly.
 */
private const val EFFECT_02_AGSL_SRC = """
uniform float u_time;
uniform float2 u_resolution;
uniform float u_intensity;
uniform float u_spread_progress;
uniform float u_margin_ratio;
uniform float u_glow_size;
uniform float3 u_gold_color;
uniform float3 u_cyan_color;
uniform float3 u_green_color;
uniform float u_bottom_mask_strength;
uniform float u_blend_ratio;

float2 rotate2d(float2 p, float a) {
  float c = cos(a);
  float s = sin(a);
  return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

float sdRoundedBox(float2 p, float2 b, float4 r) {
  r.xy = (p.x > 0.0) ? r.xy : r.zw;
  r.x = (p.y > 0.0) ? r.x : r.y;
  float2 q = abs(p) - b + r.x;
  return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r.x;
}

float hash21(float2 p) {
  p = fract(p * float2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float causticOctave(float2 uv, float t, float fi) {
  float2 p = rotate2d(uv, 0.45 + fi * 2.39996);
  for (int k = 0; k < 2; k++) {
    float fk = float(k);
    float2 w = float2(
      sin(p.y * (1.18 + fi * 0.17 + fk * 0.12) + p.x * (0.24 + fi * 0.08) + t * (0.42 + fi * 0.06) + fi * 1.57 + fk),
      cos(p.x * (0.92 + fi * 0.15 + fk * 0.10) - p.y * (0.31 + fi * 0.08) - t * (0.38 + fi * 0.05) + fi * 1.18)
    );
    p += w * (0.45 / (1.0 + fi * 0.18 + fk * 0.12));
  }
  float2 d1 = float2(1.02 + fi * 0.19, 0.35 + fi * 0.11);
  float2 d2 = float2(-0.48 - fi * 0.09, 1.12 + fi * 0.07);
  float ridge = sin(dot(p, d1) + t * 0.14) * sin(dot(p, d2) - t * 0.11);
  return 0.17 / (abs(ridge) * 5.4 + 0.22);
}

float causticField(float2 uv, float t) {
  float2 p = uv + float2(
    sin(uv.y * 0.42 + t * 0.24) * 0.38,
    cos(uv.x * 0.36 - t * 0.18) * 0.38
  );
  float a = causticOctave(p, t, 0.0);
  float2 p2 = rotate2d(p * 1.618 + float2(1.72, -1.18), 1.15);
  float b = causticOctave(p2, t * 0.82, 1.0);
  float2 p3 = rotate2d(p * 2.54 + float2(2.31, -0.65), 1.52);
  float c = causticOctave(p3, t * 0.65, 2.0);
  float acc = max(a * 1.15, b * 0.82) + c * 0.38;
  return pow(clamp(acc, 0.0, 2.0), 1.35);
}

float lightSpotsField(float2 uv, float t) {
  float2 p = uv * 3.6;
  float2 ip = floor(p);
  float2 fp = fract(p) - 0.5;
  float minDist = 1.0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      float2 neighbor = float2(float(x), float(y));
      float2 cellId = ip + neighbor;
      float h = hash21(cellId);
      float2 offset = float2(
        sin(t * 1.25 + h * 6.28) * 0.32,
        cos(t * 1.55 + h * 6.28) * 0.32
      );
      float2 diff = neighbor + offset - fp;
      minDist = min(minDist, length(diff));
    }
  }
  float spot = smoothstep(0.65, 0.08, minDist);
  return pow(spot, 1.8);
}

half4 main(float2 fragCoord) {
  float2 res = max(u_resolution, float2(1.0, 1.0));
  float2 frag = float2(fragCoord.x, res.y - fragCoord.y);
  float2 uv = (frag * 2.0 - res) / max(res.y, 1.0);
  float normY = frag.y / res.y;
  float t = u_time * 0.42;

  float revealedMask = (u_spread_progress >= 1.0)
    ? 1.0
    : smoothstep(normY - 0.14, normY + 0.02, u_spread_progress * 1.08 - 0.04);
  revealedMask = clamp(revealedMask, 0.0, 1.0);

  float2 halfRes = res * 0.5;
  float2 centeredPos = frag - halfRes;
  float margin = min(res.x, res.y) * u_margin_ratio;
  float2 boxHalf = halfRes - float2(margin);
  float cornerR = min(res.x, res.y) * 0.12;

  float perimAngle = atan(centeredPos.y, centeredPos.x);
  float perimWave = sin(perimAngle * 5.0 + t * 2.2) * 4.5 + cos(perimAngle * 9.0 - t * 1.5) * 2.6;
  float dBox = abs(sdRoundedBox(centeredPos, boxHalf, float4(cornerR)) + perimWave);

  float effectiveGlow = max(8.0, u_glow_size);
  float wideHalo = exp(-dBox * (0.07 / (effectiveGlow * 0.12)));
  float mediumBloom = exp(-dBox * (0.16 / (effectiveGlow * 0.10)));
  float sharpCore = smoothstep(2.8, 0.3, dBox);
  float perimeterAura = (wideHalo * 0.35 + mediumBloom * 0.65 + sharpCore * 0.95) * revealedMask;

  float filamentAngleT = (perimAngle + 3.14159) / 6.28318;
  float filamentGoldBias = smoothstep(0.18, 0.82, sin(filamentAngleT * 3.14159 + 0.7));
  float3 perimeterColor = mix(u_cyan_color, u_gold_color, filamentGoldBias);
  perimeterColor = mix(perimeterColor, u_green_color, sin(filamentAngleT * 6.28 + t * 0.5) * 0.2 + 0.2);

  float bottomGrad = pow(clamp(1.0 - normY * 0.85, 0.0, 1.0), 1.5);
  float causticsMask = mix(0.04, 1.0, bottomGrad * u_bottom_mask_strength) * revealedMask;

  float cVal = causticField(uv * 1.85, t);
  float veins = smoothstep(0.16, 1.10, cVal);
  float haze = smoothstep(0.03, 0.78, cVal) * 0.35;
  float causticGlow = veins * 0.92 + haze;

  float spots = lightSpotsField(uv * 1.25, t * 0.85);
  float waterLight = causticGlow * 0.88 + spots * 0.95;

  float colorFlow = sin(uv.x * 2.2 + uv.y * 1.5 + t * 0.75) * 0.5 + 0.5;
  float3 waterSpectrum = mix(u_cyan_color, u_green_color, clamp(colorFlow * 0.85 + 0.1, 0.0, 1.0));
  float sunCatch = smoothstep(0.28, 0.88, waterLight * 0.55 + colorFlow * 0.45);
  float3 bottomCausticColor = mix(waterSpectrum, u_gold_color, clamp(sunCatch * u_blend_ratio, 0.0, 1.0));

  float bottomContrib = waterLight * causticsMask * 0.68;
  float perimContrib = perimeterAura * 0.86;
  float totalLight = (bottomContrib + perimContrib) * u_intensity;
  float finalAlpha = clamp(totalLight, 0.0, 0.96);

  float totalWeight = max(0.001, bottomContrib + perimContrib);
  float3 finalRgb = (bottomCausticColor * bottomContrib + perimeterColor * perimContrib) / totalWeight;

  float coreSpark = smoothstep(0.85, 1.75, totalLight);
  finalRgb = mix(finalRgb, float3(0.99, 0.99, 0.96), coreSpark * 0.48);
  finalRgb *= (1.0 + coreSpark * 0.3);

  return half4(half3(finalRgb * finalAlpha), half(finalAlpha));
}
"""

private data class PerimeterPoint(
  val x: Float,
  val y: Float,
)

private data class SparkleNode(
  val baseParam: Float,
  val speed: Float,
  val size: Float,
  val phase: Float,
)

private fun generatePerimeterPoints(
  width: Float,
  height: Float,
  margin: Float,
  cornerR: Float,
  subdivisions: Int = 280,
): List<PerimeterPoint> {
  if (width <= 2f || height <= 2f) return emptyList()
  val left = margin
  val right = width - margin
  val top = margin
  val bottom = height - margin
  val r = min(cornerR, min((right - left) * 0.5f, (bottom - top) * 0.5f)).coerceAtLeast(1f)
  val straightW = (right - left - 2f * r).coerceAtLeast(0f)
  val straightH = (bottom - top - 2f * r).coerceAtLeast(0f)
  val cornerArc = (PI.toFloat() * 0.5f) * r
  val totalPerimeter = 2f * straightW + 2f * straightH + 4f * cornerArc
  if (totalPerimeter <= 0f) return emptyList()

  val len1 = straightW * 0.5f
  val len2 = len1 + cornerArc
  val len3 = len2 + straightH
  val len4 = len3 + cornerArc
  val len5 = len4 + straightW
  val len6 = len5 + cornerArc
  val len7 = len6 + straightH
  val len8 = len7 + cornerArc

  return List(subdivisions) { i ->
    val s = (i.toFloat() / subdivisions) * totalPerimeter
    val x: Float
    val y: Float
    when {
      s < len1 -> {
        x = left + r + straightW * 0.5f + s
        y = bottom
      }
      s < len2 -> {
        val angle = ((s - len1) / cornerArc) * (PI.toFloat() * 0.5f)
        x = (right - r) + sin(angle) * r
        y = (bottom - r) + cos(angle) * r
      }
      s < len3 -> {
        x = right
        y = bottom - r - (s - len2)
      }
      s < len4 -> {
        val angle = ((s - len3) / cornerArc) * (PI.toFloat() * 0.5f)
        x = (right - r) + cos(angle) * r
        y = (top + r) - sin(angle) * r
      }
      s < len5 -> {
        x = right - r - (s - len4)
        y = top
      }
      s < len6 -> {
        val angle = ((s - len5) / cornerArc) * (PI.toFloat() * 0.5f)
        x = (left + r) - sin(angle) * r
        y = (top + r) - cos(angle) * r
      }
      s < len7 -> {
        x = left
        y = top + r + (s - len6)
      }
      s < len8 -> {
        val angle = ((s - len7) / cornerArc) * (PI.toFloat() * 0.5f)
        x = (left + r) - cos(angle) * r
        y = (bottom - r) + sin(angle) * r
      }
      else -> {
        x = left + r + (s - len8)
        y = bottom
      }
    }
    PerimeterPoint(x, y)
  }
}

/**
 * Renders Effect 02 perimeter water caustics + 16 dew sparkle nodes + bottom flourishing motif
 * directly over the fitted photo rectangle without recreating GPU resources per frame or replay.
 */
@Composable
fun WaterOverlayEffect02Canvas(
  elapsedMs: Float,
  displayRect: ImageCoordinateMapper.DisplayPhotoRect,
  config: FrozenEffect02Config = FrozenEffect02Config(),
  modifier: Modifier = Modifier,
) {
  val sparkles = remember {
    List(16) { idx ->
      SparkleNode(
        baseParam = idx / 16f,
        speed = 0.012f + (idx % 3) * 0.007f,
        size = 2.2f + (idx % 4) * 0.7f,
        phase = ((idx * 1.37f) % (2f * PI.toFloat())),
      )
    }
  }

  val goldColor = remember(config.goldColorArgb) { Color(config.goldColorArgb) }
  val cyanColor = remember(config.cyanColorArgb) { Color(config.cyanColorArgb) }
  val greenColor = remember(config.greenColorArgb) { Color(config.greenColorArgb) }

  val cachedPerimeter = remember(displayRect.width, displayRect.height, config.marginRatio) {
    val margin = min(displayRect.width, displayRect.height) * config.marginRatio
    val cornerR = min(displayRect.width, displayRect.height) * 0.12f
    generatePerimeterPoints(displayRect.width, displayRect.height, margin, cornerR, 280)
  }

  val wingPath = remember { Path() }

  val runtimeShader = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
    remember { RuntimeShader(EFFECT_02_AGSL_SRC) }
  } else {
    null
  }
  val shaderBrush = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU && runtimeShader != null) {
    remember(runtimeShader) { ShaderBrush(runtimeShader) }
  } else {
    null
  }

  val spreadProgress = (elapsedMs / config.spreadDurationMs.toFloat().coerceAtLeast(1f)).coerceIn(0f, 1f)
  val timeSec = elapsedMs * 0.001f * config.speed

  Canvas(modifier = modifier) {
    if (displayRect.width <= 1f || displayRect.height <= 1f) return@Canvas

    clipRect(
      left = displayRect.left,
      top = displayRect.top,
      right = displayRect.right,
      bottom = displayRect.bottom,
    ) {
      // 1. AGSL Shader Layer (API 33+, including Android 16 API 36)
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU && runtimeShader != null && shaderBrush != null) {
        updateAgslUniforms(
          shader = runtimeShader,
          timeSec = timeSec,
          width = displayRect.width,
          height = displayRect.height,
          spreadProgress = spreadProgress,
          config = config,
        )
        drawRect(
          brush = shaderBrush,
          topLeft = Offset(displayRect.left, displayRect.top),
          size = androidx.compose.ui.geometry.Size(displayRect.width, displayRect.height),
          blendMode = BlendMode.Overlay,
        )
      } else {
        // Graceful Canvas fallback on pre-API 33 devices
        val fallbackMargin = min(displayRect.width, displayRect.height) * config.marginRatio
        drawRoundRect(
          color = goldColor.copy(alpha = 0.35f * spreadProgress),
          topLeft = Offset(displayRect.left + fallbackMargin, displayRect.top + fallbackMargin),
          size = androidx.compose.ui.geometry.Size(
            (displayRect.width - fallbackMargin * 2f).coerceAtLeast(1f),
            (displayRect.height - fallbackMargin * 2f).coerceAtLeast(1f),
          ),
          cornerRadius = androidx.compose.ui.geometry.CornerRadius(
            min(displayRect.width, displayRect.height) * 0.12f,
          ),
          style = Stroke(width = 3.dp.toPx()),
        )
      }

      // 2. Perimeter Water Sparkle Dew Nodes & Bottom Center Motif
      if (spreadProgress > 0.05f && cachedPerimeter.isNotEmpty()) {
        val rightBranchEnd = 0.5f * spreadProgress
        val leftBranchStart = 1.0f - 0.5f * spreadProgress

        for (sp in sparkles) {
          val pPos = (sp.baseParam + timeSec * sp.speed) % 1.0f
          if (pPos > rightBranchEnd && pPos < leftBranchStart) continue
          val idx = (pPos * cachedPerimeter.size).roundToInt().coerceIn(0, cachedPerimeter.lastIndex)
          val pt = cachedPerimeter[idx]
          val cx = displayRect.left + pt.x
          val cy = displayRect.top + pt.y
          val pulse = 0.65f + 0.35f * sin(timeSec * 3.4f + sp.phase)
          val isGold = pPos > 0.45f
          val pCore = if (isGold) Color(0xFFFEF08A) else Color(0xFFE0F2FE)
          val pGlow = if (isGold) goldColor else cyanColor

          drawCircle(
            color = pGlow.copy(alpha = (0.42f * pulse * config.intensity).coerceIn(0f, 1f)),
            radius = sp.size * 3.2f * pulse * density,
            center = Offset(cx, cy),
            blendMode = BlendMode.Screen,
          )
          drawCircle(
            color = pCore.copy(alpha = (0.92f * pulse * config.intensity).coerceIn(0f, 1f)),
            radius = sp.size * 1.15f * pulse * density,
            center = Offset(cx, cy),
            blendMode = BlendMode.Screen,
          )
        }

        if (spreadProgress > 0.15f) {
          val bottomMid = cachedPerimeter[0]
          val bx = displayRect.left + bottomMid.x
          val by = displayRect.top + bottomMid.y
          val bPulse = 0.75f + 0.25f * sin(timeSec * 2.4f)

          drawCircle(
            color = goldColor.copy(alpha = 0.45f),
            radius = 7f * bPulse * density,
            center = Offset(bx, by),
            blendMode = BlendMode.Screen,
          )
          drawCircle(
            color = Color.White.copy(alpha = 0.95f),
            radius = 3.0f * bPulse * density,
            center = Offset(bx, by),
            blendMode = BlendMode.Screen,
          )

          wingPath.reset()
          wingPath.moveTo(bx - 38f * density, by)
          wingPath.quadraticTo(bx - 19f * density, by - 2.2f * density, bx - 6f * density, by)
          wingPath.moveTo(bx + 6f * density, by)
          wingPath.quadraticTo(bx + 19f * density, by + 2.2f * density, bx + 38f * density, by)
          drawPath(
            path = wingPath,
            color = goldColor.copy(alpha = (0.65f * config.intensity).coerceIn(0f, 1f)),
            style = Stroke(width = 1.3f * density, cap = StrokeCap.Round),
            blendMode = BlendMode.Screen,
          )
          drawCircle(
            color = Color.White,
            radius = 1.4f * density,
            center = Offset(bx - 38f * density, by),
            blendMode = BlendMode.Screen,
          )
          drawCircle(
            color = Color.White,
            radius = 1.4f * density,
            center = Offset(bx + 38f * density, by),
            blendMode = BlendMode.Screen,
          )
        }
      }
    }
  }
}

@RequiresApi(Build.VERSION_CODES.TIRAMISU)
private fun updateAgslUniforms(
  shader: RuntimeShader,
  timeSec: Float,
  width: Float,
  height: Float,
  spreadProgress: Float,
  config: FrozenEffect02Config,
) {
  shader.setFloatUniform("u_time", timeSec)
  shader.setFloatUniform("u_resolution", width, height)
  shader.setFloatUniform("u_intensity", config.intensity)
  shader.setFloatUniform("u_spread_progress", spreadProgress)
  shader.setFloatUniform("u_margin_ratio", config.marginRatio)
  shader.setFloatUniform("u_glow_size", config.glowSizePx)

  val goldInt = config.goldColorArgb.toInt()
  val cyanInt = config.cyanColorArgb.toInt()
  val greenInt = config.greenColorArgb.toInt()

  shader.setFloatUniform(
    "u_gold_color",
    AndroidColor.red(goldInt) / 255f,
    AndroidColor.green(goldInt) / 255f,
    AndroidColor.blue(goldInt) / 255f,
  )
  shader.setFloatUniform(
    "u_cyan_color",
    AndroidColor.red(cyanInt) / 255f,
    AndroidColor.green(cyanInt) / 255f,
    AndroidColor.blue(cyanInt) / 255f,
  )
  shader.setFloatUniform(
    "u_green_color",
    AndroidColor.red(greenInt) / 255f,
    AndroidColor.green(greenInt) / 255f,
    AndroidColor.blue(greenInt) / 255f,
  )
  shader.setFloatUniform("u_bottom_mask_strength", config.bottomMaskStrength)
  shader.setFloatUniform("u_blend_ratio", config.blendRatio)
}
