package com.yujian.fishmasklab.domain

/**
 * Unified recognition task state machine across the app.
 */
enum class PipelineState {
  EMPTY,
  PHOTO_SELECTED,
  MODEL_LOADING,
  DETECTING,
  YOLOX_RESULT_VALIDATION,
  SAM_PREPROCESSING,
  SAM_ENCODER,
  PROMPT_BUILDING,
  SAM_DECODER,
  MASK_POSTPROCESSING,
  MASK_CANDIDATE_RANKING,
  MASK_QUALITY_GATE,
  BUILDING_CONTOUR,
  PLAYING,
  RESULT,
  ERROR,
}

enum class RecognitionErrorReason {
  NO_CLEAR_FISH,
  SEGMENTATION_FAILED,
  MODEL_INIT_FAILED,
  RECOGNITION_INCOMPLETE,
}

enum class ContourColorMode {
  BLUE_CYAN_CORE_GOLD_ACCENT,
  BLUE_GOLD_GRADIENT,
  PURE_GOLD,
}

data class PromptPoint(
  val id: String,
  val x: Int,
  val y: Int,
  val label: Int, // 1 = positive (foreground), 0 = negative (background)
)

data class PixelBox(
  val x1: Int,
  val y1: Int,
  val x2: Int,
  val y2: Int,
) {
  val width: Int get() = (x2 - x1).coerceAtLeast(0)
  val height: Int get() = (y2 - y1).coerceAtLeast(0)
  val area: Int get() = width * height
}

data class FishDetection(
  val box: PixelBox,
  val confidence: Float,
  val areaRatio: Float,
)

data class CandidateMask(
  val index: Int,
  val iouScore: Float,
  val data: ByteArray, // 0 or -1 (0xFF) binary mask per pixel
  val pixelCount: Int,
  val areaRatio: Float,
  val bbox: PixelBox,
  val isTrustworthy: Boolean = false,
  val rankScore: Float = iouScore,
)

data class MaskResult(
  val width: Int,
  val height: Int,
  val selectedCandidateIndex: Int,
  val candidates: List<CandidateMask>,
  val inferenceTimeMs: Long,
  val isMaskTrustworthy: Boolean = false,
  val isBboxTrustworthy: Boolean = false,
)

data class ContourPoint(
  val x: Float,
  val y: Float,
)

data class CubicBezierSegment(
  val cp1x: Float,
  val cp1y: Float,
  val cp2x: Float,
  val cp2y: Float,
  val endX: Float,
  val endY: Float,
)

data class GlintOverlayData(
  val width: Int,
  val height: Int,
  val argbPixels: IntArray,
)

data class ContourData(
  val rawPoints: List<ContourPoint>,
  val simplifiedPoints: List<ContourPoint>,
  val startPoint: ContourPoint,
  val bezierSegments: List<CubicBezierSegment>,
  val totalLength: Float,
  val rawPointCount: Int,
  val simplifiedPointCount: Int,
  val extractionTimeMs: Long,
  val internalGlint: GlintOverlayData? = null,
)

/**
 * Frozen Classic Gold Contour (V1.3) visual configuration matching the reference web project.
 */
data class FrozenContourConfig(
  val colorMode: ContourColorMode = ContourColorMode.BLUE_GOLD_GRADIENT,
  val underStrokeColorArgb: Long = 0xFF063B46L,
  val underStrokeWidthDp: Float = 3.8f,
  val underStrokeOpacity: Float = 0.8f,
  val coreColorArgb: Long = 0xFF7DD3FCL,
  val coreLineWidthDp: Float = 2.2f,
  val highlightColorArgb: Long = 0xFFDDFBFFL,
  val highlightOpacity: Float = 0.6f,
  val goldAccentColorArgb: Long = 0xFFFFD166L,
  val softGlowColorArgb: Long = 0xFF38BDF8L,
  val softGlowRadiusDp: Float = 32f,
  val goldGlowColorArgb: Long = 0xFFFEF08AL,
  val haloColorArgb: Long = 0xFF38BDF8L,
  val haloRadiusDp: Float = 64f,
  val glowIntensity: Float = 1.5f,
  val motionWaveIntensity: Float = 0.8f,
  val motionWaveSpeedMs: Long = 2200L,
  val smoothingPasses: Int = 3,
  val simplificationTolerance: Float = 2.2f,
  val antiAliasingSoftEdge: Boolean = true,
  val goldBrightness: Float = 0.90f,
  val softGlowIntensity: Float = 0.50f,
  val strokeDurationMs: Long = 1000L,
  val totalTimelineMs: Long = 2600L,
)

/**
 * Frozen Effect 02 perimeter water-light caustics configuration extracted from
 * `FROZEN_EFFECT_02_CONFIG` in `src/components/FishCatchWaterOverlay.ts`.
 */
data class FrozenEffect02Config(
  val spreadDurationMs: Long = 1200L,
  val intensity: Float = 1.80f,
  val speed: Float = 1.00f,
  val marginRatio: Float = 0.015f,
  val glowSizePx: Float = 45f,
  val goldColorArgb: Long = 0xFF4D9B91L,
  val cyanColorArgb: Long = 0xFFD89913L,
  val greenColorArgb: Long = 0xFF719FBCL,
  val bottomMaskStrength: Float = 1.00f,
  val blendRatio: Float = 0.80f,
  val causticsDensity: Float = 0.85f,
  val blendMode: String = "overlay",
)
