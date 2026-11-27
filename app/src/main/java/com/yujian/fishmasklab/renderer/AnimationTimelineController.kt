package com.yujian.fishmasklab.renderer

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.Stable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos

@Stable
class AnimationTimelineState(
  val totalWindowMs: Long = 2600L,
) {
  var elapsedMs by mutableFloatStateOf(0f)
    internal set

  var measuredFps by mutableFloatStateOf(0f)
    internal set

  var hasCompletedInitialWindow by mutableStateOf(false)
    internal set

  fun resetForReplay() {
    elapsedMs = 0f
    hasCompletedInitialWindow = false
  }
}

/**
 * Drives the native 0–2600ms+ animation timeline via Choreographer `withFrameNanos`.
 * Replay increments [replayToken] and resets time to 0 without recreating GPU shaders or geometry.
 */
@Composable
fun rememberAnimationTimelineState(
  isActive: Boolean,
  replayToken: Int,
  totalWindowMs: Long = 2600L,
  onInitialWindowCompleted: () -> Unit = {},
  onFpsSampled: (Float) -> Unit = {},
): AnimationTimelineState {
  val state = remember(totalWindowMs) { AnimationTimelineState(totalWindowMs) }

  LaunchedEffect(isActive, replayToken) {
    if (!isActive) return@LaunchedEffect
    state.resetForReplay()
    var startNanos = -1L
    var lastFrameNanos = -1L
    var fpsAccumFrames = 0
    var fpsAccumNanos = 0L

    while (true) {
      withFrameNanos { frameNanos ->
        if (startNanos < 0L) {
          startNanos = frameNanos
          lastFrameNanos = frameNanos
        }
        val dtNanos = (frameNanos - lastFrameNanos).coerceAtLeast(0L)
        lastFrameNanos = frameNanos

        if (dtNanos > 0L) {
          fpsAccumFrames++
          fpsAccumNanos += dtNanos
          if (fpsAccumNanos >= 500_000_000L) {
            val fps = (fpsAccumFrames * 1_000_000_000f) / fpsAccumNanos.toFloat()
            state.measuredFps = fps
            onFpsSampled(fps)
            fpsAccumFrames = 0
            fpsAccumNanos = 0L
          }
        }

        val currentMs = (frameNanos - startNanos) / 1_000_000f
        state.elapsedMs = currentMs

        if (!state.hasCompletedInitialWindow && currentMs >= totalWindowMs.toFloat()) {
          state.hasCompletedInitialWindow = true
          onInitialWindowCompleted()
        }
      }
    }
  }

  return state
}
