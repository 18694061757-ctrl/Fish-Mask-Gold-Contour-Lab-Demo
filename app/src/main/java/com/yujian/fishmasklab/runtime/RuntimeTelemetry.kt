package com.yujian.fishmasklab.runtime

import android.os.Debug
import android.util.Log

data class RuntimePerformanceSnapshot(
  val taskId: Long,
  val nativeHeapKb: Long,
  val javaHeapKb: Long,
  val detectorDurationMs: Long?,
  val samDurationMs: Long?,
  val contourDurationMs: Long?,
  val shaderFps: Float?,
)

object RuntimeTelemetry {
  private const val TAG = "FishMaskRuntime"

  fun capture(
    taskId: Long,
    detectorDurationMs: Long? = null,
    samDurationMs: Long? = null,
    contourDurationMs: Long? = null,
    shaderFps: Float? = null,
  ): RuntimePerformanceSnapshot {
    val runtime = Runtime.getRuntime()
    val javaUsedKb = (runtime.totalMemory() - runtime.freeMemory()) / 1024L
    val nativeUsedKb = Debug.getNativeHeapAllocatedSize() / 1024L
    return RuntimePerformanceSnapshot(
      taskId = taskId,
      nativeHeapKb = nativeUsedKb,
      javaHeapKb = javaUsedKb,
      detectorDurationMs = detectorDurationMs,
      samDurationMs = samDurationMs,
      contourDurationMs = contourDurationMs,
      shaderFps = shaderFps,
    ).also { snap ->
      Log.d(
        TAG,
        "task=${snap.taskId} nativeHeap=${snap.nativeHeapKb}KB javaHeap=${snap.javaHeapKb}KB " +
          "detMs=${snap.detectorDurationMs} samMs=${snap.samDurationMs} " +
          "contourMs=${snap.contourDurationMs} fps=${snap.shaderFps}",
      )
    }
  }
}
