package com.yujian.fishmasklab.runtime

import android.app.ActivityManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.Debug
import android.os.SystemClock
import com.example.BuildConfig
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.roundToLong
import org.json.JSONArray
import org.json.JSONObject

enum class StageOutcome {
  SUCCESS,
  FAILED,
  CANCELLED,
}

data class DeviceEnvironmentInfo(
  val appVersionName: String,
  val appVersionCode: Long,
  val buildSha: String,
  val apiLevel: Int,
  val deviceManufacturer: String,
  val deviceModel: String,
  val cpuAbis: List<String>,
  val cpuCores: Int,
  val javaHeapUsedKb: Long,
  val javaHeapMaxKb: Long,
  val nativeHeapAllocatedKb: Long,
  val pssTotalKb: Long,
  val availableMemoryKb: Long,
  val totalMemoryKb: Long,
  val gcCollectionCount: Long,
  val gcCollectionTimeMs: Long,
  val onnxRuntimeVersion: String,
  val executionProvider: String,
) {
  fun toJson(): JSONObject = JSONObject().apply {
    put("appVersionName", appVersionName)
    put("appVersionCode", appVersionCode)
    put("buildSha", buildSha)
    put("apiLevel", apiLevel)
    put("deviceManufacturer", deviceManufacturer)
    put("deviceModel", deviceModel)
    put("cpuAbis", JSONArray(cpuAbis))
    put("cpuCores", cpuCores)
    put("javaHeapUsedKb", javaHeapUsedKb)
    put("javaHeapMaxKb", javaHeapMaxKb)
    put("nativeHeapAllocatedKb", nativeHeapAllocatedKb)
    put("pssTotalKb", pssTotalKb)
    put("availableMemoryKb", availableMemoryKb)
    put("totalMemoryKb", totalMemoryKb)
    put("gcCollectionCount", gcCollectionCount)
    put("gcCollectionTimeMs", gcCollectionTimeMs)
    put("onnxRuntimeVersion", onnxRuntimeVersion)
    put("executionProvider", executionProvider)
  }

  companion object {
    fun capture(
      context: Context? = null,
      executionProvider: String = "CPU",
    ): DeviceEnvironmentInfo {
      val runtime = Runtime.getRuntime()
      val javaUsedKb = ((runtime.totalMemory() - runtime.freeMemory()) / 1024L).coerceAtLeast(0L)
      val javaMaxKb = (runtime.maxMemory() / 1024L).coerceAtLeast(0L)
      val nativeUsedKb = runCatching { Debug.getNativeHeapAllocatedSize() / 1024L }.getOrDefault(0L).coerceAtLeast(0L)
      val pssKb = runCatching {
        val mi = Debug.MemoryInfo()
        Debug.getMemoryInfo(mi)
        mi.totalPss.toLong().coerceAtLeast(0L)
      }.getOrDefault(0L)

      var availMemKb = 0L
      var totalMemKb = 0L
      if (context != null) {
        runCatching {
          val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
          if (am != null) {
            val memInfo = ActivityManager.MemoryInfo()
            am.getMemoryInfo(memInfo)
            availMemKb = (memInfo.availMem / 1024L).coerceAtLeast(0L)
            totalMemKb = (memInfo.totalMem / 1024L).coerceAtLeast(0L)
          }
        }
      }

      var versionName = BuildConfig.VERSION_NAME
      var versionCode = BuildConfig.VERSION_CODE.toLong()
      if (context != null) {
        runCatching {
          val pm = context.packageManager
          val pkgInfo = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            pm.getPackageInfo(context.packageName, PackageManager.PackageInfoFlags.of(0))
          } else {
            @Suppress("DEPRECATION")
            pm.getPackageInfo(context.packageName, 0)
          }
          versionName = pkgInfo.versionName ?: versionName
          @Suppress("DEPRECATION")
          versionCode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            pkgInfo.longVersionCode
          } else {
            pkgInfo.versionCode.toLong()
          }
        }
      }

      val ortVersion = runCatching {
        val pkg = ai.onnxruntime.OrtEnvironment::class.java.`package`
        pkg?.implementationVersion?.takeIf { it.isNotBlank() } ?: BuildConfig.ORT_VERSION
      }.getOrDefault(BuildConfig.ORT_VERSION)

      val manufacturer = runCatching { Build.MANUFACTURER.orEmpty().ifBlank { "unknown" } }.getOrDefault("unknown")
      val model = runCatching { Build.MODEL.orEmpty().ifBlank { "unknown" } }.getOrDefault("unknown")
      val abis = runCatching { Build.SUPPORTED_ABIS?.toList().orEmpty() }.getOrDefault(emptyList())
      val gcCount = runCatching { Debug.getRuntimeStat("art.gc.gc-count")?.toLongOrNull() ?: -1L }.getOrDefault(-1L)
      val gcTimeMs = runCatching { Debug.getRuntimeStat("art.gc.gc-time")?.toLongOrNull() ?: -1L }.getOrDefault(-1L)

      return DeviceEnvironmentInfo(
        appVersionName = versionName,
        appVersionCode = versionCode,
        buildSha = BuildConfig.GIT_BUILD_SHA,
        apiLevel = Build.VERSION.SDK_INT,
        deviceManufacturer = manufacturer,
        deviceModel = model,
        cpuAbis = abis,
        cpuCores = runtime.availableProcessors().coerceAtLeast(1),
        javaHeapUsedKb = javaUsedKb,
        javaHeapMaxKb = javaMaxKb,
        nativeHeapAllocatedKb = nativeUsedKb,
        pssTotalKb = pssKb,
        availableMemoryKb = availMemKb,
        totalMemoryKb = totalMemKb,
        gcCollectionCount = gcCount,
        gcCollectionTimeMs = gcTimeMs,
        onnxRuntimeVersion = ortVersion,
        executionProvider = executionProvider,
      )
    }

    fun fromJson(json: JSONObject): DeviceEnvironmentInfo {
      val abisArr = json.optJSONArray("cpuAbis")
      val abis = buildList {
        if (abisArr != null) {
          for (i in 0 until abisArr.length()) {
            add(abisArr.optString(i))
          }
        }
      }
      return DeviceEnvironmentInfo(
        appVersionName = json.optString("appVersionName", "1.0"),
        appVersionCode = json.optLong("appVersionCode", 1L),
        buildSha = json.optString("buildSha", "unknown"),
        apiLevel = json.optInt("apiLevel", 0),
        deviceManufacturer = json.optString("deviceManufacturer", "unknown"),
        deviceModel = json.optString("deviceModel", "unknown"),
        cpuAbis = abis,
        cpuCores = json.optInt("cpuCores", 1),
        javaHeapUsedKb = json.optLong("javaHeapUsedKb", 0L),
        javaHeapMaxKb = json.optLong("javaHeapMaxKb", 0L),
        nativeHeapAllocatedKb = json.optLong("nativeHeapAllocatedKb", 0L),
        pssTotalKb = json.optLong("pssTotalKb", 0L),
        availableMemoryKb = json.optLong("availableMemoryKb", 0L),
        totalMemoryKb = json.optLong("totalMemoryKb", 0L),
        gcCollectionCount = json.optLong("gcCollectionCount", -1L),
        gcCollectionTimeMs = json.optLong("gcCollectionTimeMs", -1L),
        onnxRuntimeVersion = json.optString("onnxRuntimeVersion", "unknown"),
        executionProvider = json.optString("executionProvider", "CPU"),
      )
    }
  }
}

data class StageTimingRecord(
  val sessionId: String,
  val taskId: Long,
  val photoGenerationId: Long,
  val runIndex: Int,
  val stageName: String,
  val startTimestampNs: Long,
  val endTimestampNs: Long,
  val durationMs: Double,
  val durationRoundedMs: Long,
  val status: StageOutcome,
  val errorCode: String? = null,
  val metrics: Map<String, Any> = emptyMap(),
) {
  fun toJson(): JSONObject = JSONObject().apply {
    put("sessionId", sessionId)
    put("taskId", taskId)
    put("photoGenerationId", photoGenerationId)
    put("runIndex", runIndex)
    put("stageName", stageName)
    put("startTimestampNs", startTimestampNs)
    put("endTimestampNs", endTimestampNs)
    put("durationMs", durationMs)
    put("durationRoundedMs", durationRoundedMs)
    put("status", status.name)
    if (errorCode != null) put("errorCode", errorCode)
    if (metrics.isNotEmpty()) {
      put("metrics", DiagnosticPrivacySanitizer.sanitizeMapToJson(metrics))
    }
  }

  companion object {
    fun fromJson(json: JSONObject): StageTimingRecord {
      val metricsObj = json.optJSONObject("metrics")
      val metricsMap = mutableMapOf<String, Any>()
      if (metricsObj != null) {
        val keys = metricsObj.keys()
        while (keys.hasNext()) {
          val k = keys.next()
          metricsObj.opt(k)?.let { v -> metricsMap[k] = v }
        }
      }
      return StageTimingRecord(
        sessionId = json.optString("sessionId"),
        taskId = json.optLong("taskId", 0L),
        photoGenerationId = json.optLong("photoGenerationId", 0L),
        runIndex = json.optInt("runIndex", 0),
        stageName = json.optString("stageName"),
        startTimestampNs = json.optLong("startTimestampNs", 0L),
        endTimestampNs = json.optLong("endTimestampNs", 0L),
        durationMs = json.optDouble("durationMs", 0.0).coerceAtLeast(0.0),
        durationRoundedMs = json.optLong("durationRoundedMs", 0L).coerceAtLeast(0L),
        status = runCatching { StageOutcome.valueOf(json.optString("status", "SUCCESS")) }
          .getOrDefault(StageOutcome.SUCCESS),
        errorCode = json.optString("errorCode").takeIf { it.isNotEmpty() },
        metrics = metricsMap,
      )
    }
  }
}

data class WarmupTimingSnapshot(
  val yoloxStateAtPick: ModelLoadLifecycleState,
  val slimSamStateAtPick: ModelLoadLifecycleState,
  val allModelsReadyAtPick: Boolean,
  val notReadyModelsAtPick: List<String>,
  val modelWaitTimeMs: Double,
  val modelWaitParallelWithPhotoLoad: Boolean,
  val warmupOverlapWithPhotoLoadMs: Double,
  val warmupTotalMs: Double?,
  val slimSamTotalLoadMs: Double?,
  val yoloxTotalLoadMs: Double?,
  val moduleStatesAtPick: Map<String, RuntimeModuleState> = emptyMap(),
) {
  fun toJson(): JSONObject = JSONObject().apply {
    put("yoloxStateAtPick", yoloxStateAtPick.name)
    put("slimSamStateAtPick", slimSamStateAtPick.name)
    put("allModelsReadyAtPick", allModelsReadyAtPick)
    put("notReadyModelsAtPick", JSONArray(notReadyModelsAtPick))
    put("modelWaitTimeMs", modelWaitTimeMs)
    put("modelWaitParallelWithPhotoLoad", modelWaitParallelWithPhotoLoad)
    put("warmupOverlapWithPhotoLoadMs", warmupOverlapWithPhotoLoadMs)
    put("moduleStatesAtPick", JSONObject().apply {
      moduleStatesAtPick.forEach { (module, state) -> put(module, state.name) }
    })
    if (warmupTotalMs != null) put("warmupTotalMs", warmupTotalMs)
    if (slimSamTotalLoadMs != null) put("slimSamTotalLoadMs", slimSamTotalLoadMs)
    if (yoloxTotalLoadMs != null) put("yoloxTotalLoadMs", yoloxTotalLoadMs)
  }

  companion object {
    fun fromJson(json: JSONObject): WarmupTimingSnapshot {
      val notReadyArr = json.optJSONArray("notReadyModelsAtPick")
      val notReadyList = buildList {
        if (notReadyArr != null) {
          for (i in 0 until notReadyArr.length()) {
            add(notReadyArr.optString(i))
          }
        }
      }
      return WarmupTimingSnapshot(
        yoloxStateAtPick = runCatching {
          ModelLoadLifecycleState.valueOf(json.optString("yoloxStateAtPick", "NOT_STARTED"))
        }.getOrDefault(ModelLoadLifecycleState.NOT_STARTED),
        slimSamStateAtPick = runCatching {
          ModelLoadLifecycleState.valueOf(json.optString("slimSamStateAtPick", "NOT_STARTED"))
        }.getOrDefault(ModelLoadLifecycleState.NOT_STARTED),
        allModelsReadyAtPick = json.optBoolean("allModelsReadyAtPick", false),
        notReadyModelsAtPick = notReadyList,
        modelWaitTimeMs = json.optDouble("modelWaitTimeMs", 0.0).coerceAtLeast(0.0),
        modelWaitParallelWithPhotoLoad = json.optBoolean("modelWaitParallelWithPhotoLoad", false),
        warmupOverlapWithPhotoLoadMs = json.optDouble("warmupOverlapWithPhotoLoadMs", 0.0).coerceAtLeast(0.0),
        warmupTotalMs = if (json.has("warmupTotalMs")) json.optDouble("warmupTotalMs") else null,
        slimSamTotalLoadMs = if (json.has("slimSamTotalLoadMs")) json.optDouble("slimSamTotalLoadMs") else null,
        yoloxTotalLoadMs = if (json.has("yoloxTotalLoadMs")) json.optDouble("yoloxTotalLoadMs") else null,
        moduleStatesAtPick = json.optJSONObject("moduleStatesAtPick")?.let(::parseRuntimeModuleStates).orEmpty(),
      )
    }
  }
}

data class RecognitionSummaryMetrics(
  val modelWarmupTimeMs: Double?,
  val modelWaitTimeMs: Double,
  val photoDecodeTimeMs: Double,
  val yoloxTimeMs: Double,
  val samProcessorTimeMs: Double,
  val samEncoderTimeMs: Double,
  val samDecoderTimeMs: Double,
  val maskPostprocessTimeMs: Double,
  val inferenceTimeMs: Double,
  val contourTimeMs: Double,
  val recognitionPipelineTimeMs: Double,
  val firstResultTimeMs: Double,
  val animationTimeMs: Double?,
  val fullExperienceTimeMs: Double?,
  val reusedEncoderEmbedding: Boolean,
  val shaderFps: Float?,
) {
  fun toJson(): JSONObject = JSONObject().apply {
    if (modelWarmupTimeMs != null) put("modelWarmupTimeMs", modelWarmupTimeMs)
    put("modelWaitTimeMs", modelWaitTimeMs)
    put("photoDecodeTimeMs", photoDecodeTimeMs)
    put("yoloxTimeMs", yoloxTimeMs)
    put("samProcessorTimeMs", samProcessorTimeMs)
    put("samEncoderTimeMs", samEncoderTimeMs)
    put("samDecoderTimeMs", samDecoderTimeMs)
    put("maskPostprocessTimeMs", maskPostprocessTimeMs)
    put("inferenceTimeMs", inferenceTimeMs)
    put("contourTimeMs", contourTimeMs)
    put("recognitionPipelineTimeMs", recognitionPipelineTimeMs)
    put("firstResultTimeMs", firstResultTimeMs)
    if (animationTimeMs != null) put("animationTimeMs", animationTimeMs)
    if (fullExperienceTimeMs != null) put("fullExperienceTimeMs", fullExperienceTimeMs)
    put("reusedEncoderEmbedding", reusedEncoderEmbedding)
    if (shaderFps != null) put("shaderFps", shaderFps.toDouble())
  }

  companion object {
    fun fromJson(json: JSONObject): RecognitionSummaryMetrics = RecognitionSummaryMetrics(
      modelWarmupTimeMs = if (json.has("modelWarmupTimeMs")) json.optDouble("modelWarmupTimeMs") else null,
      modelWaitTimeMs = json.optDouble("modelWaitTimeMs", 0.0).coerceAtLeast(0.0),
      photoDecodeTimeMs = json.optDouble("photoDecodeTimeMs", 0.0).coerceAtLeast(0.0),
      yoloxTimeMs = json.optDouble("yoloxTimeMs", 0.0).coerceAtLeast(0.0),
      samProcessorTimeMs = json.optDouble("samProcessorTimeMs", 0.0).coerceAtLeast(0.0),
      samEncoderTimeMs = json.optDouble("samEncoderTimeMs", 0.0).coerceAtLeast(0.0),
      samDecoderTimeMs = json.optDouble("samDecoderTimeMs", 0.0).coerceAtLeast(0.0),
      maskPostprocessTimeMs = json.optDouble("maskPostprocessTimeMs", 0.0).coerceAtLeast(0.0),
      inferenceTimeMs = json.optDouble("inferenceTimeMs", 0.0).coerceAtLeast(0.0),
      contourTimeMs = json.optDouble("contourTimeMs", 0.0).coerceAtLeast(0.0),
      recognitionPipelineTimeMs = json.optDouble("recognitionPipelineTimeMs", 0.0).coerceAtLeast(0.0),
      firstResultTimeMs = json.optDouble("firstResultTimeMs", 0.0).coerceAtLeast(0.0),
      animationTimeMs = if (json.has("animationTimeMs")) json.optDouble("animationTimeMs") else null,
      fullExperienceTimeMs = if (json.has("fullExperienceTimeMs")) json.optDouble("fullExperienceTimeMs") else null,
      reusedEncoderEmbedding = json.optBoolean("reusedEncoderEmbedding", false),
      shaderFps = if (json.has("shaderFps")) json.optDouble("shaderFps").toFloat() else null,
    )
  }
}

data class PerformanceSessionRecord(
  val sessionId: String,
  val taskId: Long,
  val photoGenerationId: Long,
  val runIndex: Int,
  val mode: String, // INITIAL_RECOGNITION | TAP_REFOCUS
  val startProfile: String, // COLD_START | WARM_AFTER_READY | SUBSEQUENT_HOT | TAP_REFOCUS
  val createdAtUtc: String,
  val startTimestampNs: Long,
  val endTimestampNs: Long,
  val durationMs: Double,
  val status: StageOutcome,
  val errorCode: String? = null,
  val deviceInfo: DeviceEnvironmentInfo,
  val modelVersions: Map<String, String> = emptyMap(),
  val warmupSnapshot: WarmupTimingSnapshot? = null,
  val summaryMetrics: RecognitionSummaryMetrics? = null,
  val stages: List<StageTimingRecord> = emptyList(),
) {
  fun toJson(): JSONObject = JSONObject().apply {
    put("sessionId", sessionId)
    put("taskId", taskId)
    put("photoGenerationId", photoGenerationId)
    put("runIndex", runIndex)
    put("mode", mode)
    put("startProfile", startProfile)
    put("createdAtUtc", createdAtUtc)
    put("startTimestampNs", startTimestampNs)
    put("endTimestampNs", endTimestampNs)
    put("durationMs", durationMs)
    put("status", status.name)
    if (errorCode != null) put("errorCode", errorCode)
    put("deviceInfo", deviceInfo.toJson())
    put("modelVersions", JSONObject(modelVersions))
    if (warmupSnapshot != null) put("warmupSnapshot", warmupSnapshot.toJson())
    if (summaryMetrics != null) put("summaryMetrics", summaryMetrics.toJson())
    put("stages", JSONArray().apply { stages.forEach { put(it.toJson()) } })
  }

  companion object {
    fun fromJson(json: JSONObject): PerformanceSessionRecord {
      val stagesArr = json.optJSONArray("stages")
      val stageList = buildList {
        if (stagesArr != null) {
          for (i in 0 until stagesArr.length()) {
            stagesArr.optJSONObject(i)?.let { add(StageTimingRecord.fromJson(it)) }
          }
        }
      }
      return PerformanceSessionRecord(
        sessionId = json.optString("sessionId"),
        taskId = json.optLong("taskId", 0L),
        photoGenerationId = json.optLong("photoGenerationId", 0L),
        runIndex = json.optInt("runIndex", 0),
        mode = json.optString("mode", "INITIAL_RECOGNITION"),
        startProfile = json.optString("startProfile", "COLD_START"),
        createdAtUtc = json.optString("createdAtUtc", ""),
        startTimestampNs = json.optLong("startTimestampNs", 0L),
        endTimestampNs = json.optLong("endTimestampNs", 0L),
        durationMs = json.optDouble("durationMs", 0.0).coerceAtLeast(0.0),
        status = runCatching { StageOutcome.valueOf(json.optString("status", "SUCCESS")) }
          .getOrDefault(StageOutcome.SUCCESS),
        errorCode = json.optString("errorCode").takeIf { it.isNotEmpty() },
        deviceInfo = json.optJSONObject("deviceInfo")?.let { DeviceEnvironmentInfo.fromJson(it) }
          ?: DeviceEnvironmentInfo.capture(null),
        modelVersions = json.optJSONObject("modelVersions")?.let(::parseStringMap).orEmpty(),
        warmupSnapshot = json.optJSONObject("warmupSnapshot")?.let { WarmupTimingSnapshot.fromJson(it) },
        summaryMetrics = json.optJSONObject("summaryMetrics")?.let { RecognitionSummaryMetrics.fromJson(it) },
        stages = stageList,
      )
    }
  }
}

private fun parseStringMap(json: JSONObject): Map<String, String> = buildMap {
  val keys = json.keys()
  while (keys.hasNext()) {
    val key = keys.next()
    val value = json.optString(key)
    if (value.isNotBlank()) put(key, value)
  }
}

data class RuntimeResourceSample(
  val phase: String,
  val timestampNs: Long,
  val taskId: Long,
  val photoGenerationId: Long,
  val javaHeapUsedKb: Long,
  val javaHeapMaxKb: Long,
  val nativeHeapAllocatedKb: Long,
  val pssTotalKb: Long,
  val availableMemoryKb: Long,
  val gcCollectionCount: Long,
  val gcCollectionTimeMs: Long,
) {
  fun toJson(): JSONObject = JSONObject().apply {
    put("phase", phase)
    put("timestampNs", timestampNs)
    put("taskId", taskId)
    put("photoGenerationId", photoGenerationId)
    put("javaHeapUsedKb", javaHeapUsedKb)
    put("javaHeapMaxKb", javaHeapMaxKb)
    put("nativeHeapAllocatedKb", nativeHeapAllocatedKb)
    put("pssTotalKb", pssTotalKb)
    put("availableMemoryKb", availableMemoryKb)
    put("gcCollectionCount", gcCollectionCount)
    put("gcCollectionTimeMs", gcCollectionTimeMs)
  }
}

data class GlobalWarmupRecord(
  val status: ModelLoadLifecycleState,
  val slimSamState: ModelLoadLifecycleState,
  val yoloxState: ModelLoadLifecycleState,
  val warmupStartNs: Long,
  val slimSamReadyNs: Long?,
  val yoloxReadyNs: Long?,
  val allModelsReadyNs: Long?,
  val warmupTotalMs: Double?,
  val slimSamTotalLoadMs: Double?,
  val yoloxTotalLoadMs: Double?,
  val reusedAuditVerificationForYolox: Boolean,
  val duplicateEnsureLoadedBlockedCount: Int,
  val errorCode: String? = null,
  val deviceInfo: DeviceEnvironmentInfo? = null,
  val stages: List<StageTimingRecord> = emptyList(),
  val moduleStates: Map<String, RuntimeModuleState> = emptyMap(),
) {
  val allModulesInferenceReady: Boolean
    get() = moduleStates["YOLOX"] == RuntimeModuleState.INFERENCE_READY &&
      moduleStates["PROCESSOR"] == RuntimeModuleState.PROCESSOR_READY &&
      moduleStates["ENCODER"] == RuntimeModuleState.INFERENCE_READY &&
      moduleStates["DECODER"] == RuntimeModuleState.INFERENCE_READY

  fun toJson(): JSONObject = JSONObject().apply {
    put("status", status.name)
    put("slimSamState", slimSamState.name)
    put("yoloxState", yoloxState.name)
    put("warmupStartNs", warmupStartNs)
    if (slimSamReadyNs != null) put("slimSamReadyNs", slimSamReadyNs)
    if (yoloxReadyNs != null) put("yoloxReadyNs", yoloxReadyNs)
    if (allModelsReadyNs != null) put("allModelsReadyNs", allModelsReadyNs)
    if (warmupTotalMs != null) put("warmupTotalMs", warmupTotalMs)
    if (slimSamTotalLoadMs != null) put("slimSamTotalLoadMs", slimSamTotalLoadMs)
    if (yoloxTotalLoadMs != null) put("yoloxTotalLoadMs", yoloxTotalLoadMs)
    put("reusedAuditVerificationForYolox", reusedAuditVerificationForYolox)
    put("duplicateEnsureLoadedBlockedCount", duplicateEnsureLoadedBlockedCount)
    if (errorCode != null) put("errorCode", errorCode)
    if (deviceInfo != null) put("deviceInfo", deviceInfo.toJson())
    put("stages", JSONArray().apply { stages.forEach { put(it.toJson()) } })
    put("moduleStates", JSONObject().apply {
      moduleStates.forEach { (module, state) -> put(module, state.name) }
    })
  }

  companion object {
    fun fromJson(json: JSONObject): GlobalWarmupRecord {
      val stagesArr = json.optJSONArray("stages")
      val stageList = buildList {
        if (stagesArr != null) {
          for (i in 0 until stagesArr.length()) {
            stagesArr.optJSONObject(i)?.let { add(StageTimingRecord.fromJson(it)) }
          }
        }
      }
      return GlobalWarmupRecord(
        status = runCatching { ModelLoadLifecycleState.valueOf(json.optString("status", "NOT_STARTED")) }
          .getOrDefault(ModelLoadLifecycleState.NOT_STARTED),
        slimSamState = runCatching { ModelLoadLifecycleState.valueOf(json.optString("slimSamState", "NOT_STARTED")) }
          .getOrDefault(ModelLoadLifecycleState.NOT_STARTED),
        yoloxState = runCatching { ModelLoadLifecycleState.valueOf(json.optString("yoloxState", "NOT_STARTED")) }
          .getOrDefault(ModelLoadLifecycleState.NOT_STARTED),
        warmupStartNs = json.optLong("warmupStartNs", 0L),
        slimSamReadyNs = if (json.has("slimSamReadyNs")) json.optLong("slimSamReadyNs") else null,
        yoloxReadyNs = if (json.has("yoloxReadyNs")) json.optLong("yoloxReadyNs") else null,
        allModelsReadyNs = if (json.has("allModelsReadyNs")) json.optLong("allModelsReadyNs") else null,
        warmupTotalMs = if (json.has("warmupTotalMs")) json.optDouble("warmupTotalMs") else null,
        slimSamTotalLoadMs = if (json.has("slimSamTotalLoadMs")) json.optDouble("slimSamTotalLoadMs") else null,
        yoloxTotalLoadMs = if (json.has("yoloxTotalLoadMs")) json.optDouble("yoloxTotalLoadMs") else null,
        reusedAuditVerificationForYolox = json.optBoolean("reusedAuditVerificationForYolox", false),
        duplicateEnsureLoadedBlockedCount = json.optInt("duplicateEnsureLoadedBlockedCount", 0),
        errorCode = json.optString("errorCode").takeIf { it.isNotEmpty() },
        deviceInfo = json.optJSONObject("deviceInfo")?.let { DeviceEnvironmentInfo.fromJson(it) },
        stages = stageList,
        moduleStates = json.optJSONObject("moduleStates")?.let(::parseRuntimeModuleStates).orEmpty(),
      )
    }
  }
}

private fun parseRuntimeModuleStates(json: JSONObject): Map<String, RuntimeModuleState> = buildMap {
  val keys = json.keys()
  while (keys.hasNext()) {
    val key = keys.next()
    val state = runCatching { RuntimeModuleState.valueOf(json.optString(key)) }.getOrNull()
    if (state != null) put(key, state)
  }
}

/**
 * Strips any URI schemes, storage/cache file paths, or GPS keys/terms from telemetry & diagnostics.
 */
object DiagnosticPrivacySanitizer {
  private val bannedKeyTokens = listOf("uri", "filepath", "gps", "latitude", "longitude", "location", "exiftag", "rawbytes")
  private val uriPattern = Regex("""(?:content|file|https?)://\S+""", RegexOption.IGNORE_CASE)
  private val unixPathPattern = Regex("""/(?:storage|sdcard|data/user|data/data|mnt|DCIM|Pictures|Download)/[^\s,;]+""")
  private val gpsTextPattern = Regex("""(?i)\b(?:gps(?:latitude|longitude|altitude)?|exif_gps)\b""")

  fun sanitizeText(value: String): String {
    var s = uriPattern.replace(value, "[REDACTED_URI]")
    s = unixPathPattern.replace(s, "[REDACTED_PATH]")
    s = gpsTextPattern.replace(s, "[REDACTED_GEO]")
    return s
  }

  fun isSensitiveKey(key: String): Boolean {
    val low = key.lowercase(Locale.US)
    return bannedKeyTokens.any { low.contains(it) }
  }

  fun sanitizeMap(input: Map<String, Any>): Map<String, Any> {
    if (input.isEmpty()) return emptyMap()
    val out = LinkedHashMap<String, Any>(input.size)
    for ((k, v) in input) {
      if (isSensitiveKey(k)) continue
      val cleanVal: Any = when (v) {
        is String -> sanitizeText(v)
        is Number, is Boolean -> v
        is IntArray -> v.toList()
        is LongArray -> v.toList()
        is FloatArray -> v.map { it.toDouble() }
        is DoubleArray -> v.toList()
        is List<*> -> v.map { item -> if (item is String) sanitizeText(item) else item ?: "" }
        else -> sanitizeText(v.toString())
      }
      out[k] = cleanVal
    }
    return out
  }

  fun sanitizeMapToJson(input: Map<String, Any>): JSONObject {
    val clean = sanitizeMap(input)
    val obj = JSONObject()
    for ((k, v) in clean) {
      when (v) {
        is List<*> -> obj.put(k, JSONArray(v))
        is Float -> obj.put(k, v.toDouble())
        else -> obj.put(k, v)
      }
    }
    return obj
  }
}

/**
 * Thread-safe, non-blocking monotonic performance telemetry recorder using
 * [SystemClock.elapsedRealtimeNanos] exclusively for all durations.
 */
class RecognitionPerformanceRecorder(
  private val context: Context? = null,
  internal val store: RuntimeDiagnosticsStore? = null,
  private val nanoClock: () -> Long = { SystemClock.elapsedRealtimeNanos() },
  private val utcClockMs: () -> Long = { System.currentTimeMillis() },
) {
  companion object {
    private const val MAX_RESOURCE_SAMPLES = 500

    fun nanosToMs(deltaNanos: Long): Double = deltaNanos.coerceAtLeast(0L) / 1_000_000.0
  }

  private val runCounter = AtomicInteger(0)
  private val sessionCounter = AtomicLong(0L)

  private class MutableSession(
    val sessionId: String,
    val taskId: Long,
    val photoGenerationId: Long,
    val runIndex: Int,
    val mode: String,
    val startProfile: String,
    val createdAtUtc: String,
    val startTimestampNs: Long,
    val deviceInfo: DeviceEnvironmentInfo,
    val modelVersions: Map<String, String>,
  ) {
    val lock = Any()
    val openStages = LinkedHashMap<String, Pair<Long, Map<String, Any>>>()
    val completedStages = LinkedHashMap<String, StageTimingRecord>()
    var warmupSnapshot: WarmupTimingSnapshot? = null
    var summaryMetrics: RecognitionSummaryMetrics? = null
    var finishedOutcome: StageOutcome? = null
    var endTimestampNs: Long = startTimestampNs
    var errorCode: String? = null
  }

  private val sessions = ConcurrentHashMap<String, MutableSession>()
  private val resourceSamples = CopyOnWriteArrayList<RuntimeResourceSample>()
  private val warmupLock = Any()
  private var warmupStatus: ModelLoadLifecycleState = ModelLoadLifecycleState.NOT_STARTED
  private var slimSamLoadState: ModelLoadLifecycleState = ModelLoadLifecycleState.NOT_STARTED
  private var yoloxLoadState: ModelLoadLifecycleState = ModelLoadLifecycleState.NOT_STARTED
  private var warmupStartNs: Long = 0L
  private var slimSamReadyNs: Long? = null
  private var yoloxReadyNs: Long? = null
  private var allModelsReadyNs: Long? = null
  private var slimSamTotalLoadMs: Double? = null
  private var yoloxTotalLoadMs: Double? = null
  private var warmupReusedAuditForYolox: Boolean = false
  private var warmupDuplicateGuardCount: Int = 0
  private var warmupErrorCode: String? = null
  private val warmupOpenStages = LinkedHashMap<String, Long>()
  private val warmupCompletedStages = LinkedHashMap<String, StageTimingRecord>()
  private val moduleStates = LinkedHashMap<String, RuntimeModuleState>()
  private var executionProvider: String = "CPU"

  @Volatile
  var lastFinishedSession: PerformanceSessionRecord? = null
    private set

  fun nowNanos(): Long = nanoClock()

  fun captureResourceSample(
    phase: String,
    taskId: Long,
    photoGenerationId: Long,
  ): RuntimeResourceSample {
    val device = DeviceEnvironmentInfo.capture(context, executionProvider)
    val sample = RuntimeResourceSample(
      phase = DiagnosticPrivacySanitizer.sanitizeText(phase),
      timestampNs = nanoClock(),
      taskId = taskId,
      photoGenerationId = photoGenerationId,
      javaHeapUsedKb = device.javaHeapUsedKb,
      javaHeapMaxKb = device.javaHeapMaxKb,
      nativeHeapAllocatedKb = device.nativeHeapAllocatedKb,
      pssTotalKb = device.pssTotalKb,
      availableMemoryKb = device.availableMemoryKb,
      gcCollectionCount = device.gcCollectionCount,
      gcCollectionTimeMs = device.gcCollectionTimeMs,
    )
    resourceSamples.add(sample)
    while (resourceSamples.size > MAX_RESOURCE_SAMPLES) resourceSamples.removeAt(0)
    return sample
  }

  fun resourceSamplesSnapshot(): List<RuntimeResourceSample> = resourceSamples.toList()

  fun updateExecutionProvider(provider: String) {
    executionProvider = DiagnosticPrivacySanitizer.sanitizeText(provider).ifBlank { "CPU" }
  }

  fun markWarmupStarted(): Long = synchronized(warmupLock) {
    val now = nanoClock()
    if (warmupStatus == ModelLoadLifecycleState.NOT_STARTED) {
      warmupStatus = ModelLoadLifecycleState.LOADING
      warmupStartNs = now
    }
    now
  }

  fun incrementDuplicateEnsureLoadedGuard() {
    synchronized(warmupLock) {
      warmupDuplicateGuardCount++
    }
  }

  fun updateModuleState(
    moduleName: String,
    state: RuntimeModuleState,
    errorCode: String? = null,
  ) {
    val snapshotToPersist: GlobalWarmupRecord?
    synchronized(warmupLock) {
      val now = nanoClock()
      if (warmupStartNs == 0L) warmupStartNs = now
      moduleStates[moduleName] = state
      if (state == RuntimeModuleState.ERROR) {
        warmupErrorCode = errorCode ?: "${moduleName}_ERROR"
      }
      snapshotToPersist = refreshGlobalWarmupLocked(now)
    }
    if (snapshotToPersist != null) {
      store?.persistWarmupAsync(snapshotToPersist)
    }
  }

  fun updateSlimSamLoadState(state: ModelLoadLifecycleState, totalLoadMs: Double? = null, errorCode: String? = null) {
    val snapshotToPersist: GlobalWarmupRecord?
    synchronized(warmupLock) {
      val now = nanoClock()
      if (warmupStartNs == 0L) warmupStartNs = now
      slimSamLoadState = state
      if (state == ModelLoadLifecycleState.LOADING && warmupStatus == ModelLoadLifecycleState.NOT_STARTED) {
        warmupStatus = ModelLoadLifecycleState.LOADING
      } else if (state == ModelLoadLifecycleState.READY) {
        if (slimSamReadyNs == null) slimSamReadyNs = now
        if (totalLoadMs != null) slimSamTotalLoadMs = totalLoadMs.coerceAtLeast(0.0)
      } else if (state == ModelLoadLifecycleState.ERROR) {
        warmupStatus = ModelLoadLifecycleState.ERROR
        warmupErrorCode = errorCode
      }
      snapshotToPersist = refreshGlobalWarmupLocked(now)
    }
    if (snapshotToPersist != null) {
      store?.persistWarmupAsync(snapshotToPersist)
    }
  }

  fun updateYoloxLoadState(
    state: ModelLoadLifecycleState,
    totalLoadMs: Double? = null,
    reusedAudit: Boolean = false,
    errorCode: String? = null,
  ) {
    val snapshotToPersist: GlobalWarmupRecord?
    synchronized(warmupLock) {
      val now = nanoClock()
      if (warmupStartNs == 0L) warmupStartNs = now
      yoloxLoadState = state
      if (reusedAudit) warmupReusedAuditForYolox = true
      if (state == ModelLoadLifecycleState.LOADING && warmupStatus == ModelLoadLifecycleState.NOT_STARTED) {
        warmupStatus = ModelLoadLifecycleState.LOADING
      } else if (state == ModelLoadLifecycleState.READY) {
        if (yoloxReadyNs == null) yoloxReadyNs = now
        if (totalLoadMs != null) yoloxTotalLoadMs = totalLoadMs.coerceAtLeast(0.0)
      } else if (state == ModelLoadLifecycleState.ERROR) {
        warmupStatus = ModelLoadLifecycleState.ERROR
        warmupErrorCode = errorCode
      }
      snapshotToPersist = refreshGlobalWarmupLocked(now)
    }
    if (snapshotToPersist != null) {
      store?.persistWarmupAsync(snapshotToPersist)
    }
  }

  private fun refreshGlobalWarmupLocked(nowNs: Long): GlobalWarmupRecord? {
    if (moduleStates.isNotEmpty()) {
      val ready = moduleStates["YOLOX"] == RuntimeModuleState.INFERENCE_READY &&
        moduleStates["PROCESSOR"] == RuntimeModuleState.PROCESSOR_READY &&
        moduleStates["ENCODER"] == RuntimeModuleState.INFERENCE_READY &&
        moduleStates["DECODER"] == RuntimeModuleState.INFERENCE_READY
      if (ready) {
        warmupStatus = ModelLoadLifecycleState.READY
        if (allModelsReadyNs == null) allModelsReadyNs = nowNs.coerceAtLeast(warmupStartNs)
        return buildWarmupRecordLocked()
      }
      if (moduleStates.values.any { it == RuntimeModuleState.ERROR }) {
        warmupStatus = ModelLoadLifecycleState.ERROR
        return buildWarmupRecordLocked()
      }
      if (moduleStates.values.any { it == RuntimeModuleState.SKIPPED_RESOURCE }) {
        warmupStatus = ModelLoadLifecycleState.SKIPPED_RESOURCE
        return buildWarmupRecordLocked()
      }
      if (warmupStatus == ModelLoadLifecycleState.NOT_STARTED) {
        warmupStatus = ModelLoadLifecycleState.LOADING
      }
      return null
    }

    if (slimSamLoadState == ModelLoadLifecycleState.READY && yoloxLoadState == ModelLoadLifecycleState.READY) {
      warmupStatus = ModelLoadLifecycleState.READY
      if (allModelsReadyNs == null) {
        allModelsReadyNs = nowNs.coerceAtLeast(warmupStartNs)
      }
      return buildWarmupRecordLocked()
    }
    if (warmupStatus == ModelLoadLifecycleState.ERROR) {
      return buildWarmupRecordLocked()
    }
    return null
  }

  fun beginWarmupStage(stageName: String): Long = synchronized(warmupLock) {
    val now = nanoClock()
    if (warmupStartNs == 0L) warmupStartNs = now
    if (!warmupCompletedStages.containsKey(stageName) && !warmupOpenStages.containsKey(stageName)) {
      warmupOpenStages[stageName] = now
    }
    now
  }

  fun recordWarmupStageDuration(
    stageName: String,
    startNs: Long,
    endNs: Long = nanoClock(),
    status: StageOutcome = StageOutcome.SUCCESS,
    errorCode: String? = null,
    metrics: Map<String, Any> = emptyMap(),
  ): Boolean {
    synchronized(warmupLock) {
      if (warmupCompletedStages.containsKey(stageName)) return false
      warmupOpenStages.remove(stageName)
      val safeStart = startNs
      val safeEnd = endNs.coerceAtLeast(safeStart)
      val durMs = nanosToMs(safeEnd - safeStart)
      warmupCompletedStages[stageName] = StageTimingRecord(
        sessionId = "warmup",
        taskId = 0L,
        photoGenerationId = 0L,
        runIndex = 0,
        stageName = stageName,
        startTimestampNs = safeStart,
        endTimestampNs = safeEnd,
        durationMs = durMs,
        durationRoundedMs = durMs.roundToLong(),
        status = status,
        errorCode = errorCode?.let { DiagnosticPrivacySanitizer.sanitizeText(it) },
        metrics = DiagnosticPrivacySanitizer.sanitizeMap(metrics),
      )
      return true
    }
  }

  fun completeWarmupStage(
    stageName: String,
    status: StageOutcome = StageOutcome.SUCCESS,
    errorCode: String? = null,
    metrics: Map<String, Any> = emptyMap(),
  ): Boolean {
    synchronized(warmupLock) {
      if (warmupCompletedStages.containsKey(stageName)) return false
      val start = warmupOpenStages.remove(stageName) ?: nanoClock()
      val end = nanoClock().coerceAtLeast(start)
      val durMs = nanosToMs(end - start)
      warmupCompletedStages[stageName] = StageTimingRecord(
        sessionId = "warmup",
        taskId = 0L,
        photoGenerationId = 0L,
        runIndex = 0,
        stageName = stageName,
        startTimestampNs = start,
        endTimestampNs = end,
        durationMs = durMs,
        durationRoundedMs = durMs.roundToLong(),
        status = status,
        errorCode = errorCode?.let { DiagnosticPrivacySanitizer.sanitizeText(it) },
        metrics = DiagnosticPrivacySanitizer.sanitizeMap(metrics),
      )
      return true
    }
  }

  fun currentWarmupRecord(): GlobalWarmupRecord = synchronized(warmupLock) {
    buildWarmupRecordLocked()
  }

  private fun buildWarmupRecordLocked(): GlobalWarmupRecord {
    val totalMs = if (warmupStartNs > 0L && allModelsReadyNs != null) {
      nanosToMs(allModelsReadyNs!! - warmupStartNs)
    } else {
      null
    }
    return GlobalWarmupRecord(
      status = warmupStatus,
      slimSamState = slimSamLoadState,
      yoloxState = yoloxLoadState,
      warmupStartNs = warmupStartNs,
      slimSamReadyNs = slimSamReadyNs,
      yoloxReadyNs = yoloxReadyNs,
      allModelsReadyNs = allModelsReadyNs,
      warmupTotalMs = totalMs,
      slimSamTotalLoadMs = slimSamTotalLoadMs,
      yoloxTotalLoadMs = yoloxTotalLoadMs,
      reusedAuditVerificationForYolox = warmupReusedAuditForYolox,
      duplicateEnsureLoadedBlockedCount = warmupDuplicateGuardCount,
      errorCode = warmupErrorCode,
      deviceInfo = DeviceEnvironmentInfo.capture(context, executionProvider),
      stages = warmupCompletedStages.values.toList(),
      moduleStates = moduleStates.toMap(),
    )
  }

  fun captureWarmupPickState(): Pair<ModelLoadLifecycleState, ModelLoadLifecycleState> =
    synchronized(warmupLock) { yoloxLoadState to slimSamLoadState }

  fun captureModuleStates(): Map<String, RuntimeModuleState> =
    synchronized(warmupLock) { moduleStates.toMap() }

  fun startSession(
    taskId: Long,
    photoGenerationId: Long,
    mode: String,
    startProfile: String,
    startTimestampNs: Long = nanoClock(),
    modelVersions: Map<String, String> = mapOf(
      "YOLOX" to ModelAssetAudit.YOLOX_EXPECTED_SHA256,
      "SlimSAM" to ModelAssetAudit.SLIMSAM_REVISION,
    ),
  ): String {
    val runIdx = runCounter.incrementAndGet()
    val seq = sessionCounter.incrementAndGet()
    val sessionId = "sess_${runIdx}_${seq}_${taskId}"
    val fmt = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US).apply {
      timeZone = TimeZone.getTimeZone("UTC")
    }
    val session = MutableSession(
      sessionId = sessionId,
      taskId = taskId,
      photoGenerationId = photoGenerationId,
      runIndex = runIdx,
      mode = mode,
      startProfile = startProfile,
      createdAtUtc = fmt.format(Date(utcClockMs())),
      startTimestampNs = startTimestampNs,
      deviceInfo = DeviceEnvironmentInfo.capture(context, executionProvider),
      modelVersions = modelVersions,
    )
    sessions[sessionId] = session
    return sessionId
  }

  fun beginStage(
    sessionId: String?,
    stageName: String,
    metrics: Map<String, Any> = emptyMap(),
  ): Long {
    val now = nanoClock()
    if (sessionId == null) return now
    val s = sessions[sessionId] ?: return now
    synchronized(s.lock) {
      if (s.finishedOutcome != null) return now
      if (s.completedStages.containsKey(stageName)) return now
      if (!s.openStages.containsKey(stageName)) {
        s.openStages[stageName] = now to DiagnosticPrivacySanitizer.sanitizeMap(metrics)
      }
    }
    return now
  }

  fun completeStage(
    sessionId: String?,
    stageName: String,
    status: StageOutcome = StageOutcome.SUCCESS,
    errorCode: String? = null,
    metrics: Map<String, Any> = emptyMap(),
  ): Boolean {
    if (sessionId == null) return false
    val s = sessions[sessionId] ?: return false
    val end = nanoClock()
    synchronized(s.lock) {
      if (s.completedStages.containsKey(stageName)) {
        return false
      }
      val started = s.openStages.remove(stageName)
      val safeStart = started?.first ?: end
      val safeEnd = end.coerceAtLeast(safeStart)
      val mergedMetrics = LinkedHashMap<String, Any>().apply {
        started?.second?.let { putAll(it) }
        putAll(DiagnosticPrivacySanitizer.sanitizeMap(metrics))
      }
      val durMs = nanosToMs(safeEnd - safeStart)
      s.completedStages[stageName] = StageTimingRecord(
        sessionId = s.sessionId,
        taskId = s.taskId,
        photoGenerationId = s.photoGenerationId,
        runIndex = s.runIndex,
        stageName = stageName,
        startTimestampNs = safeStart,
        endTimestampNs = safeEnd,
        durationMs = durMs,
        durationRoundedMs = durMs.roundToLong(),
        status = status,
        errorCode = errorCode?.let { DiagnosticPrivacySanitizer.sanitizeText(it) },
        metrics = mergedMetrics,
      )
      return true
    }
  }

  fun recordStageSpan(
    sessionId: String?,
    stageName: String,
    startNs: Long,
    endNs: Long,
    status: StageOutcome = StageOutcome.SUCCESS,
    errorCode: String? = null,
    metrics: Map<String, Any> = emptyMap(),
  ): Boolean {
    if (sessionId == null) return false
    val s = sessions[sessionId] ?: return false
    synchronized(s.lock) {
      if (s.completedStages.containsKey(stageName)) {
        return false
      }
      s.openStages.remove(stageName)
      val safeEnd = endNs.coerceAtLeast(startNs)
      val durMs = nanosToMs(safeEnd - startNs)
      s.completedStages[stageName] = StageTimingRecord(
        sessionId = s.sessionId,
        taskId = s.taskId,
        photoGenerationId = s.photoGenerationId,
        runIndex = s.runIndex,
        stageName = stageName,
        startTimestampNs = startNs,
        endTimestampNs = safeEnd,
        durationMs = durMs,
        durationRoundedMs = durMs.roundToLong(),
        status = status,
        errorCode = errorCode?.let { DiagnosticPrivacySanitizer.sanitizeText(it) },
        metrics = DiagnosticPrivacySanitizer.sanitizeMap(metrics),
      )
      return true
    }
  }

  fun attachWarmupSnapshot(sessionId: String?, snapshot: WarmupTimingSnapshot) {
    if (sessionId == null) return
    val s = sessions[sessionId] ?: return
    synchronized(s.lock) {
      s.warmupSnapshot = snapshot
    }
  }

  fun attachSummaryMetrics(sessionId: String?, metrics: RecognitionSummaryMetrics) {
    if (sessionId == null) return
    val s = sessions[sessionId] ?: return
    synchronized(s.lock) {
      s.summaryMetrics = metrics
    }
  }

  fun updateAnimationCompletion(
    sessionId: String?,
    animationStartNs: Long,
    animationEndNs: Long,
    shaderFps: Float?,
    photoPickStartNs: Long,
  ): PerformanceSessionRecord? {
    if (sessionId == null) return null
    val s = sessions[sessionId] ?: return null
    val safeEnd = animationEndNs.coerceAtLeast(animationStartNs)
    val animDurationMs = nanosToMs(safeEnd - animationStartNs)
    val fullExpMs = nanosToMs(safeEnd - photoPickStartNs.coerceAtMost(safeEnd))

    val updated = synchronized(s.lock) {
      if (!s.completedStages.containsKey("animation_start")) {
        s.completedStages["animation_start"] = StageTimingRecord(
          sessionId = s.sessionId,
          taskId = s.taskId,
          photoGenerationId = s.photoGenerationId,
          runIndex = s.runIndex,
          stageName = "animation_start",
          startTimestampNs = animationStartNs,
          endTimestampNs = animationStartNs,
          durationMs = 0.0,
          durationRoundedMs = 0L,
          status = StageOutcome.SUCCESS,
        )
      }
      if (!s.completedStages.containsKey("animation_complete")) {
        s.completedStages["animation_complete"] = StageTimingRecord(
          sessionId = s.sessionId,
          taskId = s.taskId,
          photoGenerationId = s.photoGenerationId,
          runIndex = s.runIndex,
          stageName = "animation_complete",
          startTimestampNs = animationStartNs,
          endTimestampNs = safeEnd,
          durationMs = animDurationMs,
          durationRoundedMs = animDurationMs.roundToLong(),
          status = StageOutcome.SUCCESS,
          metrics = buildMap {
            if (shaderFps != null) put("shaderFps", shaderFps.toDouble())
          },
        )
      }
      s.summaryMetrics = s.summaryMetrics?.copy(
        animationTimeMs = animDurationMs,
        fullExperienceTimeMs = fullExpMs,
        shaderFps = shaderFps ?: s.summaryMetrics?.shaderFps,
      )
      toImmutableRecordLocked(s)
    }
    if (updated.finishedOutcomeSaved()) {
      lastFinishedSession = updated
      store?.persistSessionAsync(updated)
    }
    return updated
  }

  private fun PerformanceSessionRecord.finishedOutcomeSaved(): Boolean = true

  fun finishSession(
    sessionId: String?,
    outcome: StageOutcome,
    errorCode: String? = null,
    endTimestampNs: Long = nanoClock(),
  ): PerformanceSessionRecord? {
    if (sessionId == null) return null
    val s = sessions[sessionId] ?: return null
    val record = synchronized(s.lock) {
      if (s.finishedOutcome != null) {
        return@synchronized toImmutableRecordLocked(s)
      }
      val safeEnd = endTimestampNs.coerceAtLeast(s.startTimestampNs)
      s.finishedOutcome = outcome
      s.endTimestampNs = safeEnd
      s.errorCode = errorCode?.let { DiagnosticPrivacySanitizer.sanitizeText(it) }

      // Close any still-open stages as FAILED or CANCELLED while preserving all already completed stages
      if (s.openStages.isNotEmpty()) {
        val openCopy = s.openStages.entries.toList()
        s.openStages.clear()
        for ((stageName, pair) in openCopy) {
          val (stNs, metrics) = pair
          val se = safeEnd.coerceAtLeast(stNs)
          val dur = nanosToMs(se - stNs)
          s.completedStages[stageName] = StageTimingRecord(
            sessionId = s.sessionId,
            taskId = s.taskId,
            photoGenerationId = s.photoGenerationId,
            runIndex = s.runIndex,
            stageName = stageName,
            startTimestampNs = stNs,
            endTimestampNs = se,
            durationMs = dur,
            durationRoundedMs = dur.roundToLong(),
            status = if (outcome == StageOutcome.SUCCESS) StageOutcome.CANCELLED else outcome,
            errorCode = s.errorCode,
            metrics = metrics,
          )
        }
      }
      toImmutableRecordLocked(s)
    }
    lastFinishedSession = record
    store?.persistSessionAsync(record)
    return record
  }

  fun getSessionSnapshot(sessionId: String?): PerformanceSessionRecord? {
    if (sessionId == null) return null
    val s = sessions[sessionId] ?: return null
    return synchronized(s.lock) { toImmutableRecordLocked(s) }
  }

  private fun toImmutableRecordLocked(s: MutableSession): PerformanceSessionRecord {
    val safeEnd = s.endTimestampNs.coerceAtLeast(s.startTimestampNs)
    val durMs = nanosToMs(safeEnd - s.startTimestampNs)
    return PerformanceSessionRecord(
      sessionId = s.sessionId,
      taskId = s.taskId,
      photoGenerationId = s.photoGenerationId,
      runIndex = s.runIndex,
      mode = s.mode,
      startProfile = s.startProfile,
      createdAtUtc = s.createdAtUtc,
      startTimestampNs = s.startTimestampNs,
      endTimestampNs = safeEnd,
      durationMs = durMs,
      status = s.finishedOutcome ?: StageOutcome.SUCCESS,
      errorCode = s.errorCode,
      deviceInfo = s.deviceInfo,
      modelVersions = s.modelVersions,
      warmupSnapshot = s.warmupSnapshot,
      summaryMetrics = s.summaryMetrics,
      stages = s.completedStages.values.toList(),
    )
  }
}
