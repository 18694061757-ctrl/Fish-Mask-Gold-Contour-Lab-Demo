package com.yujian.fishmasklab.runtime

import android.content.Context
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.OutputStream
import java.security.DigestOutputStream
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

data class DiagnosticZipExportResult(
  val fileCount: Int,
  val totalBytes: Long,
  val zipSha256: String,
  val sessionCount: Int,
)

data class GeneratedReportBundle(
  val summaryJson: String,
  val stageTimelineCsv: String,
  val performanceReportMd: String,
  val runtimeEventsJsonl: String,
  val modelWarmupJson: String,
  val deviceInfoJson: String,
  val modelManifestJson: String,
)

/**
 * Persistent rolling runtime telemetry & diagnostic store backed by `filesDir/runtime_diagnostics/`.
 * All disk writes run sequentially on a single-lane IO dispatcher so inference and Main UI never block.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class RuntimeDiagnosticsStore(
  private val context: Context,
  private val rootDir: File = File(context.filesDir, "runtime_diagnostics"),
  private val maxRetainedSessions: Int = MAX_SESSIONS,
) {
  companion object {
    const val MAX_SESSIONS = 10

    fun defaultZipFileName(timestampMs: Long = System.currentTimeMillis()): String {
      val fmt = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).apply {
        timeZone = TimeZone.getDefault()
      }
      return "FishMask_Diagnostics_${fmt.format(Date(timestampMs))}.zip"
    }

    private fun sha256Hex(bytes: ByteArray): String =
      MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    private fun csvEscape(value: String): String {
      val sanitized = DiagnosticPrivacySanitizer.sanitizeText(value)
      return if (sanitized.contains(',') || sanitized.contains('"') || sanitized.contains('\n')) {
        "\"${sanitized.replace("\"", "\"\"")}\""
      } else {
        sanitized
      }
    }

    fun buildReportBundle(
      warmup: GlobalWarmupRecord?,
      auditReport: ModelAssetAuditReport?,
      sessions: List<PerformanceSessionRecord>,
      context: Context? = null,
    ): GeneratedReportBundle {
      val ordered = sessions.sortedWith(compareBy<PerformanceSessionRecord> { it.runIndex }.thenBy { it.startTimestampNs })
      val latest = ordered.lastOrNull()
      val deviceInfo = latest?.deviceInfo
        ?: warmup?.deviceInfo
        ?: DeviceEnvironmentInfo.capture(context)

      val coldRun = ordered.firstOrNull { it.mode == "INITIAL_RECOGNITION" && it.status == StageOutcome.SUCCESS }
      val hotRun = ordered.lastOrNull {
        it.mode == "INITIAL_RECOGNITION" && it.status == StageOutcome.SUCCESS && it.sessionId != coldRun?.sessionId
      } ?: ordered.lastOrNull { it.mode == "TAP_REFOCUS" && it.status == StageOutcome.SUCCESS }

      val summaryObj = JSONObject().apply {
        put("taskId", "FISH_MASK_ANDROID_P0_V2_2_PERFORMANCE_TELEMETRY_LOG_EXPORT")
        put("retainedSessionCount", ordered.size)
        put("maxRetainedSessions", MAX_SESSIONS)
        put("deviceInfo", deviceInfo.toJson())
        if (warmup != null) {
          put("warmupTotalMs", warmup.warmupTotalMs)
          put("slimSamTotalLoadMs", warmup.slimSamTotalLoadMs)
          put("yoloxTotalLoadMs", warmup.yoloxTotalLoadMs)
          put("warmupStatus", warmup.status.name)
        }
        if (coldRun?.summaryMetrics != null) {
          put("coldStartMetrics", coldRun.summaryMetrics.toJson())
        }
        if (hotRun?.summaryMetrics != null) {
          put("hotRunMetrics", hotRun.summaryMetrics.toJson())
        }
        if (latest?.summaryMetrics != null) {
          put("latestMetrics", latest.summaryMetrics.toJson())
        }
        put(
          "sessions",
          JSONArray().apply {
            for (s in ordered) {
              put(
                JSONObject().apply {
                  put("sessionId", s.sessionId)
                  put("taskId", s.taskId)
                  put("photoGenerationId", s.photoGenerationId)
                  put("runIndex", s.runIndex)
                  put("mode", s.mode)
                  put("startProfile", s.startProfile)
                  put("createdAtUtc", s.createdAtUtc)
                  put("durationMs", s.durationMs)
                  put("status", s.status.name)
                  put("modelVersions", JSONObject(s.modelVersions))
                  if (s.errorCode != null) put("errorCode", s.errorCode)
                  if (s.warmupSnapshot != null) put("warmupSnapshot", s.warmupSnapshot.toJson())
                  if (s.summaryMetrics != null) put("summaryMetrics", s.summaryMetrics.toJson())
                },
              )
            }
          },
        )
      }

      val csvSb = StringBuilder()
      csvSb.appendLine("session_id,task_id,photo_generation_id,run_index,mode,start_profile,stage_name,start_ns,end_ns,duration_ms,status,error_code")
      if (warmup != null) {
        for (st in warmup.stages) {
          csvSb.appendLine(
            listOf(
              csvEscape(st.sessionId),
              st.taskId.toString(),
              st.photoGenerationId.toString(),
              st.runIndex.toString(),
              "WARMUP",
              "MODEL_PRELOAD",
              csvEscape(st.stageName),
              st.startTimestampNs.toString(),
              st.endTimestampNs.toString(),
              String.format(Locale.US, "%.3f", st.durationMs),
              st.status.name,
              csvEscape(st.errorCode.orEmpty()),
            ).joinToString(","),
          )
        }
      }
      for (s in ordered) {
        for (st in s.stages) {
          csvSb.appendLine(
            listOf(
              csvEscape(s.sessionId),
              s.taskId.toString(),
              s.photoGenerationId.toString(),
              s.runIndex.toString(),
              csvEscape(s.mode),
              csvEscape(s.startProfile),
              csvEscape(st.stageName),
              st.startTimestampNs.toString(),
              st.endTimestampNs.toString(),
              String.format(Locale.US, "%.3f", st.durationMs),
              st.status.name,
              csvEscape(st.errorCode.orEmpty()),
            ).joinToString(","),
          )
        }
      }

      val jsonlSb = StringBuilder()
      if (warmup != null) {
        for (st in warmup.stages) {
          jsonlSb.appendLine(
            st.toJson().apply {
              put("eventType", "WARMUP_STAGE")
              put("modelStatus", warmup.status.name)
            }.toString(),
          )
        }
      }
      for (s in ordered) {
        jsonlSb.appendLine(
          JSONObject().apply {
            put("eventType", "SESSION_SUMMARY")
            put("sessionId", s.sessionId)
            put("taskId", s.taskId)
            put("photoGenerationId", s.photoGenerationId)
            put("runIndex", s.runIndex)
            put("mode", s.mode)
            put("startProfile", s.startProfile)
            put("status", s.status.name)
            put("durationMs", s.durationMs)
            if (s.errorCode != null) put("errorCode", s.errorCode)
          }.toString(),
        )
        for (st in s.stages) {
          jsonlSb.appendLine(
            st.toJson().apply {
              put("eventType", "STAGE_TIMING")
              put("mode", s.mode)
            }.toString(),
          )
        }
      }

      val warmupJson = (warmup ?: GlobalWarmupRecord(
        status = ModelLoadLifecycleState.NOT_STARTED,
        slimSamState = ModelLoadLifecycleState.NOT_STARTED,
        yoloxState = ModelLoadLifecycleState.NOT_STARTED,
        warmupStartNs = 0L,
        slimSamReadyNs = null,
        yoloxReadyNs = null,
        allModelsReadyNs = null,
        warmupTotalMs = null,
        slimSamTotalLoadMs = null,
        yoloxTotalLoadMs = null,
        reusedAuditVerificationForYolox = false,
        duplicateEnsureLoadedBlockedCount = 0,
      )).toJson().toString(2)

      val manifestObj = JSONObject().apply {
        put("slimsamModelId", ModelAssetAudit.SLIMSAM_MODEL_ID)
        put("slimsamRevision", ModelAssetAudit.SLIMSAM_REVISION)
        put("yoloxAssetPath", ModelAssetAudit.YOLOX_ASSET_PATH)
        put("yoloxExpectedSha256", ModelAssetAudit.YOLOX_EXPECTED_SHA256)
        put("slimSamEncoderAssetPath", ModelAssetAudit.SLIMSAM_ENCODER_PATH)
        put("slimSamEncoderExpectedSha256", ModelAssetAudit.SLIMSAM_ENCODER_EXPECTED_SHA256)
        put("slimSamDecoderAssetPath", ModelAssetAudit.SLIMSAM_DECODER_PATH)
        put("slimSamDecoderExpectedSha256", ModelAssetAudit.SLIMSAM_DECODER_EXPECTED_SHA256)
        put("slimSamProcessorPath", ModelAssetAudit.SLIMSAM_PROCESSOR_PATH)
        put("slimSamProcessorExpectedSha256", ModelAssetAudit.SLIMSAM_PROCESSOR_EXPECTED_SHA256)
        if (auditReport != null) {
          put("auditDurationMs", auditReport.auditDurationMs)
          put("sha256TotalDurationMs", auditReport.sha256TotalDurationMs)
          put("yoloxActualSha256", auditReport.yoloxAsset.sha256)
          put("yoloxSizeBytes", auditReport.yoloxAsset.sizeBytes)
          put("slimSamEncoderActualSha256", auditReport.slimSamEncoderAsset.sha256)
          put("slimSamEncoderSizeBytes", auditReport.slimSamEncoderAsset.sizeBytes)
          put("slimSamDecoderActualSha256", auditReport.slimSamDecoderAsset.sha256)
          put("slimSamDecoderSizeBytes", auditReport.slimSamDecoderAsset.sizeBytes)
        }
      }

      val reportMd = buildMarkdownReport(warmup, deviceInfo, ordered, coldRun, hotRun)

      return GeneratedReportBundle(
        summaryJson = DiagnosticPrivacySanitizer.sanitizeText(summaryObj.toString(2)),
        stageTimelineCsv = DiagnosticPrivacySanitizer.sanitizeText(csvSb.toString()),
        performanceReportMd = DiagnosticPrivacySanitizer.sanitizeText(reportMd),
        runtimeEventsJsonl = DiagnosticPrivacySanitizer.sanitizeText(jsonlSb.toString()),
        modelWarmupJson = DiagnosticPrivacySanitizer.sanitizeText(warmupJson),
        deviceInfoJson = DiagnosticPrivacySanitizer.sanitizeText(deviceInfo.toJson().toString(2)),
        modelManifestJson = DiagnosticPrivacySanitizer.sanitizeText(manifestObj.toString(2)),
      )
    }

    private fun fmtMs(value: Double?): String =
      if (value == null) "N/A (not measured in sample)" else String.format(Locale.US, "%.2f ms", value)

    private fun buildMarkdownReport(
      warmup: GlobalWarmupRecord?,
      deviceInfo: DeviceEnvironmentInfo,
      sessions: List<PerformanceSessionRecord>,
      coldRun: PerformanceSessionRecord?,
      hotRun: PerformanceSessionRecord?,
    ): String {
      val sb = StringBuilder()
      sb.appendLine("# Fish Mask Recognition Performance Report (P0-V2.2)")
      sb.appendLine()
      sb.appendLine("## 1. Environment & Runtime Contract")
      sb.appendLine("- App Version: `${deviceInfo.appVersionName}` (code=${deviceInfo.appVersionCode}, buildSha=`${deviceInfo.buildSha}`)")
      sb.appendLine("- Device: `${deviceInfo.deviceManufacturer} ${deviceInfo.deviceModel}` | API `${deviceInfo.apiLevel}` | ABIs `${deviceInfo.cpuAbis.joinToString()}` | CPU Cores `${deviceInfo.cpuCores}`")
      sb.appendLine("- ONNX Runtime: `${deviceInfo.onnxRuntimeVersion}` | Execution Provider: `${deviceInfo.executionProvider}`")
      sb.appendLine("- Memory Snapshot: Java Heap `${deviceInfo.javaHeapUsedKb}/${deviceInfo.javaHeapMaxKb} KB`, Native Heap `${deviceInfo.nativeHeapAllocatedKb} KB`, PSS `${deviceInfo.pssTotalKb} KB`, Available `${deviceInfo.availableMemoryKb} KB`")
      sb.appendLine()

      sb.appendLine("## 2. Model Warmup & Load Breakdown")
      if (warmup != null) {
        sb.appendLine("- Warmup Status: `${warmup.status.name}` (SlimSAM=`${warmup.slimSamState.name}`, YOLOX=`${warmup.yoloxState.name}`)")
        sb.appendLine("- Total Model Warmup Time: `${fmtMs(warmup.warmupTotalMs)}` (SlimSAM=`${fmtMs(warmup.slimSamTotalLoadMs)}`, YOLOX=`${fmtMs(warmup.yoloxTotalLoadMs)}`)")
        sb.appendLine("- Duplicate `ensureLoaded()` Reused/Blocked Count: `${warmup.duplicateEnsureLoadedBlockedCount}` | Reused YOLOX SHA Audit: `${warmup.reusedAuditVerificationForYolox}`")
        if (warmup.stages.isNotEmpty()) {
          sb.appendLine()
          sb.appendLine("| Load Sub-Stage | Duration | Status |")
          sb.appendLine("|---|---:|---|")
          for (st in warmup.stages) {
            sb.appendLine("| `${st.stageName}` | ${String.format(Locale.US, "%.2f ms", st.durationMs)} | ${st.status.name} |")
          }
        }
      } else {
        sb.appendLine("- No global warmup telemetry recorded yet.")
      }
      sb.appendLine()

      val coldM = coldRun?.summaryMetrics
      val hotM = hotRun?.summaryMetrics
      sb.appendLine("## 3. Cold Start vs Subsequent Hot Run Summary")
      sb.appendLine("| 指标 | 首次冷启动 | 后续热启动 |")
      sb.appendLine("|---|---:|---:|")
      sb.appendLine("| 模型预热 | ${fmtMs(warmup?.warmupTotalMs ?: coldM?.modelWarmupTimeMs)} | ${fmtMs(hotM?.modelWaitTimeMs)} (wait) |")
      sb.appendLine("| 照片解码 | ${fmtMs(coldM?.photoDecodeTimeMs)} | ${fmtMs(hotM?.photoDecodeTimeMs)} |")
      sb.appendLine("| YOLOX | ${fmtMs(coldM?.yoloxTimeMs)} | ${fmtMs(hotM?.yoloxTimeMs)} |")
      sb.appendLine("| SAM Processor | ${fmtMs(coldM?.samProcessorTimeMs)} | ${fmtMs(hotM?.samProcessorTimeMs)} |")
      sb.appendLine("| SAM Encoder | ${fmtMs(coldM?.samEncoderTimeMs)} | ${fmtMs(hotM?.samEncoderTimeMs)} |")
      sb.appendLine("| SAM Decoder | ${fmtMs(coldM?.samDecoderTimeMs)} | ${fmtMs(hotM?.samDecoderTimeMs)} |")
      sb.appendLine("| Mask 后处理 | ${fmtMs(coldM?.maskPostprocessTimeMs)} | ${fmtMs(hotM?.maskPostprocessTimeMs)} |")
      sb.appendLine("| Contour | ${fmtMs(coldM?.contourTimeMs)} | ${fmtMs(hotM?.contourTimeMs)} |")
      sb.appendLine("| 完整识别 | ${fmtMs(coldM?.recognitionPipelineTimeMs)} | ${fmtMs(hotM?.recognitionPipelineTimeMs)} |")
      sb.appendLine("| 动效播放 | ${fmtMs(coldM?.animationTimeMs)} | ${fmtMs(hotM?.animationTimeMs)} |")
      sb.appendLine()

      val primarySample = coldRun ?: sessions.lastOrNull { it.status == StageOutcome.SUCCESS }
      if (primarySample?.summaryMetrics != null) {
        val m = primarySample.summaryMetrics
        val parts = listOf(
          "Photo Decode" to m.photoDecodeTimeMs,
          "Model Wait" to m.modelWaitTimeMs,
          "YOLOX Inference" to m.yoloxTimeMs,
          "SAM Processor" to m.samProcessorTimeMs,
          "SAM Encoder (Q8 CPU)" to m.samEncoderTimeMs,
          "SAM Decoder (Q8 CPU)" to m.samDecoderTimeMs,
          "Mask Postprocess" to m.maskPostprocessTimeMs,
          "Contour Extraction" to m.contourTimeMs,
        ).sortedByDescending { it.second }
        val totalMeasured = parts.sumOf { it.second }.coerceAtLeast(1e-6)
        sb.appendLine("## 4. Bottleneck Ranking (Measured from Session `${primarySample.sessionId}`)")
        for ((idx, pair) in parts.withIndex()) {
          val pct = (pair.second / totalMeasured) * 100.0
          sb.appendLine("${idx + 1}. **${pair.first}**: ${String.format(Locale.US, "%.2f ms (%.1f%%)", pair.second, pct)}")
        }
        sb.appendLine()
      }

      if (sessions.isNotEmpty()) {
        sb.appendLine("## 5. Recorded Recognition Sessions (${sessions.size})")
        sb.appendLine("| Run# | Session | Mode | Status | Pipeline Ms | First Result Ms | Reused Emb |")
        sb.appendLine("|---:|---|---|---|---:|---:|:---:|")
        for (s in sessions) {
          val sm = s.summaryMetrics
          sb.appendLine(
            "| ${s.runIndex} | `${s.sessionId}` | ${s.mode} | ${s.status.name} | " +
              "${fmtMs(sm?.recognitionPipelineTimeMs ?: s.durationMs)} | ${fmtMs(sm?.firstResultTimeMs)} | " +
              "${sm?.reusedEncoderEmbedding ?: false} |",
          )
        }
      }
      return sb.toString()
    }
  }

  private val ioScope = CoroutineScope(SupervisorJob() + Dispatchers.IO.limitedParallelism(1))
  private val diskMutex = Mutex()
  private val sessionsDir = File(rootDir, "sessions")
  private val warmupFile = File(rootDir, "model_warmup.json")
  private val indexFile = File(rootDir, "sessions_index.json")

  @Volatile
  private var lastWriteJob: Job? = null

  fun persistWarmupAsync(record: GlobalWarmupRecord): Job {
    val job = ioScope.launch {
      diskMutex.withLock {
        writeWarmupLocked(record)
      }
    }
    lastWriteJob = job
    return job
  }

  fun persistSessionAsync(record: PerformanceSessionRecord): Job {
    val job = ioScope.launch {
      diskMutex.withLock {
        writeSessionLocked(record)
      }
    }
    lastWriteJob = job
    return job
  }

  suspend fun flushPendingWrites() {
    lastWriteJob?.join()
    withContext(Dispatchers.IO) {
      diskMutex.withLock {
        // Ensures any preceding sequential write inside diskMutex has completed
      }
    }
  }

  suspend fun loadWarmupRecord(): GlobalWarmupRecord? = withContext(Dispatchers.IO) {
    diskMutex.withLock {
      if (!warmupFile.exists()) return@withLock null
      runCatching { GlobalWarmupRecord.fromJson(JSONObject(warmupFile.readText())) }.getOrNull()
    }
  }

  suspend fun loadSessions(): List<PerformanceSessionRecord> = withContext(Dispatchers.IO) {
    diskMutex.withLock {
      readSessionsLocked()
    }
  }

  suspend fun exportDiagnosticZip(
    outputStream: OutputStream,
    currentWarmup: GlobalWarmupRecord? = null,
    auditReport: ModelAssetAuditReport? = null,
    resourceSamples: List<RuntimeResourceSample> = emptyList(),
  ): DiagnosticZipExportResult = withContext(Dispatchers.IO) {
    flushPendingWrites()
    val warmup: GlobalWarmupRecord?
    val sessions: List<PerformanceSessionRecord>
    diskMutex.withLock {
      warmup = currentWarmup ?: if (warmupFile.exists()) {
        runCatching { GlobalWarmupRecord.fromJson(JSONObject(warmupFile.readText())) }.getOrNull()
      } else {
        null
      }
      sessions = readSessionsLocked()
    }

    val bundle = buildReportBundle(
      warmup = warmup,
      auditReport = auditReport,
      sessions = sessions,
      context = context,
    )

    val entries = linkedMapOf(
      "summary.json" to bundle.summaryJson.toByteArray(Charsets.UTF_8),
      "performance_report.md" to bundle.performanceReportMd.toByteArray(Charsets.UTF_8),
      "stage_timeline.csv" to bundle.stageTimelineCsv.toByteArray(Charsets.UTF_8),
      "runtime_events.jsonl" to bundle.runtimeEventsJsonl.toByteArray(Charsets.UTF_8),
      "model_warmup.json" to bundle.modelWarmupJson.toByteArray(Charsets.UTF_8),
      "device_info.json" to bundle.deviceInfoJson.toByteArray(Charsets.UTF_8),
      "model_manifest.json" to bundle.modelManifestJson.toByteArray(Charsets.UTF_8),
    )

    val startupStages = warmup?.stages.orEmpty()
    val recognitionStages = sessions.flatMap { session -> session.stages.map { session to it } }
    val startupTimeline = JSONObject().apply {
      put("status", if (startupStages.isEmpty()) "NOT_CAPTURED" else "CAPTURED")
      put("timeUnit", "ms")
      put("stages", JSONArray().apply { startupStages.forEach { put(it.toJson()) } })
    }
    val recognitionTimeline = JSONObject().apply {
      put("status", if (sessions.isEmpty()) "NOT_CAPTURED" else "CAPTURED")
      put("timeUnit", "ms")
      put("sessions", JSONArray().apply { sessions.forEach { put(it.toJson()) } })
    }
    val runtimeStatus = JSONObject().apply {
      put("status", warmup?.status?.name ?: ModelLoadLifecycleState.NOT_STARTED.name)
      put("allModulesInferenceReady", warmup?.allModulesInferenceReady ?: false)
      put("moduleStates", JSONObject().apply {
        warmup?.moduleStates?.forEach { (module, state) -> put(module, state.name) }
      })
      if (warmup == null) put("captureStatus", "NOT_CAPTURED")
    }
    val onnxStages = startupStages.filter { stage ->
      stage.stageName.contains("session", ignoreCase = true) ||
        stage.stageName.contains("warmup", ignoreCase = true) ||
        stage.stageName.contains("ort_env", ignoreCase = true)
    } + recognitionStages.map { it.second }.filter { stage ->
      stage.stageName.contains("yolox_inference", ignoreCase = true) ||
        stage.stageName.contains("encoder", ignoreCase = true) ||
        stage.stageName.contains("decoder", ignoreCase = true)
    }
    val onnxSessionMetrics = JSONObject().apply {
      put("status", if (onnxStages.isEmpty()) "NOT_CAPTURED" else "CAPTURED")
      put("timeUnit", "ms")
      put("stages", JSONArray().apply { onnxStages.forEach { put(it.toJson()) } })
    }
    val preprocessStages = recognitionStages.map { it.second }.filter { stage ->
      stage.stageName.contains("preprocess", ignoreCase = true) ||
        stage.stageName.startsWith("photo_") || stage.stageName == "exif_normalize"
    }
    val preprocessMetrics = JSONObject().apply {
      put("status", if (preprocessStages.isEmpty()) "NOT_CAPTURED" else "CAPTURED")
      put("timeUnit", "ms")
      put("stages", JSONArray().apply { preprocessStages.forEach { put(it.toJson()) } })
    }
    val memoryProfile = JSONObject().apply {
      put("status", if (resourceSamples.isEmpty()) "NOT_CAPTURED" else "CAPTURED_POINT_SAMPLES")
      put("sampleCount", resourceSamples.size)
      if (resourceSamples.isNotEmpty()) {
        put("observedSampledPssPeakKb", resourceSamples.maxOf { it.pssTotalKb })
        put("observedSampledNativeHeapPeakKb", resourceSamples.maxOf { it.nativeHeapAllocatedKb })
        put("samples", JSONArray().apply { resourceSamples.forEach { put(it.toJson()) } })
      }
      put("encoderInputTensorRetainedAfterEncoder", false)
      put("encoderInputTensorReleaseSemantics", "No photo-cache reference after Encoder call returns; heap reclamation remains GC-managed")
      put("peakDefinition", "Maximum among explicit runtime samples; this is not a continuous device-wide peak")
    }
    val gcSamples = resourceSamples.filter { it.gcCollectionCount >= 0L && it.gcCollectionTimeMs >= 0L }
    val gcMetrics = JSONObject().apply {
      put("status", if (gcSamples.isEmpty()) "NOT_CAPTURED" else "CAPTURED_POINT_SAMPLES")
      put("counterSource", "Debug.getRuntimeStat(art.gc.gc-count/art.gc.gc-time)")
      put("samples", JSONArray().apply { gcSamples.forEach { put(it.toJson()) } })
      if (gcSamples.size >= 2) {
        put("observedCollectionCountDelta", (gcSamples.last().gcCollectionCount - gcSamples.first().gcCollectionCount).coerceAtLeast(0L))
        put("observedCollectionTimeMsDelta", (gcSamples.last().gcCollectionTimeMs - gcSamples.first().gcCollectionTimeMs).coerceAtLeast(0L))
      }
    }
    val threadConfig = JSONObject().apply {
      val threadStages = startupStages.filter { it.metrics.containsKey("intraOpThreads") }
      put("status", if (threadStages.isEmpty()) "NOT_CAPTURED" else "CAPTURED_CURRENT_CONFIGURATION")
      put("cpuThreadMatrixStatus", "DEVICE_THREAD_MATRIX_NOT_RUN")
      put("nnapiComparisonStatus", "NOT_RUN_CPU_DEFAULT_RETAINED")
      put("slimSamEncoderDecoderConcurrencyStatus", "SEQUENTIAL_NO_DEVICE_REPEAT_EVIDENCE")
      put("intraOpConfigurations", JSONArray().apply {
        threadStages.forEach { stage ->
          put(JSONObject().apply {
            put("stageName", stage.stageName)
            put("intraOpThreads", stage.metrics["intraOpThreads"])
          })
        }
      })
    }
    val externalTests = JSONObject().apply {
      put("status", "NOT_CAPTURED_IN_RUNTIME")
      put("detail", "Parity and unit test results are produced by CI test artifacts, not by the app runtime export")
    }
    entries["startup_timeline.json"] = startupTimeline.toString(2).toByteArray(Charsets.UTF_8)
    entries["recognition_timeline.json"] = recognitionTimeline.toString(2).toByteArray(Charsets.UTF_8)
    entries["model_runtime_status.json"] = runtimeStatus.toString(2).toByteArray(Charsets.UTF_8)
    entries["onnx_session_metrics.json"] = onnxSessionMetrics.toString(2).toByteArray(Charsets.UTF_8)
    entries["preprocess_metrics.json"] = preprocessMetrics.toString(2).toByteArray(Charsets.UTF_8)
    entries["memory_profile.json"] = memoryProfile.toString(2).toByteArray(Charsets.UTF_8)
    entries["gc_metrics.json"] = gcMetrics.toString(2).toByteArray(Charsets.UTF_8)
    entries["thread_config.json"] = threadConfig.toString(2).toByteArray(Charsets.UTF_8)
    entries["model_asset_manifest.json"] = bundle.modelManifestJson.toByteArray(Charsets.UTF_8)
    entries["parity_test_results.json"] = externalTests.toString(2).toByteArray(Charsets.UTF_8)
    entries["test_summary.json"] = externalTests.toString(2).toByteArray(Charsets.UTF_8)

    val sumsSb = StringBuilder()
    for ((name, bytes) in entries) {
      sumsSb.append(sha256Hex(bytes)).append("  ").append(name).append('\n')
    }
    entries["SHA256SUMS"] = sumsSb.toString().toByteArray(Charsets.UTF_8)
    entries["SHA256SUMS.txt"] = sumsSb.toString().toByteArray(Charsets.UTF_8)

    val baos = ByteArrayOutputStream()
    val digest = MessageDigest.getInstance("SHA-256")
    ZipOutputStream(DigestOutputStream(baos, digest)).use { zos ->
      for ((name, bytes) in entries) {
        val entry = ZipEntry(name).apply { time = 0L }
        zos.putNextEntry(entry)
        zos.write(bytes)
        zos.closeEntry()
      }
    }
    val zipBytes = baos.toByteArray()
    outputStream.write(zipBytes)
    outputStream.flush()

    DiagnosticZipExportResult(
      fileCount = entries.size,
      totalBytes = zipBytes.size.toLong(),
      zipSha256 = sha256Hex(zipBytes),
      sessionCount = sessions.size,
    )
  }

  private fun writeWarmupLocked(record: GlobalWarmupRecord) {
    rootDir.mkdirs()
    val sanitized = DiagnosticPrivacySanitizer.sanitizeText(record.toJson().toString(2))
    atomicWriteText(warmupFile, sanitized)
  }

  private fun writeSessionLocked(record: PerformanceSessionRecord) {
    rootDir.mkdirs()
    sessionsDir.mkdirs()

    val currentIds = readIndexLocked().toMutableList()
    currentIds.remove(record.sessionId)
    currentIds.add(record.sessionId)

    val safeJson = DiagnosticPrivacySanitizer.sanitizeText(record.toJson().toString(2))
    atomicWriteText(File(sessionsDir, "${record.sessionId}.json"), safeJson)

    while (currentIds.size > maxRetainedSessions) {
      val evictedId = currentIds.removeAt(0)
      runCatching { File(sessionsDir, "$evictedId.json").delete() }
    }
    writeIndexLocked(currentIds)
  }

  private fun readIndexLocked(): List<String> {
    if (!indexFile.exists()) return emptyList()
    return runCatching {
      val arr = JSONArray(indexFile.readText())
      buildList {
        for (i in 0 until arr.length()) {
          val id = arr.optString(i)
          if (id.isNotBlank()) add(id)
        }
      }
    }.getOrDefault(emptyList())
  }

  private fun writeIndexLocked(ids: List<String>) {
    atomicWriteText(indexFile, JSONArray(ids).toString())
  }

  private fun readSessionsLocked(): List<PerformanceSessionRecord> {
    if (!sessionsDir.exists()) return emptyList()
    val orderedIds = readIndexLocked()
    val result = ArrayList<PerformanceSessionRecord>(orderedIds.size)
    val seen = HashSet<String>()

    for (id in orderedIds) {
      val f = File(sessionsDir, "$id.json")
      if (f.exists()) {
        runCatching {
          PerformanceSessionRecord.fromJson(JSONObject(f.readText()))
        }.getOrNull()?.let {
          result.add(it)
          seen.add(id)
        }
      }
    }

    // Also pick up any orphaned session files if index wasn't updated before crash
    sessionsDir.listFiles()?.filter { it.extension == "json" }?.sortedBy { it.lastModified() }?.forEach { f ->
      val id = f.nameWithoutExtension
      if (!seen.contains(id)) {
        runCatching {
          PerformanceSessionRecord.fromJson(JSONObject(f.readText()))
        }.getOrNull()?.let { result.add(it) }
      }
    }
    return result.takeLast(maxRetainedSessions)
  }

  private fun atomicWriteText(target: File, content: String) {
    target.parentFile?.mkdirs()
    val tmp = File(target.parentFile, "${target.name}.tmp.${System.nanoTime()}")
    tmp.writeText(content, Charsets.UTF_8)
    if (target.exists()) target.delete()
    if (!tmp.renameTo(target)) {
      tmp.copyTo(target, overwrite = true)
      tmp.delete()
    }
  }
}
