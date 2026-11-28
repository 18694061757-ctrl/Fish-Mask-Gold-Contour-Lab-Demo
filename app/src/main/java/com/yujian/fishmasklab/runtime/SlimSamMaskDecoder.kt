package com.yujian.fishmasklab.runtime

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.os.SystemClock
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

data class SamDecoderOutput(
  val iouScores: FloatArray,      // 3 candidate IoU scores
  val predMasksLogits: FloatArray, // [3 * 256 * 256] raw logits
  val decoderDurationMs: Long,
  val decoderStartNs: Long = 0L,
  val decoderEndNs: Long = 0L,
) {
  companion object {
    const val NUM_CANDIDATES = 3
    const val MASK_LOGIT_SIZE = 256
    const val MASK_PLANE_ELEMENTS = MASK_LOGIT_SIZE * MASK_LOGIT_SIZE
  }
}

/**
 * Local CPU ONNX Runtime session manager for `Xenova/slimsam-77-uniform` Prompt Encoder & Mask Decoder Q8.
 */
class SlimSamMaskDecoder {
  companion object {
    private const val TAG = "SlimSamMaskDecoder"
    const val INPUT_POINTS = "input_points"
    const val INPUT_LABELS = "input_labels"
    const val INPUT_IMAGE_EMBEDDINGS = "image_embeddings"
    const val INPUT_POSITIONAL_EMBEDDINGS = "image_positional_embeddings"
    const val OUTPUT_IOU_SCORES = "iou_scores"
    const val OUTPUT_PRED_MASKS = "pred_masks"
  }

  private val mutex = Mutex()
  private val inferenceMutex = Mutex()
  private val gate = NativeSessionGate("SlimSamMaskDecoder")
  val intraOpThreadCount: Int = Runtime.getRuntime().availableProcessors().coerceIn(2, 4)
  private var session: OrtSession? = null

  @Volatile
  internal var nativeRunObserver: ((String) -> Unit)? = null

  @Volatile
  internal var sessionClosedObserver: ((String) -> Unit)? = null

  @Volatile
  var contractSummary: String? = null
    private set

  fun isLoaded(): Boolean = !gate.isClosingOrClosed() && session != null

  fun isClosed(): Boolean = gate.isClosed()

  fun activeOperationCount(): Int = gate.activeOperationCount()

  suspend fun ensureLoaded(modelFile: File) = withContext(Dispatchers.IO) {
    if (gate.isClosingOrClosed()) throw CancellationException("SlimSamMaskDecoder is closed")
    if (session != null) return@withContext
    mutex.withLock {
      if (gate.isClosingOrClosed()) throw CancellationException("SlimSamMaskDecoder is closed")
      if (session != null) return@withContext
      try {
        val env = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions().apply {
          setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
          setIntraOpNumThreads(intraOpThreadCount)
        }
        val modelBytes = modelFile.readBytes()
        val created = options.use { opts ->
          env.createSession(modelBytes, opts)
        }
        try {
          val summary = validateContract(created)
          runSmokeCheck(env, created)
          val published = gate.publishIfOpen {
            contractSummary = summary
            session = created
          }
          if (!published) {
            runCatching { created.close() }
            throw CancellationException("SlimSamMaskDecoder closed during load")
          }
          Log.i(TAG, "Mask Decoder loaded, contract verified & smoke-checked: $summary")
        } catch (t: Throwable) {
          runCatching { created.close() }
          throw t
        }
      } catch (ce: CancellationException) {
        throw ce
      } catch (oe: OrtException) {
        val code = classifyOrtException(oe, defaultCode = SlimSamErrorCode.SESSION_CREATE_FAILED)
        throw SlimSamRuntimeException(code, "Mask Decoder session init failed: ${oe.message}", oe)
      } catch (sre: SlimSamRuntimeException) {
        throw sre
      } catch (t: Throwable) {
        throw SlimSamRuntimeException(
          SlimSamErrorCode.SESSION_CREATE_FAILED,
          "Mask Decoder init failed: ${t.message ?: t.javaClass.simpleName}",
          t,
        )
      }
    }
  }

  private fun validateContract(s: OrtSession): String {
    val ptsInfo = s.inputInfo[INPUT_POINTS]?.info as? TensorInfo
      ?: throw SlimSamRuntimeException(
        SlimSamErrorCode.TENSOR_CONTRACT_MISMATCH,
        "Missing input '$INPUT_POINTS'. Found: ${s.inputNames}",
      )
    if (ptsInfo.type != OnnxJavaType.FLOAT || ptsInfo.shape.size != 4 || ptsInfo.shape[3] != 2L) {
      throw SlimSamRuntimeException(
        SlimSamErrorCode.TENSOR_CONTRACT_MISMATCH,
        "Invalid '$INPUT_POINTS' contract: type=${ptsInfo.type}, shape=${ptsInfo.shape.contentToString()}",
      )
    }

    val lblInfo = s.inputInfo[INPUT_LABELS]?.info as? TensorInfo
      ?: throw SlimSamRuntimeException(
        SlimSamErrorCode.TENSOR_CONTRACT_MISMATCH,
        "Missing input '$INPUT_LABELS'. Found: ${s.inputNames}",
      )
    if (lblInfo.type != OnnxJavaType.INT64 || lblInfo.shape.size != 3) {
      throw SlimSamRuntimeException(
        SlimSamErrorCode.TENSOR_CONTRACT_MISMATCH,
        "Invalid '$INPUT_LABELS' contract: type=${lblInfo.type}, shape=${lblInfo.shape.contentToString()}",
      )
    }

    val embInfo = s.inputInfo[INPUT_IMAGE_EMBEDDINGS]?.info as? TensorInfo
      ?: throw SlimSamRuntimeException(
        SlimSamErrorCode.TENSOR_CONTRACT_MISMATCH,
        "Missing input '$INPUT_IMAGE_EMBEDDINGS'. Found: ${s.inputNames}",
      )
    if (embInfo.type != OnnxJavaType.FLOAT || embInfo.shape.size != 4 ||
      embInfo.shape[1] != 256L || embInfo.shape[2] != 64L || embInfo.shape[3] != 64L
    ) {
      throw SlimSamRuntimeException(
        SlimSamErrorCode.TENSOR_CONTRACT_MISMATCH,
        "Invalid '$INPUT_IMAGE_EMBEDDINGS' contract: type=${embInfo.type}, shape=${embInfo.shape.contentToString()}",
      )
    }

    val posInfo = s.inputInfo[INPUT_POSITIONAL_EMBEDDINGS]?.info as? TensorInfo
      ?: throw SlimSamRuntimeException(
        SlimSamErrorCode.TENSOR_CONTRACT_MISMATCH,
        "Missing input '$INPUT_POSITIONAL_EMBEDDINGS'. Found: ${s.inputNames}",
      )
    if (posInfo.type != OnnxJavaType.FLOAT || posInfo.shape.size != 4 ||
      posInfo.shape[1] != 256L || posInfo.shape[2] != 64L || posInfo.shape[3] != 64L
    ) {
      throw SlimSamRuntimeException(
        SlimSamErrorCode.TENSOR_CONTRACT_MISMATCH,
        "Invalid '$INPUT_POSITIONAL_EMBEDDINGS' contract: type=${posInfo.type}, shape=${posInfo.shape.contentToString()}",
      )
    }

    val iouInfo = s.outputInfo[OUTPUT_IOU_SCORES]?.info as? TensorInfo
      ?: throw SlimSamRuntimeException(
        SlimSamErrorCode.TENSOR_CONTRACT_MISMATCH,
        "Missing output '$OUTPUT_IOU_SCORES'. Found: ${s.outputNames}",
      )
    if (iouInfo.type != OnnxJavaType.FLOAT || iouInfo.shape.size != 3 || iouInfo.shape[2] != 3L) {
      throw SlimSamRuntimeException(
        SlimSamErrorCode.TENSOR_CONTRACT_MISMATCH,
        "Invalid '$OUTPUT_IOU_SCORES' contract: type=${iouInfo.type}, shape=${iouInfo.shape.contentToString()}",
      )
    }

    val masksInfo = s.outputInfo[OUTPUT_PRED_MASKS]?.info as? TensorInfo
      ?: throw SlimSamRuntimeException(
        SlimSamErrorCode.TENSOR_CONTRACT_MISMATCH,
        "Missing output '$OUTPUT_PRED_MASKS'. Found: ${s.outputNames}",
      )
    if (masksInfo.type != OnnxJavaType.FLOAT || masksInfo.shape.size != 5 ||
      masksInfo.shape[2] != 3L || masksInfo.shape[3] != 256L || masksInfo.shape[4] != 256L
    ) {
      throw SlimSamRuntimeException(
        SlimSamErrorCode.TENSOR_CONTRACT_MISMATCH,
        "Invalid '$OUTPUT_PRED_MASKS' contract: type=${masksInfo.type}, shape=${masksInfo.shape.contentToString()}",
      )
    }

    return "in[$INPUT_POINTS:${ptsInfo.type}${ptsInfo.shape.contentToString()}, " +
      "$INPUT_LABELS:${lblInfo.type}${lblInfo.shape.contentToString()}, " +
      "$INPUT_IMAGE_EMBEDDINGS:${embInfo.shape.contentToString()}, " +
      "$INPUT_POSITIONAL_EMBEDDINGS:${posInfo.shape.contentToString()}] -> " +
      "out[$OUTPUT_IOU_SCORES:${iouInfo.shape.contentToString()}, " +
      "$OUTPUT_PRED_MASKS:${masksInfo.shape.contentToString()}]"
  }

  /**
   * Minimal CPU smoke check verifying quantized decoder ops run cleanly at initialization time.
   */
  private fun runSmokeCheck(env: OrtEnvironment, s: OrtSession) {
    val zeroEmb = FloatArray(SamImageEmbedding.ELEMENT_COUNT)
    val pts = floatArrayOf(512f, 512f)
    val lbls = longArrayOf(1L)
    try {
      executeSession(
        env = env,
        s = s,
        imageEmbeddings = zeroEmb,
        positionalEmbeddings = zeroEmb,
        pointCoords = pts,
        pointLabels = lbls,
        pointCount = 1,
        notifyObserver = false,
      )
    } catch (oe: OrtException) {
      val code = classifyOrtException(oe, defaultCode = SlimSamErrorCode.DECODER_FAILED)
      throw SlimSamRuntimeException(code, "Mask Decoder smoke validation failed: ${oe.message}", oe)
    }
  }

  suspend fun decode(
    embedding: SamImageEmbedding,
    pointCoords: FloatArray,
    pointLabels: LongArray,
    pointCount: Int,
  ): SamDecoderOutput = withContext(Dispatchers.Default) {
    ensureActive()
    require(pointCount > 0 && pointCoords.size == pointCount * 2 && pointLabels.size == pointCount) {
      "Invalid prompt tensor counts: pointCount=$pointCount, coords=${pointCoords.size}, labels=${pointLabels.size}"
    }
    inferenceMutex.withLock {
      ensureActive()
      if (gate.isClosingOrClosed()) {
        throw CancellationException("SlimSamMaskDecoder is closing or closed")
      }
      val env = OrtEnvironment.getEnvironment()
      try {
        gate.withNativeOperation {
          val activeSession = session ?: throw SlimSamRuntimeException(
            SlimSamErrorCode.SESSION_CREATE_FAILED,
            "Mask Decoder session is not loaded",
          )
          executeSession(
            env = env,
            s = activeSession,
            imageEmbeddings = embedding.imageEmbeddings,
            positionalEmbeddings = embedding.imagePositionalEmbeddings,
            pointCoords = pointCoords,
            pointLabels = pointLabels,
            pointCount = pointCount,
            notifyObserver = true,
          )
        }
      } catch (ce: CancellationException) {
        throw ce
      } catch (sre: SlimSamRuntimeException) {
        throw sre
      } catch (oe: OrtException) {
        val code = classifyOrtException(oe, defaultCode = SlimSamErrorCode.DECODER_FAILED)
        throw SlimSamRuntimeException(code, "Mask Decoder execution failed: ${oe.message}", oe)
      } catch (t: Throwable) {
        throw SlimSamRuntimeException(
          SlimSamErrorCode.DECODER_FAILED,
          "Mask Decoder execution failed: ${t.message ?: t.javaClass.simpleName}",
          t,
        )
      }
    }
  }

  private fun executeSession(
    env: OrtEnvironment,
    s: OrtSession,
    imageEmbeddings: FloatArray,
    positionalEmbeddings: FloatArray,
    pointCoords: FloatArray,
    pointLabels: LongArray,
    pointCount: Int,
    notifyObserver: Boolean,
  ): SamDecoderOutput {
    val startNs = SystemClock.elapsedRealtimeNanos()
    val ptsShape = longArrayOf(1L, 1L, pointCount.toLong(), 2L)
    val lblShape = longArrayOf(1L, 1L, pointCount.toLong())
    val embShape = longArrayOf(1L, 256L, 64L, 64L)

    OnnxTensor.createTensor(env, FloatBuffer.wrap(pointCoords), ptsShape).use { ptsTensor ->
      OnnxTensor.createTensor(env, LongBuffer.wrap(pointLabels), lblShape).use { lblTensor ->
        OnnxTensor.createTensor(env, FloatBuffer.wrap(imageEmbeddings), embShape).use { embTensor ->
          OnnxTensor.createTensor(env, FloatBuffer.wrap(positionalEmbeddings), embShape).use { posTensor ->
            val feeds = mapOf(
              INPUT_POINTS to ptsTensor,
              INPUT_LABELS to lblTensor,
              INPUT_IMAGE_EMBEDDINGS to embTensor,
              INPUT_POSITIONAL_EMBEDDINGS to posTensor,
            )
            if (notifyObserver) {
              nativeRunObserver?.invoke("SLIMSAM_DECODER_RUN")
            }
            s.run(feeds).use { results ->
              val iouTensor = results[OUTPUT_IOU_SCORES].orElse(null) as? OnnxTensor
                ?: throw SlimSamRuntimeException(
                  SlimSamErrorCode.DECODER_FAILED,
                  "Missing '$OUTPUT_IOU_SCORES' tensor",
                )
              val masksTensor = results[OUTPUT_PRED_MASKS].orElse(null) as? OnnxTensor
                ?: throw SlimSamRuntimeException(
                  SlimSamErrorCode.DECODER_FAILED,
                  "Missing '$OUTPUT_PRED_MASKS' tensor",
                )

              val ious = FloatArray(SamDecoderOutput.NUM_CANDIDATES)
              val predMasks = FloatArray(
                SamDecoderOutput.NUM_CANDIDATES * SamDecoderOutput.MASK_PLANE_ELEMENTS,
              )
              iouTensor.floatBuffer.get(ious)
              masksTensor.floatBuffer.get(predMasks)

              for (v in ious) {
                if (!v.isFinite()) {
                  throw SlimSamRuntimeException(
                    SlimSamErrorCode.DECODER_FAILED,
                    "Decoder output '$OUTPUT_IOU_SCORES' contains non-finite value: $v",
                  )
                }
              }
              for (v in predMasks) {
                if (!v.isFinite()) {
                  throw SlimSamRuntimeException(
                    SlimSamErrorCode.DECODER_FAILED,
                    "Decoder output '$OUTPUT_PRED_MASKS' contains non-finite value: $v",
                  )
                }
              }

              val endNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(startNs)
              return SamDecoderOutput(
                iouScores = ious,
                predMasksLogits = predMasks,
                decoderDurationMs = ((endNs - startNs) / 1_000_000L).coerceAtLeast(0L),
                decoderStartNs = startNs,
                decoderEndNs = endNs,
              )
            }
          }
        }
      }
    }
  }

  fun close() {
    gate.closeIdempotent {
      runCatching { session?.close() }
      session = null
      contractSummary = null
      sessionClosedObserver?.invoke("CLOSE_SLIMSAM_DECODER_SESSION")
    }
  }
}
