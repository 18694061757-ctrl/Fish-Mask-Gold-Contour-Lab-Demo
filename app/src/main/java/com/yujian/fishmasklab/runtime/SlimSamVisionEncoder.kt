package com.yujian.fishmasklab.runtime

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

class SlimSamRuntimeException(
  val errorCode: SlimSamErrorCode,
  message: String,
  cause: Throwable? = null,
) : IllegalStateException(message, cause)

data class SamImageEmbedding(
  val imageEmbeddings: FloatArray,
  val imagePositionalEmbeddings: FloatArray,
  val embeddingsShape: LongArray,
  val positionalShape: LongArray,
  val encoderDurationMs: Long,
) {
  companion object {
    const val CHANNELS = 256
    const val GRID_SIZE = 64
    const val ELEMENT_COUNT = CHANNELS * GRID_SIZE * GRID_SIZE // 1,048,576
  }
}

/**
 * Local CPU ONNX Runtime session manager for `Xenova/slimsam-77-uniform` Vision Encoder Q8.
 */
class SlimSamVisionEncoder {
  companion object {
    private const val TAG = "SlimSamVisionEncoder"
    const val INPUT_PIXEL_VALUES = "pixel_values"
    const val OUTPUT_IMAGE_EMBEDDINGS = "image_embeddings"
    const val OUTPUT_POSITIONAL_EMBEDDINGS = "image_positional_embeddings"
    const val INPUT_SIZE = 1024
  }

  private val mutex = Mutex()
  private var session: OrtSession? = null

  @Volatile
  var contractSummary: String? = null
    private set

  fun isLoaded(): Boolean = session != null

  suspend fun ensureLoaded(modelFile: File) = withContext(Dispatchers.IO) {
    if (session != null) return@withContext
    mutex.withLock {
      if (session != null) return@withContext
      try {
        val env = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions().apply {
          setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
          setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(2, 4))
        }
        val created = options.use { opts ->
          env.createSession(modelFile.absolutePath, opts)
        }
        try {
          contractSummary = validateContract(created)
          Log.i(TAG, "Vision Encoder loaded & verified: $contractSummary")
          session = created
        } catch (t: Throwable) {
          runCatching { created.close() }
          throw t
        }
      } catch (ce: CancellationException) {
        throw ce
      } catch (oe: OrtException) {
        val code = classifyOrtException(oe, defaultCode = SlimSamErrorCode.SESSION_CREATE_FAILED)
        throw SlimSamRuntimeException(code, "Vision Encoder session init failed: ${oe.message}", oe)
      } catch (sre: SlimSamRuntimeException) {
        throw sre
      } catch (t: Throwable) {
        throw SlimSamRuntimeException(
          SlimSamErrorCode.SESSION_CREATE_FAILED,
          "Vision Encoder init failed: ${t.message ?: t.javaClass.simpleName}",
          t,
        )
      }
    }
  }

  private fun validateContract(s: OrtSession): String {
    val inInfo = s.inputInfo[INPUT_PIXEL_VALUES]?.info as? TensorInfo
      ?: throw SlimSamRuntimeException(
        SlimSamErrorCode.TENSOR_CONTRACT_MISMATCH,
        "Missing input '$INPUT_PIXEL_VALUES'. Found: ${s.inputNames}",
      )
    if (inInfo.type != OnnxJavaType.FLOAT || inInfo.shape.size != 4 ||
      inInfo.shape[1] != 3L || inInfo.shape[2] != 1024L || inInfo.shape[3] != 1024L
    ) {
      throw SlimSamRuntimeException(
        SlimSamErrorCode.TENSOR_CONTRACT_MISMATCH,
        "Invalid '$INPUT_PIXEL_VALUES' contract: type=${inInfo.type}, shape=${inInfo.shape.contentToString()}",
      )
    }

    val embInfo = s.outputInfo[OUTPUT_IMAGE_EMBEDDINGS]?.info as? TensorInfo
      ?: throw SlimSamRuntimeException(
        SlimSamErrorCode.TENSOR_CONTRACT_MISMATCH,
        "Missing output '$OUTPUT_IMAGE_EMBEDDINGS'. Found: ${s.outputNames}",
      )
    if (embInfo.type != OnnxJavaType.FLOAT || embInfo.shape.size != 4 ||
      embInfo.shape[1] != 256L || embInfo.shape[2] != 64L || embInfo.shape[3] != 64L
    ) {
      throw SlimSamRuntimeException(
        SlimSamErrorCode.TENSOR_CONTRACT_MISMATCH,
        "Invalid '$OUTPUT_IMAGE_EMBEDDINGS' contract: type=${embInfo.type}, shape=${embInfo.shape.contentToString()}",
      )
    }

    val posInfo = s.outputInfo[OUTPUT_POSITIONAL_EMBEDDINGS]?.info as? TensorInfo
      ?: throw SlimSamRuntimeException(
        SlimSamErrorCode.TENSOR_CONTRACT_MISMATCH,
        "Missing output '$OUTPUT_POSITIONAL_EMBEDDINGS'. Found: ${s.outputNames}",
      )
    if (posInfo.type != OnnxJavaType.FLOAT || posInfo.shape.size != 4 ||
      posInfo.shape[1] != 256L || posInfo.shape[2] != 64L || posInfo.shape[3] != 64L
    ) {
      throw SlimSamRuntimeException(
        SlimSamErrorCode.TENSOR_CONTRACT_MISMATCH,
        "Invalid '$OUTPUT_POSITIONAL_EMBEDDINGS' contract: type=${posInfo.type}, shape=${posInfo.shape.contentToString()}",
      )
    }

    return "in[$INPUT_PIXEL_VALUES:${inInfo.type}${inInfo.shape.contentToString()}] -> " +
      "out[$OUTPUT_IMAGE_EMBEDDINGS:${embInfo.shape.contentToString()}, " +
      "$OUTPUT_POSITIONAL_EMBEDDINGS:${posInfo.shape.contentToString()}]"
  }

  suspend fun encode(pixelValues: FloatArray): SamImageEmbedding = withContext(Dispatchers.Default) {
    ensureActive()
    require(pixelValues.size == 3 * INPUT_SIZE * INPUT_SIZE) {
      "Expected ${3 * INPUT_SIZE * INPUT_SIZE} floats for pixel_values, got ${pixelValues.size}"
    }
    val activeSession = session ?: throw SlimSamRuntimeException(
      SlimSamErrorCode.SESSION_CREATE_FAILED,
      "Vision Encoder session is not loaded",
    )
    val env = OrtEnvironment.getEnvironment()
    val start = System.currentTimeMillis()
    val shape = longArrayOf(1L, 3L, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())

    try {
      OnnxTensor.createTensor(env, FloatBuffer.wrap(pixelValues), shape).use { inputTensor ->
        ensureActive()
        activeSession.run(mapOf(INPUT_PIXEL_VALUES to inputTensor)).use { results ->
          ensureActive()
          val embTensor = results[OUTPUT_IMAGE_EMBEDDINGS].orElse(null) as? OnnxTensor
            ?: throw SlimSamRuntimeException(
              SlimSamErrorCode.ENCODER_FAILED,
              "Missing '$OUTPUT_IMAGE_EMBEDDINGS' tensor",
            )
          val posTensor = results[OUTPUT_POSITIONAL_EMBEDDINGS].orElse(null) as? OnnxTensor
            ?: throw SlimSamRuntimeException(
              SlimSamErrorCode.ENCODER_FAILED,
              "Missing '$OUTPUT_POSITIONAL_EMBEDDINGS' tensor",
            )

          val embShape = embTensor.info.shape
          val posShape = posTensor.info.shape
          val embeddings = FloatArray(SamImageEmbedding.ELEMENT_COUNT)
          val positional = FloatArray(SamImageEmbedding.ELEMENT_COUNT)
          embTensor.floatBuffer.get(embeddings)
          posTensor.floatBuffer.get(positional)

          verifyFinite(embeddings, OUTPUT_IMAGE_EMBEDDINGS)
          verifyFinite(positional, OUTPUT_POSITIONAL_EMBEDDINGS)

          SamImageEmbedding(
            imageEmbeddings = embeddings,
            imagePositionalEmbeddings = positional,
            embeddingsShape = embShape,
            positionalShape = posShape,
            encoderDurationMs = (System.currentTimeMillis() - start).coerceAtLeast(0L),
          )
        }
      }
    } catch (ce: CancellationException) {
      throw ce
    } catch (sre: SlimSamRuntimeException) {
      throw sre
    } catch (oe: OrtException) {
      val code = classifyOrtException(oe, defaultCode = SlimSamErrorCode.ENCODER_FAILED)
      throw SlimSamRuntimeException(code, "Vision Encoder execution failed: ${oe.message}", oe)
    } catch (t: Throwable) {
      throw SlimSamRuntimeException(
        SlimSamErrorCode.ENCODER_FAILED,
        "Vision Encoder execution failed: ${t.message ?: t.javaClass.simpleName}",
        t,
      )
    }
  }

  private fun verifyFinite(data: FloatArray, tensorName: String) {
    for (v in data) {
      if (!v.isFinite()) {
        throw SlimSamRuntimeException(
          SlimSamErrorCode.ENCODER_FAILED,
          "Vision Encoder output '$tensorName' contains non-finite value: $v",
        )
      }
    }
  }

  fun close() {
    runCatching { session?.close() }
    session = null
    contractSummary = null
  }
}

internal fun classifyOrtException(oe: OrtException, defaultCode: SlimSamErrorCode): SlimSamErrorCode {
  val msg = oe.message.orEmpty().lowercase()
  return if (msg.contains("unsupported") || msg.contains("not implemented") || msg.contains("opset") || msg.contains("kernel")) {
    SlimSamErrorCode.UNSUPPORTED_OPERATOR
  } else {
    defaultCode
  }
}
