package com.yujian.fishmasklab.runtime

import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.coroutines.cancellation.CancellationException

/**
 * Deterministic ordered shutdown phases for native ONNX sessions & Bitmaps.
 */
enum class ShutdownPhase {
  RUNNING,
  SHUTDOWN_REQUESTED,
  STOP_ACCEPTING_NEW_TASKS,
  CANCEL_ACTIVE_RECOGNITION,
  WAIT_ALL_NATIVE_OPERATIONS,
  WAIT_VIEWMODEL_LOAD_JOB,
  CLEAR_ENCODER_EMBEDDING_CACHE,
  CLOSE_YOLOX_SESSION,
  CLOSE_SLIMSAM_ENCODER_SESSION,
  CLOSE_SLIMSAM_DECODER_SESSION,
  RECYCLE_DISPLAY_BITMAP,
  RECYCLE_WORK_BITMAP,
  SHUTDOWN_COMPLETE,
}

/**
 * Read/write concurrency gate preventing `OrtSession.close()` from executing while any
 * `OrtSession.run()` or source `Bitmap` read is in progress, while keeping `close()` idempotent.
 */
internal class NativeSessionGate(private val sessionName: String) {
  private val rwLock = ReentrantReadWriteLock(true)
  private val activeOps = AtomicInteger(0)

  @Volatile
  private var closingOrClosed: Boolean = false

  @Volatile
  private var closed: Boolean = false

  fun isClosingOrClosed(): Boolean = closingOrClosed

  fun isClosed(): Boolean = closed

  fun activeOperationCount(): Int = activeOps.get()

  inline fun <T> withNativeOperation(block: () -> T): T {
    val readLock = rwLock.readLock()
    readLock.lock()
    try {
      if (closingOrClosed) {
        throw CancellationException("$sessionName is closing or closed")
      }
      activeOps.incrementAndGet()
      try {
        return block()
      } finally {
        activeOps.decrementAndGet()
      }
    } finally {
      readLock.unlock()
    }
  }

  inline fun publishIfOpen(block: () -> Unit): Boolean {
    val writeLock = rwLock.writeLock()
    writeLock.lock()
    try {
      if (closingOrClosed) {
        return false
      }
      block()
      return true
    } finally {
      writeLock.unlock()
    }
  }

  fun closeIdempotent(closeAction: () -> Unit): Boolean {
    closingOrClosed = true
    val writeLock = rwLock.writeLock()
    writeLock.lock()
    try {
      if (closed) {
        return false
      }
      closeAction()
      closed = true
      return true
    } finally {
      writeLock.unlock()
    }
  }
}
