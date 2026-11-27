package com.yujian.fishmasklab.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.SystemClock
import androidx.exifinterface.media.ExifInterface
import com.yujian.fishmasklab.domain.ImageCoordinateMapper
import java.io.File
import java.io.InputStream
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

data class PhotoLoadStageTiming(
  val stageName: String,
  val startNs: Long,
  val endNs: Long,
) {
  val durationMs: Double
    get() = (endNs - startNs).coerceAtLeast(0L) / 1_000_000.0
}

data class PhotoLoadTelemetry(
  val sourceKind: String,
  val pickerReturnNs: Long,
  val loadStartNs: Long,
  val loadEndNs: Long,
  val totalDurationMs: Double,
  val originalWidth: Int,
  val originalHeight: Int,
  val displayWidth: Int,
  val displayHeight: Int,
  val displayPixelCount: Int,
  val displayBitmapConfig: String,
  val workWidth: Int,
  val workHeight: Int,
  val workPixelCount: Int,
  val workBitmapConfig: String,
  val exifOrientation: Int,
  val stages: List<PhotoLoadStageTiming>,
)

data class LoadedFishPhoto(
  val displayBitmap: Bitmap,
  val workBitmap: Bitmap,
  val mapper: ImageCoordinateMapper,
  val telemetry: PhotoLoadTelemetry? = null,
)

/**
 * Decodes user photos from Photo Picker URIs or CameraX temp files with full EXIF
 * orientation correction, proportional downscaling, separation of display vs inference bitmaps,
 * and monotonic nanosecond stage telemetry (zero URI/path/GPS persistence).
 */
object ExifPhotoLoader {

  private const val MAX_DISPLAY_SIDE = 2048
  private const val MAX_WORK_SIDE = 1024

  suspend fun loadFromUri(
    context: Context,
    uri: Uri,
    pickerReturnNs: Long = SystemClock.elapsedRealtimeNanos(),
  ): LoadedFishPhoto =
    withContext(Dispatchers.IO) {
      val loadStartNs = SystemClock.elapsedRealtimeNanos()
      val stages = ArrayList<PhotoLoadStageTiming>(9)
      stages.add(
        PhotoLoadStageTiming(
          stageName = "photo_picker_uri_received",
          startNs = pickerReturnNs,
          endNs = loadStartNs.coerceAtLeast(pickerReturnNs),
        ),
      )

      val exifStartNs = SystemClock.elapsedRealtimeNanos()
      val orientation = context.contentResolver.openInputStream(uri)?.use { stream ->
        readExifOrientation(stream)
      } ?: ExifInterface.ORIENTATION_NORMAL
      val exifEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(exifStartNs)
      stages.add(PhotoLoadStageTiming("photo_exif_read", exifStartNs, exifEndNs))

      val boundsStartNs = SystemClock.elapsedRealtimeNanos()
      val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
      context.contentResolver.openInputStream(uri)?.use { stream ->
        BitmapFactory.decodeStream(stream, null, bounds)
      }
      val boundsEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(boundsStartNs)
      if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
        throw IllegalArgumentException("无法读取照片尺寸")
      }
      stages.add(PhotoLoadStageTiming("photo_bounds_decode", boundsStartNs, boundsEndNs))

      val decodeStartNs = SystemClock.elapsedRealtimeNanos()
      val decodeOpts = BitmapFactory.Options().apply {
        inSampleSize = computeInSampleSize(bounds.outWidth, bounds.outHeight, MAX_DISPLAY_SIDE)
        inPreferredConfig = Bitmap.Config.ARGB_8888
      }
      val rawBitmap = context.contentResolver.openInputStream(uri)?.use { stream ->
        BitmapFactory.decodeStream(stream, null, decodeOpts)
      } ?: throw IllegalStateException("照片解码失败")
      val decodeEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(decodeStartNs)
      stages.add(PhotoLoadStageTiming("photo_bitmap_decode", decodeStartNs, decodeEndNs))

      buildLoadedPhoto(
        rawBitmap = rawBitmap,
        exifOrientation = orientation,
        originalWidth = bounds.outWidth,
        originalHeight = bounds.outHeight,
        sourceKind = "PHOTO_PICKER",
        pickerReturnNs = pickerReturnNs,
        loadStartNs = loadStartNs,
        stages = stages,
      )
    }

  suspend fun loadFromFileAndDeleteTemp(
    file: File,
    captureReturnNs: Long = SystemClock.elapsedRealtimeNanos(),
  ): LoadedFishPhoto =
    withContext(Dispatchers.IO) {
      val loadStartNs = SystemClock.elapsedRealtimeNanos()
      val stages = ArrayList<PhotoLoadStageTiming>(9)
      stages.add(
        PhotoLoadStageTiming(
          stageName = "photo_picker_uri_received",
          startNs = captureReturnNs,
          endNs = loadStartNs.coerceAtLeast(captureReturnNs),
        ),
      )

      try {
        val exifStartNs = SystemClock.elapsedRealtimeNanos()
        val orientation = file.inputStream().use { stream -> readExifOrientation(stream) }
        val exifEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(exifStartNs)
        stages.add(PhotoLoadStageTiming("photo_exif_read", exifStartNs, exifEndNs))

        val boundsStartNs = SystemClock.elapsedRealtimeNanos()
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        val boundsEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(boundsStartNs)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
          throw IllegalArgumentException("无法读取拍摄照片尺寸")
        }
        stages.add(PhotoLoadStageTiming("photo_bounds_decode", boundsStartNs, boundsEndNs))

        val decodeStartNs = SystemClock.elapsedRealtimeNanos()
        val decodeOpts = BitmapFactory.Options().apply {
          inSampleSize = computeInSampleSize(bounds.outWidth, bounds.outHeight, MAX_DISPLAY_SIDE)
          inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val rawBitmap = BitmapFactory.decodeFile(file.absolutePath, decodeOpts)
          ?: throw IllegalStateException("拍摄照片解码失败")
        val decodeEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(decodeStartNs)
        stages.add(PhotoLoadStageTiming("photo_bitmap_decode", decodeStartNs, decodeEndNs))

        buildLoadedPhoto(
          rawBitmap = rawBitmap,
          exifOrientation = orientation,
          originalWidth = bounds.outWidth,
          originalHeight = bounds.outHeight,
          sourceKind = "CAMERA_CAPTURE",
          pickerReturnNs = captureReturnNs,
          loadStartNs = loadStartNs,
          stages = stages,
        )
      } finally {
        runCatching {
          if (file.exists()) file.delete()
        }
      }
    }

  private fun buildLoadedPhoto(
    rawBitmap: Bitmap,
    exifOrientation: Int,
    originalWidth: Int,
    originalHeight: Int,
    sourceKind: String,
    pickerReturnNs: Long,
    loadStartNs: Long,
    stages: MutableList<PhotoLoadStageTiming>,
  ): LoadedFishPhoto {
    val transformStartNs = SystemClock.elapsedRealtimeNanos()
    val oriented = applyExifOrientation(rawBitmap, exifOrientation)
    if (oriented !== rawBitmap && !rawBitmap.isRecycled) {
      rawBitmap.recycle()
    }
    val transformEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(transformStartNs)
    stages.add(PhotoLoadStageTiming("photo_exif_transform", transformStartNs, transformEndNs))

    val displayScaleStartNs = SystemClock.elapsedRealtimeNanos()
    val displayBitmap = scaleDownIfNeeded(oriented, MAX_DISPLAY_SIDE)
    if (displayBitmap !== oriented && !oriented.isRecycled) {
      oriented.recycle()
    }
    val displayScaleEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(displayScaleStartNs)
    stages.add(PhotoLoadStageTiming("photo_display_scale", displayScaleStartNs, displayScaleEndNs))

    val mapperStartNs = SystemClock.elapsedRealtimeNanos()
    val mapper = ImageCoordinateMapper.create(
      origWidth = displayBitmap.width,
      origHeight = displayBitmap.height,
      maxWorkDimension = MAX_WORK_SIDE,
    )
    val mapperEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(mapperStartNs)
    stages.add(PhotoLoadStageTiming("photo_coordinate_mapper_create", mapperStartNs, mapperEndNs))

    val workBuildStartNs = SystemClock.elapsedRealtimeNanos()
    val workBitmap = if (
      mapper.workWidth == displayBitmap.width &&
      mapper.workHeight == displayBitmap.height
    ) {
      displayBitmap.copy(Bitmap.Config.ARGB_8888, false)
    } else {
      Bitmap.createScaledBitmap(displayBitmap, mapper.workWidth, mapper.workHeight, true)
    }
    val workBuildEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(workBuildStartNs)
    stages.add(PhotoLoadStageTiming("photo_work_bitmap_build", workBuildStartNs, workBuildEndNs))

    val loadEndNs = SystemClock.elapsedRealtimeNanos().coerceAtLeast(loadStartNs)
    stages.add(PhotoLoadStageTiming("photo_load_total", loadStartNs, loadEndNs))

    val telemetry = PhotoLoadTelemetry(
      sourceKind = sourceKind,
      pickerReturnNs = pickerReturnNs,
      loadStartNs = loadStartNs,
      loadEndNs = loadEndNs,
      totalDurationMs = (loadEndNs - loadStartNs).coerceAtLeast(0L) / 1_000_000.0,
      originalWidth = originalWidth,
      originalHeight = originalHeight,
      displayWidth = displayBitmap.width,
      displayHeight = displayBitmap.height,
      displayPixelCount = displayBitmap.width * displayBitmap.height,
      displayBitmapConfig = displayBitmap.config?.name ?: Bitmap.Config.ARGB_8888.name,
      workWidth = workBitmap.width,
      workHeight = workBitmap.height,
      workPixelCount = workBitmap.width * workBitmap.height,
      workBitmapConfig = workBitmap.config?.name ?: Bitmap.Config.ARGB_8888.name,
      exifOrientation = exifOrientation,
      stages = stages,
    )

    return LoadedFishPhoto(
      displayBitmap = displayBitmap,
      workBitmap = workBitmap,
      mapper = mapper,
      telemetry = telemetry,
    )
  }

  private fun readExifOrientation(stream: InputStream): Int {
    return runCatching {
      ExifInterface(stream).getAttributeInt(
        ExifInterface.TAG_ORIENTATION,
        ExifInterface.ORIENTATION_NORMAL,
      )
    }.getOrDefault(ExifInterface.ORIENTATION_NORMAL)
  }

  private fun computeInSampleSize(width: Int, height: Int, maxSide: Int): Int {
    var sample = 1
    val longest = max(width, height)
    while (longest / (sample * 2) >= maxSide) {
      sample *= 2
    }
    return sample
  }

  private fun scaleDownIfNeeded(bitmap: Bitmap, maxSide: Int): Bitmap {
    val longest = max(bitmap.width, bitmap.height)
    if (longest <= maxSide) return bitmap
    val scale = min(1f, maxSide.toFloat() / longest.toFloat())
    val w = max(1, (bitmap.width * scale).roundToInt())
    val h = max(1, (bitmap.height * scale).roundToInt())
    return Bitmap.createScaledBitmap(bitmap, w, h, true)
  }

  private fun applyExifOrientation(bitmap: Bitmap, orientation: Int): Bitmap {
    val matrix = Matrix()
    when (orientation) {
      ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
      ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
      ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
      ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
      ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
      ExifInterface.ORIENTATION_TRANSPOSE -> {
        matrix.postRotate(90f)
        matrix.postScale(-1f, 1f)
      }
      ExifInterface.ORIENTATION_TRANSVERSE -> {
        matrix.postRotate(270f)
        matrix.postScale(-1f, 1f)
      }
      else -> return bitmap
    }
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
  }
}
