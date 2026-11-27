package com.yujian.fishmasklab.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import com.yujian.fishmasklab.domain.ImageCoordinateMapper
import java.io.File
import java.io.InputStream
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

data class LoadedFishPhoto(
  val displayBitmap: Bitmap,
  val workBitmap: Bitmap,
  val mapper: ImageCoordinateMapper,
)

/**
 * Decodes user photos from Photo Picker URIs or CameraX temp files with full EXIF
 * orientation correction, proportional downscaling, and separation of display vs inference bitmaps.
 */
object ExifPhotoLoader {

  private const val MAX_DISPLAY_SIDE = 2048
  private const val MAX_WORK_SIDE = 1024

  suspend fun loadFromUri(context: Context, uri: Uri): LoadedFishPhoto =
    withContext(Dispatchers.IO) {
      val orientation = context.contentResolver.openInputStream(uri)?.use { stream ->
        readExifOrientation(stream)
      } ?: ExifInterface.ORIENTATION_NORMAL

      val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
      context.contentResolver.openInputStream(uri)?.use { stream ->
        BitmapFactory.decodeStream(stream, null, bounds)
      }
      if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
        throw IllegalArgumentException("无法读取照片尺寸")
      }

      val decodeOpts = BitmapFactory.Options().apply {
        inSampleSize = computeInSampleSize(bounds.outWidth, bounds.outHeight, MAX_DISPLAY_SIDE)
        inPreferredConfig = Bitmap.Config.ARGB_8888
      }
      val rawBitmap = context.contentResolver.openInputStream(uri)?.use { stream ->
        BitmapFactory.decodeStream(stream, null, decodeOpts)
      } ?: throw IllegalStateException("照片解码失败")

      buildLoadedPhoto(rawBitmap, orientation)
    }

  suspend fun loadFromFileAndDeleteTemp(file: File): LoadedFishPhoto =
    withContext(Dispatchers.IO) {
      try {
        val orientation = file.inputStream().use { stream -> readExifOrientation(stream) }
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
          throw IllegalArgumentException("无法读取拍摄照片尺寸")
        }
        val decodeOpts = BitmapFactory.Options().apply {
          inSampleSize = computeInSampleSize(bounds.outWidth, bounds.outHeight, MAX_DISPLAY_SIDE)
          inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val rawBitmap = BitmapFactory.decodeFile(file.absolutePath, decodeOpts)
          ?: throw IllegalStateException("拍摄照片解码失败")
        buildLoadedPhoto(rawBitmap, orientation)
      } finally {
        runCatching {
          if (file.exists()) file.delete()
        }
      }
    }

  private fun buildLoadedPhoto(rawBitmap: Bitmap, exifOrientation: Int): LoadedFishPhoto {
    val oriented = applyExifOrientation(rawBitmap, exifOrientation)
    if (oriented !== rawBitmap && !rawBitmap.isRecycled) {
      rawBitmap.recycle()
    }

    val displayBitmap = scaleDownIfNeeded(oriented, MAX_DISPLAY_SIDE)
    if (displayBitmap !== oriented && !oriented.isRecycled) {
      oriented.recycle()
    }

    val mapper = ImageCoordinateMapper.create(
      origWidth = displayBitmap.width,
      origHeight = displayBitmap.height,
      maxWorkDimension = MAX_WORK_SIDE,
    )
    val workBitmap = if (
      mapper.workWidth == displayBitmap.width &&
      mapper.workHeight == displayBitmap.height
    ) {
      displayBitmap.copy(Bitmap.Config.ARGB_8888, false)
    } else {
      Bitmap.createScaledBitmap(displayBitmap, mapper.workWidth, mapper.workHeight, true)
    }

    return LoadedFishPhoto(
      displayBitmap = displayBitmap,
      workBitmap = workBitmap,
      mapper = mapper,
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
