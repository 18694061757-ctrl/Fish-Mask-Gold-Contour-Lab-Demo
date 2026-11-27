package com.yujian.fishmasklab.app

import android.content.Context
import android.view.ViewGroup
import androidx.activity.compose.BackHandler
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.R
import com.example.ui.theme.ChampagneGold
import com.example.ui.theme.DeepLakeBackground
import com.example.ui.theme.DeepLakeSurface
import java.io.File

@Composable
fun CameraCaptureScreen(
  onPhotoCaptured: (File) -> Unit,
  onClose: () -> Unit,
) {
  BackHandler(onBack = onClose)

  val context = LocalContext.current
  val lifecycleOwner = LocalLifecycleOwner.current
  var isCapturing by remember { mutableStateOf(false) }
  var captureError by remember { mutableStateOf<String?>(null) }

  val imageCapture = remember {
    ImageCapture.Builder()
      .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
      .build()
  }

  val previewView = remember {
    PreviewView(context).apply {
      layoutParams = ViewGroup.LayoutParams(
        ViewGroup.LayoutParams.MATCH_PARENT,
        ViewGroup.LayoutParams.MATCH_PARENT,
      )
      scaleType = PreviewView.ScaleType.FIT_CENTER
    }
  }

  DisposableEffect(lifecycleOwner) {
    val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
    val executor = ContextCompat.getMainExecutor(context)
    val listener = Runnable {
      runCatching {
        val cameraProvider = cameraProviderFuture.get()
        val preview = Preview.Builder().build().also {
          it.surfaceProvider = previewView.surfaceProvider
        }
        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(
          lifecycleOwner,
          CameraSelector.DEFAULT_BACK_CAMERA,
          preview,
          imageCapture,
        )
      }.onFailure { err ->
        captureError = err.localizedMessage ?: "相机初始化失败"
      }
    }
    cameraProviderFuture.addListener(listener, executor)

    onDispose {
      runCatching {
        if (cameraProviderFuture.isDone) {
          cameraProviderFuture.get().unbindAll()
        }
      }
    }
  }

  Box(
    modifier = Modifier
      .fillMaxSize()
      .background(Color.Black),
  ) {
    AndroidView(
      factory = { previewView },
      modifier = Modifier.fillMaxSize(),
    )

    // Top bar
    Row(
      modifier = Modifier
        .fillMaxWidth()
        .statusBarsPadding()
        .padding(horizontal = 16.dp, vertical = 12.dp),
      horizontalArrangement = Arrangement.SpaceBetween,
      verticalAlignment = Alignment.CenterVertically,
    ) {
      Surface(
        color = DeepLakeBackground.copy(alpha = 0.68f),
        shape = CircleShape,
      ) {
        IconButton(
          onClick = onClose,
          modifier = Modifier
            .minimumInteractiveComponentSize()
            .testTag("camera_back_button"),
        ) {
          Icon(
            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
            contentDescription = stringResource(R.string.action_cancel),
            tint = ChampagneGold,
          )
        }
      }

      Surface(
        color = DeepLakeBackground.copy(alpha = 0.68f),
        shape = RoundedCornerShape(20.dp),
      ) {
        Text(
          text = stringResource(R.string.camera_capture_hint),
          style = MaterialTheme.typography.bodyMedium,
          color = Color.White,
          modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
        )
      }
    }

    // Bottom shutter controls
    Column(
      modifier = Modifier
        .align(Alignment.BottomCenter)
        .fillMaxWidth()
        .navigationBarsPadding()
        .padding(bottom = 32.dp),
      horizontalAlignment = Alignment.CenterHorizontally,
      verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
      captureError?.let { err ->
        Surface(
          color = DeepLakeSurface.copy(alpha = 0.9f),
          shape = RoundedCornerShape(16.dp),
        ) {
          Text(
            text = err,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.error,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
          )
        }
      }

      Surface(
        onClick = {
          if (!isCapturing) {
            isCapturing = true
            captureError = null
            takePhotoToCache(
              context = context,
              imageCapture = imageCapture,
              onSuccess = { file ->
                isCapturing = false
                onPhotoCaptured(file)
              },
              onFailure = { msg ->
                isCapturing = false
                captureError = msg
              },
            )
          }
        },
        shape = CircleShape,
        color = ChampagneGold,
        border = BorderStroke(3.dp, Color.White.copy(alpha = 0.85f)),
        modifier = Modifier
          .size(76.dp)
          .border(2.dp, DeepLakeBackground.copy(alpha = 0.5f), CircleShape)
          .testTag("camera_shutter_button"),
      ) {
        Box(contentAlignment = Alignment.Center) {
          Icon(
            imageVector = Icons.Default.CameraAlt,
            contentDescription = stringResource(R.string.action_capture),
            tint = DeepLakeBackground,
            modifier = Modifier.size(30.dp),
          )
        }
      }
    }
  }
}

private fun takePhotoToCache(
  context: Context,
  imageCapture: ImageCapture,
  onSuccess: (File) -> Unit,
  onFailure: (String) -> Unit,
) {
  val tempFile = File(context.cacheDir, "fish_capture_${System.currentTimeMillis()}.jpg")
  val outputOptions = ImageCapture.OutputFileOptions.Builder(tempFile).build()
  imageCapture.takePicture(
    outputOptions,
    ContextCompat.getMainExecutor(context),
    object : ImageCapture.OnImageSavedCallback {
      override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
        onSuccess(tempFile)
      }

      override fun onError(exception: ImageCaptureException) {
        runCatching { if (tempFile.exists()) tempFile.delete() }
        onFailure(exception.localizedMessage ?: "拍摄失败")
      }
    },
  )
}
