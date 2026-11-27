package com.yujian.fishmasklab.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.SwapHoriz
import androidx.compose.material.icons.outlined.ErrorOutline
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.R
import com.example.ui.theme.ChampagneGold
import com.example.ui.theme.ChampagneGoldBright
import com.example.ui.theme.CyanAura
import com.example.ui.theme.DeepLakeBackground
import com.example.ui.theme.DeepLakeBorder
import com.example.ui.theme.DeepLakeSurface
import com.example.ui.theme.DeepLakeSurfaceElevated
import com.example.ui.theme.ErrorCoral
import com.example.ui.theme.IceCyanHighlight
import com.example.ui.theme.LakeTealAccent
import com.example.ui.theme.TextPrimary
import com.example.ui.theme.TextSecondary
import com.yujian.fishmasklab.domain.ImageCoordinateMapper
import com.yujian.fishmasklab.domain.PipelineState
import com.yujian.fishmasklab.domain.RecognitionErrorReason
import com.yujian.fishmasklab.renderer.FishContourGlowCanvas
import com.yujian.fishmasklab.renderer.WaterOverlayEffect02Canvas
import com.yujian.fishmasklab.renderer.rememberAnimationTimelineState

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FishMaskHomeScreen(
  viewModel: FishMaskViewModel = viewModel(),
) {
  val uiState by viewModel.uiState.collectAsStateWithLifecycle()
  val context = LocalContext.current
  val snackbarHostState = remember { SnackbarHostState() }

  val photoPickerLauncher = rememberLauncherForActivityResult(
    contract = ActivityResultContracts.PickVisualMedia(),
  ) { uri ->
    if (uri != null) {
      viewModel.onPhotoPicked(uri)
    }
  }

  val cameraPermissionLauncher = rememberLauncherForActivityResult(
    contract = ActivityResultContracts.RequestPermission(),
  ) { granted ->
    if (granted) {
      viewModel.openCamera()
    } else {
      viewModel.onCameraPermissionDenied()
    }
  }

  val launchCameraFlow = {
    val hasPerm = ContextCompat.checkSelfPermission(
      context,
      Manifest.permission.CAMERA,
    ) == PackageManager.PERMISSION_GRANTED
    if (hasPerm) {
      viewModel.openCamera()
    } else {
      cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
    }
  }

  val launchGalleryFlow = {
    viewModel.dismissSourceSheet()
    photoPickerLauncher.launch(
      PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
    )
  }

  val deniedMsg = stringResource(R.string.camera_permission_denied)
  LaunchedEffect(uiState.cameraPermissionDenied) {
    if (uiState.cameraPermissionDenied) {
      snackbarHostState.showSnackbar(deniedMsg)
      viewModel.dismissPermissionNotice()
    }
  }

  if (uiState.isCameraOpen) {
    CameraCaptureScreen(
      onPhotoCaptured = viewModel::onPhotoCaptured,
      onClose = viewModel::closeCamera,
    )
    return
  }

  // System Back cancels processing or returns from error cleanly
  BackHandler(enabled = uiState.isProcessing || uiState.pipelineState == PipelineState.ERROR) {
    viewModel.cancelAndResetToEmpty()
  }

  Box(
    modifier = Modifier
      .fillMaxSize()
      .background(
        Brush.radialGradient(
          colors = listOf(
            Color(0xFF0B3540),
            DeepLakeBackground,
            Color(0xFF020D11),
          ),
        ),
      ),
  ) {
    Column(
      modifier = Modifier
        .fillMaxSize()
        .statusBarsPadding()
        .navigationBarsPadding(),
      horizontalAlignment = Alignment.CenterHorizontally,
    ) {
      // Top Brand Header
      BrandHeader(
        showSubtitle = uiState.pipelineState == PipelineState.EMPTY,
      )

      // Core Workspace Area
      Box(
        modifier = Modifier
          .weight(1f)
          .fillMaxWidth(),
        contentAlignment = Alignment.Center,
      ) {
        AnimatedContent(
          targetState = uiState.pipelineState == PipelineState.EMPTY,
          transitionSpec = { fadeIn() togetherWith fadeOut() },
          label = "screen_stage_transition",
        ) { isEmpty ->
          if (isEmpty) {
            EmptyHeroSection(
              modifier = Modifier.fillMaxSize(),
            )
          } else {
            PhotoStageSection(
              uiState = uiState,
              onTapViewport = viewModel::onTapViewport,
              onTimelineInitialCompleted = viewModel::onTimelineInitialWindowCompleted,
              onFpsSampled = viewModel::onFpsSampled,
            )
          }
        }
      }

      // Bottom Controls per Page State
      BottomActionArea(
        uiState = uiState,
        onTakePhoto = launchCameraFlow,
        onPickPhoto = launchGalleryFlow,
        onChangePhoto = viewModel::openSourceSheet,
        onReplay = viewModel::onReplayClicked,
        onRetrySelect = viewModel::openSourceSheet,
        onBackHome = viewModel::cancelAndResetToEmpty,
      )
    }

    SnackbarHost(
      hostState = snackbarHostState,
      modifier = Modifier
        .align(Alignment.BottomCenter)
        .navigationBarsPadding()
        .padding(16.dp),
    )

    if (uiState.isSourceSheetVisible) {
      val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
      ModalBottomSheet(
        onDismissRequest = viewModel::dismissSourceSheet,
        sheetState = sheetState,
        containerColor = DeepLakeSurface,
        contentColor = TextPrimary,
      ) {
        Column(
          modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp)
            .padding(bottom = 32.dp),
          verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
          Text(
            text = stringResource(R.string.source_sheet_title),
            style = MaterialTheme.typography.titleMedium,
            color = ChampagneGold,
          )
          Spacer(modifier = Modifier.height(4.dp))
          PrimaryChampagneButton(
            text = stringResource(R.string.action_take_photo),
            icon = Icons.Default.CameraAlt,
            onClick = launchCameraFlow,
            modifier = Modifier
              .fillMaxWidth()
              .testTag("sheet_take_photo"),
          )
          SecondaryTealButton(
            text = stringResource(R.string.action_pick_photo),
            icon = Icons.Default.PhotoLibrary,
            onClick = launchGalleryFlow,
            modifier = Modifier
              .fillMaxWidth()
              .testTag("sheet_pick_photo"),
          )
        }
      }
    }
  }
}

@Composable
private fun BrandHeader(showSubtitle: Boolean) {
  Column(
    modifier = Modifier
      .fillMaxWidth()
      .padding(top = 16.dp, bottom = 8.dp),
    horizontalAlignment = Alignment.CenterHorizontally,
  ) {
    Text(
      text = stringResource(R.string.brand_title),
      style = MaterialTheme.typography.displaySmall.copy(
        fontSize = if (showSubtitle) 26.sp else 20.sp,
        letterSpacing = 5.sp,
      ),
      color = TextPrimary,
    )
    if (showSubtitle) {
      Spacer(modifier = Modifier.height(4.dp))
      Text(
        text = stringResource(R.string.brand_subtitle),
        style = MaterialTheme.typography.labelSmall,
        color = ChampagneGold.copy(alpha = 0.88f),
      )
    }
  }
}

@Composable
private fun EmptyHeroSection(modifier: Modifier = Modifier) {
  Column(
    modifier = modifier
      .widthIn(max = 560.dp)
      .padding(horizontal = 28.dp),
    horizontalAlignment = Alignment.CenterHorizontally,
    verticalArrangement = Arrangement.Center,
  ) {
    // Minimalist fish catch luminous emblem
    FishEmblemArtwork(
      modifier = Modifier
        .size(210.dp)
        .padding(bottom = 12.dp),
    )

    Spacer(modifier = Modifier.height(20.dp))

    Text(
      text = stringResource(R.string.empty_headline),
      style = MaterialTheme.typography.headlineMedium,
      color = TextPrimary,
      textAlign = TextAlign.Center,
    )

    Spacer(modifier = Modifier.height(10.dp))

    Text(
      text = stringResource(R.string.empty_subhead),
      style = MaterialTheme.typography.bodyLarge,
      color = TextSecondary,
      textAlign = TextAlign.Center,
    )
  }
}

@Composable
private fun FishEmblemArtwork(modifier: Modifier = Modifier) {
  val cd = stringResource(R.string.cd_brand_emblem)
  Canvas(modifier = modifier) {
    val w = size.width
    val h = size.height
    val cx = w * 0.5f
    val cy = h * 0.48f

    // Deep lake ambient halo
    drawCircle(
      brush = Brush.radialGradient(
        colors = listOf(
          LakeTealAccent.copy(alpha = 0.30f),
          CyanAura.copy(alpha = 0.10f),
          Color.Transparent,
        ),
        center = Offset(cx, cy),
        radius = w * 0.48f,
      ),
      radius = w * 0.48f,
      center = Offset(cx, cy),
    )

    // Subtle concentric water rings
    drawArc(
      color = ChampagneGold.copy(alpha = 0.32f),
      startAngle = 15f,
      sweepAngle = 150f,
      useCenter = false,
      topLeft = Offset(cx - w * 0.34f, cy + h * 0.16f),
      size = androidx.compose.ui.geometry.Size(w * 0.68f, h * 0.16f),
      style = Stroke(width = 1.6.dp.toPx(), cap = StrokeCap.Round),
    )
    drawArc(
      color = CyanAura.copy(alpha = 0.24f),
      startAngle = 25f,
      sweepAngle = 130f,
      useCenter = false,
      topLeft = Offset(cx - w * 0.24f, cy + h * 0.20f),
      size = androidx.compose.ui.geometry.Size(w * 0.48f, h * 0.10f),
      style = Stroke(width = 1.2.dp.toPx(), cap = StrokeCap.Round),
    )

    // Blue-gold sleek fish silhouette contour
    val fishPath = Path().apply {
      moveTo(cx - w * 0.28f, cy + h * 0.04f)
      cubicTo(
        cx - w * 0.14f,
        cy - h * 0.20f,
        cx + w * 0.14f,
        cy - h * 0.18f,
        cx + w * 0.28f,
        cy - h * 0.02f,
      )
      cubicTo(
        cx + w * 0.14f,
        cy + h * 0.14f,
        cx - w * 0.12f,
        cy + h * 0.15f,
        cx - w * 0.28f,
        cy + h * 0.04f,
      )
      // Tail fin
      moveTo(cx - w * 0.26f, cy + h * 0.03f)
      lineTo(cx - w * 0.36f, cy - h * 0.09f)
      cubicTo(
        cx - w * 0.32f,
        cy + h * 0.02f,
        cx - w * 0.32f,
        cy + h * 0.05f,
        cx - w * 0.36f,
        cy + h * 0.14f,
      )
      close()
    }

    val blueGoldBrush = Brush.linearGradient(
      colors = listOf(
        CyanAura,
        IceCyanHighlight,
        ChampagneGoldBright,
        CyanAura,
      ),
      start = Offset(cx - w * 0.35f, cy - h * 0.20f),
      end = Offset(cx + w * 0.30f, cy + h * 0.18f),
    )

    drawPath(
      path = fishPath,
      color = Color(0xFF063B46).copy(alpha = 0.85f),
      style = Stroke(width = 5.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round),
    )
    drawPath(
      path = fishPath,
      brush = blueGoldBrush,
      style = Stroke(width = 2.4.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round),
    )
    // Eye jewel
    drawCircle(
      color = ChampagneGoldBright,
      radius = 2.8.dp.toPx(),
      center = Offset(cx + w * 0.18f, cy - h * 0.03f),
    )
    val unused = cd
  }
}

@Composable
private fun PhotoStageSection(
  uiState: FishMaskUiState,
  onTapViewport: (Float, Float, Float, Float) -> Unit,
  onTimelineInitialCompleted: () -> Unit,
  onFpsSampled: (Float) -> Unit,
) {
  val bitmap: Bitmap? = uiState.displayBitmap
  val mapper: ImageCoordinateMapper? = uiState.coordinateMapper

  val timeline = rememberAnimationTimelineState(
    isActive = uiState.isResultVisible && uiState.contourData != null,
    replayToken = uiState.replayToken,
    totalWindowMs = uiState.contourConfig.totalTimelineMs,
    onInitialWindowCompleted = onTimelineInitialCompleted,
    onFpsSampled = onFpsSampled,
  )

  BoxWithConstraints(
    modifier = Modifier
      .fillMaxSize()
      .padding(horizontal = 16.dp, vertical = 8.dp),
    contentAlignment = Alignment.Center,
  ) {
    val density = LocalDensity.current
    val viewportWidthPx = with(density) { maxWidth.toPx() }
    val viewportHeightPx = with(density) { maxHeight.toPx() }

    val displayRect = remember(mapper, viewportWidthPx, viewportHeightPx) {
      mapper?.computeDisplayRect(viewportWidthPx, viewportHeightPx)
    }

    Box(
      modifier = Modifier
        .fillMaxSize()
        .testTag("result_photo_viewport")
        .pointerInput(uiState.isResultVisible, mapper, viewportWidthPx, viewportHeightPx) {
          if (uiState.isResultVisible && mapper != null) {
            detectTapGestures { offset ->
              onTapViewport(offset.x, offset.y, viewportWidthPx, viewportHeightPx)
            }
          }
        },
      contentAlignment = Alignment.Center,
    ) {
      if (bitmap != null && !bitmap.isRecycled) {
        val imageBitmap = remember(bitmap) { bitmap.asImageBitmap() }
        Image(
          bitmap = imageBitmap,
          contentDescription = stringResource(R.string.cd_catch_photo),
          modifier = Modifier.fillMaxSize(),
          contentScale = ContentScale.Fit,
        )
      }

      // Effect 02 perimeter water light + Fish Contour Glow V1.3 (only when real verified contour exists)
      if (uiState.isResultVisible && uiState.contourData != null && displayRect != null) {
        WaterOverlayEffect02Canvas(
          elapsedMs = timeline.elapsedMs,
          displayRect = displayRect,
          config = uiState.effect02Config,
          modifier = Modifier.fillMaxSize(),
        )
        FishContourGlowCanvas(
          contourData = uiState.contourData,
          elapsedMs = timeline.elapsedMs,
          displayRect = displayRect,
          config = uiState.contourConfig,
          modifier = Modifier.fillMaxSize(),
        )
      }

      // Page B: Real event-driven processing status pill (no BBox, no debug mask, no percentage)
      if (uiState.isProcessing) {
        ProcessingStatusPill(
          pipelineState = uiState.pipelineState,
          modifier = Modifier
            .align(Alignment.BottomCenter)
            .padding(bottom = 16.dp),
        )
      }

      // Page D: Honest error overlay keeping original photo visible
      if (uiState.pipelineState == PipelineState.ERROR) {
        ErrorStatusCard(
          reason = uiState.errorReason ?: RecognitionErrorReason.RECOGNITION_INCOMPLETE,
          diagnosticDetail = uiState.diagnosticDetail,
          modifier = Modifier
            .align(Alignment.BottomCenter)
            .padding(bottom = 12.dp)
            .widthIn(max = 460.dp),
        )
      }
    }
  }
}

@Composable
private fun ProcessingStatusPill(
  pipelineState: PipelineState,
  modifier: Modifier = Modifier,
) {
  val label = when (pipelineState) {
    PipelineState.PHOTO_SELECTED,
    PipelineState.MODEL_LOADING,
    -> stringResource(R.string.status_preparing_photo)
    PipelineState.DETECTING -> stringResource(R.string.status_finding_fish)
    PipelineState.SEGMENTING -> stringResource(R.string.status_segmenting_edge)
    PipelineState.BUILDING_CONTOUR -> stringResource(R.string.status_building_contour)
    PipelineState.PLAYING -> stringResource(R.string.status_lighting_catch)
    else -> stringResource(R.string.status_preparing_photo)
  }

  Surface(
    modifier = modifier.testTag("processing_status_pill"),
    color = DeepLakeBackground.copy(alpha = 0.82f),
    shape = RoundedCornerShape(28.dp),
    border = BorderStroke(1.dp, ChampagneGold.copy(alpha = 0.35f)),
  ) {
    Row(
      modifier = Modifier.padding(horizontal = 20.dp, vertical = 12.dp),
      verticalAlignment = Alignment.CenterVertically,
      horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
      CircularProgressIndicator(
        modifier = Modifier.size(18.dp),
        color = ChampagneGoldBright,
        strokeWidth = 2.dp,
      )
      Text(
        text = label,
        style = MaterialTheme.typography.titleMedium,
        color = TextPrimary,
      )
    }
  }
}

@Composable
private fun ErrorStatusCard(
  reason: RecognitionErrorReason,
  diagnosticDetail: String?,
  modifier: Modifier = Modifier,
) {
  val headline = when (reason) {
    RecognitionErrorReason.NO_CLEAR_FISH -> stringResource(R.string.error_no_clear_fish)
    RecognitionErrorReason.SEGMENTATION_FAILED -> stringResource(R.string.error_segmentation_failed)
    RecognitionErrorReason.MODEL_INIT_FAILED -> stringResource(R.string.error_model_init_failed)
    RecognitionErrorReason.RECOGNITION_INCOMPLETE -> stringResource(R.string.error_recognition_incomplete)
  }

  Surface(
    modifier = modifier
      .fillMaxWidth()
      .testTag("error_status_card"),
    color = DeepLakeBackground.copy(alpha = 0.90f),
    shape = RoundedCornerShape(20.dp),
    border = BorderStroke(1.dp, ErrorCoral.copy(alpha = 0.45f)),
  ) {
    Row(
      modifier = Modifier.padding(16.dp),
      verticalAlignment = Alignment.Top,
      horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
      Icon(
        imageVector = Icons.Outlined.ErrorOutline,
        contentDescription = headline,
        tint = ErrorCoral,
        modifier = Modifier.size(22.dp),
      )
      Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(
          text = headline,
          style = MaterialTheme.typography.titleMedium,
          color = TextPrimary,
        )
        if (!diagnosticDetail.isNullOrBlank()) {
          Text(
            text = diagnosticDetail,
            style = MaterialTheme.typography.bodyMedium,
            color = TextSecondary,
          )
        }
      }
    }
  }
}

@Composable
private fun BottomActionArea(
  uiState: FishMaskUiState,
  onTakePhoto: () -> Unit,
  onPickPhoto: () -> Unit,
  onChangePhoto: () -> Unit,
  onReplay: () -> Unit,
  onRetrySelect: () -> Unit,
  onBackHome: () -> Unit,
) {
  Box(
    modifier = Modifier
      .fillMaxWidth()
      .widthIn(max = 560.dp)
      .padding(horizontal = 24.dp, vertical = 16.dp),
  ) {
    when {
      uiState.pipelineState == PipelineState.EMPTY -> {
        Column(
          modifier = Modifier.fillMaxWidth(),
          verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
          PrimaryChampagneButton(
            text = stringResource(R.string.action_take_photo),
            icon = Icons.Default.CameraAlt,
            onClick = onTakePhoto,
            modifier = Modifier
              .fillMaxWidth()
              .testTag("btn_take_photo"),
          )
          SecondaryTealButton(
            text = stringResource(R.string.action_pick_photo),
            icon = Icons.Default.PhotoLibrary,
            onClick = onPickPhoto,
            modifier = Modifier
              .fillMaxWidth()
              .testTag("btn_pick_photo"),
          )
        }
      }

      uiState.isProcessing -> {
        Row(
          modifier = Modifier.fillMaxWidth(),
          horizontalArrangement = Arrangement.Center,
        ) {
          TextButton(
            onClick = onBackHome,
            modifier = Modifier
              .minimumInteractiveComponentSize()
              .testTag("btn_cancel_processing"),
          ) {
            Text(
              text = stringResource(R.string.action_cancel),
              style = MaterialTheme.typography.labelLarge,
              color = TextSecondary,
            )
          }
        }
      }

      uiState.isResultVisible -> {
        Row(
          modifier = Modifier.fillMaxWidth(),
          horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
          SecondaryTealButton(
            text = stringResource(R.string.action_change_photo),
            icon = Icons.Default.SwapHoriz,
            onClick = onChangePhoto,
            modifier = Modifier
              .weight(1f)
              .testTag("btn_change_photo"),
          )
          PrimaryChampagneButton(
            text = stringResource(R.string.action_replay),
            icon = Icons.Default.Refresh,
            onClick = onReplay,
            modifier = Modifier
              .weight(1f)
              .testTag("btn_replay"),
          )
        }
      }

      uiState.pipelineState == PipelineState.ERROR -> {
        Row(
          modifier = Modifier.fillMaxWidth(),
          horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
          SecondaryTealButton(
            text = stringResource(R.string.action_back_home),
            icon = null,
            onClick = onBackHome,
            modifier = Modifier
              .weight(1f)
              .testTag("btn_back_home"),
          )
          PrimaryChampagneButton(
            text = stringResource(R.string.action_retry_pick),
            icon = Icons.Default.SwapHoriz,
            onClick = onRetrySelect,
            modifier = Modifier
              .weight(1.3f)
              .testTag("btn_retry_pick"),
          )
        }
      }
    }
  }
}

@Composable
private fun PrimaryChampagneButton(
  text: String,
  icon: androidx.compose.ui.graphics.vector.ImageVector,
  onClick: () -> Unit,
  modifier: Modifier = Modifier,
) {
  Button(
    onClick = onClick,
    modifier = modifier.height(54.dp),
    shape = RoundedCornerShape(16.dp),
    colors = ButtonDefaults.buttonColors(
      containerColor = ChampagneGold,
      contentColor = DeepLakeBackground,
    ),
    contentPadding = PaddingValues(horizontal = 20.dp),
  ) {
    Icon(
      imageVector = icon,
      contentDescription = null,
      modifier = Modifier.size(20.dp),
    )
    Spacer(modifier = Modifier.size(8.dp))
    Text(
      text = text,
      style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.SemiBold),
    )
  }
}

@Composable
private fun SecondaryTealButton(
  text: String,
  icon: androidx.compose.ui.graphics.vector.ImageVector?,
  onClick: () -> Unit,
  modifier: Modifier = Modifier,
) {
  OutlinedButton(
    onClick = onClick,
    modifier = modifier
      .height(54.dp)
      .border(1.dp, DeepLakeBorder, RoundedCornerShape(16.dp)),
    shape = RoundedCornerShape(16.dp),
    colors = ButtonDefaults.outlinedButtonColors(
      containerColor = DeepLakeSurfaceElevated.copy(alpha = 0.72f),
      contentColor = TextPrimary,
    ),
    border = BorderStroke(1.dp, ChampagneGold.copy(alpha = 0.35f)),
    contentPadding = PaddingValues(horizontal = 20.dp),
  ) {
    if (icon != null) {
      Icon(
        imageVector = icon,
        contentDescription = null,
        tint = ChampagneGold,
        modifier = Modifier.size(20.dp),
      )
      Spacer(modifier = Modifier.size(8.dp))
    }
    Text(
      text = text,
      style = MaterialTheme.typography.labelLarge,
    )
  }
}
