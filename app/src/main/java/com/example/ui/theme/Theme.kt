package com.example.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val FishMaskColorScheme =
  darkColorScheme(
    primary = ChampagneGold,
    onPrimary = DeepLakeBackground,
    primaryContainer = DeepLakeSurfaceElevated,
    onPrimaryContainer = ChampagneGoldBright,
    secondary = CyanAura,
    onSecondary = DeepLakeBackground,
    secondaryContainer = DeepLakeSurface,
    onSecondaryContainer = IceCyanHighlight,
    tertiary = LakeTealAccent,
    onTertiary = TextPrimary,
    background = DeepLakeBackground,
    onBackground = TextPrimary,
    surface = DeepLakeSurface,
    onSurface = TextPrimary,
    surfaceVariant = DeepLakeSurfaceElevated,
    onSurfaceVariant = TextSecondary,
    outline = DeepLakeBorder,
    error = ErrorCoral,
    onError = DeepLakeBackground,
  )

@Composable
fun FishMaskLabTheme(content: @Composable () -> Unit) {
  MaterialTheme(
    colorScheme = FishMaskColorScheme,
    typography = Typography,
    content = content,
  )
}
