package com.example

import android.content.Context
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.test.core.app.ApplicationProvider
import com.yujian.fishmasklab.runtime.ModelAssetAudit
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class ExampleRobolectricTest {

  @get:Rule
  val composeRule = createAndroidComposeRule<MainActivity>()

  @Test
  fun `app name matches Fish Mask Gold Contour Lab`() {
    val context = ApplicationProvider.getApplicationContext<Context>()
    val appName = context.getString(R.string.app_name)
    assertEquals("Fish Mask Gold Contour Lab", appName)
  }

  @Test
  fun `model asset audit detects bundled yolox and verifies slimsam q8 assets`() {
    val context = ApplicationProvider.getApplicationContext<Context>()
    val report = ModelAssetAudit.runAudit(context.assets)
    assertTrue("YOLOX-Nano ONNX should be bundled and checksum verified", report.isYoloxReady)
    assertTrue("SlimSAM-77 Q8 ONNX and configs should be bundled and checksum verified", report.isSlimSamReady)
    assertTrue("Expected no missing assets: ${report.missingAssets}", report.missingAssets.isEmpty())
    assertTrue("Expected no SHA mismatches: ${report.shaMismatchedAssets}", report.shaMismatchedAssets.isEmpty())
  }

  @Test
  fun `empty home screen shows capture and gallery actions without debug or export UI`() {
    composeRule.onNodeWithText("FISH MASK").assertIsDisplayed()
    composeRule.onNodeWithText("GOLD CONTOUR EXPERIENCE").assertIsDisplayed()
    composeRule.onNodeWithTag("btn_take_photo").assertIsDisplayed()
    composeRule.onNodeWithTag("btn_pick_photo").assertIsDisplayed()
    composeRule.onNodeWithTag("processing_status_pill").assertDoesNotExist()
    composeRule.onNodeWithTag("result_photo_viewport").assertDoesNotExist()
  }
}
