package com.yujian.fishmasklab.runtime

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.security.MessageDigest
import java.util.zip.ZipInputStream
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class RuntimeDiagnosticsZipExportTest {

  @Test
  fun `diagnostic zip includes required privacy-safe files and verifiable sha256 manifest`() = runBlocking {
    val context = ApplicationProvider.getApplicationContext<Context>()
    val tempRoot = File.createTempFile("fishmask-diagnostics", "dir")
    assertTrue(tempRoot.delete())
    assertTrue(tempRoot.mkdirs())

    try {
      val store = RuntimeDiagnosticsStore(context, rootDir = tempRoot)
      val output = ByteArrayOutputStream()
      val result = store.exportDiagnosticZip(output)
      val entries = linkedMapOf<String, ByteArray>()
      ZipInputStream(ByteArrayInputStream(output.toByteArray())).use { zip ->
        while (true) {
          val entry = zip.nextEntry ?: break
          entries[entry.name] = zip.readBytes()
          zip.closeEntry()
        }
      }

      val required = setOf(
        "startup_timeline.json",
        "recognition_timeline.json",
        "model_runtime_status.json",
        "onnx_session_metrics.json",
        "preprocess_metrics.json",
        "memory_profile.json",
        "gc_metrics.json",
        "thread_config.json",
        "device_info.json",
        "model_asset_manifest.json",
        "parity_test_results.json",
        "test_summary.json",
        "SHA256SUMS",
      )
      assertTrue("missing required entries: ${required - entries.keys}", entries.keys.containsAll(required))
      assertEquals(entries.size, result.fileCount)
      assertNotNull(entries["device_info.json"])
      assertEquals("NOT_CAPTURED", JSONObject(entries.getValue("memory_profile.json").decodeToString()).getString("status"))
      assertEquals("NOT_CAPTURED_IN_RUNTIME", JSONObject(entries.getValue("test_summary.json").decodeToString()).getString("status"))
      assertFalse("diagnostics must not include user photos", entries.keys.any { it.endsWith(".jpg") || it.endsWith(".png") })

      val checksumText = entries.getValue("SHA256SUMS").decodeToString()
      val checksums = checksumText.lineSequence().filter { it.isNotBlank() }.associate { line ->
        val separator = line.indexOf("  ")
        assertTrue("malformed checksum line: $line", separator > 0)
        line.substring(separator + 2) to line.substring(0, separator)
      }
      assertEquals("checksum manifest must cover every non-manifest entry", entries.keys - setOf("SHA256SUMS", "SHA256SUMS.txt"), checksums.keys)
      for ((name, expectedSha) in checksums) {
        val actualSha = MessageDigest.getInstance("SHA-256")
          .digest(entries.getValue(name))
          .joinToString("") { "%02x".format(it) }
        assertEquals("checksum mismatch for $name", expectedSha, actualSha)
      }
    } finally {
      tempRoot.deleteRecursively()
    }
  }
}
