package com.yujian.fishmasklab.runtime

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import com.yujian.fishmasklab.app.ExifPhotoLoader
import com.yujian.fishmasklab.app.FishMaskViewModel
import com.yujian.fishmasklab.app.LoadedFishPhoto
import com.yujian.fishmasklab.domain.ContourPoint
import com.yujian.fishmasklab.domain.ImageCoordinateMapper
import com.yujian.fishmasklab.domain.PipelineState
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import kotlinx.coroutines.withTimeout
import org.junit.After
import org.junit.Before
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class NativeLifecycleTest {

  @Before
  fun setUpMainDispatcher() {
    Dispatchers.setMain(Dispatchers.Default)
  }

  @After
  fun tearDownMainDispatcher() {
    Dispatchers.resetMain()
  }

  private fun loadFixtureViaProductionLoader(resourceName: String): LoadedFishPhoto = runBlocking {
    val stream = javaClass.classLoader!!.getResourceAsStream(resourceName)
      ?: error("Missing test fixture $resourceName")
    val tempFile = File.createTempFile("lifecycle_", "_$resourceName")
    stream.use { input -> tempFile.outputStream().use { output -> input.copyTo(output) } }
    ExifPhotoLoader.loadFromFileAndDeleteTemp(tempFile)
  }

  private fun assertPhaseOrder(log: List<String>, vararg expectedInOrder: ShutdownPhase) {
    var cursor = -1
    for (phase in expectedInOrder) {
      val idx = log.indexOf(phase.name)
      assertTrue("Missing phase ${phase.name} in log=$log", idx >= 0)
      assertTrue("Phase ${phase.name} out of order in log=$log", idx > cursor)
      cursor = idx
    }
  }

  @Test
  fun `viewmodel shutdown waits deterministically for active YOLOX native run before closing sessions and recycling bitmaps`() =
    runBlocking {
      val app = ApplicationProvider.getApplicationContext<Application>()
      val yolox = YoloxFishDetector(app)
      val encoder = SlimSamVisionEncoder()
      val decoder = SlimSamMaskDecoder()
      val segmenter = SlimSamSegmenter(app, encoder = encoder, decoder = decoder)
      val vm = FishMaskViewModel(app, yolox, segmenter)

      val yoloxEntered = CountDownLatch(1)
      val allowYoloxFinish = CountDownLatch(1)
      val yoloxExited = AtomicBoolean(false)

      yolox.nativeRunObserver = { stage ->
        if (stage == "YOLOX_RUN") {
          yoloxEntered.countDown()
          assertTrue(allowYoloxFinish.await(15, TimeUnit.SECONDS))
          yoloxExited.set(true)
        }
      }

      val loaded = loadFixtureViaProductionLoader("preset_fish.jpg")
      vm.loadPhotoWithLoaderForTest { loaded }
      assertTrue("YOLOX native run must start", yoloxEntered.await(25, TimeUnit.SECONDS))

      // Trigger ViewModel destruction while YOLOX OrtSession.run() is paused mid-flight
      val shutdownDeferred = async(Dispatchers.Default) {
        vm.clearForTestAndAwait()
      }

      // Deterministic polling of coordinator phase until SHUTDOWN_REQUESTED is recorded,
      // then verify native sessions & bitmaps are NOT closed/recycled while YOLOX is running.
      withTimeout(5_000L) {
        while (!vm.coordinator.isShutdownRequested()) {
          kotlinx.coroutines.yield()
        }
      }
      Thread.sleep(40L) // give cleanup coroutine chance to reach awaitAllNativeOperations()
      assertFalse("YOLOX must NOT close while OrtSession.run is active", yolox.isClosed())
      assertFalse("Encoder must NOT close before native ops drain", encoder.isClosed())
      assertFalse("Decoder must NOT close before native ops drain", decoder.isClosed())
      assertFalse("Display bitmap must NOT recycle before native ops drain", loaded.displayBitmap.isRecycled)
      assertFalse("Work bitmap must NOT recycle before native ops drain", loaded.workBitmap.isRecycled)
      assertFalse("YOLOX should still be paused", yoloxExited.get())

      // Unblock native run and verify ordered shutdown completion
      allowYoloxFinish.countDown()
      withTimeout(15_000L) { shutdownDeferred.await() }

      assertTrue("YOLOX native run must have exited cleanly", yoloxExited.get())
      assertTrue(yolox.isClosed())
      assertTrue(encoder.isClosed())
      assertTrue(decoder.isClosed())
      assertTrue(segmenter.isClosed())
      assertTrue(loaded.displayBitmap.isRecycled)
      assertTrue(loaded.workBitmap.isRecycled)
      assertTrue(vm.coordinator.isShutdownComplete())

      assertPhaseOrder(
        vm.coordinator.lifecycleLog,
        ShutdownPhase.RUNNING,
        ShutdownPhase.SHUTDOWN_REQUESTED,
        ShutdownPhase.STOP_ACCEPTING_NEW_TASKS,
        ShutdownPhase.CANCEL_ACTIVE_RECOGNITION,
        ShutdownPhase.WAIT_ALL_NATIVE_OPERATIONS,
        ShutdownPhase.WAIT_VIEWMODEL_LOAD_JOB,
        ShutdownPhase.CLEAR_ENCODER_EMBEDDING_CACHE,
        ShutdownPhase.CLOSE_YOLOX_SESSION,
        ShutdownPhase.CLOSE_SLIMSAM_ENCODER_SESSION,
        ShutdownPhase.CLOSE_SLIMSAM_DECODER_SESSION,
        ShutdownPhase.RECYCLE_DISPLAY_BITMAP,
        ShutdownPhase.RECYCLE_WORK_BITMAP,
        ShutdownPhase.SHUTDOWN_COMPLETE,
      )

      // Idempotence & post-shutdown task rejection
      yolox.close()
      encoder.close()
      decoder.close()
      assertEquals(-1L, vm.coordinator.startRecognition(99L, loaded.workBitmap, loaded.displayBitmap, loaded.mapper))
    }

  @Test
  fun `viewmodel shutdown waits deterministically for active SlimSAM Vision Encoder native run`() =
    runBlocking {
      val app = ApplicationProvider.getApplicationContext<Application>()
      val yolox = YoloxFishDetector(app)
      val encoder = SlimSamVisionEncoder()
      val decoder = SlimSamMaskDecoder()
      val segmenter = SlimSamSegmenter(app, encoder = encoder, decoder = decoder)
      val vm = FishMaskViewModel(app, yolox, segmenter)

      val encoderEntered = CountDownLatch(1)
      val allowEncoderFinish = CountDownLatch(1)
      val encoderExited = AtomicBoolean(false)

      encoder.nativeRunObserver = { stage ->
        if (stage == "SLIMSAM_ENCODER_RUN") {
          encoderEntered.countDown()
          assertTrue(allowEncoderFinish.await(15, TimeUnit.SECONDS))
          encoderExited.set(true)
        }
      }

      val loaded = loadFixtureViaProductionLoader("fish_crop.jpg")
      vm.loadPhotoWithLoaderForTest { loaded }
      assertTrue("Encoder native run must start", encoderEntered.await(30, TimeUnit.SECONDS))

      val shutdownDeferred = async(Dispatchers.Default) {
        vm.clearForTestAndAwait()
      }
      withTimeout(5_000L) {
        while (!vm.coordinator.isShutdownRequested()) {
          kotlinx.coroutines.yield()
        }
      }
      Thread.sleep(40L)

      assertFalse("Encoder must NOT close while OrtSession.run is active", encoder.isClosed())
      assertFalse("Display bitmap must NOT recycle while encoder active", loaded.displayBitmap.isRecycled)
      assertFalse(encoderExited.get())

      allowEncoderFinish.countDown()
      withTimeout(20_000L) { shutdownDeferred.await() }

      assertTrue(encoderExited.get())
      assertTrue(encoder.isClosed())
      assertTrue(decoder.isClosed())
      assertTrue(yolox.isClosed())
      assertTrue(loaded.displayBitmap.isRecycled)
      assertTrue(loaded.workBitmap.isRecycled)
    }

  @Test
  fun `viewmodel shutdown waits deterministically for active SlimSAM Mask Decoder during tap refocus`() =
    runBlocking {
      val app = ApplicationProvider.getApplicationContext<Application>()
      val yolox = YoloxFishDetector(app)
      val encoder = SlimSamVisionEncoder()
      val decoder = SlimSamMaskDecoder()
      val segmenter = SlimSamSegmenter(app, encoder = encoder, decoder = decoder)
      val vm = FishMaskViewModel(app, yolox, segmenter)

      val loaded = loadFixtureViaProductionLoader("fish_crop.jpg")
      vm.loadPhotoWithLoaderForTest { loaded }

      val playing = withTimeout(45_000L) {
        vm.coordinator.events
          .filterIsInstance<PipelineProgressEvent.StateChanged>()
          .first { it.state == PipelineState.PLAYING || it.state == PipelineState.ERROR }
      }
      assertEquals("Initial recognition must reach PLAYING", PipelineState.PLAYING, playing.state)
      val det = playing.detection!!

      val decoderEntered = CountDownLatch(1)
      val allowDecoderFinish = CountDownLatch(1)
      val decoderExited = AtomicBoolean(false)

      decoder.nativeRunObserver = { stage ->
        if (stage == "SLIMSAM_DECODER_RUN") {
          decoderEntered.countDown()
          assertTrue(allowDecoderFinish.await(15, TimeUnit.SECONDS))
          decoderExited.set(true)
        }
      }

      val tapX = (det.box.x1 + det.box.x2) / 2f
      val tapY = (det.box.y1 + det.box.y2) / 2f
      val refocusTask = vm.coordinator.startRecognition(
        photoGenerationId = 1L,
        workBitmap = loaded.workBitmap,
        samBitmap = loaded.displayBitmap,
        mapper = loaded.mapper,
        tapPoint = ContourPoint(tapX, tapY),
        existingDetection = det,
      )
      assertTrue(refocusTask > 0L)
      assertTrue("Decoder run must start", decoderEntered.await(15, TimeUnit.SECONDS))

      val shutdownDeferred = async(Dispatchers.Default) {
        vm.clearForTestAndAwait()
      }
      withTimeout(5_000L) {
        while (!vm.coordinator.isShutdownRequested()) {
          kotlinx.coroutines.yield()
        }
      }
      Thread.sleep(40L)

      assertFalse("Decoder must NOT close while OrtSession.run is active", decoder.isClosed())
      assertFalse("Display bitmap must NOT recycle while decoder active", loaded.displayBitmap.isRecycled)
      assertFalse(decoderExited.get())

      allowDecoderFinish.countDown()
      withTimeout(15_000L) { shutdownDeferred.await() }

      assertTrue(decoderExited.get())
      assertTrue(decoder.isClosed())
      assertTrue(encoder.isClosed())
      assertTrue(yolox.isClosed())
      assertTrue(loaded.displayBitmap.isRecycled)
      assertTrue(loaded.workBitmap.isRecycled)
    }

  @Test
  fun `continuous rapid tap refocus serializes decoder execution with zero overlap and reuses cached embedding`() =
    runBlocking {
      val app = ApplicationProvider.getApplicationContext<Application>()
      val yolox = YoloxFishDetector(app)
      val encoder = SlimSamVisionEncoder()
      val decoder = SlimSamMaskDecoder()
      val segmenter = SlimSamSegmenter(app, encoder = encoder, decoder = decoder)
      val coordinator = RecognitionPipelineCoordinator(yolox, segmenter)

      try {
        val loaded = loadFixtureViaProductionLoader("fish_crop.jpg")
        val t1 = coordinator.startRecognition(
          photoGenerationId = 77L,
          workBitmap = loaded.workBitmap,
          samBitmap = loaded.displayBitmap,
          mapper = loaded.mapper,
        )
        val playing1 = withTimeout(45_000L) {
          coordinator.events
            .filterIsInstance<PipelineProgressEvent.StateChanged>()
            .first { it.taskId == t1 && (it.state == PipelineState.PLAYING || it.state == PipelineState.ERROR) }
        }
        assertEquals(PipelineState.PLAYING, playing1.state)
        val det = playing1.detection!!

        val activeDecodes = AtomicInteger(0)
        val maxConcurrentDecodes = AtomicInteger(0)
        decoder.nativeRunObserver = {
          val cur = activeDecodes.incrementAndGet()
          maxConcurrentDecodes.updateAndGet { prev -> maxOf(prev, cur) }
          Thread.sleep(25L)
          activeDecodes.decrementAndGet()
        }

        val cx = (det.box.x1 + det.box.x2) / 2f
        val cy = (det.box.y1 + det.box.y2) / 2f
        var lastTapTask = -1L
        repeat(5) { step ->
          lastTapTask = coordinator.startRecognition(
            photoGenerationId = 77L,
            workBitmap = loaded.workBitmap,
            samBitmap = loaded.displayBitmap,
            mapper = loaded.mapper,
            tapPoint = ContourPoint(cx + step * 2f, cy + step * 2f),
            existingDetection = det,
          )
        }

        val finalEvent = withTimeout(25_000L) {
          coordinator.events
            .filterIsInstance<PipelineProgressEvent.StateChanged>()
            .first { it.taskId == lastTapTask && (it.state == PipelineState.PLAYING || it.state == PipelineState.ERROR) }
        }
        assertEquals(
          "Final rapid refocus failed: ${finalEvent.errorReason} ${finalEvent.diagnosticMessage}",
          PipelineState.PLAYING,
          finalEvent.state,
        )
        assertTrue("Final tap refocus must reuse cached embedding", finalEvent.reusedEncoderEmbedding)
        assertEquals("Decoder calls must never overlap concurrently", 1, maxConcurrentDecodes.get())
      } finally {
        coordinator.shutdownAndAwait()
      }
    }

  @Test
  fun `switching photos mid-flight and cancelAndResetToEmpty recycle superseded bitmaps cleanly`() =
    runBlocking {
      val app = ApplicationProvider.getApplicationContext<Application>()
      val yolox = YoloxFishDetector(app)
      val segmenter = SlimSamSegmenter(app)
      val vm = FishMaskViewModel(app, yolox, segmenter)

      val photo1 = loadFixtureViaProductionLoader("fish_crop.jpg")
      val photo2 = loadFixtureViaProductionLoader("preset_fish.jpg")

      val yoloxEntered = CountDownLatch(1)
      val allowYoloxFinish = CountDownLatch(1)
      yolox.nativeRunObserver = {
        yoloxEntered.countDown()
        allowYoloxFinish.await(15, TimeUnit.SECONDS)
        yolox.nativeRunObserver = null
      }

      vm.loadPhotoWithLoaderForTest { photo1 }
      assertTrue(yoloxEntered.await(25, TimeUnit.SECONDS))

      // Immediately switch to photo2 while photo1 is inside YOLOX
      val gatePhoto2Load = CompletableDeferred<Unit>()
      vm.loadPhotoWithLoaderForTest {
        gatePhoto2Load.await()
        photo2
      }
      assertFalse("photo1 bitmaps must NOT recycle until YOLOX finishes", photo1.displayBitmap.isRecycled)
      allowYoloxFinish.countDown()
      gatePhoto2Load.complete(Unit)

      val photo2Done = withTimeout(45_000L) {
        vm.coordinator.events
          .filterIsInstance<PipelineProgressEvent.StateChanged>()
          .first { it.photoGenerationId == 2L && (it.state == PipelineState.PLAYING || it.state == PipelineState.ERROR) }
      }
      assertEquals(PipelineState.PLAYING, photo2Done.state)
      assertTrue("Superseded photo1 displayBitmap must be recycled", photo1.displayBitmap.isRecycled)
      assertTrue("Superseded photo1 workBitmap must be recycled", photo1.workBitmap.isRecycled)

      vm.cancelAndResetToEmpty()
      assertEquals(PipelineState.EMPTY, vm.uiState.value.pipelineState)
      vm.clearForTestAndAwait()
      assertTrue("photo2 displayBitmap must be recycled on clear", photo2.displayBitmap.isRecycled)
      assertTrue("photo2 workBitmap must be recycled on clear", photo2.workBitmap.isRecycled)
    }
}
