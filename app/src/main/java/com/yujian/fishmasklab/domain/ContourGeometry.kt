package com.yujian.fishmasklab.domain

import kotlin.math.ceil
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

data class ConnectedComponentResult(
  val cleanMask: ByteArray,
  val pixelCount: Int,
  val bbox: PixelBox,
  val startX: Int,
  val startY: Int,
)

/**
 * Pure Kotlin geometry algorithms ported directly from `src/utils/contourExtraction.ts`.
 */
object ContourGeometry {

  private inline fun isForeground(b: Byte): Boolean = (b.toInt() and 0xFF) > 127

  /**
   * Extracts the Largest Connected Component (LCC) from a binary mask using 4-connected BFS.
   */
  fun getLargestConnectedComponent(
    mask: ByteArray,
    width: Int,
    height: Int,
  ): ConnectedComponentResult {
    val total = width * height
    if (total <= 0 || mask.size < total) {
      return ConnectedComponentResult(ByteArray(0), 0, PixelBox(0, 0, 0, 0), -1, -1)
    }

    val visited = ByteArray(total)
    val queue = IntArray(total)
    val componentBuffer = IntArray(total)

    var maxCount = 0
    var bestPixels = IntArray(0)
    var bestMinX = width
    var bestMinY = height
    var bestMaxX = 0
    var bestMaxY = 0
    var bestStartX = -1
    var bestStartY = -1

    for (y in 0 until height) {
      val rowOffset = y * width
      for (x in 0 until width) {
        val idx = rowOffset + x
        if (!isForeground(mask[idx]) || visited[idx].toInt() != 0) continue

        var head = 0
        var tail = 0
        var compSize = 0
        queue[tail++] = idx
        visited[idx] = 1

        var minX = x
        var maxX = x
        var minY = y
        var maxY = y

        while (head < tail) {
          val curr = queue[head++]
          componentBuffer[compSize++] = curr
          val cx = curr % width
          val cy = curr / width

          if (cx < minX) minX = cx
          if (cx > maxX) maxX = cx
          if (cy < minY) minY = cy
          if (cy > maxY) maxY = cy

          if (cx > 0) {
            val n = curr - 1
            if (isForeground(mask[n]) && visited[n].toInt() == 0) {
              visited[n] = 1
              queue[tail++] = n
            }
          }
          if (cx < width - 1) {
            val n = curr + 1
            if (isForeground(mask[n]) && visited[n].toInt() == 0) {
              visited[n] = 1
              queue[tail++] = n
            }
          }
          if (cy > 0) {
            val n = curr - width
            if (isForeground(mask[n]) && visited[n].toInt() == 0) {
              visited[n] = 1
              queue[tail++] = n
            }
          }
          if (cy < height - 1) {
            val n = curr + width
            if (isForeground(mask[n]) && visited[n].toInt() == 0) {
              visited[n] = 1
              queue[tail++] = n
            }
          }
        }

        if (compSize > maxCount) {
          maxCount = compSize
          bestPixels = componentBuffer.copyOf(compSize)
          bestMinX = minX
          bestMinY = minY
          bestMaxX = maxX
          bestMaxY = maxY
          bestStartX = x
          bestStartY = y
        }
      }
    }

    val cleanMask = ByteArray(total)
    for (i in bestPixels.indices) {
      cleanMask[bestPixels[i]] = 0xFF.toByte()
    }

    return ConnectedComponentResult(
      cleanMask = cleanMask,
      pixelCount = maxCount,
      bbox = if (maxCount > 0) PixelBox(bestMinX, bestMinY, bestMaxX, bestMaxY) else PixelBox(0, 0, 0, 0),
      startX = bestStartX,
      startY = bestStartY,
    )
  }

  /**
   * Moore-Neighbor Tracing algorithm to trace the exterior perimeter of the primary mask component.
   */
  fun extractBoundaryPoints(
    mask: ByteArray,
    width: Int,
    height: Int,
    startHintX: Int = -1,
    startHintY: Int = -1,
  ): List<ContourPoint> {
    var startX = startHintX
    var startY = startHintY

    if (
      startX < 0 ||
      startY < 0 ||
      startX >= width ||
      startY >= height ||
      !isForeground(mask[startY * width + startX])
    ) {
      startX = -1
      startY = -1
      outer@ for (y in 0 until height) {
        val rowOff = y * width
        for (x in 0 until width) {
          if (isForeground(mask[rowOff + x])) {
            startX = x
            startY = y
            break@outer
          }
        }
      }
    }

    if (startX == -1) return emptyList()

    val dx = intArrayOf(0, 1, 1, 1, 0, -1, -1, -1)
    val dy = intArrayOf(-1, -1, 0, 1, 1, 1, 0, -1)

    val boundary = ArrayList<ContourPoint>(512)
    var currX = startX
    var currY = startY
    var backtrackDir = 6 // West

    boundary.add(ContourPoint(currX.toFloat(), currY.toFloat()))
    val maxSteps = width * height
    var steps = 0

    while (steps < maxSteps) {
      steps++
      var nextX = -1
      var nextY = -1
      var nextDir = -1

      for (i in 0 until 8) {
        val dir = (backtrackDir + 1 + i) % 8
        val nx = currX + dx[dir]
        val ny = currY + dy[dir]
        if (nx in 0 until width && ny in 0 until height) {
          if (isForeground(mask[ny * width + nx])) {
            nextX = nx
            nextY = ny
            nextDir = dir
            break
          }
        }
      }

      if (nextX == -1) break
      if (nextX == startX && nextY == startY && boundary.size > 2) break

      boundary.add(ContourPoint(nextX.toFloat(), nextY.toFloat()))
      currX = nextX
      currY = nextY
      backtrackDir = (nextDir + 4) % 8
    }

    return boundary
  }

  private fun perpendicularDistance(p: ContourPoint, a: ContourPoint, b: ContourPoint): Float {
    val dx = b.x - a.x
    val dy = b.y - a.y
    val mag = hypot(dx, dy)
    if (mag == 0f) {
      return hypot(p.x - a.x, p.y - a.y)
    }
    val u = ((p.x - a.x) * dx + (p.y - a.y) * dy) / (mag * mag)
    val ix = a.x + u * dx
    val iy = a.y + u * dy
    return hypot(p.x - ix, p.y - iy)
  }

  /**
   * Iterative Ramer-Douglas-Peucker simplification (safe against deep recursion on high-res contours).
   */
  fun simplifyContourRDP(points: List<ContourPoint>, epsilon: Float = 2.2f): List<ContourPoint> {
    if (points.size <= 3) return points

    val keep = BooleanArray(points.size)
    keep[0] = true
    keep[points.lastIndex] = true

    val stack = ArrayDeque<Pair<Int, Int>>()
    stack.addLast(0 to points.lastIndex)

    while (stack.isNotEmpty()) {
      val (start, end) = stack.removeLast()
      if (end - start <= 1) continue

      var maxDist = 0f
      var index = start
      val a = points[start]
      val b = points[end]
      for (i in (start + 1) until end) {
        val d = perpendicularDistance(points[i], a, b)
        if (d > maxDist) {
          maxDist = d
          index = i
        }
      }

      if (maxDist > epsilon) {
        keep[index] = true
        stack.addLast(start to index)
        stack.addLast(index to end)
      }
    }

    val result = ArrayList<ContourPoint>(points.size / 4)
    for (i in points.indices) {
      if (keep[i]) result.add(points[i])
    }
    return result
  }

  /**
   * Closed cyclic weighted Laplacian smoothing filter:
   * P'_i = 0.20 * P_{i-1} + 0.60 * P_i + 0.20 * P_{i+1}
   */
  fun smoothClosedContour(points: List<ContourPoint>, passes: Int = 3): List<ContourPoint> {
    if (points.size < 4 || passes <= 0) return points
    var curr = points
    val n = curr.size
    for (p in 0 until passes) {
      val next = ArrayList<ContourPoint>(n)
      for (i in 0 until n) {
        val prev = curr[(i - 1 + n) % n]
        val self = curr[i]
        val nxt = curr[(i + 1) % n]
        next.add(
          ContourPoint(
            x = 0.20f * prev.x + 0.60f * self.x + 0.20f * nxt.x,
            y = 0.20f * prev.y + 0.60f * self.y + 0.20f * nxt.y,
          ),
        )
      }
      curr = next
    }
    return curr
  }

  /**
   * Converts closed contour vertices to smooth Catmull-Rom Cubic Bézier segments.
   */
  fun pointsToCatmullRomBeziers(
    points: List<ContourPoint>,
  ): Pair<List<CubicBezierSegment>, Float> {
    if (points.size < 3) return emptyList<CubicBezierSegment>() to 0f

    val n = points.size
    val segments = ArrayList<CubicBezierSegment>(n)
    var approxLength = 0f

    for (i in 0 until n) {
      val p0 = points[(i - 1 + n) % n]
      val p1 = points[i]
      val p2 = points[(i + 1) % n]
      val p3 = points[(i + 2) % n]

      val cp1x = p1.x + (p2.x - p0.x) / 6f
      val cp1y = p1.y + (p2.y - p0.y) / 6f
      val cp2x = p2.x - (p3.x - p1.x) / 6f
      val cp2y = p2.y - (p3.y - p1.y) / 6f

      segments.add(CubicBezierSegment(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y))
      approxLength += hypot(p2.x - p1.x, p2.y - p1.y)
    }

    return segments to max(1f, approxLength)
  }

  /**
   * Full contour extraction pipeline matching `extractContourFromMask` in the Web codebase.
   */
  fun extractContourFromMask(
    mask: ByteArray,
    width: Int,
    height: Int,
    simplificationTolerance: Float = 2.2f,
    smoothingPasses: Int = 3,
    glintOverlay: GlintOverlayData? = null,
    startTimeMs: Long = System.currentTimeMillis(),
  ): ContourData? {
    val lcc = getLargestConnectedComponent(mask, width, height)
    if (lcc.pixelCount < 60) return null

    val rawPoints = extractBoundaryPoints(lcc.cleanMask, width, height, lcc.startX, lcc.startY)
    if (rawPoints.size < 3) return null

    val clampedPasses = min(5, max(1, smoothingPasses))
    val smoothedRaw = smoothClosedContour(rawPoints, clampedPasses)
    val simplified = simplifyContourRDP(smoothedRaw, simplificationTolerance)
    val secondPassCount = max(1, min(3, ceil(clampedPasses / 2.0).toInt()))
    val finalSmoothed = smoothClosedContour(simplified, secondPassCount)

    if (finalSmoothed.size < 3) return null

    val (segments, totalLength) = pointsToCatmullRomBeziers(finalSmoothed)
    val elapsed = (System.currentTimeMillis() - startTimeMs).coerceAtLeast(0L)

    return ContourData(
      rawPoints = rawPoints,
      simplifiedPoints = finalSmoothed,
      startPoint = finalSmoothed.first(),
      bezierSegments = segments,
      totalLength = totalLength,
      rawPointCount = rawPoints.size,
      simplifiedPointCount = finalSmoothed.size,
      extractionTimeMs = elapsed,
      internalGlint = glintOverlay,
    )
  }
}
