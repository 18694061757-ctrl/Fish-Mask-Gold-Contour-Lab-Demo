import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
import { pathToFileURL } from 'url';
import { createHash } from 'crypto';
import { deflateSync } from 'zlib';

const ROOT = process.env.FISHMASK_ROOT_DIR || process.cwd();
const MODELS_DIR = path.join(ROOT, 'app/src/main/assets/models');
const FIXTURES_DIR = path.join(ROOT, 'app/src/test/resources');
const OUT_DIR = process.env.WEB_PARITY_OUT_DIR || path.join(ROOT, 'build/web_parity_out');
fs.mkdirSync(OUT_DIR, { recursive: true });

function tensorBytes(data) {
  return Buffer.from(data.buffer, data.byteOffset || 0, data.byteLength);
}

function writeTensor(outDir, name, data) {
  const outPath = path.join(outDir, name);
  fs.writeFileSync(outPath, tensorBytes(data));
  return outPath;
}

function readFloat32(file) {
  const bytes = fs.readFileSync(file);
  if (bytes.length % 4 !== 0) throw new Error(`Float32 tensor byte length is not divisible by 4: ${file}`);
  const out = new Float32Array(bytes.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = bytes.readFloatLE(i * 4);
  return out;
}

function readInt64(file) {
  const bytes = fs.readFileSync(file);
  if (bytes.length % 8 !== 0) throw new Error(`Int64 tensor byte length is not divisible by 8: ${file}`);
  const out = new BigInt64Array(bytes.length / 8);
  for (let i = 0; i < out.length; i++) out[i] = bytes.readBigInt64LE(i * 8);
  return out;
}

function sha256File(file) {
  return createHash('sha256').update(fs.readFileSync(file)).digest('hex');
}

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngChunk(type, payload) {
  const name = Buffer.from(type, 'ascii');
  const length = Buffer.alloc(4);
  length.writeUInt32BE(payload.length);
  const crcInput = Buffer.concat([name, payload]);
  const checksum = Buffer.alloc(4);
  checksum.writeUInt32BE(crc32(crcInput));
  return Buffer.concat([length, crcInput, checksum]);
}

function writeMaskPng(file, mask, width, height) {
  const scanlines = Buffer.alloc(height * (width + 1));
  for (let y = 0; y < height; y++) {
    const row = y * (width + 1);
    scanlines[row] = 0;
    for (let x = 0; x < width; x++) scanlines[row + 1 + x] = mask[y * width + x] > 127 ? 255 : 0;
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8; // bit depth
  header[9] = 0; // grayscale
  header[10] = 0;
  header[11] = 0;
  header[12] = 0;
  const png = Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    pngChunk('IHDR', header),
    pngChunk('IDAT', deflateSync(scanlines)),
    pngChunk('IEND', Buffer.alloc(0)),
  ]);
  fs.writeFileSync(file, png);
}

const nodeModulesDir = [
  process.env.WEB_PARITY_NODE_MODULES,
  path.join(process.cwd(), 'node_modules'),
  path.join(ROOT, 'node_modules'),
].find((p) => p && fs.existsSync(p));

let ortWasm;
let SamImageProcessor;
let RawImage;
let Tensor;
if (nodeModulesDir) {
  const req = createRequire(path.join(nodeModulesDir, '_resolver.js'));
  ortWasm = req('onnxruntime-web/wasm');
  const tfEntry = path.join(nodeModulesDir, '@huggingface/transformers/dist/transformers.node.mjs');
  const tfMod = fs.existsSync(tfEntry)
    ? await import(pathToFileURL(tfEntry).href)
    : req('@huggingface/transformers');
  ({ SamImageProcessor, RawImage, Tensor } = tfMod);
} else {
  ortWasm = await import('onnxruntime-web/wasm');
  ({ SamImageProcessor, RawImage, Tensor } = await import('@huggingface/transformers'));
}

if (process.env.ORT_WASM_PATHS) {
  ortWasm.env.wasm.wasmPaths = process.env.ORT_WASM_PATHS;
} else if (nodeModulesDir) {
  const localWasmDir = path.join(nodeModulesDir, 'onnxruntime-web/dist/');
  if (fs.existsSync(localWasmDir)) {
    ortWasm.env.wasm.wasmPaths = localWasmDir;
  }
}
ortWasm.env.wasm.numThreads = 1;

function resizeRgbaBilinear(rgba, srcW, srcH, dstW, dstH) {
  const out = new Uint8ClampedArray(dstW * dstH * 4);
  const xScale = srcW / dstW;
  const yScale = srcH / dstH;
  for (let y = 0; y < dstH; y++) {
    const sy = Math.min(srcH - 1, Math.max(0, (y + 0.5) * yScale - 0.5));
    const y0 = Math.floor(sy);
    const y1 = Math.min(srcH - 1, y0 + 1);
    const wy = sy - y0;
    for (let x = 0; x < dstW; x++) {
      const sx = Math.min(srcW - 1, Math.max(0, (x + 0.5) * xScale - 0.5));
      const x0 = Math.floor(sx);
      const x1 = Math.min(srcW - 1, x0 + 1);
      const wx = sx - x0;
      const i00 = (y0 * srcW + x0) * 4;
      const i10 = (y0 * srcW + x1) * 4;
      const i01 = (y1 * srcW + x0) * 4;
      const i11 = (y1 * srcW + x1) * 4;
      const dstIdx = (y * dstW + x) * 4;
      for (let c = 0; c < 4; c++) {
        const v = (rgba[i00 + c] * (1 - wx) + rgba[i10 + c] * wx) * (1 - wy) +
                  (rgba[i01 + c] * (1 - wx) + rgba[i11 + c] * wx) * wy;
        out[dstIdx + c] = Math.round(v);
      }
    }
  }
  return out;
}

async function detectYolox(detSess, rgba, sourceW, sourceH, diagnosticOut = null) {
  const inputSize = 416;
  const scale = Math.min(inputSize / sourceW, inputSize / sourceH);
  const drawW = Math.max(1, Math.floor(sourceW * scale));
  const drawH = Math.max(1, Math.floor(sourceH * scale));
  const resized = resizeRgbaBilinear(rgba, sourceW, sourceH, drawW, drawH);
  const tensorData = new Float32Array(3 * inputSize * inputSize).fill(114.0);
  const plane = inputSize * inputSize;
  for (let y = 0; y < drawH; y++) {
    for (let x = 0; x < drawW; x++) {
      const srcIdx = (y * drawW + x) * 4;
      const dstIdx = y * inputSize + x;
      tensorData[dstIdx] = resized[srcIdx + 2];
      tensorData[plane + dstIdx] = resized[srcIdx + 1];
      tensorData[2 * plane + dstIdx] = resized[srcIdx];
    }
  }
  if (diagnosticOut) {
    writeTensor(diagnosticOut, 'yolox_input_bgr_web.f32', tensorData);
    fs.writeFileSync(
      path.join(diagnosticOut, 'yolox_preprocess.json'),
      JSON.stringify({ sourceWidth: sourceW, sourceHeight: sourceH, resizedWidth: drawW, resizedHeight: drawH, fill: 114, layout: 'NCHW-BGR', shape: [1, 3, inputSize, inputSize] }, null, 2),
    );
  }
  const inputName = detSess.inputNames[0] || 'images';
  const outputName = detSess.outputNames[0] || 'output';
  const outMap = await detSess.run({
    [inputName]: new ortWasm.Tensor('float32', tensorData, [1, 3, inputSize, inputSize]),
  });
  const outTensor = outMap[outputName];
  const data = outTensor.data;
  if (diagnosticOut) {
    writeTensor(diagnosticOut, 'yolox_raw_output_web.f32', data);
    fs.writeFileSync(
      path.join(diagnosticOut, 'yolox_output_shape.json'),
      JSON.stringify({ shape: outTensor.dims, dtype: 'float32' }, null, 2),
    );
  }
  const numRows = outTensor.dims[1];
  const stride = outTensor.dims[2] || 6;
  const raw = [];
  for (let i = 0; i < numRows; i++) {
    const off = i * stride;
    const conf = data[off + 4] * data[off + 5];
    if (conf >= 0.20) {
      const cx = data[off];
      const cy = data[off + 1];
      const w = data[off + 2];
      const h = data[off + 3];
      const x1 = Math.max(0, Math.min(sourceW, Math.round((cx - w * 0.5) / scale)));
      const y1 = Math.max(0, Math.min(sourceH, Math.round((cy - h * 0.5) / scale)));
      const x2 = Math.max(0, Math.min(sourceW, Math.round((cx + w * 0.5) / scale)));
      const y2 = Math.max(0, Math.min(sourceH, Math.round((cy + h * 0.5) / scale)));
      if ((x2 - x1) > 25 && (y2 - y1) > 25) {
        raw.push({
          box: [x1, y1, x2, y2],
          confidence: Number(conf.toFixed(4)),
          width: x2 - x1,
          height: y2 - y1,
          areaRatio: Number(((x2 - x1) * (y2 - y1) / (sourceW * sourceH)).toFixed(4)),
        });
      }
    }
  }
  raw.sort((a, b) => b.confidence - a.confidence);
  return raw[0] || null;
}

function generatePromptsFromBox(box, w, h) {
  const [bx1, by1, bx2, by2] = box;
  const bw = Math.max(20, bx2 - bx1);
  const bh = Math.max(20, by2 - by1);
  const bcx = Math.round((bx1 + bx2) / 2);
  const bcy = Math.round((by1 + by2) / 2);
  let pts;
  if (bh > bw * 1.15) {
    pts = [
      { id: 'box-center', x: bcx, y: Math.round(by1 + bh * 0.56), label: 1 },
      { id: 'box-upper', x: bcx, y: Math.round(by1 + bh * 0.28), label: 1 },
      { id: 'box-mid-up', x: bcx, y: Math.round(by1 + bh * 0.44), label: 1 },
      { id: 'box-mid-down', x: bcx, y: Math.round(by1 + bh * 0.72), label: 1 },
      { id: 'box-lower', x: bcx, y: Math.round(by1 + bh * 0.88), label: 1 },
      { id: 'box-lat-left', x: Math.max(bx1 + 8, Math.round(bcx - bw * 0.22)), y: Math.round(by1 + bh * 0.55), label: 1 },
      { id: 'box-lat-right', x: Math.min(bx2 - 8, Math.round(bcx + bw * 0.22)), y: Math.round(by1 + bh * 0.55), label: 1 },
      { id: 'box-gripper-neg', x: bcx, y: Math.max(0, Math.round(by1 - 10)), label: 0 },
    ];
  } else {
    pts = [
      { id: 'box-center', x: bcx, y: bcy, label: 1 },
      { id: 'box-head', x: Math.round(bx1 + bw * 0.20), y: bcy, label: 1 },
      { id: 'box-mid-left', x: Math.round(bx1 + bw * 0.38), y: bcy, label: 1 },
      { id: 'box-mid-right', x: Math.round(bx1 + bw * 0.62), y: bcy, label: 1 },
      { id: 'box-tail', x: Math.round(bx1 + bw * 0.80), y: bcy, label: 1 },
      { id: 'box-dorsal', x: bcx, y: Math.max(by1 + 5, Math.round(bcy - bh * 0.2)), label: 1 },
      { id: 'box-ventral', x: bcx, y: Math.min(by2 - 5, Math.round(bcy + bh * 0.2)), label: 1 },
    ];
  }
  return pts.map(p => ({ ...p, x: Math.max(0, Math.min(w - 1, p.x)), y: Math.max(0, Math.min(h - 1, p.y)) }));
}

function getLargestConnectedComponent(mask, width, height) {
  const total = width * height;
  const visited = new Uint8Array(total);
  const queue = new Int32Array(total);
  let maxCount = 0;
  let bestPixels = [];
  let bestBbox = [0, 0, 0, 0];
  let bestStartX = -1;
  let bestStartY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (mask[idx] > 127 && !visited[idx]) {
        let head = 0;
        let tail = 0;
        queue[tail++] = idx;
        visited[idx] = 1;
        const comp = [];
        let minX = x, maxX = x, minY = y, maxY = y;
        while (head < tail) {
          const curr = queue[head++];
          comp.push(curr);
          const cx = curr % width;
          const cy = Math.floor(curr / width);
          if (cx < minX) minX = cx;
          if (cx > maxX) maxX = cx;
          if (cy < minY) minY = cy;
          if (cy > maxY) maxY = cy;
          if (cx > 0 && mask[curr - 1] > 127 && !visited[curr - 1]) { visited[curr - 1] = 1; queue[tail++] = curr - 1; }
          if (cx < width - 1 && mask[curr + 1] > 127 && !visited[curr + 1]) { visited[curr + 1] = 1; queue[tail++] = curr + 1; }
          if (cy > 0 && mask[curr - width] > 127 && !visited[curr - width]) { visited[curr - width] = 1; queue[tail++] = curr - width; }
          if (cy < height - 1 && mask[curr + width] > 127 && !visited[curr + width]) { visited[curr + width] = 1; queue[tail++] = curr + width; }
        }
        if (comp.length > maxCount) {
          maxCount = comp.length;
          bestPixels = comp;
          bestBbox = [minX, minY, maxX, maxY];
          bestStartX = x;
          bestStartY = y;
        }
      }
    }
  }
  const cleanMask = new Uint8Array(total);
  for (const idx of bestPixels) cleanMask[idx] = 255;
  return { cleanMask, pixelCount: maxCount, bbox: bestBbox, startX: bestStartX, startY: bestStartY };
}

function summarizeFloatTensor(values) {
  let min = Infinity, max = -Infinity, sum = 0, nearZero = 0, positive = 0, negative = 0;
  for (const value of values) {
    if (value < min) min = value;
    if (value > max) max = value;
    sum += value;
    if (Math.abs(value) < 0.01) nearZero++;
    if (value > 0) positive++;
    if (value < 0) negative++;
  }
  const sorted = Float32Array.from(values).sort();
  const q = (p) => sorted[Math.floor((sorted.length - 1) * p)];
  return {
    elementCount: values.length, min, max, mean: sum / Math.max(1, values.length),
    nearZeroAbsLt0_01Count: nearZero, nearZeroAbsLt0_01Percent: 100 * nearZero / Math.max(1, values.length),
    positiveCount: positive, negativeCount: negative, percentileMethod: 'floor_index_on_sorted_float32',
    p1: q(0.01), p5: q(0.05), p50: q(0.5), p95: q(0.95), p99: q(0.99),
  };
}

function basicFloatStats(values) {
  let min = Infinity, max = -Infinity, sum = 0;
  for (const value of values) {
    if (value < min) min = value;
    if (value > max) max = value;
    sum += value;
  }
  return { elementCount: values.length, min, max, mean: sum / Math.max(1, values.length) };
}

function binaryMaskMetrics(a, b) {
  if (a.length !== b.length) throw new Error(`Mask sizes differ: ${a.length} vs ${b.length}`);
  let intersection = 0, union = 0, differing = 0;
  for (let i = 0; i < a.length; i++) {
    const av = a[i] > 127, bv = b[i] > 127;
    if (av && bv) intersection++;
    if (av || bv) union++;
    if (av !== bv) differing++;
  }
  return { iou: union === 0 ? 1 : intersection / union, pixelDiffPercent: 100 * differing / Math.max(1, a.length) };
}

function writeMaskStage(stageDir, stageName, mask, width, height, includePng = true) {
  fs.writeFileSync(path.join(stageDir, `${stageName}.bin`), Buffer.from(mask));
  if (includePng) writeMaskPng(path.join(stageDir, `${stageName}.png`), mask, width, height);
}

function writeDecoderOutputFiles(tensorOut, variant, iouScores, logits, outputShape) {
  const decoderDir = path.join(tensorOut, 'decoder', variant);
  fs.mkdirSync(decoderDir, { recursive: true });
  const plane = 256 * 256;
  const candidateLogits = [];
  for (let candidateIndex = 0; candidateIndex < 3; candidateIndex++) {
    const values = logits.subarray(candidateIndex * plane, (candidateIndex + 1) * plane);
    writeTensor(decoderDir, `candidate_${candidateIndex}_logits.f32`, values);
    candidateLogits.push({ candidateIndex, logits: summarizeFloatTensor(values) });
  }
  fs.writeFileSync(path.join(decoderDir, 'iou_scores.json'), JSON.stringify({ shape: [1, 3], values: Array.from(iouScores) }, null, 2));
  fs.writeFileSync(path.join(decoderDir, 'output_metadata.json'), JSON.stringify({
    predMasksShape: outputShape, predMasksDtype: 'float32', iouScoresShape: [1, 3], iouScoresDtype: 'float32', candidateLogits,
  }, null, 2));
}

function emulateKotlinPostprocessStages(logits, resizedWidth, resizedHeight, width, height, detectionBox, stageRoot, includeMaskPng = true) {
  const f = Math.fround, side = 1024, sourceSide = 256;
  const candidateSummaries = [];
  for (let candidateIndex = 0; candidateIndex < 3; candidateIndex++) {
    const stageDir = path.join(stageRoot, `candidate_${candidateIndex}`);
    fs.mkdirSync(stageDir, { recursive: true });
    const plane = sourceSide * sourceSide;
    const src = logits.subarray(candidateIndex * plane, (candidateIndex + 1) * plane);
    writeTensor(stageDir, 'logits_256.f32', src);
    fs.writeFileSync(path.join(stageDir, 'logits_256.bin'), tensorBytes(src));

    const upsampled = new Float32Array(side * side);
    const scale = f(sourceSide / side);
    const x0 = new Int32Array(side), x1 = new Int32Array(side), wx = new Float32Array(side), ix = new Float32Array(side);
    const y0 = new Int32Array(side), y1 = new Int32Array(side), wy = new Float32Array(side), iy = new Float32Array(side);
    for (let x = 0; x < side; x++) {
      const sx = f(Math.max(0, Math.min(sourceSide - 1, f(f(f(x + 0.5) * scale) - 0.5))));
      x0[x] = Math.floor(sx); x1[x] = Math.min(sourceSide - 1, x0[x] + 1); wx[x] = f(sx - x0[x]); ix[x] = f(1 - wx[x]);
    }
    for (let y = 0; y < side; y++) {
      const sy = f(Math.max(0, Math.min(sourceSide - 1, f(f(f(y + 0.5) * scale) - 0.5))));
      y0[y] = Math.floor(sy); y1[y] = Math.min(sourceSide - 1, y0[y] + 1); wy[y] = f(sy - y0[y]); iy[y] = f(1 - wy[y]);
    }
    for (let y = 0; y < side; y++) {
      const row0 = y0[y] * sourceSide, row1 = y1[y] * sourceSide, rowOut = y * side;
      for (let x = 0; x < side; x++) {
        const top = f(f(src[row0 + x0[x]] * ix[x]) + f(src[row0 + x1[x]] * wx[x]));
        const bottom = f(f(src[row1 + x0[x]] * ix[x]) + f(src[row1 + x1[x]] * wx[x]));
        upsampled[rowOut + x] = f(f(top * iy[y]) + f(bottom * wy[y]));
      }
    }
    writeTensor(stageDir, 'upsampled_1024.f32', upsampled);

    const cropped = new Float32Array(resizedWidth * resizedHeight);
    for (let y = 0; y < resizedHeight; y++) cropped.set(upsampled.subarray(y * side, y * side + resizedWidth), y * resizedWidth);
    writeTensor(stageDir, 'padding_cropped.f32', cropped);

    const resized = new Float32Array(width * height);
    const sxScale = f(resizedWidth / width), syScale = f(resizedHeight / height);
    const xx0 = new Int32Array(width), xx1 = new Int32Array(width), xwx = new Float32Array(width), xix = new Float32Array(width);
    const yy0 = new Int32Array(height), yy1 = new Int32Array(height), ywy = new Float32Array(height), yiy = new Float32Array(height);
    for (let x = 0; x < width; x++) {
      const sx = f(Math.max(0, Math.min(resizedWidth - 1, f(f(f(x + 0.5) * sxScale) - 0.5))));
      xx0[x] = Math.floor(sx); xx1[x] = Math.min(resizedWidth - 1, xx0[x] + 1); xwx[x] = f(sx - xx0[x]); xix[x] = f(1 - xwx[x]);
    }
    for (let y = 0; y < height; y++) {
      const sy = f(Math.max(0, Math.min(resizedHeight - 1, f(f(f(y + 0.5) * syScale) - 0.5))));
      yy0[y] = Math.floor(sy); yy1[y] = Math.min(resizedHeight - 1, yy0[y] + 1); ywy[y] = f(sy - yy0[y]); yiy[y] = f(1 - ywy[y]);
    }
    for (let y = 0; y < height; y++) {
      const row0 = yy0[y] * resizedWidth, row1 = yy1[y] * resizedWidth, rowOut = y * width;
      for (let x = 0; x < width; x++) {
        const top = f(f(cropped[row0 + xx0[x]] * xix[x]) + f(cropped[row0 + xx1[x]] * xwx[x]));
        const bottom = f(f(cropped[row1 + xx0[x]] * xix[x]) + f(cropped[row1 + xx1[x]] * xwx[x]));
        resized[rowOut + x] = f(f(top * yiy[y]) + f(bottom * ywy[y]));
      }
    }
    writeTensor(stageDir, 'resized_original.f32', resized);
    const threshold = new Uint8Array(width * height);
    for (let i = 0; i < threshold.length; i++) if (resized[i] > 0) threshold[i] = 255;
    writeMaskStage(stageDir, 'threshold_mask', threshold, width, height, includeMaskPng);

    const [bx1, by1, bx2, by2] = detectionBox;
    const padX = Math.round(f(f(bx2 - bx1) * f(0.45))), padY = Math.round(f(f(by2 - by1) * f(0.45)));
    const cx1 = Math.max(0, bx1 - padX), cy1 = Math.max(0, by1 - padY);
    const cx2 = Math.min(width, bx2 + padX), cy2 = Math.min(height, by2 + padY);
    const clipped = threshold.slice();
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      if (y < cy1 || y > cy2 || x < cx1 || x > cx2) clipped[y * width + x] = 0;
    }
    writeMaskStage(stageDir, 'clipped_mask', clipped, width, height, includeMaskPng);
    const lcc = getLargestConnectedComponent(clipped, width, height);
    writeMaskStage(stageDir, 'lcc_mask', lcc.cleanMask, width, height, includeMaskPng);
    writeMaskStage(stageDir, 'final_mask', lcc.cleanMask, width, height, includeMaskPng);
    candidateSummaries.push({ candidateIndex, logitStats: summarizeFloatTensor(src), thresholdPixels: threshold.reduce((n, v) => n + (v > 127), 0), clippedPixels: clipped.reduce((n, v) => n + (v > 127), 0), lccPixels: lcc.pixelCount, lccBbox: lcc.bbox });
  }
  fs.writeFileSync(path.join(stageRoot, 'stage_summary.json'), JSON.stringify({
    resizedWidth, resizedHeight, originalWidth: width, originalHeight: height,
    upsampledShape: [1, 1, 1024, 1024], croppedShape: [1, 1, resizedHeight, resizedWidth],
    resizedOriginalShape: [1, 1, height, width], candidates: candidateSummaries,
  }, null, 2));
  return candidateSummaries;
}

function extractBoundaryPoints(mask, width, height, hintX, hintY) {
  const dx = [0, 1, 1, 1, 0, -1, -1, -1];
  const dy = [-1, -1, 0, 1, 1, 1, 0, -1];
  const startX = hintX;
  const startY = hintY;
  const pts = [{ x: startX, y: startY }];
  let cx = startX, cy = startY, backtrackDir = 6;
  const maxSteps = Math.max(width * height * 2, 16);
  let fx = -1, fy = -1, fdir = -1, steps = 0;
  while (steps < maxSteps) {
    steps++;
    let nx = -1, ny = -1, ndir = -1;
    for (let i = 0; i < 8; i++) {
      const dir = (backtrackDir + 1 + i) % 8;
      const tx = cx + dx[dir];
      const ty = cy + dy[dir];
      if (tx >= 0 && tx < width && ty >= 0 && ty < height && mask[ty * width + tx] > 127) {
        nx = tx; ny = ty; ndir = dir;
        break;
      }
    }
    if (nx === -1) break;
    if (steps === 1) {
      fx = nx; fy = ny; fdir = ndir;
    } else if (cx === startX && cy === startY && nx === fx && ny === fy && ndir === fdir) {
      if (pts.length > 1 && pts[pts.length - 1].x === startX && pts[pts.length - 1].y === startY) pts.pop();
      break;
    }
    pts.push({ x: nx, y: ny });
    cx = nx; cy = ny; backtrackDir = (ndir + 4) % 8;
  }
  return pts;
}

async function scoreCandidatesAndSave(photoOut, prefix, decOut, postMasks, detection, width, height) {
  const ious = Array.from(decOut.iou_scores.data);
  const rawMaskData = postMasks[0].data;
  const totalPixels = width * height;
  const padX = Math.round((detection.box[2] - detection.box[0]) * 0.45);
  const padY = Math.round((detection.box[3] - detection.box[1]) * 0.45);
  const bx1 = Math.max(0, detection.box[0] - padX);
  const by1 = Math.max(0, detection.box[1] - padY);
  const bx2 = Math.min(width, detection.box[2] + padX);
  const by2 = Math.min(height, detection.box[3] + padY);
  const targetW = detection.width;
  const targetH = detection.height;
  const targetArea = targetW * targetH;

  const candidates = [];
  for (let c = 0; c < 3; c++) {
    const offset = c * totalPixels;
    const binaryMask = new Uint8Array(totalPixels);
    for (let i = 0; i < totalPixels; i++) {
      if (rawMaskData[offset + i] > 0) {
        const px = i % width;
        const py = Math.floor(i / width);
        if (px >= bx1 && px <= bx2 && py >= by1 && py <= by2) binaryMask[i] = 255;
      }
    }
    const lcc = getLargestConnectedComponent(binaryMask, width, height);
    fs.writeFileSync(path.join(photoOut, `${prefix}candidate_${c}.bin`), Buffer.from(lcc.cleanMask));
    const candidateContour = extractBoundaryPoints(lcc.cleanMask, width, height, lcc.startX, lcc.startY);
    fs.writeFileSync(
      path.join(photoOut, `${prefix}contour_candidate_${c}.json`),
      JSON.stringify({ rawPoints: candidateContour }),
    );
    const iou = Number(ious[c]);
    const covRatio = lcc.pixelCount / Math.max(1, targetArea);
    const isVert = targetH > targetW * 1.1;
    const majorSpan = isVert
      ? (lcc.bbox[3] - lcc.bbox[1]) / Math.max(1, targetH)
      : (lcc.bbox[2] - lcc.bbox[0]) / Math.max(1, targetW);
    const isTrustworthy = covRatio >= 0.18 && covRatio <= 0.85 && majorSpan >= 0.48 && iou >= 0.60;
    const aCov = Math.min(lcc.pixelCount / Math.max(1, targetArea * 0.35), 1.0);
    const aSpan = Math.min((lcc.bbox[3] - lcc.bbox[1]) / Math.max(1, targetH * 0.7), 1.0) *
      Math.min((lcc.bbox[2] - lcc.bbox[0]) / Math.max(1, targetW * 0.7), 1.0);
    const samServiceScore = iou * 0.4 + aCov * 0.3 + aSpan * 0.3;
    const sizeScore = Math.min(covRatio / 0.35, 1.0) * Math.min(1.8 / Math.max(0.01, covRatio), 1.0);
    const appScore = iou * 0.5 + sizeScore * 0.5;
    candidates.push({
      index: c,
      iouScore: iou,
      pixelCount: lcc.pixelCount,
      bbox: lcc.bbox,
      covRatio,
      majorSpan,
      isTrustworthy,
      samServiceScore,
      appScore,
    });
  }
  const samBest = [...candidates].sort((a, b) => b.samServiceScore - a.samServiceScore)[0];
  const appSelected = [...candidates].sort((a, b) => b.appScore - a.appScore)[0];
  const trustworthy = candidates.filter((candidate) => candidate.pixelCount > 0 && candidate.isTrustworthy);
  const androidCompatibleSelected = (trustworthy.length > 0
    ? [...trustworthy].sort((a, b) => b.samServiceScore - a.samServiceScore)
    : [...candidates].filter((candidate) => candidate.pixelCount > 0).sort((a, b) => b.samServiceScore - a.samServiceScore))[0];
  const primaryIdx = prefix === 'bil_' ? samBest.index : appSelected.index;
  const lccPrimary = getLargestConnectedComponent(
    fs.readFileSync(path.join(photoOut, `${prefix}candidate_${primaryIdx}.bin`)),
    width,
    height,
  );
  const rawPts = extractBoundaryPoints(lccPrimary.cleanMask, width, height, lccPrimary.startX, lccPrimary.startY);
  fs.writeFileSync(path.join(photoOut, `${prefix}contour_app.json`), JSON.stringify({ rawPoints: rawPts }));
  return {
    candidates,
    samServiceBestIndex: samBest.index,
    appSelectedIndex: appSelected.index,
    androidCompatibleSelectedIndex: androidCompatibleSelected?.index ?? 0,
    androidCompatibleUsedTrustworthyGate: trustworthy.length > 0,
  };
}

const procCfg = JSON.parse(fs.readFileSync(path.join(MODELS_DIR, 'slimsam/preprocessor_config.json'), 'utf8'));
const processor = new SamImageProcessor(procCfg);
const detSess = await ortWasm.InferenceSession.create(
  fs.readFileSync(path.join(MODELS_DIR, 'fish_detector_yolox_nano_v0_1.onnx')),
  { executionProviders: ['wasm'] },
);
const encSess = await ortWasm.InferenceSession.create(
  fs.readFileSync(path.join(MODELS_DIR, 'slimsam/onnx/vision_encoder_quantized.onnx')),
  { executionProviders: ['wasm'] },
);
const decSess = await ortWasm.InferenceSession.create(
  fs.readFileSync(path.join(MODELS_DIR, 'slimsam/onnx/prompt_encoder_mask_decoder_quantized.onnx')),
  { executionProviders: ['wasm'] },
);

const photos = ['preset_fish.jpg', 'preset_catfish.jpg', 'preset_snakehead.jpg', 'fish_crop.jpg'];
const summary = {};
const androidEvidenceDir = process.env.ANDROID_EVIDENCE_DIR || null;

for (const fname of photos) {
  const photoId = path.basename(fname, '.jpg');
  const photoOut = path.join(OUT_DIR, photoId);
  fs.mkdirSync(photoOut, { recursive: true });
  const tensorOut = path.join(photoOut, 'tensors');
  fs.mkdirSync(tensorOut, { recursive: true });

  const rawImg = await RawImage.read(path.join(FIXTURES_DIR, fname));
  const rgbaImg = rawImg.rgba();
  const width = rgbaImg.width;
  const height = rgbaImg.height;
  const fixtureSha256 = sha256File(path.join(FIXTURES_DIR, fname));
  const detection = await detectYolox(detSess, rgbaImg.data, width, height, tensorOut);
  if (!detection) throw new Error(`Web YOLOX failed on ${fname}`);

  const prompts = generatePromptsFromBox(detection.box, width, height);
  const inputPoints = [prompts.map(p => [p.x, p.y])];
  const inputLabels = [prompts.map(p => p.label)];

  const procInputs = await processor(rawImg, { input_points: inputPoints, input_labels: inputLabels });
  writeTensor(tensorOut, 'sam_pixel_values_node.f32', procInputs.pixel_values.data);
  writeTensor(tensorOut, 'sam_input_points_node.f32', procInputs.input_points.data);
  writeTensor(tensorOut, 'sam_input_labels_node.i64', procInputs.input_labels.data);
  fs.writeFileSync(path.join(photoOut, 'pixel_values.f32'), tensorBytes(procInputs.pixel_values.data));
  fs.writeFileSync(path.join(tensorOut, 'sam_node_input_shapes.json'), JSON.stringify({
    pixelValues: { shape: procInputs.pixel_values.dims, dtype: 'float32' },
    inputPoints: { shape: procInputs.input_points.dims, dtype: 'float32' },
    inputLabels: { shape: procInputs.input_labels.dims, dtype: 'int64' },
    originalSizes: procInputs.original_sizes,
    reshapedInputSizes: procInputs.reshaped_input_sizes,
    fixtureSha256,
    prompts,
  }, null, 2));
  const encOut = await encSess.run({
    pixel_values: new ortWasm.Tensor('float32', procInputs.pixel_values.data, procInputs.pixel_values.dims),
  });
  writeTensor(tensorOut, 'sam_image_embeddings_node.f32', encOut.image_embeddings.data);
  writeTensor(tensorOut, 'sam_positional_embeddings_node.f32', encOut.image_positional_embeddings.data);
  fs.writeFileSync(path.join(photoOut, 'image_embeddings.f32'), tensorBytes(encOut.image_embeddings.data));

  const decOut = await decSess.run({
    input_points: new ortWasm.Tensor('float32', procInputs.input_points.data, procInputs.input_points.dims),
    input_labels: new ortWasm.Tensor('int64', procInputs.input_labels.data, procInputs.input_labels.dims),
    image_embeddings: encOut.image_embeddings,
    image_positional_embeddings: encOut.image_positional_embeddings,
  });
  writeTensor(tensorOut, 'sam_iou_scores_node.f32', decOut.iou_scores.data);
  writeTensor(tensorOut, 'sam_pred_masks_node.f32', decOut.pred_masks.data);
  fs.writeFileSync(path.join(tensorOut, 'sam_node_output_shapes.json'), JSON.stringify({
    imageEmbeddings: encOut.image_embeddings.dims,
    positionalEmbeddings: encOut.image_positional_embeddings.dims,
    iouScores: decOut.iou_scores.dims,
    predMasks: decOut.pred_masks.dims,
  }, null, 2));
  const postMasks = await processor.post_process_masks(
    new Tensor('float32', decOut.pred_masks.data, decOut.pred_masks.dims),
    procInputs.original_sizes,
    procInputs.reshaped_input_sizes,
  );
  const sharpMeta = await scoreCandidatesAndSave(photoOut, '', decOut, postMasks, detection, width, height);

  const [rH, rW] = procInputs.reshaped_input_sizes[0];
  const rgbData = rawImg.rgb().data;
  const pvBil = new Float32Array(3 * 1024 * 1024);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const plane = 1024 * 1024;
  for (let y = 0; y < rH; y++) {
    const sy = Math.min(height - 1, Math.max(0, (y + 0.5) * (height / rH) - 0.5));
    const y0 = Math.floor(sy);
    const y1 = Math.min(height - 1, y0 + 1);
    const wy = sy - y0;
    for (let x = 0; x < rW; x++) {
      const sx = Math.min(width - 1, Math.max(0, (x + 0.5) * (width / rW) - 0.5));
      const x0 = Math.floor(sx);
      const x1 = Math.min(width - 1, x0 + 1);
      const wx = sx - x0;
      const i00 = (y0 * width + x0) * 3;
      const i10 = (y0 * width + x1) * 3;
      const i01 = (y1 * width + x0) * 3;
      const i11 = (y1 * width + x1) * 3;
      for (let c = 0; c < 3; c++) {
        const v = ((rgbData[i00 + c] * (1 - wx) + rgbData[i10 + c] * wx) * (1 - wy) +
                   (rgbData[i01 + c] * (1 - wx) + rgbData[i11 + c] * wx) * wy) / 255;
        pvBil[c * plane + y * 1024 + x] = (v - mean[c]) / std[c];
      }
    }
  }
  const encBil = await encSess.run({ pixel_values: new ortWasm.Tensor('float32', pvBil, [1, 3, 1024, 1024]) });
  writeTensor(tensorOut, 'sam_pixel_values_bilinear.f32', pvBil);
  writeTensor(tensorOut, 'sam_image_embeddings_bilinear.f32', encBil.image_embeddings.data);
  writeTensor(tensorOut, 'sam_positional_embeddings_bilinear.f32', encBil.image_positional_embeddings.data);
  fs.writeFileSync(path.join(photoOut, 'pv_bilinear.f32'), tensorBytes(pvBil));
  fs.writeFileSync(path.join(photoOut, 'emb_bilinear.f32'), tensorBytes(encBil.image_embeddings.data));

  const decBil = await decSess.run({
    input_points: new ortWasm.Tensor('float32', procInputs.input_points.data, procInputs.input_points.dims),
    input_labels: new ortWasm.Tensor('int64', procInputs.input_labels.data, procInputs.input_labels.dims),
    image_embeddings: encBil.image_embeddings,
    image_positional_embeddings: encBil.image_positional_embeddings,
  });
  writeTensor(tensorOut, 'sam_iou_scores_bilinear.f32', decBil.iou_scores.data);
  writeTensor(tensorOut, 'sam_pred_masks_bilinear.f32', decBil.pred_masks.data);
  writeTensor(tensorOut, 'sam_input_points_bilinear.f32', procInputs.input_points.data);
  writeTensor(tensorOut, 'sam_input_labels_bilinear.i64', procInputs.input_labels.data);
  fs.writeFileSync(path.join(tensorOut, 'processor_tensor_summary.json'), JSON.stringify({
    webNode: {
      shape: procInputs.pixel_values.dims,
      dtype: 'float32',
      sha256: sha256File(path.join(photoOut, 'pixel_values.f32')),
      stats: basicFloatStats(procInputs.pixel_values.data),
    },
    webBilinear: {
      shape: [1, 3, 1024, 1024],
      dtype: 'float32',
      sha256: sha256File(path.join(tensorOut, 'sam_pixel_values_bilinear.f32')),
      stats: basicFloatStats(pvBil),
    },
    resizedInputSize: procInputs.reshaped_input_sizes,
    padding: { right: 1024 - rW, bottom: 1024 - rH },
  }, null, 2));
  const postMasksBil = await processor.post_process_masks(
    new Tensor('float32', decBil.pred_masks.data, decBil.pred_masks.dims),
    procInputs.original_sizes,
    procInputs.reshaped_input_sizes,
  );
  const bilMeta = await scoreCandidatesAndSave(photoOut, 'bil_', decBil, postMasksBil, detection, width, height);

  writeDecoderOutputFiles(tensorOut, 'web_node', decOut.iou_scores.data, decOut.pred_masks.data, decOut.pred_masks.dims);
  writeDecoderOutputFiles(tensorOut, 'web_bilinear', decBil.iou_scores.data, decBil.pred_masks.data, decBil.pred_masks.dims);
  const nodeStageRoot = path.join(tensorOut, 'mask_stages/web_node');
  const bilStageRoot = path.join(tensorOut, 'mask_stages/web_bilinear');
  const nodeStages = emulateKotlinPostprocessStages(
    decOut.pred_masks.data, rW, rH, width, height, detection.box, nodeStageRoot,
  );
  const bilStages = emulateKotlinPostprocessStages(
    decBil.pred_masks.data, rW, rH, width, height, detection.box, bilStageRoot,
  );
  const compareCustomStagesToTransformers = (variant, summaries, prefix) => summaries.map(({ candidateIndex }) => {
    const customMask = fs.readFileSync(path.join(tensorOut, 'mask_stages', variant, `candidate_${candidateIndex}`, 'final_mask.bin'));
    const transformersMask = fs.readFileSync(path.join(photoOut, `${prefix}candidate_${candidateIndex}.bin`));
    return { candidateIndex, ...binaryMaskMetrics(customMask, transformersMask) };
  });
  fs.writeFileSync(path.join(tensorOut, 'mask_stages', 'postprocess_reference_validation.json'), JSON.stringify({
    note: 'Custom float32 stage implementation is compared against Transformers.js post_process_masks plus clipping/LCC output.',
    webNode: compareCustomStagesToTransformers('web_node', nodeStages, ''),
    webBilinear: compareCustomStagesToTransformers('web_bilinear', bilStages, 'bil_'),
  }, null, 2));

  let androidMatched = null;
  if (androidEvidenceDir) {
    const androidTensorDir = path.join(androidEvidenceDir, 'photos', photoId, 'tensors');
    const androidPvPath = path.join(androidTensorDir, 'sam_pixel_values_android.f32');
    const androidEmbPath = path.join(androidTensorDir, 'sam_image_embeddings_android.f32');
    const androidPosPath = path.join(androidTensorDir, 'sam_positional_embeddings_android.f32');
    const androidPointsPath = path.join(androidTensorDir, 'sam_input_points_android.f32');
    const androidLabelsPath = path.join(androidTensorDir, 'sam_input_labels_android.i64');
    for (const requiredPath of [androidPvPath, androidEmbPath, androidPosPath, androidPointsPath, androidLabelsPath]) {
      if (!fs.existsSync(requiredPath)) throw new Error(`Missing Android ONNX input capture: ${requiredPath}`);
    }

    const androidPixelValues = readFloat32(androidPvPath);
    const androidEmbeddings = readFloat32(androidEmbPath);
    const androidPositional = readFloat32(androidPosPath);
    const androidPoints = readFloat32(androidPointsPath);
    const androidLabels = readInt64(androidLabelsPath);
    if (androidPixelValues.length !== 3 * 1024 * 1024 || androidEmbeddings.length !== 256 * 64 * 64 ||
        androidPositional.length !== 256 * 64 * 64 || androidPoints.length === 0 || androidLabels.length * 2 !== androidPoints.length) {
      throw new Error(`Android ONNX tensor shape mismatch for ${photoId}: pixel=${androidPixelValues.length}, emb=${androidEmbeddings.length}, pos=${androidPositional.length}, points=${androidPoints.length}, labels=${androidLabels.length}`);
    }

    const androidLogitsPath = path.join(androidTensorDir, 'sam_pred_masks_android.f32');
    const androidIousPath = path.join(androidTensorDir, 'sam_iou_scores_android.f32');
    const androidShapeInfo = JSON.parse(fs.readFileSync(path.join(androidTensorDir, 'sam_preprocess_shapes.json'), 'utf8'));
    const androidDetection = JSON.parse(fs.readFileSync(path.join(androidEvidenceDir, 'photos', photoId, 'detection.json'), 'utf8'));
    const androidDims = androidDetection.origDims;
    const [androidResizedHeight, androidResizedWidth] = androidShapeInfo.resizedInputSize;
    const androidRawLogits = readFloat32(androidLogitsPath);
    const androidRawIous = readFloat32(androidIousPath);
    if (androidRawLogits.length !== 3 * 256 * 256 || androidRawIous.length !== 3) {
      throw new Error(`Android decoder output shape mismatch for ${photoId}: logits=${androidRawLogits.length}, ious=${androidRawIous.length}`);
    }
    writeDecoderOutputFiles(tensorOut, 'android_logits_web_postprocess', androidRawIous, androidRawLogits, [1, 3, 256, 256]);
    const androidMaskStageRoot = path.join(tensorOut, 'mask_stages/android_logits_web_postprocess');
    const androidLogitStages = emulateKotlinPostprocessStages(
      androidRawLogits,
      androidResizedWidth,
      androidResizedHeight,
      androidDims[0],
      androidDims[1],
      androidDetection.box,
      androidMaskStageRoot,
      false,
    );
    fs.writeFileSync(path.join(androidMaskStageRoot, 'mask_shape_and_identity.json'), JSON.stringify({
      fixtureSha256: androidDetection.fixtureSha256,
      sourceDims: androidDims,
      resizedInputSize: [androidResizedHeight, androidResizedWidth],
      detectionBox: androidDetection.box,
      decoderLogitsSha256: sha256File(androidLogitsPath),
      decoderIouScoresSha256: sha256File(androidIousPath),
      candidates: androidLogitStages,
    }, null, 2));

    const encoderOnAndroidInput = await encSess.run({
      pixel_values: new ortWasm.Tensor('float32', androidPixelValues, [1, 3, 1024, 1024]),
    });
    writeTensor(tensorOut, 'sam_image_embeddings_web_on_android_input.f32', encoderOnAndroidInput.image_embeddings.data);
    writeTensor(tensorOut, 'sam_positional_embeddings_web_on_android_input.f32', encoderOnAndroidInput.image_positional_embeddings.data);

    const decoderOnAndroidInputs = await decSess.run({
      input_points: new ortWasm.Tensor('float32', androidPoints, [1, 1, androidLabels.length, 2]),
      input_labels: new ortWasm.Tensor('int64', androidLabels, [1, 1, androidLabels.length]),
      image_embeddings: new ortWasm.Tensor('float32', androidEmbeddings, [1, 256, 64, 64]),
      image_positional_embeddings: new ortWasm.Tensor('float32', androidPositional, [1, 256, 64, 64]),
    });
    writeTensor(tensorOut, 'sam_iou_scores_web_on_android_inputs.f32', decoderOnAndroidInputs.iou_scores.data);
    writeTensor(tensorOut, 'sam_pred_masks_web_on_android_inputs.f32', decoderOnAndroidInputs.pred_masks.data);
    writeDecoderOutputFiles(
      tensorOut,
      'web_decoder_android_inputs',
      decoderOnAndroidInputs.iou_scores.data,
      decoderOnAndroidInputs.pred_masks.data,
      decoderOnAndroidInputs.pred_masks.dims,
    );
    emulateKotlinPostprocessStages(
      decoderOnAndroidInputs.pred_masks.data,
      androidResizedWidth,
      androidResizedHeight,
      androidDims[0],
      androidDims[1],
      androidDetection.box,
      path.join(tensorOut, 'mask_stages/web_decoder_android_inputs'),
      false,
    );

    const decoderOnAndroidPixels = await decSess.run({
      input_points: new ortWasm.Tensor('float32', androidPoints, [1, 1, androidLabels.length, 2]),
      input_labels: new ortWasm.Tensor('int64', androidLabels, [1, 1, androidLabels.length]),
      image_embeddings: encoderOnAndroidInput.image_embeddings,
      image_positional_embeddings: encoderOnAndroidInput.image_positional_embeddings,
    });
    writeTensor(tensorOut, 'sam_iou_scores_web_encoder_on_android_input.f32', decoderOnAndroidPixels.iou_scores.data);
    writeTensor(tensorOut, 'sam_pred_masks_web_encoder_on_android_input.f32', decoderOnAndroidPixels.pred_masks.data);

    androidMatched = {
      androidPixelValuesSha256: sha256File(androidPvPath),
      androidImageEmbeddingsSha256: sha256File(androidEmbPath),
      androidPositionalEmbeddingsSha256: sha256File(androidPosPath),
      androidPromptCoordsSha256: sha256File(androidPointsPath),
      androidPromptLabelsSha256: sha256File(androidLabelsPath),
      webOnAndroidImageEmbeddingsShape: encoderOnAndroidInput.image_embeddings.dims,
      webOnAndroidPositionalEmbeddingsShape: encoderOnAndroidInput.image_positional_embeddings.dims,
      webDecoderOnAndroidInputsShape: decoderOnAndroidInputs.pred_masks.dims,
      webDecoderOnAndroidInputIoUShape: decoderOnAndroidInputs.iou_scores.dims,
      webDecoderOnAndroidEncoderOutputShape: decoderOnAndroidPixels.pred_masks.dims,
    };
    fs.writeFileSync(path.join(tensorOut, 'android_matched_web_run.json'), JSON.stringify(androidMatched, null, 2));
  }

  const tensorFiles = [];
  const visitTensors = (dir) => {
    for (const name of fs.readdirSync(dir).sort()) {
      const file = path.join(dir, name);
      if (fs.statSync(file).isDirectory()) visitTensors(file);
      else tensorFiles.push(path.relative(tensorOut, file).split(path.sep).join('/'));
    }
  };
  visitTensors(tensorOut);
  fs.writeFileSync(path.join(tensorOut, 'tensor_manifest.json'), JSON.stringify({
    photoId,
    fixtureSha256,
    fixtureSize: [width, height],
    tensors: tensorFiles.filter((name) => name !== 'tensor_manifest.json').map((name) => ({
      file: name,
      sizeBytes: fs.statSync(path.join(tensorOut, name)).size,
      sha256: sha256File(path.join(tensorOut, name)),
    })),
  }, null, 2));

  summary[photoId] = {
    width,
    height,
    fixtureSha256,
    detection,
    prompts,
    reshapedInputSize: [rH, rW],
    candidates: sharpMeta.candidates,
    samServiceBestIndex: sharpMeta.samServiceBestIndex,
    appSelectedIndex: sharpMeta.appSelectedIndex,
    androidCompatibleSelectedIndex: sharpMeta.androidCompatibleSelectedIndex,
    androidCompatibleUsedTrustworthyGate: sharpMeta.androidCompatibleUsedTrustworthyGate,
    bilCandidates: bilMeta.candidates,
    bilSamServiceBestIndex: bilMeta.samServiceBestIndex,
    bilAppSelectedIndex: bilMeta.appSelectedIndex,
    bilAndroidCompatibleSelectedIndex: bilMeta.androidCompatibleSelectedIndex,
    bilAndroidCompatibleUsedTrustworthyGate: bilMeta.androidCompatibleUsedTrustworthyGate,
    androidMatched,
  };
  console.log(`[WEB_PARITY] ${photoId} r=${rW}x${rH} det=${JSON.stringify(detection.box)} samBest=${sharpMeta.samServiceBestIndex} bilSamBest=${bilMeta.samServiceBestIndex} androidCompatible=${sharpMeta.androidCompatibleSelectedIndex}/${bilMeta.androidCompatibleSelectedIndex}`);
}
fs.writeFileSync(path.join(OUT_DIR, 'web_results.json'), JSON.stringify(summary, null, 2));
