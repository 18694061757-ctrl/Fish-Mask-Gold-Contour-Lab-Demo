import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
import { pathToFileURL } from 'url';

const ROOT = process.env.FISHMASK_ROOT_DIR || process.cwd();
const MODELS_DIR = path.join(ROOT, 'app/src/main/assets/models');
const FIXTURES_DIR = path.join(ROOT, 'app/src/test/resources');
const OUT_DIR = process.env.WEB_PARITY_OUT_DIR || path.join(ROOT, 'build/web_parity_out');
fs.mkdirSync(OUT_DIR, { recursive: true });

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

async function detectYolox(detSess, rgba, sourceW, sourceH) {
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
  const inputName = detSess.inputNames[0] || 'images';
  const outputName = detSess.outputNames[0] || 'output';
  const outMap = await detSess.run({
    [inputName]: new ortWasm.Tensor('float32', tensorData, [1, 3, inputSize, inputSize]),
  });
  const outTensor = outMap[outputName];
  const data = outTensor.data;
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
  const primaryIdx = prefix === 'bil_' ? samBest.index : appSelected.index;
  const lccPrimary = getLargestConnectedComponent(
    fs.readFileSync(path.join(photoOut, `${prefix}candidate_${primaryIdx}.bin`)),
    width,
    height,
  );
  const rawPts = extractBoundaryPoints(lccPrimary.cleanMask, width, height, lccPrimary.startX, lccPrimary.startY);
  fs.writeFileSync(path.join(photoOut, `${prefix}contour_app.json`), JSON.stringify({ rawPoints: rawPts }));
  return { candidates, samServiceBestIndex: samBest.index, appSelectedIndex: appSelected.index };
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

for (const fname of photos) {
  const photoId = path.basename(fname, '.jpg');
  const photoOut = path.join(OUT_DIR, photoId);
  fs.mkdirSync(photoOut, { recursive: true });

  const rawImg = await RawImage.read(path.join(FIXTURES_DIR, fname));
  const rgbaImg = rawImg.rgba();
  const width = rgbaImg.width;
  const height = rgbaImg.height;
  const detection = await detectYolox(detSess, rgbaImg.data, width, height);
  if (!detection) throw new Error(`Web YOLOX failed on ${fname}`);

  const prompts = generatePromptsFromBox(detection.box, width, height);
  const inputPoints = [prompts.map(p => [p.x, p.y])];
  const inputLabels = [prompts.map(p => p.label)];

  const procInputs = await processor(rawImg, { input_points: inputPoints, input_labels: inputLabels });
  fs.writeFileSync(
    path.join(photoOut, 'pixel_values.f32'),
    Buffer.from(procInputs.pixel_values.data.buffer, procInputs.pixel_values.data.byteOffset, procInputs.pixel_values.data.byteLength),
  );
  const encOut = await encSess.run({
    pixel_values: new ortWasm.Tensor('float32', procInputs.pixel_values.data, procInputs.pixel_values.dims),
  });
  fs.writeFileSync(
    path.join(photoOut, 'image_embeddings.f32'),
    Buffer.from(encOut.image_embeddings.data.buffer, encOut.image_embeddings.data.byteOffset, encOut.image_embeddings.data.byteLength),
  );

  const decOut = await decSess.run({
    input_points: new ortWasm.Tensor('float32', procInputs.input_points.data, procInputs.input_points.dims),
    input_labels: new ortWasm.Tensor('int64', procInputs.input_labels.data, procInputs.input_labels.dims),
    image_embeddings: encOut.image_embeddings,
    image_positional_embeddings: encOut.image_positional_embeddings,
  });
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
  fs.writeFileSync(path.join(photoOut, 'pv_bilinear.f32'), Buffer.from(pvBil.buffer));
  fs.writeFileSync(path.join(photoOut, 'emb_bilinear.f32'), Buffer.from(encBil.image_embeddings.data.buffer));

  const decBil = await decSess.run({
    input_points: new ortWasm.Tensor('float32', procInputs.input_points.data, procInputs.input_points.dims),
    input_labels: new ortWasm.Tensor('int64', procInputs.input_labels.data, procInputs.input_labels.dims),
    image_embeddings: encBil.image_embeddings,
    image_positional_embeddings: encBil.image_positional_embeddings,
  });
  const postMasksBil = await processor.post_process_masks(
    new Tensor('float32', decBil.pred_masks.data, decBil.pred_masks.dims),
    procInputs.original_sizes,
    procInputs.reshaped_input_sizes,
  );
  const bilMeta = await scoreCandidatesAndSave(photoOut, 'bil_', decBil, postMasksBil, detection, width, height);

  summary[photoId] = {
    width,
    height,
    detection,
    prompts,
    reshapedInputSize: [rH, rW],
    candidates: sharpMeta.candidates,
    samServiceBestIndex: sharpMeta.samServiceBestIndex,
    appSelectedIndex: sharpMeta.appSelectedIndex,
    bilCandidates: bilMeta.candidates,
    bilSamServiceBestIndex: bilMeta.samServiceBestIndex,
    bilAppSelectedIndex: bilMeta.appSelectedIndex,
  };
  console.log(`[WEB_PARITY] ${photoId} r=${rW}x${rH} det=${JSON.stringify(detection.box)} samBest=${sharpMeta.samServiceBestIndex} bilSamBest=${bilMeta.samServiceBestIndex}`);
}
fs.writeFileSync(path.join(OUT_DIR, 'web_results.json'), JSON.stringify(summary, null, 2));
