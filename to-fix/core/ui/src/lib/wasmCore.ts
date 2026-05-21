import { ALPHABET, FEATURE_COUNT, type LabSample, type LabSvmSnapshot, type StrokeInput } from './types.ts';

const DEFAULT_SVM_C = 10;
const DEFAULT_SVM_GAMMA = 1 / FEATURE_COUNT;

interface HandwritingWasmExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  alloc(size: number): number;
  dealloc(ptr: number, size: number): void;
  extract_features(pointsPtr: number, pointCount: number, strokeLengthsPtr: number, strokeCount: number, outPtr: number): void;
  knn_predict(
    featuresPtr: number,
    cacheFeaturesPtr: number,
    cacheLabelsPtr: number,
    cacheSize: number,
    k: number,
    farNeighborDistance: number,
    outPtr: number,
  ): number;
  train_svm_classifier(
    sampleFeaturesPtr: number,
    sampleLabelsPtr: number,
    sampleCount: number,
    readyLabelsPtr: number,
    readyCount: number,
    c: number,
    gamma: number,
    outLabelsPtr: number,
    outBiasesPtr: number,
    outStartsPtr: number,
    outCountsPtr: number,
    outCoefficientsPtr: number,
    outSupportFeaturesPtr: number,
    outFeatureMeanPtr: number,
    outFeatureStdPtr: number,
  ): number;
  predict_svm_classifier(
    featuresPtr: number,
    labelsPtr: number,
    biasesPtr: number,
    startsPtr: number,
    countsPtr: number,
    coefficientsPtr: number,
    supportFeaturesPtr: number,
    classifierCount: number,
    totalSupportCount: number,
    gamma: number,
    featureMeanPtr: number,
    featureStdPtr: number,
    outPtr: number,
  ): void;
}

let wasmExports: HandwritingWasmExports | null = null;
let initPromise: Promise<void> | null = null;

function labelToIndex(label: string): number {
  const code = label.slice(0, 1).toUpperCase().charCodeAt(0) - 65;
  return code >= 0 && code < ALPHABET.length ? code : 255;
}

function indexToLabel(index: number): string {
  return ALPHABET[index] ?? '?';
}

function defaultWasmUrl(): string {
  return new URL('wasm/handwriting_core.wasm', new URL(import.meta.env.BASE_URL, window.location.origin)).toString();
}

async function instantiate(wasmUrl = defaultWasmUrl()): Promise<HandwritingWasmExports> {
  const response = await fetch(wasmUrl);
  if (!response.ok) {
    throw new Error(`Failed to load handwriting_core.wasm: HTTP ${response.status}`);
  }
  try {
    const result = await WebAssembly.instantiateStreaming(response, {});
    return result.instance.exports as unknown as HandwritingWasmExports;
  } catch {
    const result = await WebAssembly.instantiate(await response.arrayBuffer(), {});
    return result.instance.exports as unknown as HandwritingWasmExports;
  }
}

export async function initWasmCore(wasmUrl?: string): Promise<void> {
  if (wasmExports) {
    return;
  }
  initPromise ??= instantiate(wasmUrl).then((exports) => {
    wasmExports = exports;
  });
  await initPromise;
}

function requireExports(): HandwritingWasmExports {
  if (!wasmExports) {
    throw new Error('handwriting_core.wasm has not been initialized.');
  }
  return wasmExports;
}

function mallocCopy(data: ArrayBufferView): { ptr: number; bytes: number } {
  const exports = requireExports();
  const ptr = exports.alloc(data.byteLength);
  new Uint8Array(exports.memory.buffer, ptr, data.byteLength).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  return { ptr, bytes: data.byteLength };
}

function mallocZeroed(bytes: number): { ptr: number; bytes: number } {
  const exports = requireExports();
  const ptr = exports.alloc(bytes);
  new Uint8Array(exports.memory.buffer, ptr, bytes).fill(0);
  return { ptr, bytes };
}

function free(allocation: { ptr: number; bytes: number } | null | undefined): void {
  if (allocation) {
    requireExports().dealloc(allocation.ptr, allocation.bytes);
  }
}

function readFloat64(ptr: number, length: number): number[] {
  return Array.from(new Float64Array(requireExports().memory.buffer, ptr, length));
}

function readUint32(ptr: number, length: number): number[] {
  return Array.from(new Uint32Array(requireExports().memory.buffer, ptr, length));
}

function readUint8(ptr: number, length: number): number[] {
  return Array.from(new Uint8Array(requireExports().memory.buffer, ptr, length));
}

function flattenStrokes(strokes: StrokeInput): { points: Float64Array; strokeLengths: Uint32Array } {
  const points = new Float64Array(strokes.reduce((total, stroke) => total + stroke.length, 0) * 3);
  const strokeLengths = new Uint32Array(strokes.length);
  let cursor = 0;
  strokes.forEach((stroke, strokeIndex) => {
    strokeLengths[strokeIndex] = stroke.length;
    stroke.forEach((point) => {
      points[cursor] = point.x;
      points[cursor + 1] = point.y;
      points[cursor + 2] = point.t;
      cursor += 3;
    });
  });
  return { points, strokeLengths };
}

function featureArray(features: number[]): Float64Array {
  const result = new Float64Array(FEATURE_COUNT);
  result.set(features.slice(0, FEATURE_COUNT));
  return result;
}

export function extractFeatures(strokes: StrokeInput): number[] {
  const exports = requireExports();
  const { points, strokeLengths } = flattenStrokes(strokes);
  const pointsAlloc = mallocCopy(points);
  const lengthsAlloc = mallocCopy(strokeLengths);
  const outAlloc = mallocZeroed(FEATURE_COUNT * Float64Array.BYTES_PER_ELEMENT);
  try {
    exports.extract_features(pointsAlloc.ptr, points.length / 3, lengthsAlloc.ptr, strokeLengths.length, outAlloc.ptr);
    return readFloat64(outAlloc.ptr, FEATURE_COUNT);
  } finally {
    free(pointsAlloc);
    free(lengthsAlloc);
    free(outAlloc);
  }
}

export function predictKnn(features: number[], samples: LabSample[], k = 5): Record<string, number> {
  const scores = Object.fromEntries(ALPHABET.map((label) => [label, 0])) as Record<string, number>;
  if (samples.length === 0) {
    return scores;
  }
  const exports = requireExports();
  const queryAlloc = mallocCopy(featureArray(features));
  const cacheFeatures = new Float64Array(samples.length * FEATURE_COUNT);
  const cacheLabels = new Uint8Array(samples.length);
  samples.forEach((sample, index) => {
    cacheFeatures.set(featureArray(sample.features), index * FEATURE_COUNT);
    cacheLabels[index] = labelToIndex(sample.label);
  });
  const cacheFeaturesAlloc = mallocCopy(cacheFeatures);
  const cacheLabelsAlloc = mallocCopy(cacheLabels);
  const outAlloc = mallocZeroed(ALPHABET.length * Float64Array.BYTES_PER_ELEMENT);
  try {
    const nonZero = exports.knn_predict(queryAlloc.ptr, cacheFeaturesAlloc.ptr, cacheLabelsAlloc.ptr, samples.length, k, 4.5, outAlloc.ptr);
    if (nonZero === 0) {
      return scores;
    }
    readFloat64(outAlloc.ptr, ALPHABET.length).forEach((score, index) => {
      scores[indexToLabel(index)] = score;
    });
    return scores;
  } finally {
    free(queryAlloc);
    free(cacheFeaturesAlloc);
    free(cacheLabelsAlloc);
    free(outAlloc);
  }
}

export function trainSvm(samples: LabSample[], readyLetters: string[]): LabSvmSnapshot | null {
  if (samples.length === 0 || readyLetters.length < 2) {
    return null;
  }
  const exports = requireExports();
  const sampleFeatures = new Float64Array(samples.length * FEATURE_COUNT);
  samples.forEach((sample, index) => sampleFeatures.set(featureArray(sample.features), index * FEATURE_COUNT));
  const sampleLabels = new Uint8Array(samples.map((sample) => labelToIndex(sample.label)));
  const readyLabels = new Uint8Array(readyLetters.map(labelToIndex));
  const maxSupportCount = samples.length * readyLetters.length;

  const sampleFeaturesAlloc = mallocCopy(sampleFeatures);
  const sampleLabelsAlloc = mallocCopy(sampleLabels);
  const readyLabelsAlloc = mallocCopy(readyLabels);
  const outLabelsAlloc = mallocZeroed(readyLetters.length * Uint8Array.BYTES_PER_ELEMENT);
  const outBiasesAlloc = mallocZeroed(readyLetters.length * Float64Array.BYTES_PER_ELEMENT);
  const outStartsAlloc = mallocZeroed(readyLetters.length * Uint32Array.BYTES_PER_ELEMENT);
  const outCountsAlloc = mallocZeroed(readyLetters.length * Uint32Array.BYTES_PER_ELEMENT);
  const outCoefficientsAlloc = mallocZeroed(maxSupportCount * Float64Array.BYTES_PER_ELEMENT);
  const outSupportAlloc = mallocZeroed(maxSupportCount * FEATURE_COUNT * Float64Array.BYTES_PER_ELEMENT);
  const outMeanAlloc = mallocZeroed(FEATURE_COUNT * Float64Array.BYTES_PER_ELEMENT);
  const outStdAlloc = mallocZeroed(FEATURE_COUNT * Float64Array.BYTES_PER_ELEMENT);

  try {
    const supportCount = exports.train_svm_classifier(
      sampleFeaturesAlloc.ptr,
      sampleLabelsAlloc.ptr,
      samples.length,
      readyLabelsAlloc.ptr,
      readyLetters.length,
      DEFAULT_SVM_C,
      DEFAULT_SVM_GAMMA,
      outLabelsAlloc.ptr,
      outBiasesAlloc.ptr,
      outStartsAlloc.ptr,
      outCountsAlloc.ptr,
      outCoefficientsAlloc.ptr,
      outSupportAlloc.ptr,
      outMeanAlloc.ptr,
      outStdAlloc.ptr,
    );
    if (supportCount === 0) {
      return null;
    }
    const labels = readUint8(outLabelsAlloc.ptr, readyLetters.length).map(indexToLabel);
    return {
      id: `svm-${Date.now().toString(36)}`,
      version: 'svm-rbf-v1',
      createdAt: Date.now(),
      c: DEFAULT_SVM_C,
      gamma: DEFAULT_SVM_GAMMA,
      labels,
      biases: readFloat64(outBiasesAlloc.ptr, readyLetters.length),
      starts: readUint32(outStartsAlloc.ptr, readyLetters.length),
      counts: readUint32(outCountsAlloc.ptr, readyLetters.length),
      coefficients: readFloat64(outCoefficientsAlloc.ptr, supportCount),
      supportVectors: readFloat64(outSupportAlloc.ptr, supportCount * FEATURE_COUNT),
      featureMean: readFloat64(outMeanAlloc.ptr, FEATURE_COUNT),
      featureStd: readFloat64(outStdAlloc.ptr, FEATURE_COUNT),
      supportCount,
      datasetSize: samples.length,
      readyLetters: [...readyLetters],
      metrics: { user_inputtedAccuracy: 0, implicitAccuracy: 0, overallAccuracy: 0 },
    };
  } finally {
    [
      sampleFeaturesAlloc,
      sampleLabelsAlloc,
      readyLabelsAlloc,
      outLabelsAlloc,
      outBiasesAlloc,
      outStartsAlloc,
      outCountsAlloc,
      outCoefficientsAlloc,
      outSupportAlloc,
      outMeanAlloc,
      outStdAlloc,
    ].forEach(free);
  }
}

export function predictSvm(features: number[], snapshot: LabSvmSnapshot): Record<string, number> {
  const scores = Object.fromEntries(ALPHABET.map((label) => [label, 0])) as Record<string, number>;
  const exports = requireExports();
  const featureAlloc = mallocCopy(featureArray(features));
  const labelsAlloc = mallocCopy(new Uint8Array(snapshot.labels.map(labelToIndex)));
  const biasesAlloc = mallocCopy(new Float64Array(snapshot.biases));
  const startsAlloc = mallocCopy(new Uint32Array(snapshot.starts));
  const countsAlloc = mallocCopy(new Uint32Array(snapshot.counts));
  const coeffAlloc = mallocCopy(new Float64Array(snapshot.coefficients));
  const supportAlloc = mallocCopy(new Float64Array(snapshot.supportVectors));
  const meanAlloc = mallocCopy(new Float64Array(snapshot.featureMean));
  const stdAlloc = mallocCopy(new Float64Array(snapshot.featureStd));
  const outAlloc = mallocZeroed(ALPHABET.length * Float64Array.BYTES_PER_ELEMENT);
  try {
    exports.predict_svm_classifier(
      featureAlloc.ptr,
      labelsAlloc.ptr,
      biasesAlloc.ptr,
      startsAlloc.ptr,
      countsAlloc.ptr,
      coeffAlloc.ptr,
      supportAlloc.ptr,
      snapshot.labels.length,
      snapshot.supportCount,
      snapshot.gamma,
      meanAlloc.ptr,
      stdAlloc.ptr,
      outAlloc.ptr,
    );
    readFloat64(outAlloc.ptr, ALPHABET.length).forEach((score, index) => {
      scores[indexToLabel(index)] = score;
    });
    return scores;
  } finally {
    [featureAlloc, labelsAlloc, biasesAlloc, startsAlloc, countsAlloc, coeffAlloc, supportAlloc, meanAlloc, stdAlloc, outAlloc].forEach(free);
  }
}
