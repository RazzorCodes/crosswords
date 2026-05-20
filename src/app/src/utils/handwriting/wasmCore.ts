import type { RecognitionCandidate, StrokeInput } from '../recognizers/types';
import type {
  AcceptedSampleRecord,
  BalancedDataset,
  FeatureClassifierCentroid,
  FeatureClassifierSnapshot,
  LetterAcceptanceCounts,
  SnapshotMetrics,
  TrainingTriggerReason,
} from './types';

const ALPHABET = Array.from({ length: 26 }, (_, index) => String.fromCharCode(65 + index));
const FEATURE_COUNT = 30;

interface HandwritingWasmExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  alloc(size: number): number;
  dealloc(ptr: number, size: number): void;
  extract_features(
    pointsPtr: number,
    pointCount: number,
    strokeLengthsPtr: number,
    strokeCount: number,
    outPtr: number,
  ): void;
  knn_predict(
    featuresPtr: number,
    cacheFeaturesPtr: number,
    cacheLabelsPtr: number,
    cacheSize: number,
    k: number,
    farNeighborDistance: number,
    outPtr: number,
  ): number;
  train_centroid_classifier(
    sampleFeaturesPtr: number,
    sampleLabelsPtr: number,
    sampleCount: number,
    readyLabelsPtr: number,
    readyCount: number,
    outCentroidsPtr: number,
    outCountsPtr: number,
  ): number;
  predict_centroid_classifier(
    centroidsPtr: number,
    centroidLabelsPtr: number,
    centroidCount: number,
    featuresPtr: number,
    outPtr: number,
  ): void;
  compute_letter_stats(
    labelsPtr: number,
    acceptancesPtr: number,
    sampleCount: number,
    outCountsPtr: number,
    outReadyPtr: number,
    outPriorityPtr: number,
  ): number;
  build_balanced_dataset(
    labelsPtr: number,
    acceptancesPtr: number,
    createdAtPtr: number,
    sampleCount: number,
    outTrainingMaskPtr: number,
    outHoldoutMaskPtr: number,
    outReadyPtr: number,
  ): number;
  evaluate_snapshot(
    holdoutFeaturesPtr: number,
    holdoutLabelsPtr: number,
    holdoutAcceptancesPtr: number,
    holdoutCount: number,
    centroidsPtr: number,
    centroidLabelsPtr: number,
    centroidCount: number,
    outMetricsPtr: number,
  ): void;
}

let exportsPromise: Promise<HandwritingWasmExports> | null = null;
let wasmExports: HandwritingWasmExports | null = null;

function labelToIndex(label: string): number {
  const normalized = label.slice(0, 1).toUpperCase();
  const code = normalized.charCodeAt(0) - 65;
  return code >= 0 && code < 26 ? code : 255;
}

function indexToLabel(index: number): string {
  return ALPHABET[index] ?? '?';
}

function acceptanceToByte(acceptance: AcceptedSampleRecord['acceptance']): number {
  return acceptance === 'user_inputted' ? 1 : 0;
}

function requireExports(): HandwritingWasmExports {
  if (!wasmExports) {
    throw new Error('Rust handwriting core has not been initialized.');
  }
  return wasmExports;
}

async function instantiate(): Promise<HandwritingWasmExports> {
  const response = await fetch('/wasm/handwriting_core.wasm');
  if (!response.ok) {
    throw new Error(`Failed to load handwriting_core.wasm: ${response.status}`);
  }

  let instance: WebAssembly.Instance;
  if ('instantiateStreaming' in WebAssembly) {
    try {
      const result = await WebAssembly.instantiateStreaming(response, {});
      instance = result.instance;
    } catch {
      const buffer = await response.arrayBuffer();
      const result = await WebAssembly.instantiate(buffer, {});
      instance = result.instance;
    }
  } else {
    const buffer = await response.arrayBuffer();
    const result = await WebAssembly.instantiate(buffer, {});
    instance = result.instance;
  }

  return instance.exports as unknown as HandwritingWasmExports;
}

export async function initHandwritingRustCore(): Promise<void> {
  if (wasmExports) {
    return;
  }
  if (!exportsPromise) {
    exportsPromise = instantiate().then((exports) => {
      wasmExports = exports;
      return exports;
    });
  }
  await exportsPromise;
}

export function isHandwritingRustCoreReady(): boolean {
  return wasmExports !== null;
}

function mallocCopy(data: ArrayBufferView): { ptr: number; bytes: number } {
  const exports = requireExports();
  const bytes = data.byteLength;
  const ptr = exports.alloc(bytes);
  const source = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const target = new Uint8Array(exports.memory.buffer, ptr, bytes);
  target.set(source);
  return { ptr, bytes };
}

function mallocZeroed(byteLength: number): { ptr: number; bytes: number } {
  const exports = requireExports();
  const ptr = exports.alloc(byteLength);
  const target = new Uint8Array(exports.memory.buffer, ptr, byteLength);
  target.fill(0);
  return { ptr, bytes: byteLength };
}

function freeAllocation(allocation: { ptr: number; bytes: number } | null | undefined) {
  if (!allocation) {
    return;
  }
  requireExports().dealloc(allocation.ptr, allocation.bytes);
}

function readFloat64(ptr: number, length: number): number[] {
  const exports = requireExports();
  return Array.from(new Float64Array(exports.memory.buffer, ptr, length));
}

function readUint32(ptr: number, length: number): number[] {
  const exports = requireExports();
  return Array.from(new Uint32Array(exports.memory.buffer, ptr, length));
}

function readUint8(ptr: number, length: number): number[] {
  const exports = requireExports();
  return Array.from(new Uint8Array(exports.memory.buffer, ptr, length));
}

function flattenStrokes(strokes: StrokeInput): {
  points: Float64Array;
  strokeLengths: Uint32Array;
} {
  const flattened = new Float64Array(strokes.flat().length * 3);
  const strokeLengths = new Uint32Array(strokes.length);
  let cursor = 0;
  strokes.forEach((stroke, strokeIndex) => {
    strokeLengths[strokeIndex] = stroke.length;
    stroke.forEach((point) => {
      flattened[cursor] = point.x;
      flattened[cursor + 1] = point.y;
      flattened[cursor + 2] = point.t;
      cursor += 3;
    });
  });
  return { points: flattened, strokeLengths };
}

function featuresArray(features: number[]): Float64Array {
  const result = new Float64Array(FEATURE_COUNT);
  result.set(features.slice(0, FEATURE_COUNT));
  return result;
}

function flattenSampleFeatures(samples: AcceptedSampleRecord[]): Float64Array {
  const flattened = new Float64Array(samples.length * FEATURE_COUNT);
  samples.forEach((sample, index) => {
    flattened.set(featuresArray(sample.features), index * FEATURE_COUNT);
  });
  return flattened;
}

function flattenCentroids(centroids: FeatureClassifierCentroid[]): Float64Array {
  const flattened = new Float64Array(centroids.length * FEATURE_COUNT);
  centroids.forEach((centroid, index) => {
    flattened.set(featuresArray(centroid.centroid), index * FEATURE_COUNT);
  });
  return flattened;
}

function probabilitiesToCandidates(
  probabilities: number[],
  source: string,
): RecognitionCandidate[] {
  return probabilities
    .map((score, index) => ({
      char: indexToLabel(index),
      score,
      source,
    }))
    .sort((left, right) => right.score - left.score);
}

export function extractFeaturesRust(strokes: StrokeInput): number[] {
  const exports = requireExports();
  const { points, strokeLengths } = flattenStrokes(strokes);
  const pointsAllocation = mallocCopy(points);
  const strokeLengthsAllocation = mallocCopy(strokeLengths);
  const outAllocation = mallocZeroed(FEATURE_COUNT * Float64Array.BYTES_PER_ELEMENT);

  try {
    exports.extract_features(
      pointsAllocation.ptr,
      points.length / 3,
      strokeLengthsAllocation.ptr,
      strokeLengths.length,
      outAllocation.ptr,
    );
    return readFloat64(outAllocation.ptr, FEATURE_COUNT);
  } finally {
    freeAllocation(pointsAllocation);
    freeAllocation(strokeLengthsAllocation);
    freeAllocation(outAllocation);
  }
}

export function predictKnnRust(
  features: number[],
  cache: Array<{ label: string; features: number[] }>,
  k = 5,
  farNeighborDistance = 4.5,
): RecognitionCandidate[] {
  const exports = requireExports();
  const featureArray = featuresArray(features);
  const featureAllocation = mallocCopy(featureArray);
  const outAllocation = mallocZeroed(ALPHABET.length * Float64Array.BYTES_PER_ELEMENT);

  if (cache.length === 0) {
    freeAllocation(featureAllocation);
    freeAllocation(outAllocation);
    return [];
  }

  const cacheFeatures = new Float64Array(cache.length * FEATURE_COUNT);
  const cacheLabels = new Uint8Array(cache.length);
  cache.forEach((entry, index) => {
    cacheFeatures.set(featuresArray(entry.features), index * FEATURE_COUNT);
    cacheLabels[index] = labelToIndex(entry.label);
  });

  const cacheFeaturesAllocation = mallocCopy(cacheFeatures);
  const cacheLabelsAllocation = mallocCopy(cacheLabels);

  try {
    const nonZero = exports.knn_predict(
      featureAllocation.ptr,
      cacheFeaturesAllocation.ptr,
      cacheLabelsAllocation.ptr,
      cache.length,
      k,
      farNeighborDistance,
      outAllocation.ptr,
    );
    if (nonZero === 0) {
      return [];
    }
    return probabilitiesToCandidates(readFloat64(outAllocation.ptr, ALPHABET.length), 'knn')
      .filter((candidate) => candidate.score > 0);
  } finally {
    freeAllocation(featureAllocation);
    freeAllocation(outAllocation);
    freeAllocation(cacheFeaturesAllocation);
    freeAllocation(cacheLabelsAllocation);
  }
}

export function trainFeatureClassifierRust(
  samples: AcceptedSampleRecord[],
  readyLetters: string[],
  reason: TrainingTriggerReason,
): FeatureClassifierSnapshot | null {
  if (samples.length === 0 || readyLetters.length < 2) {
    return null;
  }

  const exports = requireExports();
  const sampleFeatures = flattenSampleFeatures(samples);
  const sampleLabels = new Uint8Array(samples.map((sample) => labelToIndex(sample.label)));
  const readyLabels = new Uint8Array(readyLetters.map((label) => labelToIndex(label)));

  const sampleFeaturesAllocation = mallocCopy(sampleFeatures);
  const sampleLabelsAllocation = mallocCopy(sampleLabels);
  const readyLabelsAllocation = mallocCopy(readyLabels);
  const outCentroidsAllocation = mallocZeroed(readyLetters.length * FEATURE_COUNT * Float64Array.BYTES_PER_ELEMENT);
  const outCountsAllocation = mallocZeroed(readyLetters.length * Uint32Array.BYTES_PER_ELEMENT);

  try {
    const centroidCount = exports.train_centroid_classifier(
      sampleFeaturesAllocation.ptr,
      sampleLabelsAllocation.ptr,
      samples.length,
      readyLabelsAllocation.ptr,
      readyLetters.length,
      outCentroidsAllocation.ptr,
      outCountsAllocation.ptr,
    );
    if (centroidCount < 2) {
      return null;
    }

    const centroidValues = readFloat64(outCentroidsAllocation.ptr, centroidCount * FEATURE_COUNT);
    const counts = readUint32(outCountsAllocation.ptr, centroidCount);
    const centroids: FeatureClassifierCentroid[] = [];
    for (let index = 0; index < centroidCount; index += 1) {
      const count = counts[index] ?? 0;
      if (count <= 0) {
        continue;
      }
      const start = index * FEATURE_COUNT;
      centroids.push({
        label: readyLetters[index],
        centroid: centroidValues.slice(start, start + FEATURE_COUNT),
        count,
      });
    }

    if (centroids.length < 2) {
      return null;
    }

    return {
      id: `snapshot-${Math.random().toString(36).slice(2, 10)}`,
      version: 'rust-centroid-v1',
      createdAt: Date.now(),
      centroids,
      labelMap: [...readyLetters],
      metrics: {
        user_inputtedAccuracy: 0,
        implicitAccuracy: 0,
        overallAccuracy: 0,
      },
      datasetSize: samples.length,
      readyLetters: [...readyLetters],
      reason,
    };
  } finally {
    freeAllocation(sampleFeaturesAllocation);
    freeAllocation(sampleLabelsAllocation);
    freeAllocation(readyLabelsAllocation);
    freeAllocation(outCentroidsAllocation);
    freeAllocation(outCountsAllocation);
  }
}

export function predictFeatureClassifierProbabilitiesRust(
  snapshot: FeatureClassifierSnapshot | null,
  features: number[],
): Record<string, number> {
  const probabilities = Object.fromEntries(ALPHABET.map((label) => [label, 0])) as Record<string, number>;
  if (!snapshot || snapshot.centroids.length === 0) {
    const uniform = 1 / ALPHABET.length;
    ALPHABET.forEach((label) => {
      probabilities[label] = uniform;
    });
    return probabilities;
  }

  const exports = requireExports();
  const centroidValues = flattenCentroids(snapshot.centroids);
  const centroidLabels = new Uint8Array(snapshot.centroids.map((centroid) => labelToIndex(centroid.label)));
  const featureValues = featuresArray(features);

  const centroidsAllocation = mallocCopy(centroidValues);
  const labelsAllocation = mallocCopy(centroidLabels);
  const featuresAllocation = mallocCopy(featureValues);
  const outAllocation = mallocZeroed(ALPHABET.length * Float64Array.BYTES_PER_ELEMENT);

  try {
    exports.predict_centroid_classifier(
      centroidsAllocation.ptr,
      labelsAllocation.ptr,
      snapshot.centroids.length,
      featuresAllocation.ptr,
      outAllocation.ptr,
    );
    readFloat64(outAllocation.ptr, ALPHABET.length).forEach((score, index) => {
      probabilities[indexToLabel(index)] = score;
    });
    return probabilities;
  } finally {
    freeAllocation(centroidsAllocation);
    freeAllocation(labelsAllocation);
    freeAllocation(featuresAllocation);
    freeAllocation(outAllocation);
  }
}

export function computeLetterStatsRust(ledger: AcceptedSampleRecord[]): {
  countsByLetter: Record<string, LetterAcceptanceCounts>;
  readyLetters: string[];
  priorityLetters: string[];
  mostNeededLetter: string | null;
} {
  const countsByLetter = Object.fromEntries(
    ALPHABET.map((label) => [label, { user_inputted: 0, implicit: 0 }]),
  ) as Record<string, LetterAcceptanceCounts>;

  if (ledger.length === 0) {
    return {
      countsByLetter,
      readyLetters: [],
      priorityLetters: [],
      mostNeededLetter: null,
    };
  }

  const exports = requireExports();
  const labels = new Uint8Array(ledger.map((sample) => labelToIndex(sample.label)));
  const acceptances = new Uint8Array(ledger.map((sample) => acceptanceToByte(sample.acceptance)));
  const labelsAllocation = mallocCopy(labels);
  const acceptancesAllocation = mallocCopy(acceptances);
  const countsAllocation = mallocZeroed(ALPHABET.length * 2 * Uint32Array.BYTES_PER_ELEMENT);
  const readyAllocation = mallocZeroed(ALPHABET.length * Uint8Array.BYTES_PER_ELEMENT);
  const priorityAllocation = mallocZeroed(ALPHABET.length * Uint8Array.BYTES_PER_ELEMENT);

  try {
    const mostNeededIndex = exports.compute_letter_stats(
      labelsAllocation.ptr,
      acceptancesAllocation.ptr,
      ledger.length,
      countsAllocation.ptr,
      readyAllocation.ptr,
      priorityAllocation.ptr,
    );
    const counts = readUint32(countsAllocation.ptr, ALPHABET.length * 2);
    const readyFlags = readUint8(readyAllocation.ptr, ALPHABET.length);
    const priorityFlags = readUint8(priorityAllocation.ptr, ALPHABET.length);

    ALPHABET.forEach((label, index) => {
      countsByLetter[label] = {
        user_inputted: counts[index * 2] ?? 0,
        implicit: counts[(index * 2) + 1] ?? 0,
      };
    });

    return {
      countsByLetter,
      readyLetters: ALPHABET.filter((_, index) => readyFlags[index] === 1),
      priorityLetters: ALPHABET.filter((_, index) => priorityFlags[index] === 1),
      mostNeededLetter: mostNeededIndex >= 0 ? indexToLabel(mostNeededIndex) : null,
    };
  } finally {
    freeAllocation(labelsAllocation);
    freeAllocation(acceptancesAllocation);
    freeAllocation(countsAllocation);
    freeAllocation(readyAllocation);
    freeAllocation(priorityAllocation);
  }
}

export function buildBalancedDatasetRust(ledger: AcceptedSampleRecord[]): BalancedDataset {
  if (ledger.length === 0) {
    return {
      training: [],
      holdout: [],
      readyLetters: [],
      perLetterTarget: 0,
    };
  }

  const exports = requireExports();
  const labels = new Uint8Array(ledger.map((sample) => labelToIndex(sample.label)));
  const acceptances = new Uint8Array(ledger.map((sample) => acceptanceToByte(sample.acceptance)));
  const createdAt = new Float64Array(ledger.map((sample) => sample.createdAt));
  const labelsAllocation = mallocCopy(labels);
  const acceptancesAllocation = mallocCopy(acceptances);
  const createdAtAllocation = mallocCopy(createdAt);
  const trainingMaskAllocation = mallocZeroed(ledger.length * Uint8Array.BYTES_PER_ELEMENT);
  const holdoutMaskAllocation = mallocZeroed(ledger.length * Uint8Array.BYTES_PER_ELEMENT);
  const readyAllocation = mallocZeroed(ALPHABET.length * Uint8Array.BYTES_PER_ELEMENT);

  try {
    const perLetterTarget = exports.build_balanced_dataset(
      labelsAllocation.ptr,
      acceptancesAllocation.ptr,
      createdAtAllocation.ptr,
      ledger.length,
      trainingMaskAllocation.ptr,
      holdoutMaskAllocation.ptr,
      readyAllocation.ptr,
    );
    const trainingMask = readUint8(trainingMaskAllocation.ptr, ledger.length);
    const holdoutMask = readUint8(holdoutMaskAllocation.ptr, ledger.length);
    const readyFlags = readUint8(readyAllocation.ptr, ALPHABET.length);

    return {
      training: ledger.filter((_, index) => trainingMask[index] === 1),
      holdout: ledger.filter((_, index) => holdoutMask[index] === 1),
      readyLetters: ALPHABET.filter((_, index) => readyFlags[index] === 1),
      perLetterTarget,
    };
  } finally {
    freeAllocation(labelsAllocation);
    freeAllocation(acceptancesAllocation);
    freeAllocation(createdAtAllocation);
    freeAllocation(trainingMaskAllocation);
    freeAllocation(holdoutMaskAllocation);
    freeAllocation(readyAllocation);
  }
}

export function evaluateSnapshotRust(
  snapshot: FeatureClassifierSnapshot | null,
  holdout: AcceptedSampleRecord[],
): SnapshotMetrics {
  if (!snapshot || snapshot.centroids.length === 0 || holdout.length === 0) {
    return {
      user_inputtedAccuracy: 0,
      implicitAccuracy: 0,
      overallAccuracy: 0,
    };
  }

  const exports = requireExports();
  const holdoutFeatures = flattenSampleFeatures(holdout);
  const holdoutLabels = new Uint8Array(holdout.map((sample) => labelToIndex(sample.label)));
  const holdoutAcceptances = new Uint8Array(holdout.map((sample) => acceptanceToByte(sample.acceptance)));
  const centroidValues = flattenCentroids(snapshot.centroids);
  const centroidLabels = new Uint8Array(snapshot.centroids.map((centroid) => labelToIndex(centroid.label)));

  const holdoutFeaturesAllocation = mallocCopy(holdoutFeatures);
  const holdoutLabelsAllocation = mallocCopy(holdoutLabels);
  const holdoutAcceptancesAllocation = mallocCopy(holdoutAcceptances);
  const centroidValuesAllocation = mallocCopy(centroidValues);
  const centroidLabelsAllocation = mallocCopy(centroidLabels);
  const metricsAllocation = mallocZeroed(3 * Float64Array.BYTES_PER_ELEMENT);

  try {
    exports.evaluate_snapshot(
      holdoutFeaturesAllocation.ptr,
      holdoutLabelsAllocation.ptr,
      holdoutAcceptancesAllocation.ptr,
      holdout.length,
      centroidValuesAllocation.ptr,
      centroidLabelsAllocation.ptr,
      snapshot.centroids.length,
      metricsAllocation.ptr,
    );
    const [userInputtedAccuracy, implicitAccuracy, overallAccuracy] = readFloat64(metricsAllocation.ptr, 3);
    return {
      user_inputtedAccuracy: userInputtedAccuracy ?? 0,
      implicitAccuracy: implicitAccuracy ?? 0,
      overallAccuracy: overallAccuracy ?? 0,
    };
  } finally {
    freeAllocation(holdoutFeaturesAllocation);
    freeAllocation(holdoutLabelsAllocation);
    freeAllocation(holdoutAcceptancesAllocation);
    freeAllocation(centroidValuesAllocation);
    freeAllocation(centroidLabelsAllocation);
    freeAllocation(metricsAllocation);
  }
}
