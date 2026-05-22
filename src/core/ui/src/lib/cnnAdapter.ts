import type { BaselineManifest, LabCandidate, LabCnnArtifacts, LabSample, SnapshotMetrics, StrokeInput } from './types.ts';

export const MIN_CNN_FINE_TUNE_SAMPLES = 8;
export const RECOMMENDED_CNN_SAMPLES_PER_LETTER = 10;
export const STRONG_CNN_SAMPLES_PER_LETTER = 20;

const DEFAULT_BATCH_SIZE = 8;
const DEFAULT_HEAD_EPOCHS = 4;
const DEFAULT_PARTIAL_EPOCHS = 2;
const DEFAULT_LEARNING_RATE = 0.001;
const BATCH_MAGIC = [0x42, 0x43, 0x4e, 0x4e]; // BCNN
const BATCH_VERSION = 1;

export interface CnnAvailability {
  available: boolean;
  reasons: string[];
}

export interface CnnTrainingProgress {
  step: 'preparing' | 'downloading' | 'training' | 'evaluating' | 'exporting' | 'ready';
  progress: number;
  message: string;
}

interface BurnFineTuneResult {
  model_bytes: Uint8Array;
  optimizer_bytes: Uint8Array;
  metadata_json: string;
  average_loss: number;
  trained_samples: number;
}

interface BurnWasmCnn {
  predict_strokes(strokesJson: string): Promise<Float32Array>;
  fine_tune_head(
    batchBytes: Uint8Array,
    epochs?: number,
    learningRate?: number,
    batchSize?: number,
  ): Promise<BurnFineTuneResult>;
}

interface BurnCnnModule {
  default?: (input?: string | URL | Request) => Promise<unknown>;
  WasmCnn: {
    from_safetensors(
      modelBytes: Uint8Array,
      configJson: string,
      optimizerBytes?: Uint8Array | null,
    ): Promise<BurnWasmCnn>;
  };
}

function labelIndex(label: string): number {
  const index = label.charCodeAt(0) - 65;
  return index >= 0 && index < 26 ? index : 0;
}

function copyArrayBuffer(value: ArrayBuffer | Uint8Array): ArrayBuffer {
  const source = value instanceof Uint8Array ? value : new Uint8Array(value);
  const copy = new Uint8Array(source.byteLength);
  copy.set(source);
  return copy.buffer;
}

async function fetchArrayBuffer(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
  }
  return await response.arrayBuffer();
}

async function fetchText(url: string): Promise<string | null> {
  const response = await fetch(url);
  return response.ok ? await response.text() : null;
}

export async function loadBaselineManifest(modelBaseUrl = '/models'): Promise<BaselineManifest | null> {
  const response = await fetch(`${modelBaseUrl.replace(/\/$/, '')}/manifest.json`);
  if (!response.ok) {
    return null;
  }
  return (await response.json()) as BaselineManifest;
}

export function getCnnAvailability(manifest: BaselineManifest | null): CnnAvailability {
  const reasons: string[] = [];
  if (!manifest) {
    reasons.push('baseline manifest is unavailable');
  } else {
    if (!manifest.cnn.supportsTraining) reasons.push('manifest disables browser CNN fine-tuning');
    if (!manifest.cnn.trainingRuntime.moduleUrl) reasons.push('missing Burn CNN JS module URL');
    if (!manifest.cnn.trainingRuntime.wasmUrl) reasons.push('missing Burn CNN WASM URL');
    if (!manifest.cnn.trainingArtifacts.checkpointUrl) reasons.push('missing baseline safetensors URL');
  }
  return { available: reasons.length === 0, reasons };
}

async function configJson(manifest: BaselineManifest, artifacts: LabCnnArtifacts | null): Promise<string> {
  if (artifacts?.modelConfigJson) {
    return artifacts.modelConfigJson;
  }
  const configured = manifest.cnn.trainingArtifacts.exportMetadataUrl
    ? await fetchText(manifest.cnn.trainingArtifacts.exportMetadataUrl)
    : null;
  return configured ?? JSON.stringify({
    architecture: 'pico-dual-cnn-v1',
    labels: manifest.labelMap,
    hidden_size: 128,
    one_d_shape: [3, 64],
    two_d_shape: [1, 28, 28],
    trained_epochs: 0,
    batch_size: 0,
    learning_rate: 0,
    seed: 0,
    train_samples: 0,
    validation_samples: 0,
  });
}

async function loadBurnModule(manifest: BaselineManifest): Promise<BurnCnnModule> {
  const moduleUrl = manifest.cnn.trainingRuntime.moduleUrl;
  if (!moduleUrl) {
    throw new Error('Missing Burn CNN JS module URL.');
  }
  const module = await import(/* @vite-ignore */ moduleUrl) as BurnCnnModule;
  await module.default?.(manifest.cnn.trainingRuntime.wasmUrl ?? undefined);
  if (!module.WasmCnn?.from_safetensors) {
    throw new Error('Burn CNN module does not expose WasmCnn.from_safetensors.');
  }
  return module;
}

async function createBurnCnn(manifest: BaselineManifest, artifacts: LabCnnArtifacts | null): Promise<BurnWasmCnn> {
  const checkpointUrl = manifest.cnn.trainingArtifacts.checkpointUrl;
  if (!artifacts && !checkpointUrl) {
    throw new Error('No baseline safetensors URL configured.');
  }
  const module = await loadBurnModule(manifest);
  const modelBytes = artifacts?.modelSafetensors ?? await fetchArrayBuffer(checkpointUrl!);
  const optimizerBytes = artifacts?.optimizerState?.byteLength ? new Uint8Array(artifacts.optimizerState) : null;
  return await module.WasmCnn.from_safetensors(
    new Uint8Array(modelBytes),
    await configJson(manifest, artifacts),
    optimizerBytes,
  );
}

function serializeTrainingBatch(samples: LabSample[]): Uint8Array {
  let byteLength = 12;
  for (const sample of samples) {
    byteLength += 5;
    for (const stroke of sample.strokes) {
      byteLength += 4 + stroke.length * 12;
    }
  }

  const bytes = new Uint8Array(byteLength);
  const view = new DataView(bytes.buffer);
  let offset = 0;
  for (const value of BATCH_MAGIC) {
    bytes[offset] = value;
    offset += 1;
  }
  view.setUint32(offset, BATCH_VERSION, true);
  offset += 4;
  view.setUint32(offset, samples.length, true);
  offset += 4;

  for (const sample of samples) {
    bytes[offset] = labelIndex(sample.label);
    offset += 1;
    view.setUint32(offset, sample.strokes.length, true);
    offset += 4;
    for (const stroke of sample.strokes) {
      view.setUint32(offset, stroke.length, true);
      offset += 4;
      for (const point of stroke) {
        view.setFloat32(offset, point.x, true);
        view.setFloat32(offset + 4, point.y, true);
        view.setFloat32(offset + 8, point.t, true);
        offset += 12;
      }
    }
  }
  return bytes;
}

function candidatesFromScores(scores: ArrayLike<number>, labelMap: string[]): LabCandidate[] {
  return Array.from(scores)
    .map((score, index) => ({ label: labelMap[index] ?? '?', score }))
    .filter((candidate) => /^[A-Z]$/.test(candidate.label))
    .sort((left, right) => right.score - left.score);
}

async function evaluate(cnn: BurnWasmCnn, samples: LabSample[], labelMap: string[]): Promise<SnapshotMetrics> {
  if (samples.length === 0) {
    return { user_inputtedAccuracy: 0, implicitAccuracy: 0, overallAccuracy: 0 };
  }
  let correct = 0;
  for (const sample of samples) {
    const scores = await cnn.predict_strokes(JSON.stringify({ strokes: sample.strokes }));
    const top = candidatesFromScores(scores, labelMap)[0]?.label ?? null;
    if (top === sample.label) {
      correct += 1;
    }
  }
  const overallAccuracy = correct / samples.length;
  return { user_inputtedAccuracy: overallAccuracy, implicitAccuracy: 0, overallAccuracy };
}

export async function predictCnn(
  strokes: StrokeInput,
  manifest: BaselineManifest | null,
  artifacts: LabCnnArtifacts | null,
): Promise<{ candidates: LabCandidate[]; source: 'baseline' | 'personalized' } | null> {
  if (!manifest || (!artifacts && !getCnnAvailability(manifest).available)) {
    return null;
  }
  const cnn = await createBurnCnn(manifest, artifacts);
  const labelMap = manifest.labelMap?.length ? manifest.labelMap : Array.from({ length: 26 }, (_, index) => String.fromCharCode(65 + index));
  const scores = await cnn.predict_strokes(JSON.stringify({ strokes }));
  return {
    source: artifacts ? 'personalized' : 'baseline',
    candidates: candidatesFromScores(scores, labelMap),
  };
}

export async function trainCnn(
  manifest: BaselineManifest | null,
  samples: LabSample[],
  previousArtifacts: LabCnnArtifacts | null,
  onProgress?: (progress: CnnTrainingProgress) => void,
): Promise<{ artifacts: LabCnnArtifacts | null; metrics: SnapshotMetrics | null; rejectionReason: string | null }> {
  const availability = getCnnAvailability(manifest);
  if (!availability.available || !manifest) {
    return { artifacts: null, metrics: null, rejectionReason: availability.reasons.join('; ') };
  }
  if (samples.length < MIN_CNN_FINE_TUNE_SAMPLES) {
    return { artifacts: null, metrics: null, rejectionReason: `need at least ${MIN_CNN_FINE_TUNE_SAMPLES} labeled samples to run a browser fine-tune` };
  }

  try {
    const stage = previousArtifacts?.stage === 'head-only' ? 'partial-finetune' : 'head-only';
    onProgress?.({ step: 'preparing', progress: 0, message: 'loading Burn CNN runtime' });
    const cnn = await createBurnCnn(manifest, previousArtifacts);
    onProgress?.({ step: 'training', progress: 0.2, message: `fine-tuning ${stage} on ${samples.length} samples` });
    const result = await cnn.fine_tune_head(
      serializeTrainingBatch(samples),
      stage === 'partial-finetune' ? DEFAULT_PARTIAL_EPOCHS : DEFAULT_HEAD_EPOCHS,
      DEFAULT_LEARNING_RATE,
      DEFAULT_BATCH_SIZE,
    );

    onProgress?.({ step: 'exporting', progress: 0.78, message: `exported average loss ${result.average_loss.toFixed(4)}` });
    const artifacts: LabCnnArtifacts = {
      runtime: 'burn-wasm',
      modelSafetensors: copyArrayBuffer(result.model_bytes),
      optimizerState: copyArrayBuffer(result.optimizer_bytes),
      modelConfigJson: result.metadata_json,
      metrics: null,
      stage,
      updatedAt: Date.now(),
    };
    onProgress?.({ step: 'evaluating', progress: 0.88, message: 'evaluating tuned model on local ledger' });
    artifacts.metrics = await evaluate(cnn, samples, manifest.labelMap);
    onProgress?.({ step: 'ready', progress: 1, message: 'CNN fine-tune complete' });
    return { artifacts, metrics: artifacts.metrics, rejectionReason: null };
  } catch (error) {
    return { artifacts: null, metrics: null, rejectionReason: error instanceof Error ? error.message : 'CNN fine-tuning failed' };
  }
}
