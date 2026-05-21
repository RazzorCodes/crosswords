import { InferenceSession, Tensor, env as ortEnv } from 'onnxruntime-web';
import { renderStrokesToPixels } from './canvas.ts';
import type { BaselineManifest, LabCandidate, LabCnnArtifacts, LabSample, SnapshotMetrics, StrokeInput } from './types.ts';

const IMAGE_SIZE = 64;

export interface CnnAvailability {
  available: boolean;
  reasons: string[];
}

export interface CnnTrainingProgress {
  step: 'preparing' | 'downloading' | 'training' | 'evaluating' | 'exporting' | 'ready';
  progress: number;
  message: string;
}

interface OrtTrainingSession {
  trainStep(feeds: Record<string, Tensor>): Promise<unknown>;
  optimizerStep?(): Promise<void>;
  lazyResetGrad?(): Promise<void>;
  exportModelForInferencing?(outputNames?: string[]): Promise<ArrayBuffer | Uint8Array>;
  getContiguousParameters?(trainableOnly?: boolean): Promise<ArrayBuffer | Uint8Array>;
}

interface OrtTrainingModule {
  env?: { wasm?: { wasmPaths?: string | Record<string, string>; numThreads?: number } };
  Tensor?: typeof Tensor;
  TrainingSession?: {
    create(options: Record<string, unknown>): Promise<OrtTrainingSession>;
  };
}

function configureOrt(): void {
  if (ortEnv.wasm) {
    ortEnv.wasm.proxy = false;
    ortEnv.wasm.numThreads = 1;
  }
}

function labelIndex(label: string): number {
  const index = label.charCodeAt(0) - 65;
  return index >= 0 && index < 26 ? index : 0;
}

function toArrayBuffer(value: ArrayBuffer | Uint8Array): ArrayBuffer {
  if (value instanceof ArrayBuffer) {
    return value;
  }
  const copy = new Uint8Array(value.byteLength);
  copy.set(value);
  return copy.buffer;
}

function createBatch(samples: LabSample[], TensorCtor: typeof Tensor): Record<string, Tensor> {
  const pixels = new Float32Array(samples.length * IMAGE_SIZE * IMAGE_SIZE);
  const labels = new BigInt64Array(samples.length);
  samples.forEach((sample, index) => {
    pixels.set(renderStrokesToPixels(sample.strokes), index * IMAGE_SIZE * IMAGE_SIZE);
    labels[index] = BigInt(labelIndex(sample.label));
  });
  return {
    input: new TensorCtor('float32', pixels, [samples.length, 1, IMAGE_SIZE, IMAGE_SIZE]),
    labels: new TensorCtor('int64', labels, [samples.length]),
  };
}

async function fetchArrayBuffer(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
  }
  return await response.arrayBuffer();
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
    if (!manifest.cnn.supportsTraining) reasons.push('manifest disables CNN training');
    if (!manifest.cnn.trainingRuntime.moduleUrl) reasons.push('missing ORT Web training module');
    if (!manifest.cnn.trainingArtifacts.trainUrl) reasons.push('missing CNN training model');
    if (!manifest.cnn.trainingArtifacts.evalUrl) reasons.push('missing CNN eval model');
    if (!manifest.cnn.trainingArtifacts.optimizerUrl) reasons.push('missing CNN optimizer model');
    if (!manifest.cnn.trainingArtifacts.checkpointUrl) reasons.push('missing CNN checkpoint');
  }
  return { available: reasons.length === 0, reasons };
}

export async function predictCnn(
  strokes: StrokeInput,
  manifest: BaselineManifest | null,
  artifacts: LabCnnArtifacts | null,
): Promise<{ candidates: LabCandidate[]; source: 'baseline' | 'personalized' } | null> {
  const model = artifacts?.inferenceModel ?? manifest?.cnn.inferenceUrl ?? null;
  if (!model) {
    return null;
  }
  configureOrt();
  const session = typeof model === 'string'
    ? await InferenceSession.create(model)
    : await InferenceSession.create(model);
  const input = new Tensor('float32', renderStrokesToPixels(strokes), [1, 1, IMAGE_SIZE, IMAGE_SIZE]);
  const output = await session.run({ [session.inputNames[0]]: input });
  const scores = Array.from((output[session.outputNames[0]] as { data: ArrayLike<number> }).data);
  const labelMap = manifest?.labelMap?.length ? manifest.labelMap : Array.from({ length: 26 }, (_, index) => String.fromCharCode(65 + index));
  return {
    source: artifacts?.inferenceModel ? 'personalized' : 'baseline',
    candidates: scores
      .map((score, index) => ({ label: labelMap[index] ?? '?', score }))
      .filter((candidate) => /^[A-Z]$/.test(candidate.label))
      .sort((left, right) => right.score - left.score),
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
  if (samples.length < 4) {
    return { artifacts: null, metrics: null, rejectionReason: 'need at least four lab samples for CNN training' };
  }

  onProgress?.({ step: 'preparing', progress: 0, message: 'preparing CNN training' });
  try {
    const ort = (await import(/* @vite-ignore */ manifest.cnn.trainingRuntime.moduleUrl!)) as OrtTrainingModule;
    if (ort.env?.wasm) {
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = manifest.cnn.trainingRuntime.wasmUrl ?? manifest.cnn.trainingRuntime.simdWasmUrl ?? undefined;
    }
    if (!ort.TrainingSession?.create) {
      throw new Error('ORT runtime does not expose TrainingSession.create');
    }
    onProgress?.({ step: 'downloading', progress: 0.1, message: 'downloading CNN artifacts' });
    const [trainModel, evalModel, optimizerModel, checkpoint] = await Promise.all([
      fetchArrayBuffer(manifest.cnn.trainingArtifacts.trainUrl!),
      fetchArrayBuffer(manifest.cnn.trainingArtifacts.evalUrl!),
      fetchArrayBuffer(manifest.cnn.trainingArtifacts.optimizerUrl!),
      fetchArrayBuffer(manifest.cnn.trainingArtifacts.checkpointUrl!),
    ]);
    const session = await ort.TrainingSession.create({
      trainModel,
      evalModel,
      optimizerModel,
      checkpoint: previousArtifacts?.checkpoint ?? checkpoint,
      stage: previousArtifacts?.stage === 'head-only' ? 'partial-finetune' : 'head-only',
    });
    const TensorCtor = ort.Tensor ?? Tensor;
    onProgress?.({ step: 'training', progress: 0.35, message: 'training CNN candidate' });
    await session.trainStep(createBatch(samples, TensorCtor));
    await session.optimizerStep?.();
    await session.lazyResetGrad?.();
    onProgress?.({ step: 'exporting', progress: 0.75, message: 'exporting CNN inference model' });
    const exported = await session.exportModelForInferencing?.(['output']);
    if (!exported) {
      throw new Error('CNN training did not export an inference model');
    }
    const exportedCheckpoint = session.getContiguousParameters
      ? toArrayBuffer(await session.getContiguousParameters(false))
      : checkpoint;
    const artifacts: LabCnnArtifacts = {
      checkpoint: exportedCheckpoint,
      inferenceModel: toArrayBuffer(exported),
      exportMetadata: null,
      metrics: { user_inputtedAccuracy: 0, implicitAccuracy: 0, overallAccuracy: 0 },
      stage: previousArtifacts?.stage === 'head-only' ? 'partial-finetune' : 'head-only',
      updatedAt: Date.now(),
    };
    onProgress?.({ step: 'ready', progress: 1, message: 'CNN candidate ready' });
    return { artifacts, metrics: artifacts.metrics ?? null, rejectionReason: null };
  } catch (error) {
    return { artifacts: null, metrics: null, rejectionReason: error instanceof Error ? error.message : 'CNN training failed' };
  }
}
