import { InferenceSession, Tensor, env as ortEnv } from 'onnxruntime-web';
import { renderStrokesToPixels } from '../recognizers/rasterizer';
import type {
  AcceptedSampleRecord,
  BaselineArtifactManifest,
  CnnTrainingAvailability,
  CnnTrainingProgress,
  CnnTrainingStage,
  PersonalizedCnnArtifacts,
  SnapshotMetrics,
} from './types';

const IMAGE_SIZE = 64;
const DEFAULT_BATCH_SIZE = 8;
const DEFAULT_HEAD_EPOCHS = 4;
const DEFAULT_PARTIAL_EPOCHS = 2;
const MAX_TRAINING_MS = 15_000;
const PARTIAL_FINETUNE_MIN_SAMPLES_PER_LETTER = 12;

function configureOrtInferenceRuntime(): void {
  if (!ortEnv.wasm) {
    return;
  }
  ortEnv.wasm.proxy = false;
  ortEnv.wasm.numThreads = 1;
}

interface OrtTrainingSession {
  trainStep(feeds: Record<string, Tensor>): Promise<unknown>;
  optimizerStep?(): Promise<void>;
  lazyResetGrad?(): Promise<void>;
  evalStep?(feeds: Record<string, Tensor>): Promise<Record<string, { data: ArrayLike<number> }>>;
  exportModelForInferencing?(outputNames?: string[]): Promise<ArrayBuffer | Uint8Array>;
  getContiguousParameters?(trainableOnly?: boolean): Promise<ArrayBuffer | Uint8Array>;
}

interface OrtTrainingModule {
  env?: {
    wasm?: {
      wasmPaths?: string | Record<string, string>;
      numThreads?: number;
    };
  };
  Tensor?: typeof Tensor;
  TrainingSession?: {
    create(options: Record<string, unknown>): Promise<OrtTrainingSession>;
  };
}

export interface CnnTrainingRuntimeResult {
  artifacts: PersonalizedCnnArtifacts | null;
  metrics: SnapshotMetrics;
  stage: CnnTrainingStage;
  accepted: boolean;
  rejectionReason: string | null;
}

export interface CnnTrainingRuntimeOptions {
  onProgress?: (progress: CnnTrainingProgress) => void;
}

export interface CnnTrainingRuntimeLike {
  isAvailable(): boolean;
  getAvailability?(): CnnTrainingAvailability;
  trainCandidate(
    training: AcceptedSampleRecord[],
    holdout: AcceptedSampleRecord[],
    previousArtifacts: PersonalizedCnnArtifacts | null,
    options?: CnnTrainingRuntimeOptions,
  ): Promise<CnnTrainingRuntimeResult>;
}

function labelIndex(label: string): number {
  const index = label.charCodeAt(0) - 65;
  return index >= 0 && index < 26 ? index : 0;
}

export function chooseCnnTrainingStage(
  training: AcceptedSampleRecord[],
  previousArtifacts: PersonalizedCnnArtifacts | null,
): CnnTrainingStage {
  const counts = new Map<string, number>();
  for (const sample of training) {
    counts.set(sample.label, (counts.get(sample.label) ?? 0) + 1);
  }
  const canPartialFinetune = previousArtifacts?.stage === 'head-only'
    && [...counts.values()].filter((count) => count >= PARTIAL_FINETUNE_MIN_SAMPLES_PER_LETTER).length >= 2;
  return canPartialFinetune ? 'partial-finetune' : 'head-only';
}

function createBatch(samples: AcceptedSampleRecord[], TensorCtor: typeof Tensor): Record<string, Tensor> {
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

function batches(samples: AcceptedSampleRecord[], size = DEFAULT_BATCH_SIZE): AcceptedSampleRecord[][] {
  const result: AcceptedSampleRecord[][] = [];
  for (let index = 0; index < samples.length; index += size) {
    result.push(samples.slice(index, index + size));
  }
  return result;
}

async function evaluateExportedModel(
  inferenceModel: ArrayBuffer,
  holdout: AcceptedSampleRecord[],
): Promise<SnapshotMetrics> {
  if (holdout.length === 0) {
    return {
      user_inputtedAccuracy: 0,
      implicitAccuracy: 0,
      overallAccuracy: 0,
    };
  }

  configureOrtInferenceRuntime();
  const session = await InferenceSession.create(inferenceModel);
  let userTotal = 0;
  let userCorrect = 0;
  let implicitTotal = 0;
  let implicitCorrect = 0;
  let overallCorrect = 0;

  for (const sample of holdout) {
    const input = new Tensor('float32', renderStrokesToPixels(sample.strokes), [1, 1, IMAGE_SIZE, IMAGE_SIZE]);
    const output = await session.run({ [session.inputNames[0]]: input });
    const scores = Array.from((output[session.outputNames[0]] as { data: ArrayLike<number> }).data);
    const predicted = scores.reduce((best, score, index) => (score > scores[best] ? index : best), 0);
    const correct = predicted === labelIndex(sample.label);
    if (correct) {
      overallCorrect += 1;
    }
    if (sample.acceptance === 'user_inputted') {
      userTotal += 1;
      if (correct) {
        userCorrect += 1;
      }
    } else {
      implicitTotal += 1;
      if (correct) {
        implicitCorrect += 1;
      }
    }
  }

  return {
    user_inputtedAccuracy: userTotal > 0 ? userCorrect / userTotal : 0,
    implicitAccuracy: implicitTotal > 0 ? implicitCorrect / implicitTotal : 0,
    overallAccuracy: overallCorrect / holdout.length,
  };
}

async function fetchArrayBuffer(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
  }
  return await response.arrayBuffer();
}

function toArrayBuffer(value: ArrayBuffer | Uint8Array): ArrayBuffer {
  if (value instanceof ArrayBuffer) {
    return value;
  }
  const copy = new Uint8Array(value.byteLength);
  copy.set(value);
  return copy.buffer;
}

function unavailableResult(stage: CnnTrainingStage, reason: string): CnnTrainingRuntimeResult {
  return {
    artifacts: null,
    metrics: { user_inputtedAccuracy: 0, implicitAccuracy: 0, overallAccuracy: 0 },
    stage,
    accepted: false,
    rejectionReason: reason,
  };
}

export function getCnnTrainingAvailability(manifest: BaselineArtifactManifest): CnnTrainingAvailability {
  const reasons: string[] = [];
  const { cnn } = manifest;
  if (!cnn.supportsTraining) {
    reasons.push('manifest disables CNN training');
  }
  if (!cnn.trainingRuntime.moduleUrl) {
    reasons.push('missing ORT Web training module');
  }
  if (!cnn.trainingRuntime.wasmUrl && !cnn.trainingRuntime.simdWasmUrl && !cnn.trainingRuntime.threadedWasmUrl) {
    reasons.push('missing ORT Web training WASM');
  }
  if (!cnn.trainingArtifacts.trainUrl) {
    reasons.push('missing CNN training model');
  }
  if (!cnn.trainingArtifacts.evalUrl) {
    reasons.push('missing CNN eval model');
  }
  if (!cnn.trainingArtifacts.optimizerUrl) {
    reasons.push('missing CNN optimizer model');
  }
  if (!cnn.trainingArtifacts.checkpointUrl) {
    reasons.push('missing CNN checkpoint');
  }
  return {
    available: reasons.length === 0,
    reasons,
  };
}

function progressMessage(progress: CnnTrainingProgress): CnnTrainingProgress {
  return progress;
}

export async function trainCnnCandidateDirect(
  manifest: BaselineArtifactManifest,
  training: AcceptedSampleRecord[],
  holdout: AcceptedSampleRecord[],
  previousArtifacts: PersonalizedCnnArtifacts | null,
  options: CnnTrainingRuntimeOptions = {},
): Promise<CnnTrainingRuntimeResult> {
  const stage = chooseCnnTrainingStage(training, previousArtifacts);
  const startedAt = Date.now();
  const availability = getCnnTrainingAvailability(manifest);
  if (!availability.available) {
    return unavailableResult(stage, availability.reasons.join('; '));
  }

  const emit = (progress: Omit<CnnTrainingProgress, 'stage' | 'elapsedMs'>) => {
    options.onProgress?.(progressMessage({
      ...progress,
      stage,
      elapsedMs: Date.now() - startedAt,
    }));
  };

  try {
    emit({ step: 'preparing', progress: 0, message: 'cnn preparing...' });
    const moduleUrl = manifest.cnn.trainingRuntime.moduleUrl!;
    const ort = await import(/* @vite-ignore */ moduleUrl) as OrtTrainingModule;
    if (ort.env?.wasm) {
      const wasmBase = manifest.cnn.trainingRuntime.wasmUrl;
      if (wasmBase) {
        ort.env.wasm.wasmPaths = wasmBase;
      }
      ort.env.wasm.numThreads = 1;
    }
    const TrainingSession = ort.TrainingSession;
    if (!TrainingSession?.create) {
      throw new Error('Release ORT runtime does not expose TrainingSession.create.');
    }

    emit({ step: 'downloading', progress: 0.08, message: 'cnn downloading artifacts...' });
    const artifacts = manifest.cnn.trainingArtifacts;
    const [trainModel, evalModel, optimizerModel, baselineCheckpoint] = await Promise.all([
      fetchArrayBuffer(artifacts.trainUrl!),
      fetchArrayBuffer(artifacts.evalUrl!),
      fetchArrayBuffer(artifacts.optimizerUrl!),
      fetchArrayBuffer(artifacts.checkpointUrl!),
    ]);
    const checkpoint = previousArtifacts?.checkpoint ?? baselineCheckpoint;
    emit({ step: 'preparing', progress: 0.18, message: 'cnn creating training session...' });
    const session = await TrainingSession.create({
      trainModel,
      evalModel,
      optimizerModel,
      checkpoint,
      stage,
    });

    const TensorCtor = ort.Tensor ?? Tensor;
    const epochs = stage === 'partial-finetune' ? DEFAULT_PARTIAL_EPOCHS : DEFAULT_HEAD_EPOCHS;
    const trainingBatches = batches(training);
    const batchCount = Math.max(trainingBatches.length, 1);
    for (let epoch = 0; epoch < epochs; epoch += 1) {
      for (let batchIndex = 0; batchIndex < trainingBatches.length; batchIndex += 1) {
        if (Date.now() - startedAt > MAX_TRAINING_MS) {
          throw new Error('CNN training exceeded the browser time budget.');
        }
        const trainResult = await session.trainStep(createBatch(trainingBatches[batchIndex], TensorCtor));
        await session.optimizerStep?.();
        await session.lazyResetGrad?.();
        const rawLoss = trainResult && typeof trainResult === 'object' && 'loss' in trainResult
          ? Number((trainResult as { loss: unknown }).loss)
          : undefined;
        const completed = epoch * batchCount + batchIndex + 1;
        const total = epochs * batchCount;
        emit({
          step: 'training',
          progress: 0.18 + (0.58 * completed / total),
          message: `cnn epoch ${epoch + 1}/${epochs} batch ${batchIndex + 1}/${batchCount}`,
          epoch: epoch + 1,
          epochs,
          batch: batchIndex + 1,
          batches: batchCount,
          loss: Number.isFinite(rawLoss) ? rawLoss : undefined,
        });
      }
    }

    emit({ step: 'exporting', progress: 0.82, message: 'cnn exporting inference model...' });
    const exported = await session.exportModelForInferencing?.(['output']);
    if (!exported) {
      throw new Error('CNN training runtime did not export an inference model.');
    }
    const inferenceModel = toArrayBuffer(exported);
    const exportedCheckpoint = session.getContiguousParameters
      ? toArrayBuffer(await session.getContiguousParameters(false))
      : checkpoint;
    const exportMetadata = artifacts.exportMetadataUrl
      ? await fetch(artifacts.exportMetadataUrl).then((response) => response.ok ? response.json() as Promise<Record<string, unknown>> : null)
      : null;
    emit({ step: 'evaluating', progress: 0.90, message: 'cnn evaluating candidate...' });
    const metrics = await evaluateExportedModel(inferenceModel, holdout);
    emit({ step: 'ready', progress: 1, message: 'cnn ready' });
    return {
      artifacts: {
        checkpoint: exportedCheckpoint,
        inferenceModel,
        exportMetadata,
        metrics,
        stage,
        updatedAt: Date.now(),
      },
      metrics,
      stage,
      accepted: true,
      rejectionReason: null,
    };
  } catch (error) {
    return unavailableResult(stage, error instanceof Error ? error.message : 'CNN training failed.');
  }
}

type CnnTrainingWorkerResponse =
  | { type: 'progress'; payload: CnnTrainingProgress }
  | { type: 'completed'; payload: CnnTrainingRuntimeResult }
  | { type: 'failed'; reason: string };

class WorkerCnnTrainingRuntime implements CnnTrainingRuntimeLike {
  constructor(private readonly manifest: BaselineArtifactManifest) {}

  getAvailability(): CnnTrainingAvailability {
    const availability = getCnnTrainingAvailability(this.manifest);
    if (typeof Worker === 'undefined') {
      return {
        available: false,
        reasons: ['browser Worker API is unavailable'],
      };
    }
    return availability;
  }

  isAvailable(): boolean {
    return this.getAvailability().available;
  }

  async trainCandidate(
    training: AcceptedSampleRecord[],
    holdout: AcceptedSampleRecord[],
    previousArtifacts: PersonalizedCnnArtifacts | null,
    options: CnnTrainingRuntimeOptions = {},
  ): Promise<CnnTrainingRuntimeResult> {
    const stage = chooseCnnTrainingStage(training, previousArtifacts);
    const availability = this.getAvailability();
    if (!availability.available) {
      return unavailableResult(stage, availability.reasons.join('; '));
    }

    return await new Promise<CnnTrainingRuntimeResult>((resolve) => {
      let worker: Worker;
      try {
        worker = new Worker(new URL('./cnnTrainingWorker.ts', import.meta.url), { type: 'module' });
      } catch (error) {
        resolve(unavailableResult(stage, error instanceof Error ? error.message : 'Failed to start CNN training worker.'));
        return;
      }

      worker.onmessage = (event: MessageEvent<CnnTrainingWorkerResponse>) => {
        if (event.data.type === 'progress') {
          options.onProgress?.(event.data.payload);
          return;
        }
        worker.terminate();
        if (event.data.type === 'completed') {
          resolve(event.data.payload);
          return;
        }
        resolve(unavailableResult(stage, event.data.reason));
      };
      worker.onerror = (event) => {
        worker.terminate();
        resolve(unavailableResult(stage, event.message || 'CNN training worker failed.'));
      };
      worker.postMessage({
        manifest: this.manifest,
        training,
        holdout,
        previousArtifacts,
      });
    });
  }
}

export class CnnTrainingRuntime implements CnnTrainingRuntimeLike {
  private readonly delegate: CnnTrainingRuntimeLike | null;

  constructor(private readonly manifest: BaselineArtifactManifest) {
    this.delegate = typeof window !== 'undefined' && typeof Worker !== 'undefined'
      ? new WorkerCnnTrainingRuntime(manifest)
      : null;
  }

  getAvailability(): CnnTrainingAvailability {
    return this.delegate?.getAvailability?.() ?? getCnnTrainingAvailability(this.manifest);
  }

  isAvailable(): boolean {
    return this.getAvailability().available;
  }

  async trainCandidate(
    training: AcceptedSampleRecord[],
    holdout: AcceptedSampleRecord[],
    previousArtifacts: PersonalizedCnnArtifacts | null,
    options: CnnTrainingRuntimeOptions = {},
  ): Promise<CnnTrainingRuntimeResult> {
    if (this.delegate) {
      return await this.delegate.trainCandidate(training, holdout, previousArtifacts, options);
    }
    return await trainCnnCandidateDirect(this.manifest, training, holdout, previousArtifacts, options);
  }
}
