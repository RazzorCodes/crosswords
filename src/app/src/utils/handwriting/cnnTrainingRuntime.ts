import { InferenceSession, Tensor } from 'onnxruntime-web';
import { renderStrokesToPixels } from '../recognizers/rasterizer';
import type {
  AcceptedSampleRecord,
  BaselineArtifactManifest,
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

export interface CnnTrainingRuntimeLike {
  isAvailable(): boolean;
  trainCandidate(
    training: AcceptedSampleRecord[],
    holdout: AcceptedSampleRecord[],
    previousArtifacts: PersonalizedCnnArtifacts | null,
  ): Promise<CnnTrainingRuntimeResult>;
}

function labelIndex(label: string): number {
  const index = label.charCodeAt(0) - 65;
  return index >= 0 && index < 26 ? index : 0;
}

function chooseStage(
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

export class CnnTrainingRuntime implements CnnTrainingRuntimeLike {
  constructor(private readonly manifest: BaselineArtifactManifest) {}

  isAvailable(): boolean {
    const { cnn } = this.manifest;
    return Boolean(
      cnn.supportsTraining
      && cnn.trainingRuntime.moduleUrl
      && cnn.trainingArtifacts.trainUrl
      && cnn.trainingArtifacts.evalUrl
      && cnn.trainingArtifacts.optimizerUrl
      && cnn.trainingArtifacts.checkpointUrl,
    );
  }

  async trainCandidate(
    training: AcceptedSampleRecord[],
    holdout: AcceptedSampleRecord[],
    previousArtifacts: PersonalizedCnnArtifacts | null,
  ): Promise<CnnTrainingRuntimeResult> {
    const stage = chooseStage(training, previousArtifacts);
    if (!this.isAvailable()) {
      return {
        artifacts: null,
        metrics: { user_inputtedAccuracy: 0, implicitAccuracy: 0, overallAccuracy: 0 },
        stage,
        accepted: false,
        rejectionReason: 'CNN training runtime is unavailable.',
      };
    }

    try {
      const moduleUrl = this.manifest.cnn.trainingRuntime.moduleUrl!;
      const ort = await import(/* @vite-ignore */ moduleUrl) as OrtTrainingModule;
      if (ort.env?.wasm) {
        const wasmBase = this.manifest.cnn.trainingRuntime.wasmUrl;
        if (wasmBase) {
          ort.env.wasm.wasmPaths = wasmBase;
        }
        ort.env.wasm.numThreads = 1;
      }
      const TrainingSession = ort.TrainingSession;
      if (!TrainingSession?.create) {
        throw new Error('Release ORT runtime does not expose TrainingSession.create.');
      }

      const artifacts = this.manifest.cnn.trainingArtifacts;
      const [trainModel, evalModel, optimizerModel, baselineCheckpoint] = await Promise.all([
        fetchArrayBuffer(artifacts.trainUrl!),
        fetchArrayBuffer(artifacts.evalUrl!),
        fetchArrayBuffer(artifacts.optimizerUrl!),
        fetchArrayBuffer(artifacts.checkpointUrl!),
      ]);
      const checkpoint = previousArtifacts?.checkpoint ?? baselineCheckpoint;
      const session = await TrainingSession.create({
        trainModel,
        evalModel,
        optimizerModel,
        checkpoint,
        stage,
      });

      const TensorCtor = ort.Tensor ?? Tensor;
      const startedAt = Date.now();
      const epochs = stage === 'partial-finetune' ? DEFAULT_PARTIAL_EPOCHS : DEFAULT_HEAD_EPOCHS;
      for (let epoch = 0; epoch < epochs; epoch += 1) {
        for (const batch of batches(training)) {
          if (Date.now() - startedAt > MAX_TRAINING_MS) {
            throw new Error('CNN training exceeded the browser time budget.');
          }
          await session.trainStep(createBatch(batch, TensorCtor));
          await session.optimizerStep?.();
          await session.lazyResetGrad?.();
        }
      }

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
      const metrics = await evaluateExportedModel(inferenceModel, holdout);
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
      return {
        artifacts: null,
        metrics: { user_inputtedAccuracy: 0, implicitAccuracy: 0, overallAccuracy: 0 },
        stage,
        accepted: false,
        rejectionReason: error instanceof Error ? error.message : 'CNN training failed.',
      };
    }
  }
}
