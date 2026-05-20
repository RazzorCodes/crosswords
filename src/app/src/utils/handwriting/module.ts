import { InferenceSession, Tensor } from 'onnxruntime-web';
import { renderStrokesToPixels } from '../recognizers/rasterizer';
import type { EngineResult, RecognitionResult, StrokeInput } from '../recognizers/types';
import {
  buildBalancedDataset,
  createAcceptedSampleRecord,
  createDefaultBaselineManifest,
  createExportBundle,
  createPersistedState,
  decideTrainingTrigger,
  estimatePersistedStateBytes,
  extractFeaturesFromStrokes,
  maybeIncrementPendingUserInputted,
  normalizeLabel,
  predictFeatureClassifierProbabilities,
  predictKnnFromFeatures,
  rebuildKnnCache,
  sanitizeImportedState,
  shouldAcceptCandidateSnapshot,
  toCandidatesFromProbabilities,
  trainFeatureClassifier,
  updateTrainingStateFromLedger,
  evaluateSnapshot,
} from './core';
import { createDefaultKeyValueStore } from './storage';
import type {
  AcceptedSampleInput,
  BaselineArtifactManifest,
  ExportedHandwritingBundle,
  FeatureClassifierSnapshot,
  HandwritingModuleConfig,
  HandwritingModuleEvent,
  HandwritingModuleInitResult,
  HandwritingModuleListener,
  HandwritingPrediction,
  KeyValueStore,
  PersonalizedCnnArtifacts,
  PersistedHandwritingState,
  TrainingState,
  TrainingTriggerReason,
} from './types';

const STORAGE_KEYS = {
  baseline: 'baseline',
  ledger: 'ledger',
  classifierSnapshot: 'classifierSnapshot',
  personalizedCnn: 'personalizedCnn',
  trainingState: 'trainingState',
};

const AUTO_ACCEPT_SCORE = 0.92;
const AUTO_ACCEPT_MARGIN = 0.12;
const DEFAULT_SNAPSHOT_BUDGET_BYTES = 2_000_000;

interface ResolvedConfig {
  modelBaseUrl: string;
  baselineManifestUrl: string;
  snapshotBudgetBytes: number;
}

function cloneTrainingState(state: TrainingState): TrainingState {
  return JSON.parse(JSON.stringify(state)) as TrainingState;
}

function resolveConfig(config: HandwritingModuleConfig): ResolvedConfig {
  const runtimeConfig = typeof window !== 'undefined' ? (window as Window & {
    CROSSWORDS_CONFIG?: { MODEL_BASE_URL?: string };
  }).CROSSWORDS_CONFIG : undefined;
  const modelBaseUrl = (config.modelBaseUrl ?? runtimeConfig?.MODEL_BASE_URL ?? '/models').replace(/\/$/, '');
  return {
    modelBaseUrl,
    baselineManifestUrl: config.baselineManifestUrl ?? `${modelBaseUrl}/manifest.json`,
    snapshotBudgetBytes: config.snapshotBudgetBytes ?? DEFAULT_SNAPSHOT_BUDGET_BYTES,
  };
}

function pickTopLabel(probabilities: Record<string, number>): { label: string | null; score: number } {
  let label: string | null = null;
  let score = -1;
  for (const [candidate, value] of Object.entries(probabilities)) {
    if (value > score) {
      label = candidate;
      score = value;
    }
  }
  return { label, score: Math.max(score, 0) };
}

function parseBaselineManifest(raw: unknown, fallback: BaselineArtifactManifest): BaselineArtifactManifest {
  if (!raw || typeof raw !== 'object') {
    return fallback;
  }

  const candidate = raw as Record<string, unknown>;
  const labelMap = Array.isArray(candidate.labelMap)
    ? candidate.labelMap.filter((value): value is string => typeof value === 'string' && /^[A-Z]$/.test(value))
    : fallback.labelMap;
  const cnnCandidate = candidate.cnn as Record<string, unknown> | undefined;

  return {
    version: typeof candidate.version === 'string' ? candidate.version : fallback.version,
    labelMap: labelMap.length > 0 ? labelMap : fallback.labelMap,
    cnn: {
      inferenceUrl: typeof cnnCandidate?.inferenceUrl === 'string' ? cnnCandidate.inferenceUrl : fallback.cnn.inferenceUrl,
      supportsTraining: Boolean(cnnCandidate?.supportsTraining),
      trainingArtifactsUrl: typeof cnnCandidate?.trainingArtifactsUrl === 'string' ? cnnCandidate.trainingArtifactsUrl : null,
    },
    featureClassifier: fallback.featureClassifier,
  };
}

export class BrowserHandwritingModule {
  private readonly listeners = new Set<HandwritingModuleListener>();

  private readonly store: KeyValueStore;

  private initPromise: Promise<HandwritingModuleInitResult> | null = null;

  private config: ResolvedConfig | null = null;

  private baseline: BaselineArtifactManifest | null = null;

  private ledger = [] as PersistedHandwritingState['ledger'];

  private classifierSnapshot: FeatureClassifierSnapshot | null = null;

  private personalizedCnn: PersonalizedCnnArtifacts | null = null;

  private trainingState: TrainingState | null = null;

  private cnnSession: InferenceSession | null = null;

  private cnnSessionKey: string | null = null;

  constructor(store: KeyValueStore = createDefaultKeyValueStore()) {
    this.store = store;
  }

  async init(config: HandwritingModuleConfig = {}): Promise<HandwritingModuleInitResult> {
    if (this.initPromise) {
      return await this.initPromise;
    }

    this.initPromise = this.initialize(resolveConfig(config));
    return await this.initPromise;
  }

  private async initialize(config: ResolvedConfig): Promise<HandwritingModuleInitResult> {
    this.config = config;
    const fallbackBaseline = createDefaultBaselineManifest(config.modelBaseUrl);
    const baseline = await this.loadBaseline(fallbackBaseline, config.baselineManifestUrl);
    this.baseline = baseline;

    const incoming = await this.readPersistedPieces();
    const persisted = sanitizeImportedState(incoming, baseline, config.snapshotBudgetBytes);
    this.ledger = persisted.ledger;
    this.classifierSnapshot = persisted.classifierSnapshot ?? baseline.featureClassifier;
    this.personalizedCnn = persisted.personalizedCnn;
    this.trainingState = {
      ...persisted.trainingState,
      baselineVersion: baseline.version,
      trainerStatus: {
        ...persisted.trainingState.trainerStatus,
        cnnTrainingAvailable: baseline.cnn.supportsTraining,
        personalizedCnnAvailable: Boolean(persisted.personalizedCnn?.inferenceModel),
      },
    };

    await this.persist();
    return {
      trainingState: this.getTrainingState(),
    };
  }

  private async loadBaseline(
    fallback: BaselineArtifactManifest,
    manifestUrl: string,
  ): Promise<BaselineArtifactManifest> {
    try {
      const response = await fetch(manifestUrl);
      if (!response.ok) {
        return fallback;
      }
      const manifest = parseBaselineManifest(await response.json(), fallback);
      return manifest;
    } catch {
      return fallback;
    }
  }

  private async readPersistedPieces(): Promise<Partial<PersistedHandwritingState>> {
    const [
      baseline,
      ledger,
      classifierSnapshot,
      personalizedCnn,
      trainingState,
    ] = await Promise.all([
      this.store.get<BaselineArtifactManifest>(STORAGE_KEYS.baseline),
      this.store.get<PersistedHandwritingState['ledger']>(STORAGE_KEYS.ledger),
      this.store.get<FeatureClassifierSnapshot | null>(STORAGE_KEYS.classifierSnapshot),
      this.store.get<PersonalizedCnnArtifacts | null>(STORAGE_KEYS.personalizedCnn),
      this.store.get<TrainingState>(STORAGE_KEYS.trainingState),
    ]);

    return {
      baseline: baseline ?? undefined,
      ledger: ledger ?? undefined,
      classifierSnapshot: classifierSnapshot ?? undefined,
      personalizedCnn: personalizedCnn ?? undefined,
      trainingState: trainingState ?? undefined,
    };
  }

  private emit(event: HandwritingModuleEvent) {
    for (const listener of this.listeners) {
      listener(event);
    }
  }

  subscribe(listener: HandwritingModuleListener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  getTrainingState(): TrainingState {
    if (!this.trainingState) {
      throw new Error('Handwriting module has not been initialized.');
    }
    return cloneTrainingState(this.trainingState);
  }

  private getResolvedState(): {
    baseline: BaselineArtifactManifest;
    trainingState: TrainingState;
  } {
    if (!this.baseline || !this.trainingState || !this.config) {
      throw new Error('Handwriting module has not been initialized.');
    }
    return {
      baseline: this.baseline,
      trainingState: this.trainingState,
    };
  }

  private async persist(): Promise<void> {
    const { baseline, trainingState } = this.getResolvedState();
    const persisted = createPersistedState(
      baseline,
      this.ledger,
      this.classifierSnapshot,
      this.personalizedCnn,
      trainingState,
    );
    const persistedBytes = estimatePersistedStateBytes(persisted);
    this.trainingState = {
      ...trainingState,
      persistedBytes,
    };
    await Promise.all([
      this.store.set(STORAGE_KEYS.baseline, baseline),
      this.store.set(STORAGE_KEYS.ledger, this.ledger),
      this.store.set(STORAGE_KEYS.classifierSnapshot, this.classifierSnapshot),
      this.store.set(STORAGE_KEYS.personalizedCnn, this.personalizedCnn),
      this.store.set(STORAGE_KEYS.trainingState, this.trainingState),
    ]);
  }

  private async ensureInitialized(): Promise<void> {
    if (!this.initPromise) {
      await this.init({});
      return;
    }
    await this.initPromise;
  }

  private async ensureCnnSession(): Promise<InferenceSession | null> {
    await this.ensureInitialized();
    const { baseline } = this.getResolvedState();

    const sourceKey = this.personalizedCnn?.inferenceModel
      ? `personalized:${this.personalizedCnn.updatedAt}`
      : `baseline:${baseline.version}`;

    if (this.cnnSession && this.cnnSessionKey === sourceKey) {
      return this.cnnSession;
    }

    try {
      this.cnnSession = this.personalizedCnn?.inferenceModel
        ? await InferenceSession.create(this.personalizedCnn.inferenceModel)
        : (baseline.cnn.inferenceUrl ? await InferenceSession.create(baseline.cnn.inferenceUrl) : null);
      this.cnnSessionKey = sourceKey;
      return this.cnnSession;
    } catch (error) {
      console.warn('Failed to initialize CNN session', error);
      this.cnnSession = null;
      this.cnnSessionKey = null;
      return null;
    }
  }

  async predict(strokes: StrokeInput): Promise<HandwritingPrediction> {
    await this.ensureInitialized();

    const { baseline } = this.getResolvedState();
    const features = extractFeaturesFromStrokes(strokes);
    const knnCandidates = predictKnnFromFeatures(features, rebuildKnnCache(this.ledger));
    const featureSnapshot = this.classifierSnapshot ?? baseline.featureClassifier;
    const featureProbs = predictFeatureClassifierProbabilities(featureSnapshot, features);
    const featureCandidates = toCandidatesFromProbabilities(featureProbs, 'feature-classifier');

    let cnnProbs = Object.fromEntries(baseline.labelMap.map((label) => [label, 1 / baseline.labelMap.length])) as Record<string, number>;
    let cnnStatus: EngineResult['status'] = 'unavailable';
    let cnnDetail = baseline.cnn.inferenceUrl ? undefined : 'No CNN baseline configured';
    const cnnSession = await this.ensureCnnSession();
    if (cnnSession) {
      try {
        const pixels = renderStrokesToPixels(strokes);
        const input = new Tensor('float32', pixels, [1, 1, 64, 64]);
        const results = await cnnSession.run({ [cnnSession.inputNames[0]]: input });
        const outputName = cnnSession.outputNames[0];
        const outputValue = results[outputName];
        const data = outputValue && typeof outputValue === 'object' && 'data' in outputValue
          ? Array.from((outputValue as { data: ArrayLike<number> }).data)
          : [];
        if (data.length > 0) {
          cnnStatus = 'ready';
          cnnProbs = Object.fromEntries(
            baseline.labelMap.map((label, index) => [label, Number(data[index] ?? 0)]),
          ) as Record<string, number>;
        }
      } catch (error) {
        console.error('CNN inference failed', error);
        cnnStatus = 'error';
        cnnDetail = 'Inference failed';
      }
    }

    const knnProbs = Object.fromEntries(baseline.labelMap.map((label) => [label, 0])) as Record<string, number>;
    for (const candidate of knnCandidates) {
      knnProbs[candidate.char] = candidate.score;
    }

    const engines: Array<{
      key: string;
      weight: number;
      probs: Record<string, number>;
      topLabel: string | null;
      topScore: number;
      available: boolean;
      result: EngineResult;
    }> = [
      {
        key: 'knn',
        weight: 0.45,
        probs: knnProbs,
        topLabel: knnCandidates[0]?.char ?? null,
        topScore: knnCandidates[0]?.score ?? 0,
        available: knnCandidates.length > 0,
        result: knnCandidates.length > 0
          ? {
              name: 'k-NN',
              char: knnCandidates[0].char,
              score: knnCandidates[0].score,
              status: 'ready',
            }
          : {
              name: 'k-NN',
              char: null,
              score: null,
              status: 'unavailable',
              detail: 'No local samples yet',
            },
      },
      {
        key: 'feature-classifier',
        weight: 0.25,
        probs: featureProbs,
        topLabel: featureCandidates[0]?.char ?? null,
        topScore: featureCandidates[0]?.score ?? 0,
        available: Boolean(featureSnapshot),
        result: featureSnapshot
          ? {
              name: 'Feature',
              char: featureCandidates[0]?.char ?? null,
              score: featureCandidates[0]?.score ?? null,
              status: 'ready',
            }
          : {
              name: 'Feature',
              char: null,
              score: null,
              status: 'unavailable',
              detail: 'Using baseline backstops only',
            },
      },
      {
        key: 'cnn',
        weight: 0.30,
        probs: cnnProbs,
        topLabel: pickTopLabel(cnnProbs).label,
        topScore: pickTopLabel(cnnProbs).score,
        available: cnnStatus === 'ready',
        result: {
          name: 'CNN',
          char: pickTopLabel(cnnProbs).label,
          score: cnnStatus === 'ready' ? pickTopLabel(cnnProbs).score : null,
          status: cnnStatus,
          detail: cnnDetail,
        },
      },
    ];

    const totalWeight = engines
      .filter((engine) => engine.available)
      .reduce((sum, engine) => sum + engine.weight, 0) || 1;

    const weighted = Object.fromEntries(baseline.labelMap.map((label) => [label, 0])) as Record<string, number>;
    for (const label of baseline.labelMap) {
      for (const engine of engines) {
        if (!engine.available) {
          continue;
        }
        weighted[label] += (engine.weight / totalWeight) * (engine.probs[label] ?? 0);
      }
    }

    const candidates = toCandidatesFromProbabilities(weighted, 'ensemble').slice(0, 3);
    const bestScore = candidates[0]?.score ?? 0;
    const secondScore = candidates[1]?.score ?? 0;
    const bestLabel = candidates[0]?.char ?? null;
    const agreementCount = engines
      .filter((engine) => engine.available && engine.topLabel === bestLabel)
      .length;
    const autoAccepted = (
      Boolean(bestLabel) &&
      bestScore >= AUTO_ACCEPT_SCORE &&
      bestScore - secondScore >= AUTO_ACCEPT_MARGIN &&
      agreementCount >= Math.min(2, engines.filter((engine) => engine.available).length)
    );

    const prediction: HandwritingPrediction = {
      status: autoAccepted ? 'confirmed' : 'uncertain',
      candidates,
      chosenLabel: autoAccepted ? bestLabel : null,
      confidence: bestScore,
      sourceTrail: [...engines.map((engine) => engine.key), 'weighted-vote'],
      engineResults: engines.map((engine) => engine.result),
      trainingState: this.getTrainingState(),
    };

    this.emit({ type: 'prediction', payload: prediction });
    return prediction;
  }

  async recordAcceptedSample(input: AcceptedSampleInput): Promise<void> {
    await this.ensureInitialized();
    const normalized = normalizeLabel(input.label);
    if (!normalized || !/^[A-Z]$/.test(normalized)) {
      return;
    }

    const sample = createAcceptedSampleRecord({
      ...input,
      label: normalized,
    });

    this.ledger = [...this.ledger, sample];
    this.trainingState = maybeIncrementPendingUserInputted(this.getTrainingState(), sample.acceptance);
    this.trainingState = updateTrainingStateFromLedger(this.trainingState, this.ledger, this.personalizedCnn);
    this.trainingState = {
      ...this.trainingState,
      trainerStatus: {
        ...this.trainingState.trainerStatus,
        lastEventAt: Date.now(),
      },
    };

    await this.persist();
    this.emit({
      type: 'sample-recorded',
      payload: {
        sample,
        trainingState: this.getTrainingState(),
      },
    });

    const decision = decideTrainingTrigger(this.trainingState);
    if (decision.milestoneReached !== null) {
      this.emit({
        type: 'milestone-reached',
        payload: {
          milestone: decision.milestoneReached,
          trainingState: this.getTrainingState(),
        },
      });
    }

    if (decision.shouldTrain && decision.reason) {
      await this.runTraining(decision.reason, decision.milestoneReached);
      return;
    }

    this.emit({
      type: 'artifacts-updated',
      payload: {
        trainingState: this.getTrainingState(),
      },
    });
  }

  private async runTraining(reason: TrainingTriggerReason, milestoneReached: number | null): Promise<void> {
    this.trainingState = {
      ...this.getTrainingState(),
      lastTrainingReason: reason,
      trainerStatus: {
        ...this.getTrainingState().trainerStatus,
        trainingInFlight: true,
        lastEventAt: Date.now(),
      },
    };
    await this.persist();
    this.emit({
      type: 'training-started',
      payload: {
        reason,
        trainingState: this.getTrainingState(),
      },
    });

    const dataset = buildBalancedDataset(this.ledger);
    const baseMilestones = milestoneReached !== null
      ? [...new Set([...this.trainingState.milestonesCompleted, milestoneReached])].sort((left, right) => left - right)
      : this.trainingState.milestonesCompleted;

    if (dataset.training.length === 0 || dataset.readyLetters.length < 2) {
      this.trainingState = {
        ...this.getTrainingState(),
        milestonesCompleted: baseMilestones,
        lastTrainingOutcome: 'rejected',
        lastRejectedReason: 'Not enough balanced per-letter coverage yet.',
        trainerStatus: {
          ...this.getTrainingState().trainerStatus,
          trainingInFlight: false,
          lastEventAt: Date.now(),
        },
      };
      await this.persist();
      this.emit({
        type: 'training-rejected',
        payload: {
          reason: 'Not enough balanced per-letter coverage yet.',
          trainingState: this.getTrainingState(),
        },
      });
      return;
    }

    const candidate = trainFeatureClassifier(dataset.training, reason, dataset.readyLetters);
    if (!candidate) {
      this.trainingState = {
        ...this.getTrainingState(),
        milestonesCompleted: baseMilestones,
        lastTrainingOutcome: 'rejected',
        lastRejectedReason: 'Could not build a feature-classifier snapshot.',
        trainerStatus: {
          ...this.getTrainingState().trainerStatus,
          trainingInFlight: false,
          lastEventAt: Date.now(),
        },
      };
      await this.persist();
      this.emit({
        type: 'training-rejected',
        payload: {
          reason: 'Could not build a feature-classifier snapshot.',
          trainingState: this.getTrainingState(),
        },
      });
      return;
    }

    candidate.metrics = evaluateSnapshot(candidate, dataset.holdout);
    const currentMetrics = this.classifierSnapshot?.metrics ?? this.trainingState.latestMetrics;

    if (!shouldAcceptCandidateSnapshot(currentMetrics, candidate.metrics)) {
      this.trainingState = {
        ...this.getTrainingState(),
        milestonesCompleted: baseMilestones,
        latestMetrics: currentMetrics,
        lastTrainingOutcome: 'rejected',
        lastRejectedReason: 'Regression gate rejected the candidate snapshot.',
        trainerStatus: {
          ...this.getTrainingState().trainerStatus,
          trainingInFlight: false,
          lastEventAt: Date.now(),
        },
      };
      await this.persist();
      this.emit({
        type: 'training-rejected',
        payload: {
          reason: 'Regression gate rejected the candidate snapshot.',
          trainingState: this.getTrainingState(),
        },
      });
      return;
    }

    this.classifierSnapshot = candidate;
    this.trainingState = updateTrainingStateFromLedger(
      {
        ...this.getTrainingState(),
        milestonesCompleted: baseMilestones,
        latestSnapshotId: candidate.id,
        lastCompletedTrainingAt: Date.now(),
        lastTrainingReason: reason,
        lastTrainingOutcome: 'accepted',
        lastRejectedReason: null,
        latestMetrics: candidate.metrics,
        pendingUserInputtedSinceTraining: 0,
        trainerStatus: {
          ...this.getTrainingState().trainerStatus,
          trainingInFlight: false,
          lastEventAt: Date.now(),
        },
      },
      this.ledger,
      this.personalizedCnn,
    );
    await this.persist();
    this.emit({
      type: 'training-completed',
      payload: {
        snapshot: candidate,
        trainingState: this.getTrainingState(),
      },
    });
    this.emit({
      type: 'artifacts-updated',
      payload: {
        trainingState: this.getTrainingState(),
      },
    });
  }

  async exportDevBundle(): Promise<ExportedHandwritingBundle> {
    await this.ensureInitialized();
    const { baseline } = this.getResolvedState();
    return createExportBundle(createPersistedState(
      baseline,
      this.ledger,
      this.classifierSnapshot,
      this.personalizedCnn,
      this.getTrainingState(),
    ));
  }

  async importDevBundle(bundle: ExportedHandwritingBundle): Promise<void> {
    await this.ensureInitialized();
    const { baseline } = this.getResolvedState();
    const persisted = sanitizeImportedState(
      {
        baseline: bundle.baseline,
        ledger: bundle.ledger,
        classifierSnapshot: bundle.classifierSnapshot,
        personalizedCnn: bundle.personalizedCnn,
        trainingState: bundle.trainingState,
      },
      baseline,
      this.getTrainingState().snapshotBudgetBytes,
    );
    this.baseline = persisted.baseline;
    this.ledger = persisted.ledger;
    this.classifierSnapshot = persisted.classifierSnapshot;
    this.personalizedCnn = persisted.personalizedCnn;
    this.trainingState = persisted.trainingState;
    this.cnnSession = null;
    this.cnnSessionKey = null;
    await this.persist();
    this.emit({
      type: 'artifacts-updated',
      payload: {
        trainingState: this.getTrainingState(),
      },
    });
  }

  async asRecognitionResult(strokes: StrokeInput): Promise<RecognitionResult> {
    const prediction = await this.predict(strokes);
    return {
      status: prediction.status,
      candidates: prediction.candidates,
      chosenChar: prediction.chosenLabel,
      sourceTrail: prediction.sourceTrail,
      engineResults: prediction.engineResults,
      teacher: {
        label: prediction.candidates[0]?.char ?? '?',
        confidence: prediction.confidence,
        action: prediction.chosenLabel ? 'accept' : 'prompt',
      },
    };
  }
}

export const handwritingModule = new BrowserHandwritingModule();

export async function initHandwritingModule(config?: HandwritingModuleConfig) {
  return await handwritingModule.init(config);
}
