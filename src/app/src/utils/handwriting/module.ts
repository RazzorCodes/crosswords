import { InferenceSession, Tensor } from 'onnxruntime-web';
import {
  CnnTrainingRuntime,
  type CnnTrainingRuntimeLike,
  type CnnTrainingRuntimeResult,
} from './cnnTrainingRuntime';
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
import { initHandwritingRustCore } from './wasmCore';
import type {
  AcceptedSampleInput,
  AcceptedSampleRecord,
  BaselineArtifactManifest,
  CnnTrainingArtifactUrls,
  CnnTrainingRuntimeUrls,
  DevSyncQueueItem,
  DevSyncQueueStatus,
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
  baseline: 'baselineManifest',
  coreState: 'coreState',
  personalizedCnn: 'personalizedCnn',
  trainingState: 'trainingState',
  devSyncQueue: 'devSyncQueue',
  legacyBaseline: 'baseline',
  legacyLedger: 'ledger',
  legacyClassifierSnapshot: 'classifierSnapshot',
};

const AUTO_ACCEPT_SCORE = 0.92;
const AUTO_ACCEPT_MARGIN = 0.12;
const DEFAULT_SNAPSHOT_BUDGET_BYTES = 2_000_000;

interface ResolvedConfig {
  modelBaseUrl: string;
  baselineManifestUrl: string;
  cnnTrainingRuntime: CnnTrainingRuntimeLike | null;
  snapshotBudgetBytes: number;
  devSync: Required<NonNullable<HandwritingModuleConfig['devSync']>>;
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
    cnnTrainingRuntime: isCnnTrainingRuntimeLike(config.cnnTrainingRuntime)
      ? config.cnnTrainingRuntime
      : null,
    snapshotBudgetBytes: config.snapshotBudgetBytes ?? DEFAULT_SNAPSHOT_BUDGET_BYTES,
    devSync: {
      enabled: config.devSync?.enabled ?? false,
      mode: config.devSync?.mode ?? 'legacy-server',
      endpointBaseUrl: (config.devSync?.endpointBaseUrl ?? '/api').replace(/\/$/, ''),
      flushPolicy: config.devSync?.flushPolicy ?? 'immediate',
    },
  };
}

function isCnnTrainingRuntimeLike(value: unknown): value is CnnTrainingRuntimeLike {
  return Boolean(
    value
    && typeof value === 'object'
    && typeof (value as CnnTrainingRuntimeLike).trainCandidate === 'function',
  );
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

function normalizeModelScores(rawScores: number[], labelCount: number): number[] {
  const scores = rawScores.slice(0, labelCount).map((value) => (
    Number.isFinite(value) ? value : 0
  ));
  if (scores.length === 0) {
    return [];
  }

  const sum = scores.reduce((total, value) => total + value, 0);
  const alreadyProbabilities = scores.every((value) => value >= 0 && value <= 1)
    && Math.abs(sum - 1) <= 0.05;
  if (alreadyProbabilities) {
    return scores;
  }

  const max = Math.max(...scores);
  const expScores = scores.map((value) => Math.exp(value - max));
  const expSum = expScores.reduce((total, value) => total + value, 0);
  if (expSum <= 0 || !Number.isFinite(expSum)) {
    return new Array(labelCount).fill(1 / labelCount);
  }
  return expScores.map((value) => value / expSum);
}

function parseTrainingArtifacts(raw: unknown): CnnTrainingArtifactUrls {
  const candidate = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {};
  return {
    trainUrl: typeof candidate.trainUrl === 'string' ? candidate.trainUrl : null,
    evalUrl: typeof candidate.evalUrl === 'string' ? candidate.evalUrl : null,
    optimizerUrl: typeof candidate.optimizerUrl === 'string' ? candidate.optimizerUrl : null,
    checkpointUrl: typeof candidate.checkpointUrl === 'string' ? candidate.checkpointUrl : null,
    exportMetadataUrl: typeof candidate.exportMetadataUrl === 'string' ? candidate.exportMetadataUrl : null,
  };
}

function parseTrainingRuntime(raw: unknown): CnnTrainingRuntimeUrls {
  const candidate = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {};
  return {
    moduleUrl: typeof candidate.moduleUrl === 'string' ? candidate.moduleUrl : null,
    wasmUrl: typeof candidate.wasmUrl === 'string' ? candidate.wasmUrl : null,
    simdWasmUrl: typeof candidate.simdWasmUrl === 'string' ? candidate.simdWasmUrl : null,
    threadedWasmUrl: typeof candidate.threadedWasmUrl === 'string' ? candidate.threadedWasmUrl : null,
  };
}

function isFeatureClassifierSnapshot(value: unknown): value is FeatureClassifierSnapshot {
  return Boolean(
    value
    && typeof value === 'object'
    && Array.isArray((value as FeatureClassifierSnapshot).centroids)
    && Array.isArray((value as FeatureClassifierSnapshot).labelMap),
  );
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
      trainingArtifacts: parseTrainingArtifacts(cnnCandidate?.trainingArtifacts),
      trainingRuntime: parseTrainingRuntime(cnnCandidate?.trainingRuntime),
    },
    featureClassifier: isFeatureClassifierSnapshot(candidate.featureClassifier)
      ? candidate.featureClassifier
      : fallback.featureClassifier,
  };
}

function mergeFeatureProbabilities(
  baseline: FeatureClassifierSnapshot | null,
  personalized: FeatureClassifierSnapshot | null,
  features: number[],
): Record<string, number> {
  const baselineProbs = predictFeatureClassifierProbabilities(baseline, features);
  if (!personalized || personalized.centroids.length === 0) {
    return baselineProbs;
  }

  const personalizedProbs = predictFeatureClassifierProbabilities(personalized, features);
  const cohort = new Set(personalized.labelMap);
  return Object.fromEntries(
    Object.keys(baselineProbs).map((label) => [
      label,
      cohort.has(label) ? (personalizedProbs[label] ?? 0) : baselineProbs[label],
    ]),
  ) as Record<string, number>;
}

export class BrowserHandwritingModule {
  private readonly listeners = new Set<HandwritingModuleListener>();

  private readonly store: KeyValueStore;

  private initPromise: Promise<HandwritingModuleInitResult> | null = null;

  private config: ResolvedConfig | null = null;

  private baseline: BaselineArtifactManifest | null = null;

  private ledger: AcceptedSampleRecord[] = [];

  private classifierSnapshot: FeatureClassifierSnapshot | null = null;

  private personalizedCnn: PersonalizedCnnArtifacts | null = null;

  private trainingState: TrainingState | null = null;

  private devSyncQueue: DevSyncQueueItem[] = [];

  private cnnSession: InferenceSession | null = null;

  private cnnSessionKey: string | null = null;

  private cnnTrainer: CnnTrainingRuntimeLike | null = null;

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
    try {
      await initHandwritingRustCore();
    } catch (error) {
      console.warn('Rust handwriting core unavailable, using TS fallback core.', error);
    }
    const fallbackBaseline = createDefaultBaselineManifest(config.modelBaseUrl);
    const baseline = await this.loadBaseline(fallbackBaseline, config.baselineManifestUrl);
    this.baseline = baseline;
    this.cnnTrainer = config.cnnTrainingRuntime ?? new CnnTrainingRuntime(baseline);

    const incoming = await this.readPersistedPieces();
    const persisted = sanitizeImportedState(incoming, baseline, config.snapshotBudgetBytes);
    this.ledger = persisted.coreState.ledger;
    this.classifierSnapshot = persisted.coreState.latestAcceptedFeatureClassifier;
    this.personalizedCnn = persisted.personalizedCnn;
    this.devSyncQueue = persisted.devSyncQueue;
    this.trainingState = {
      ...persisted.trainingState,
      baselineVersion: baseline.version,
      personalizedCohortLabelMap: persisted.coreState.latestAcceptedFeatureClassifier?.labelMap ?? [],
      trainerStatus: {
        ...persisted.trainingState.trainerStatus,
        activeModelGeneration: persisted.trainingState.personalizationGeneration,
        cnnTrainingAvailable: baseline.cnn.supportsTraining && this.cnnTrainer.isAvailable(),
        cnnTrainingStatus: baseline.cnn.supportsTraining && this.cnnTrainer.isAvailable() ? 'ready' : 'unavailable',
        cnnInferenceSource: persisted.personalizedCnn?.inferenceModel ? 'personalized' : 'baseline',
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
      baselineManifest,
      coreState,
      personalizedCnn,
      trainingState,
      devSyncQueue,
      legacyBaseline,
      legacyLedger,
      legacyClassifierSnapshot,
    ] = await Promise.all([
      this.store.get<BaselineArtifactManifest>(STORAGE_KEYS.baseline),
      this.store.get<PersistedHandwritingState['coreState']>(STORAGE_KEYS.coreState),
      this.store.get<PersonalizedCnnArtifacts | null>(STORAGE_KEYS.personalizedCnn),
      this.store.get<TrainingState>(STORAGE_KEYS.trainingState),
      this.store.get<DevSyncQueueItem[]>(STORAGE_KEYS.devSyncQueue),
      this.store.get<BaselineArtifactManifest>(STORAGE_KEYS.legacyBaseline),
      this.store.get<AcceptedSampleRecord[]>(STORAGE_KEYS.legacyLedger),
      this.store.get<FeatureClassifierSnapshot | null>(STORAGE_KEYS.legacyClassifierSnapshot),
    ]);

    return {
      baselineManifest: baselineManifest ?? legacyBaseline ?? undefined,
      coreState: coreState ?? (legacyLedger ? {
        version: 1,
        baselineVersion: (baselineManifest ?? legacyBaseline)?.version ?? 'unknown',
        ledger: legacyLedger,
        milestonesCompleted: trainingState?.milestonesCompleted ?? [],
        pendingUserInputtedSinceTraining: trainingState?.pendingUserInputtedSinceTraining ?? 0,
        latestAcceptedFeatureClassifier: legacyClassifierSnapshot ?? null,
        knnCache: rebuildKnnCache(legacyLedger),
      } : undefined),
      personalizedCnn: personalizedCnn ?? undefined,
      trainingState: trainingState ?? undefined,
      devSyncQueue: devSyncQueue ?? undefined,
    };
  }

  private emit(event: HandwritingModuleEvent) {
    for (const listener of this.listeners) {
      listener(event);
    }
  }

  private emitTrainingProgress(
    phase: 'feature' | 'cnn' | 'finalizing',
    status: 'running' | 'ready' | 'skipped' | 'rejected',
    progress: number,
    message: string,
    generation = this.getTrainingState().personalizationGeneration,
  ) {
    this.emit({
      type: 'training-progress',
      payload: {
        phase,
        status,
        progress,
        message,
        generation,
        trainingState: this.getTrainingState(),
      },
    });
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

  getDiagnostics(): {
    cnnSessionKey: string | null;
    cnnInferenceSource: 'baseline' | 'personalized';
    personalizedCnnAvailable: boolean;
    activeModelGeneration: number;
  } {
    const state = this.getTrainingState();
    return {
      cnnSessionKey: this.cnnSessionKey,
      cnnInferenceSource: state.trainerStatus.cnnInferenceSource,
      personalizedCnnAvailable: state.trainerStatus.personalizedCnnAvailable,
      activeModelGeneration: state.trainerStatus.activeModelGeneration,
    };
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
      this.devSyncQueue,
    );
    const persistedBytes = estimatePersistedStateBytes(persisted);
    this.trainingState = {
      ...trainingState,
      persistedBytes,
      snapshotBudgetStatus: {
        budgetBytes: trainingState.snapshotBudgetBytes,
        usedBytes: persistedBytes,
        withinBudget: persistedBytes <= trainingState.snapshotBudgetBytes,
        lastRejectedBytes: trainingState.snapshotBudgetStatus?.lastRejectedBytes ?? null,
      },
      devSyncQueueStatus: this.getDevSyncQueueStatus(trainingState.devSyncQueueStatus?.lastFlushAt ?? null),
    };
    await Promise.all([
      this.store.set(STORAGE_KEYS.baseline, baseline),
      this.store.set(STORAGE_KEYS.coreState, persisted.coreState),
      this.store.set(STORAGE_KEYS.personalizedCnn, this.personalizedCnn),
      this.store.set(STORAGE_KEYS.trainingState, this.trainingState),
      this.store.set(STORAGE_KEYS.devSyncQueue, this.devSyncQueue),
    ]);
  }

  private getDevSyncQueueStatus(lastFlushAt: number | null = this.trainingState?.devSyncQueueStatus.lastFlushAt ?? null): DevSyncQueueStatus {
    return {
      pending: this.devSyncQueue.length,
      failed: this.devSyncQueue.filter((item) => item.lastError).length,
      lastFlushAt,
    };
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
    const featureSnapshot = this.classifierSnapshot;
    const featureProbs = mergeFeatureProbabilities(baseline.featureClassifier, featureSnapshot, features);
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
          const probabilities = normalizeModelScores(data.map(Number), baseline.labelMap.length);
          cnnProbs = Object.fromEntries(
            baseline.labelMap.map((label, index) => [label, probabilities[index] ?? 0]),
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
        available: Boolean(featureSnapshot ?? baseline.featureClassifier),
        result: featureSnapshot
          ? {
              name: 'Feature',
              char: featureCandidates[0]?.char ?? null,
              score: featureCandidates[0]?.score ?? null,
              status: 'ready',
            }
          : {
              name: 'Feature',
              char: featureCandidates[0]?.char ?? null,
              score: featureCandidates[0]?.score ?? null,
              status: baseline.featureClassifier ? 'ready' : 'unavailable',
              detail: baseline.featureClassifier ? 'Using baseline feature snapshot' : 'No feature snapshot configured',
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
    this.enqueueDevSync(sample);
    this.trainingState = maybeIncrementPendingUserInputted(this.getTrainingState(), sample.acceptance);
    this.trainingState = updateTrainingStateFromLedger(
      this.trainingState,
      this.ledger,
      this.personalizedCnn,
      this.getDevSyncQueueStatus(),
    );
    this.trainingState = {
      ...this.trainingState,
      trainerStatus: {
        ...this.trainingState.trainerStatus,
        lastEventAt: Date.now(),
      },
    };

    await this.persist();
    void this.flushDevSyncQueue();
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

  private enqueueDevSync(sample: AcceptedSampleRecord): void {
    if (!this.config?.devSync.enabled) {
      return;
    }
    this.devSyncQueue = [
      ...this.devSyncQueue,
      {
        id: `devsync-${sample.id}`,
        sample,
        legacyQuality: sample.acceptance === 'user_inputted' ? 'high_quality' : 'regular',
        attempts: 0,
        lastAttemptAt: null,
        lastError: null,
      },
    ];
  }

  private async flushDevSyncQueue(): Promise<void> {
    if (
      !this.config?.devSync.enabled
      || this.config.devSync.flushPolicy !== 'immediate'
      || this.devSyncQueue.length === 0
    ) {
      return;
    }

    const pending = [...this.devSyncQueue];
    const remaining: DevSyncQueueItem[] = [];
    for (const item of pending) {
      try {
        const response = await fetch(`${this.config.devSync.endpointBaseUrl}/samples`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            id: item.sample.id,
            label: item.sample.label,
            quality: item.legacyQuality,
            source: item.sample.source,
            createdAt: item.sample.createdAt,
            strokes: item.sample.strokes,
            features: item.sample.features,
          }),
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
      } catch (error) {
        remaining.push({
          ...item,
          attempts: item.attempts + 1,
          lastAttemptAt: Date.now(),
          lastError: error instanceof Error ? error.message : 'Unknown dev sync error',
        });
      }
    }

    this.devSyncQueue = remaining;
    if (this.trainingState) {
      this.trainingState = {
        ...this.trainingState,
        devSyncQueueStatus: this.getDevSyncQueueStatus(Date.now()),
      };
      await this.persist();
    }
  }

  private async runTraining(reason: TrainingTriggerReason, milestoneReached: number | null): Promise<void> {
    this.trainingState = {
      ...this.getTrainingState(),
      lastTrainingReason: reason,
      trainerStatus: {
        ...this.getTrainingState().trainerStatus,
        trainingInFlight: true,
        cnnTrainingStatus: this.cnnTrainer?.isAvailable() ? 'training' : 'unavailable',
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
    this.emitTrainingProgress('feature', 'running', 15, 'svm/feature training...');

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
        lastCandidateRejectionReason: 'Not enough balanced per-letter coverage yet.',
        trainerStatus: {
          ...this.getTrainingState().trainerStatus,
          trainingInFlight: false,
          lastEventAt: Date.now(),
        },
      };
      await this.persist();
      this.emitTrainingProgress('feature', 'rejected', 100, 'rejected: not enough balanced coverage');
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
        lastCandidateRejectionReason: 'Could not build a feature-classifier snapshot.',
        trainerStatus: {
          ...this.getTrainingState().trainerStatus,
          trainingInFlight: false,
          lastEventAt: Date.now(),
        },
      };
      await this.persist();
      this.emitTrainingProgress('feature', 'rejected', 100, 'rejected: could not build feature snapshot');
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
    const nextGeneration = this.getTrainingState().personalizationGeneration + 1;
    this.emitTrainingProgress('feature', 'ready', 45, `svm/feature ready! v${nextGeneration}`, nextGeneration);
    const currentMetrics = this.classifierSnapshot?.metrics ?? this.trainingState.latestMetrics;
    let cnnCandidate: CnnTrainingRuntimeResult | null = null;
    if (this.cnnTrainer?.isAvailable()) {
      this.emitTrainingProgress('cnn', 'running', 60, 'cnn training...', nextGeneration);
      cnnCandidate = await this.cnnTrainer.trainCandidate(dataset.training, dataset.holdout, this.personalizedCnn);
      if (!cnnCandidate.accepted || !cnnCandidate.artifacts) {
        const rejectionReason = cnnCandidate.rejectionReason ?? 'CNN training rejected the candidate.';
        this.trainingState = {
          ...this.getTrainingState(),
          milestonesCompleted: baseMilestones,
          lastTrainingOutcome: 'rejected',
          lastRejectedReason: rejectionReason,
          lastCandidateRejectionReason: rejectionReason,
          trainerStatus: {
            ...this.getTrainingState().trainerStatus,
            trainingInFlight: false,
            cnnTrainingStatus: 'rejected',
            cnnTrainingStage: cnnCandidate.stage,
            lastEventAt: Date.now(),
          },
        };
        await this.persist();
        this.emitTrainingProgress('cnn', 'rejected', 100, `rejected: ${rejectionReason}`, this.getTrainingState().personalizationGeneration);
        this.emit({
          type: 'training-rejected',
          payload: {
            reason: rejectionReason,
            trainingState: this.getTrainingState(),
          },
        });
        return;
      }

      const currentCnnMetrics = this.personalizedCnn?.metrics ?? null;
      if (!shouldAcceptCandidateSnapshot(currentCnnMetrics, cnnCandidate.metrics)) {
        this.trainingState = {
          ...this.getTrainingState(),
          milestonesCompleted: baseMilestones,
          lastTrainingOutcome: 'rejected',
          lastRejectedReason: 'CNN regression gate rejected the paired candidate.',
          lastCandidateRejectionReason: 'CNN regression gate rejected the paired candidate.',
          trainerStatus: {
            ...this.getTrainingState().trainerStatus,
            trainingInFlight: false,
            cnnTrainingStatus: 'rejected',
            cnnTrainingStage: cnnCandidate.stage,
            lastEventAt: Date.now(),
          },
        };
        await this.persist();
        this.emitTrainingProgress('cnn', 'rejected', 100, 'rejected: CNN regression gate', this.getTrainingState().personalizationGeneration);
        this.emit({
          type: 'training-rejected',
          payload: {
            reason: 'CNN regression gate rejected the paired candidate.',
            trainingState: this.getTrainingState(),
          },
        });
        return;
      }
      this.emitTrainingProgress('cnn', 'ready', 85, `cnn ready! v${nextGeneration}`, nextGeneration);
    } else {
      this.emitTrainingProgress('cnn', 'skipped', 60, 'cnn skipped', nextGeneration);
    }

    if (!shouldAcceptCandidateSnapshot(currentMetrics, candidate.metrics)) {
      this.trainingState = {
        ...this.getTrainingState(),
        milestonesCompleted: baseMilestones,
        latestMetrics: currentMetrics,
        lastTrainingOutcome: 'rejected',
        lastRejectedReason: 'Regression gate rejected the candidate snapshot.',
        lastCandidateRejectionReason: 'Regression gate rejected the candidate snapshot.',
        trainerStatus: {
          ...this.getTrainingState().trainerStatus,
          trainingInFlight: false,
          cnnTrainingStatus: cnnCandidate ? 'rejected' : this.getTrainingState().trainerStatus.cnnTrainingStatus,
          cnnTrainingStage: cnnCandidate?.stage ?? this.getTrainingState().trainerStatus.cnnTrainingStage,
          lastEventAt: Date.now(),
        },
      };
      await this.persist();
      this.emitTrainingProgress('feature', 'rejected', 100, 'rejected: feature regression gate', this.getTrainingState().personalizationGeneration);
      this.emit({
        type: 'training-rejected',
        payload: {
          reason: 'Regression gate rejected the candidate snapshot.',
          trainingState: this.getTrainingState(),
        },
      });
      return;
    }

    const prospectiveTrainingState = {
      ...this.getTrainingState(),
      milestonesCompleted: baseMilestones,
      latestSnapshotId: candidate.id,
      personalizationGeneration: this.getTrainingState().personalizationGeneration + 1,
      lastCompletedTrainingAt: Date.now(),
      lastTrainingReason: reason,
      lastTrainingOutcome: 'accepted' as const,
      lastRejectedReason: null,
      lastCandidateRejectionReason: null,
      latestMetrics: candidate.metrics,
      pendingUserInputtedSinceTraining: 0,
      personalizedCohortLabelMap: candidate.labelMap,
      trainerStatus: {
        ...this.getTrainingState().trainerStatus,
        trainingInFlight: false,
        activeModelGeneration: this.getTrainingState().personalizationGeneration + 1,
        cnnTrainingStatus: cnnCandidate ? 'accepted' : this.getTrainingState().trainerStatus.cnnTrainingStatus,
        cnnTrainingStage: cnnCandidate?.stage ?? this.getTrainingState().trainerStatus.cnnTrainingStage,
        cnnInferenceSource: cnnCandidate?.artifacts?.inferenceModel ? 'personalized' : this.getTrainingState().trainerStatus.cnnInferenceSource,
        personalizedCnnAvailable: Boolean(cnnCandidate?.artifacts?.inferenceModel ?? this.personalizedCnn?.inferenceModel),
        lastEventAt: Date.now(),
      },
    };
    const prospectiveBytes = estimatePersistedStateBytes(createPersistedState(
      this.baseline!,
      this.ledger,
      candidate,
      cnnCandidate?.artifacts ?? this.personalizedCnn,
      prospectiveTrainingState,
      this.devSyncQueue,
    ));
    if (prospectiveBytes > prospectiveTrainingState.snapshotBudgetBytes) {
      this.trainingState = {
        ...this.getTrainingState(),
        milestonesCompleted: baseMilestones,
        lastTrainingOutcome: 'rejected',
        lastRejectedReason: 'Personalized snapshot exceeds the configured budget.',
        lastCandidateRejectionReason: 'Personalized snapshot exceeds the configured budget.',
        snapshotBudgetStatus: {
          budgetBytes: prospectiveTrainingState.snapshotBudgetBytes,
          usedBytes: this.getTrainingState().persistedBytes,
          withinBudget: false,
          lastRejectedBytes: prospectiveBytes,
        },
        trainerStatus: {
          ...this.getTrainingState().trainerStatus,
          trainingInFlight: false,
          cnnTrainingStatus: cnnCandidate ? 'rejected' : this.getTrainingState().trainerStatus.cnnTrainingStatus,
          cnnTrainingStage: cnnCandidate?.stage ?? this.getTrainingState().trainerStatus.cnnTrainingStage,
          lastEventAt: Date.now(),
        },
      };
      await this.persist();
      this.emitTrainingProgress('finalizing', 'rejected', 100, 'rejected: snapshot budget exceeded', this.getTrainingState().personalizationGeneration);
      this.emit({
        type: 'training-rejected',
        payload: {
          reason: 'Personalized snapshot exceeds the configured budget.',
          trainingState: this.getTrainingState(),
        },
      });
      return;
    }

    this.classifierSnapshot = candidate;
    if (cnnCandidate?.artifacts) {
      this.personalizedCnn = cnnCandidate.artifacts;
      this.cnnSession = null;
      this.cnnSessionKey = null;
    }
    this.emitTrainingProgress('finalizing', 'running', 95, 'finalizing...', nextGeneration);
    this.trainingState = updateTrainingStateFromLedger(
      prospectiveTrainingState,
      this.ledger,
      this.personalizedCnn,
      this.getDevSyncQueueStatus(),
    );
    await this.persist();
    this.emitTrainingProgress('finalizing', 'ready', 100, `ready! v${this.getTrainingState().personalizationGeneration}`);
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
      this.devSyncQueue,
    ));
  }

  async importDevBundle(bundle: ExportedHandwritingBundle): Promise<void> {
    await this.ensureInitialized();
    const { baseline } = this.getResolvedState();
    const persisted = sanitizeImportedState(
      {
        baselineManifest: bundle.baselineManifest,
        coreState: bundle.coreState,
        personalizedCnn: bundle.personalizedCnn,
        trainingState: bundle.trainingState,
        devSyncQueue: bundle.devSyncQueue,
      },
      baseline,
      this.getTrainingState().snapshotBudgetBytes,
    );
    this.baseline = persisted.baselineManifest;
    this.ledger = persisted.coreState.ledger;
    this.classifierSnapshot = persisted.coreState.latestAcceptedFeatureClassifier;
    this.personalizedCnn = persisted.personalizedCnn;
    this.devSyncQueue = persisted.devSyncQueue;
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
