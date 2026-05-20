import { extractFeatures } from '../recognizers/features';
import type { RecognitionCandidate, StrokeInput } from '../recognizers/types';
import {
  buildBalancedDatasetRust,
  computeLetterStatsRust,
  evaluateSnapshotRust,
  extractFeaturesRust,
  isHandwritingRustCoreReady,
  predictFeatureClassifierProbabilitiesRust,
  predictKnnRust,
  trainFeatureClassifierRust,
} from './wasmCore';
import type {
  AcceptedSampleInput,
  AcceptedSampleRecord,
  BalancedDataset,
  BaselineArtifactManifest,
  ExportedHandwritingBundle,
  FeatureClassifierCentroid,
  FeatureClassifierSnapshot,
  KNNExampleRecord,
  LetterAcceptanceCounts,
  PersistedHandwritingState,
  PersonalizedCnnArtifacts,
  SampleAcceptance,
  SerializedRustCoreState,
  SnapshotMetrics,
  TrainingState,
  TrainingTriggerDecision,
  TrainingTriggerReason,
} from './types';

const ALPHABET = Array.from({ length: 26 }, (_, index) => String.fromCharCode(65 + index));
const MILESTONES = [50, 100, 200] as const;
const MIN_READY_SAMPLES_PER_LETTER = 5;
const MIN_READY_USER_INPUTTED_PER_LETTER = 1;
const HOLDOUT_FRACTION = 0.2;
const OVERALL_TOLERANCE = 0.03;
const IMPLICIT_TOLERANCE = 0.05;
const USER_INPUTTED_TOLERANCE = 0.02;
const RECENT_SAMPLE_LIMIT = 12;

type LegacyPersistedHandwritingState = Partial<{
  baseline: BaselineArtifactManifest;
  ledger: AcceptedSampleRecord[];
  classifierSnapshot: FeatureClassifierSnapshot | null;
}>;

type ImportablePersistedHandwritingState = Partial<PersistedHandwritingState> & LegacyPersistedHandwritingState;

function randomId(prefix: string) {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

export function getAlphabet(): string[] {
  return [...ALPHABET];
}

export function normalizeLabel(label: string): string {
  return label.slice(0, 1).toUpperCase();
}

export function isLetterLabel(label: string): boolean {
  return /^[A-Z]$/.test(label);
}

export function createCountsByLetter(): Record<string, LetterAcceptanceCounts> {
  return Object.fromEntries(
    ALPHABET.map((label) => [label, { user_inputted: 0, implicit: 0 }]),
  );
}

export function createDefaultBaselineManifest(modelBaseUrl: string): BaselineArtifactManifest {
  const baseUrl = modelBaseUrl.replace(/\/$/, '');
  return {
    version: 'builtin-v1',
    labelMap: getAlphabet(),
    cnn: {
      inferenceUrl: `${baseUrl}/cnn.onnx`,
      supportsTraining: false,
      trainingArtifacts: {
        trainUrl: null,
        evalUrl: null,
        optimizerUrl: null,
        checkpointUrl: null,
        exportMetadataUrl: null,
      },
      trainingRuntime: {
        moduleUrl: null,
        wasmUrl: null,
        simdWasmUrl: null,
        threadedWasmUrl: null,
      },
    },
    featureClassifier: null,
  };
}

export function createInitialTrainingState(
  baseline: BaselineArtifactManifest,
  snapshotBudgetBytes: number,
): TrainingState {
  return {
    initialized: false,
    baselineVersion: baseline.version,
    totalAcceptedSamples: 0,
    countsByLetter: createCountsByLetter(),
    readyLetters: [],
    milestonesCompleted: [],
    nextMilestone: MILESTONES[0],
    pendingUserInputtedSinceTraining: 0,
    latestSnapshotId: null,
    personalizationGeneration: 0,
    lastCompletedTrainingAt: null,
    lastTrainingReason: null,
    lastTrainingOutcome: 'idle',
    lastRejectedReason: null,
    latestMetrics: null,
    persistedBytes: 0,
    snapshotBudgetBytes,
    personalizedCohortLabelMap: baseline.featureClassifier?.labelMap ?? [],
    lastCandidateRejectionReason: null,
    snapshotBudgetStatus: {
      budgetBytes: snapshotBudgetBytes,
      usedBytes: 0,
      withinBudget: true,
      lastRejectedBytes: null,
    },
    devSyncQueueStatus: {
      pending: 0,
      failed: 0,
      lastFlushAt: null,
    },
    recentAcceptedSamples: [],
    trainerStatus: {
      trainingInFlight: false,
      lastEventAt: null,
      activeModelGeneration: 0,
      cnnTrainingAvailable: baseline.cnn.supportsTraining,
      cnnTrainingStatus: baseline.cnn.supportsTraining ? 'ready' : 'unavailable',
      cnnTrainingStage: null,
      cnnInferenceSource: 'baseline',
      personalizedCnnAvailable: false,
    },
  };
}

export function estimatePersistedStateBytes(state: PersistedHandwritingState): number {
  const encoder = new TextEncoder();
  return encoder.encode(JSON.stringify(state)).byteLength;
}

export function createAcceptedSampleRecord(input: AcceptedSampleInput): AcceptedSampleRecord {
  const label = normalizeLabel(input.label);
  return {
    id: randomId('sample'),
    label,
    strokes: input.strokes,
    acceptance: input.acceptance,
    source: input.source,
    createdAt: input.createdAt ?? Date.now(),
    features: extractFeaturesFromStrokes(input.strokes),
  };
}

export function rebuildKnnCache(ledger: AcceptedSampleRecord[]): KNNExampleRecord[] {
  return ledger.map((sample) => ({
    id: sample.id,
    label: sample.label,
    features: sample.features,
  }));
}

function euclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let index = 0; index < a.length; index += 1) {
    sum += (a[index] - b[index]) ** 2;
  }
  return Math.sqrt(sum);
}

export function predictKnnFromFeatures(
  features: number[],
  cache: KNNExampleRecord[],
  k = 5,
  farNeighborDistance = 4.5,
): RecognitionCandidate[] {
  if (isHandwritingRustCoreReady()) {
    return predictKnnRust(features, cache, k, farNeighborDistance);
  }

  if (cache.length === 0) {
    return [];
  }

  const nearest = cache
    .map((entry) => ({
      label: entry.label,
      distance: euclideanDistance(features, entry.features),
    }))
    .sort((left, right) => left.distance - right.distance)
    .slice(0, k);

  if (nearest.length === 0 || nearest[0].distance > farNeighborDistance) {
    return [];
  }

  let totalWeight = 0;
  const scores: Record<string, number> = {};
  for (const neighbor of nearest) {
    const weight = 1 / Math.max(neighbor.distance, 1e-6);
    totalWeight += weight;
    scores[neighbor.label] = (scores[neighbor.label] ?? 0) + weight;
  }

  return Object.entries(scores)
    .map(([label, score]) => ({
      char: label,
      score: totalWeight > 0 ? score / totalWeight : 0,
      source: 'knn',
    }))
    .sort((left, right) => right.score - left.score);
}

function averageVectors(vectors: number[][]): number[] {
  const result = new Array(vectors[0]?.length ?? 0).fill(0);
  if (vectors.length === 0) {
    return result;
  }
  for (const vector of vectors) {
    for (let index = 0; index < vector.length; index += 1) {
      result[index] += vector[index];
    }
  }
  return result.map((value) => value / vectors.length);
}

export function trainFeatureClassifier(
  samples: AcceptedSampleRecord[],
  reason: TrainingTriggerReason,
  readyLetters: string[],
): FeatureClassifierSnapshot | null {
  if (isHandwritingRustCoreReady()) {
    return trainFeatureClassifierRust(samples, readyLetters, reason);
  }

  if (samples.length === 0 || readyLetters.length < 2) {
    return null;
  }

  const grouped = new Map<string, number[][]>();
  for (const sample of samples) {
    if (!grouped.has(sample.label)) {
      grouped.set(sample.label, []);
    }
    grouped.get(sample.label)?.push(sample.features);
  }

  const centroids: FeatureClassifierCentroid[] = [];
  for (const label of readyLetters) {
    const featureSet = grouped.get(label);
    if (!featureSet || featureSet.length === 0) {
      continue;
    }
    centroids.push({
      label,
      centroid: averageVectors(featureSet),
      count: featureSet.length,
    });
  }

  if (centroids.length < 2) {
    return null;
  }

  return {
    id: randomId('snapshot'),
    version: 'prototype-v1',
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
}

export function predictFeatureClassifierProbabilities(
  snapshot: FeatureClassifierSnapshot | null,
  features: number[],
): Record<string, number> {
  if (isHandwritingRustCoreReady()) {
    return predictFeatureClassifierProbabilitiesRust(snapshot, features);
  }

  const probabilities = Object.fromEntries(ALPHABET.map((label) => [label, 0])) as Record<string, number>;

  if (!snapshot || snapshot.centroids.length === 0) {
    const uniform = 1 / ALPHABET.length;
    for (const label of ALPHABET) {
      probabilities[label] = uniform;
    }
    return probabilities;
  }

  const scores = snapshot.centroids.map((centroid) => ({
    label: centroid.label,
    score: Math.exp(-euclideanDistance(features, centroid.centroid)),
  }));
  const total = scores.reduce((sum, entry) => sum + entry.score, 0);
  if (total <= 0) {
    const uniform = 1 / snapshot.centroids.length;
    for (const entry of snapshot.centroids) {
      probabilities[entry.label] = uniform;
    }
    return probabilities;
  }

  for (const entry of scores) {
    probabilities[entry.label] = entry.score / total;
  }

  return probabilities;
}

export function toCandidatesFromProbabilities(
  probabilities: Record<string, number>,
  source: string,
): RecognitionCandidate[] {
  return Object.entries(probabilities)
    .map(([label, score]) => ({
      char: label,
      score,
      source,
    }))
    .sort((left, right) => right.score - left.score);
}

function selectHoldoutCount(count: number): number {
  if (count < MIN_READY_SAMPLES_PER_LETTER) {
    return 0;
  }
  return Math.max(1, Math.floor(count * HOLDOUT_FRACTION));
}

export function buildBalancedDataset(ledger: AcceptedSampleRecord[]): BalancedDataset {
  if (isHandwritingRustCoreReady()) {
    return buildBalancedDatasetRust(ledger);
  }

  const byLetter = new Map<string, { user_inputted: AcceptedSampleRecord[]; implicit: AcceptedSampleRecord[] }>();
  for (const label of ALPHABET) {
    byLetter.set(label, { user_inputted: [], implicit: [] });
  }

  for (const sample of [...ledger].sort((left, right) => right.createdAt - left.createdAt)) {
    if (!isLetterLabel(sample.label)) {
      continue;
    }
    byLetter.get(sample.label)?.[sample.acceptance].push(sample);
  }

  const readyLetters = ALPHABET.filter((label) => {
    const buckets = byLetter.get(label);
    if (!buckets) {
      return false;
    }
    return (
      buckets.user_inputted.length >= MIN_READY_USER_INPUTTED_PER_LETTER &&
      (buckets.user_inputted.length + buckets.implicit.length) >= MIN_READY_SAMPLES_PER_LETTER
    );
  });

  if (readyLetters.length < 2) {
    return {
      training: [],
      holdout: [],
      readyLetters: [],
      perLetterTarget: 0,
    };
  }

  const perLetterTarget = Math.min(
    ...readyLetters.map((label) => {
      const buckets = byLetter.get(label)!;
      return buckets.user_inputted.length + buckets.implicit.length;
    }),
  );

  const training: AcceptedSampleRecord[] = [];
  const holdout: AcceptedSampleRecord[] = [];
  const targetUserInputted = Math.max(1, Math.round(perLetterTarget * 0.2));

  for (const label of readyLetters) {
    const buckets = byLetter.get(label)!;
    const userHoldoutCount = selectHoldoutCount(buckets.user_inputted.length);
    const implicitHoldoutCount = selectHoldoutCount(buckets.implicit.length);

    const userHoldout = userHoldoutCount > 0
      ? buckets.user_inputted.slice(-userHoldoutCount)
      : [];
    const implicitHoldout = implicitHoldoutCount > 0
      ? buckets.implicit.slice(-implicitHoldoutCount)
      : [];
    holdout.push(...userHoldout, ...implicitHoldout);

    const userPool = buckets.user_inputted.slice(0, buckets.user_inputted.length - userHoldoutCount);
    const implicitPool = buckets.implicit.slice(0, buckets.implicit.length - implicitHoldoutCount);

    const chosenUser = userPool.slice(0, targetUserInputted);
    const implicitTarget = Math.max(0, perLetterTarget - chosenUser.length);
    const chosenImplicit = implicitPool.slice(0, implicitTarget);

    const selected = [...chosenUser, ...chosenImplicit];
    if (selected.length < perLetterTarget) {
      const seenIds = new Set(selected.map((sample) => sample.id));
      const overflow = [...implicitPool.slice(chosenImplicit.length), ...userPool.slice(chosenUser.length)]
        .filter((sample) => !seenIds.has(sample.id))
        .slice(0, perLetterTarget - selected.length);
      selected.push(...overflow);
    }

    training.push(...selected.slice(0, perLetterTarget));
  }

  return {
    training,
    holdout,
    readyLetters,
    perLetterTarget,
  };
}

function accuracyForSubset(samples: AcceptedSampleRecord[], snapshot: FeatureClassifierSnapshot | null): number {
  if (samples.length === 0 || !snapshot) {
    return 0;
  }

  let correct = 0;
  for (const sample of samples) {
    const probabilities = predictFeatureClassifierProbabilities(snapshot, sample.features);
    const [prediction] = toCandidatesFromProbabilities(probabilities, 'feature-classifier');
    if (prediction?.char === sample.label) {
      correct += 1;
    }
  }
  return correct / samples.length;
}

export function evaluateSnapshot(
  snapshot: FeatureClassifierSnapshot | null,
  holdout: AcceptedSampleRecord[],
): SnapshotMetrics {
  if (isHandwritingRustCoreReady()) {
    return evaluateSnapshotRust(snapshot, holdout);
  }

  const userInputted = holdout.filter((sample) => sample.acceptance === 'user_inputted');
  const implicit = holdout.filter((sample) => sample.acceptance === 'implicit');
  return {
    user_inputtedAccuracy: accuracyForSubset(userInputted, snapshot),
    implicitAccuracy: accuracyForSubset(implicit, snapshot),
    overallAccuracy: accuracyForSubset(holdout, snapshot),
  };
}

export function shouldAcceptCandidateSnapshot(
  currentMetrics: SnapshotMetrics | null,
  candidateMetrics: SnapshotMetrics,
): boolean {
  if (!currentMetrics) {
    return true;
  }

  return (
    candidateMetrics.user_inputtedAccuracy >= currentMetrics.user_inputtedAccuracy - USER_INPUTTED_TOLERANCE &&
    candidateMetrics.implicitAccuracy >= currentMetrics.implicitAccuracy - IMPLICIT_TOLERANCE &&
    candidateMetrics.overallAccuracy >= currentMetrics.overallAccuracy - OVERALL_TOLERANCE
  );
}

export function computeCountsByLetter(ledger: AcceptedSampleRecord[]): Record<string, LetterAcceptanceCounts> {
  if (isHandwritingRustCoreReady()) {
    return computeLetterStatsRust(ledger).countsByLetter;
  }

  const counts = createCountsByLetter();
  for (const sample of ledger) {
    if (!isLetterLabel(sample.label)) {
      continue;
    }
    counts[sample.label][sample.acceptance] += 1;
  }
  return counts;
}

export function getReadyLettersFromCounts(countsByLetter: Record<string, LetterAcceptanceCounts>): string[] {
  return ALPHABET.filter((label) => {
    const counts = countsByLetter[label];
    return counts.user_inputted >= MIN_READY_USER_INPUTTED_PER_LETTER
      && (counts.user_inputted + counts.implicit) >= MIN_READY_SAMPLES_PER_LETTER;
  });
}

export function getPriorityLetters(countsByLetter: Record<string, LetterAcceptanceCounts>): string[] {
  const totals = ALPHABET.map((label) => countsByLetter[label].user_inputted + countsByLetter[label].implicit);
  const totalSamples = totals.reduce((sum, value) => sum + value, 0);
  if (totalSamples === 0) {
    return [];
  }
  const average = totalSamples / ALPHABET.length;
  return ALPHABET.filter((label) => {
    const total = countsByLetter[label].user_inputted + countsByLetter[label].implicit;
    return total < average * 0.85;
  });
}

export function getMostNeededLetter(countsByLetter: Record<string, LetterAcceptanceCounts>): string | null {
  let observedSamples = 0;
  let letter: string | null = null;
  let minCount = Number.POSITIVE_INFINITY;
  for (const candidate of ALPHABET) {
    const total = countsByLetter[candidate].user_inputted + countsByLetter[candidate].implicit;
    observedSamples += total;
    if (total < minCount) {
      minCount = total;
      letter = candidate;
    }
  }
  return observedSamples > 0 && Number.isFinite(minCount) ? letter : null;
}

export function decideTrainingTrigger(state: TrainingState): TrainingTriggerDecision {
  for (const milestone of MILESTONES) {
    if (state.totalAcceptedSamples >= milestone && !state.milestonesCompleted.includes(milestone)) {
      return {
        shouldTrain: true,
        reason: `milestone-${milestone}` as TrainingTriggerReason,
        milestoneReached: milestone,
      };
    }
  }

  if (
    state.totalAcceptedSamples >= 200 &&
    state.pendingUserInputtedSinceTraining >= 10 &&
    state.readyLetters.length >= 2
  ) {
    return {
      shouldTrain: true,
      reason: 'user-batch-10',
      milestoneReached: null,
    };
  }

  return {
    shouldTrain: false,
    reason: null,
    milestoneReached: null,
  };
}

export function updateTrainingStateFromLedger(
  state: TrainingState,
  ledger: AcceptedSampleRecord[],
  personalizedCnn: PersonalizedCnnArtifacts | null,
  devSyncQueueStatus = state.devSyncQueueStatus,
): TrainingState {
  const countsByLetter = computeCountsByLetter(ledger);
  const readyLetters = getReadyLettersFromCounts(countsByLetter);
  const recentAcceptedSamples = [...ledger]
    .sort((left, right) => right.createdAt - left.createdAt)
    .slice(0, RECENT_SAMPLE_LIMIT)
    .map((sample) => ({
      id: sample.id,
      label: sample.label,
      acceptance: sample.acceptance,
      source: sample.source,
      createdAt: sample.createdAt,
    }));

  const nextMilestone = MILESTONES.find((milestone) => !state.milestonesCompleted.includes(milestone)) ?? null;

  return {
    ...state,
    initialized: true,
    totalAcceptedSamples: ledger.length,
    countsByLetter,
    readyLetters,
    personalizedCohortLabelMap: state.personalizedCohortLabelMap ?? [],
    lastCandidateRejectionReason: state.lastCandidateRejectionReason ?? state.lastRejectedReason,
    devSyncQueueStatus,
    nextMilestone,
    recentAcceptedSamples,
    trainerStatus: {
      ...state.trainerStatus,
      cnnInferenceSource: personalizedCnn?.inferenceModel ? 'personalized' : 'baseline',
      personalizedCnnAvailable: Boolean(personalizedCnn?.inferenceModel),
      activeModelGeneration: state.personalizationGeneration,
    },
  };
}

export function createSerializedCoreState(
  baseline: BaselineArtifactManifest,
  ledger: AcceptedSampleRecord[],
  classifierSnapshot: FeatureClassifierSnapshot | null,
  trainingState: TrainingState,
): SerializedRustCoreState {
  return {
    version: 1,
    baselineVersion: baseline.version,
    ledger,
    milestonesCompleted: trainingState.milestonesCompleted,
    pendingUserInputtedSinceTraining: trainingState.pendingUserInputtedSinceTraining,
    latestAcceptedFeatureClassifier: classifierSnapshot,
    knnCache: rebuildKnnCache(ledger),
  };
}

export function createPersistedState(
  baseline: BaselineArtifactManifest,
  ledger: AcceptedSampleRecord[],
  classifierSnapshot: FeatureClassifierSnapshot | null,
  personalizedCnn: PersonalizedCnnArtifacts | null,
  trainingState: TrainingState,
  devSyncQueue: PersistedHandwritingState['devSyncQueue'] = [],
): PersistedHandwritingState {
  return {
    baselineManifest: baseline,
    coreState: createSerializedCoreState(baseline, ledger, classifierSnapshot, trainingState),
    personalizedCnn,
    trainingState,
    devSyncQueue,
  };
}

export function createExportBundle(state: PersistedHandwritingState): ExportedHandwritingBundle {
  return {
    version: 2,
    exportedAt: Date.now(),
    baselineManifest: state.baselineManifest,
    coreState: state.coreState,
    personalizedCnn: state.personalizedCnn,
    trainingState: state.trainingState,
    devSyncQueue: state.devSyncQueue,
  };
}

function coerceIncomingBaseline(
  incoming: ImportablePersistedHandwritingState | null | undefined,
  fallbackBaseline: BaselineArtifactManifest,
): BaselineArtifactManifest {
  return incoming?.baselineManifest
    ?? incoming?.baseline
    ?? fallbackBaseline;
}

function coerceIncomingLedger(
  incoming: ImportablePersistedHandwritingState | null | undefined,
): AcceptedSampleRecord[] {
  const maybeLedger = incoming?.coreState?.ledger
    ?? incoming?.ledger
    ?? [];
  return Array.isArray(maybeLedger)
    ? maybeLedger.filter((sample): sample is AcceptedSampleRecord => (
      sample != null
      && typeof sample.id === 'string'
      && typeof sample.label === 'string'
      && /^[A-Z]$/.test(sample.label)
      && Array.isArray(sample.features)
      && sample.features.length === 30
      && Array.isArray(sample.strokes)
      && (sample.acceptance === 'user_inputted' || sample.acceptance === 'implicit')
      && typeof sample.source === 'string'
      && typeof sample.createdAt === 'number'
    ))
    : [];
}

function coerceIncomingSnapshot(
  incoming: ImportablePersistedHandwritingState | null | undefined,
): FeatureClassifierSnapshot | null {
  return incoming?.coreState?.latestAcceptedFeatureClassifier
    ?? incoming?.classifierSnapshot
    ?? null;
}

export function sanitizeImportedState(
  incoming: ImportablePersistedHandwritingState | null | undefined,
  fallbackBaseline: BaselineArtifactManifest,
  snapshotBudgetBytes: number,
): PersistedHandwritingState {
  const baseline = coerceIncomingBaseline(incoming, fallbackBaseline);
  const ledger = coerceIncomingLedger(incoming);
  const classifierSnapshot = coerceIncomingSnapshot(incoming);
  const personalizedCnn = incoming?.personalizedCnn ?? null;
  const devSyncQueue = Array.isArray(incoming?.devSyncQueue) ? incoming.devSyncQueue : [];
  const baseState = createInitialTrainingState(baseline, snapshotBudgetBytes);
  const trainingState = updateTrainingStateFromLedger(
    {
      ...baseState,
      ...incoming?.trainingState,
      baselineVersion: baseline.version,
      personalizationGeneration: incoming?.trainingState?.personalizationGeneration ?? baseState.personalizationGeneration,
      snapshotBudgetBytes,
      personalizedCohortLabelMap: classifierSnapshot?.labelMap ?? incoming?.trainingState?.personalizedCohortLabelMap ?? [],
      lastCandidateRejectionReason: incoming?.trainingState?.lastCandidateRejectionReason ?? incoming?.trainingState?.lastRejectedReason ?? null,
      trainerStatus: {
        ...baseState.trainerStatus,
        ...incoming?.trainingState?.trainerStatus,
        cnnTrainingAvailable: baseline.cnn.supportsTraining,
        cnnTrainingStatus: baseline.cnn.supportsTraining
          ? incoming?.trainingState?.trainerStatus?.cnnTrainingStatus ?? 'ready'
          : 'unavailable',
        cnnInferenceSource: personalizedCnn?.inferenceModel ? 'personalized' : 'baseline',
      },
    },
    ledger,
    personalizedCnn,
    {
      pending: devSyncQueue.length,
      failed: devSyncQueue.filter((item) => item.lastError).length,
      lastFlushAt: incoming?.trainingState?.devSyncQueueStatus?.lastFlushAt ?? null,
    },
  );
  const persisted = createPersistedState(baseline, ledger, classifierSnapshot, personalizedCnn, trainingState, devSyncQueue);
  const persistedBytes = estimatePersistedStateBytes(persisted);
  return {
    ...persisted,
    trainingState: {
      ...persisted.trainingState,
      persistedBytes,
      snapshotBudgetStatus: {
        budgetBytes: snapshotBudgetBytes,
        usedBytes: persistedBytes,
        withinBudget: persistedBytes <= snapshotBudgetBytes,
        lastRejectedBytes: incoming?.trainingState?.snapshotBudgetStatus?.lastRejectedBytes ?? null,
      },
    },
  };
}

export function extractFeaturesFromStrokes(strokes: StrokeInput): number[] {
  if (isHandwritingRustCoreReady()) {
    return extractFeaturesRust(strokes);
  }
  return extractFeatures(strokes);
}

export function maybeIncrementPendingUserInputted(
  state: TrainingState,
  acceptance: SampleAcceptance,
): TrainingState {
  if (acceptance !== 'user_inputted') {
    return state;
  }
  return {
    ...state,
    pendingUserInputtedSinceTraining: state.pendingUserInputtedSinceTraining + 1,
  };
}
