import type { EngineResult, RecognitionCandidate, RecognizerStatus, StrokeInput } from '../recognizers/types';

export type SampleAcceptance = 'user_inputted' | 'implicit';

export interface AcceptedSampleInput {
  label: string;
  strokes: StrokeInput;
  acceptance: SampleAcceptance;
  source: string;
  createdAt?: number;
}

export interface AcceptedSampleRecord extends AcceptedSampleInput {
  id: string;
  label: string;
  createdAt: number;
  features: number[];
}

export interface KNNExampleRecord {
  id: string;
  label: string;
  features: number[];
}

export interface LetterAcceptanceCounts {
  user_inputted: number;
  implicit: number;
}

export interface FeatureClassifierCentroid {
  label: string;
  centroid: number[];
  count: number;
}

export interface FeatureClassifierSnapshot {
  id: string;
  version: string;
  createdAt: number;
  centroids: FeatureClassifierCentroid[];
  labelMap: string[];
  metrics: SnapshotMetrics;
  datasetSize: number;
  readyLetters: string[];
  reason: TrainingTriggerReason;
}

export interface SnapshotMetrics {
  user_inputtedAccuracy: number;
  implicitAccuracy: number;
  overallAccuracy: number;
}

export interface PersonalizedCnnArtifacts {
  checkpoint?: ArrayBuffer | null;
  inferenceModel?: ArrayBuffer | null;
  exportMetadata?: Record<string, unknown> | null;
  metrics?: SnapshotMetrics | null;
  stage?: CnnTrainingStage | null;
  updatedAt: number;
}

export interface CnnTrainingArtifactUrls {
  trainUrl: string | null;
  evalUrl: string | null;
  optimizerUrl: string | null;
  checkpointUrl: string | null;
  exportMetadataUrl: string | null;
}

export interface CnnTrainingRuntimeUrls {
  moduleUrl: string | null;
  wasmUrl: string | null;
  simdWasmUrl: string | null;
  threadedWasmUrl: string | null;
}

export type CnnTrainingStage = 'head-only' | 'partial-finetune';

export interface BaselineArtifactManifest {
  version: string;
  labelMap: string[];
  cnn: {
    inferenceUrl: string | null;
    supportsTraining: boolean;
    trainingArtifacts: CnnTrainingArtifactUrls;
    trainingRuntime: CnnTrainingRuntimeUrls;
  };
  featureClassifier: FeatureClassifierSnapshot | null;
}

export interface SerializedRustCoreState {
  version: number;
  baselineVersion: string;
  ledger: AcceptedSampleRecord[];
  milestonesCompleted: number[];
  pendingUserInputtedSinceTraining: number;
  latestAcceptedFeatureClassifier: FeatureClassifierSnapshot | null;
  knnCache: KNNExampleRecord[];
}

export interface DevSyncQueueItem {
  id: string;
  sample: AcceptedSampleRecord;
  legacyQuality: 'high_quality' | 'regular';
  attempts: number;
  lastAttemptAt: number | null;
  lastError: string | null;
}

export interface DevSyncQueueStatus {
  pending: number;
  failed: number;
  lastFlushAt: number | null;
}

export interface TrainingState {
  initialized: boolean;
  baselineVersion: string;
  totalAcceptedSamples: number;
  countsByLetter: Record<string, LetterAcceptanceCounts>;
  readyLetters: string[];
  milestonesCompleted: number[];
  nextMilestone: number | null;
  pendingUserInputtedSinceTraining: number;
  latestSnapshotId: string | null;
  personalizationGeneration: number;
  lastCompletedTrainingAt: number | null;
  lastTrainingReason: TrainingTriggerReason | null;
  lastTrainingOutcome: 'idle' | 'accepted' | 'rejected';
  lastRejectedReason: string | null;
  latestMetrics: SnapshotMetrics | null;
  persistedBytes: number;
  snapshotBudgetBytes: number;
  personalizedCohortLabelMap: string[];
  lastCandidateRejectionReason: string | null;
  snapshotBudgetStatus: {
    budgetBytes: number;
    usedBytes: number;
    withinBudget: boolean;
    lastRejectedBytes: number | null;
  };
  devSyncQueueStatus: DevSyncQueueStatus;
  recentAcceptedSamples: Array<{
    id: string;
    label: string;
    acceptance: SampleAcceptance;
    source: string;
    createdAt: number;
  }>;
  trainerStatus: {
    trainingInFlight: boolean;
    lastEventAt: number | null;
    activeModelGeneration: number;
    cnnTrainingAvailable: boolean;
    cnnTrainingStatus: 'unavailable' | 'ready' | 'training' | 'accepted' | 'rejected' | 'error';
    cnnTrainingStage: CnnTrainingStage | null;
    cnnInferenceSource: 'baseline' | 'personalized';
    personalizedCnnAvailable: boolean;
  };
}

export interface HandwritingPrediction {
  status: RecognizerStatus;
  candidates: RecognitionCandidate[];
  chosenLabel: string | null;
  confidence: number;
  sourceTrail: string[];
  engineResults: EngineResult[];
  trainingState: TrainingState;
}

export interface HandwritingModuleConfig {
  modelBaseUrl?: string;
  baselineManifestUrl?: string;
  cnnTrainingRuntime?: unknown;
  snapshotBudgetBytes?: number;
  persistDebounceMs?: number;
  pendingImplicitWindowMs?: number;
  devSync?: {
    enabled?: boolean;
    mode?: 'legacy-server';
    endpointBaseUrl?: string;
    flushPolicy?: 'immediate' | 'manual';
  };
}

export interface HandwritingModuleInitResult {
  trainingState: TrainingState;
}

export interface HandwritingModuleEventMap {
  prediction: HandwritingPrediction;
  'sample-recorded': {
    sample: AcceptedSampleRecord;
    trainingState: TrainingState;
  };
  'milestone-reached': {
    milestone: number;
    trainingState: TrainingState;
  };
  'training-started': {
    reason: TrainingTriggerReason;
    trainingState: TrainingState;
  };
  'training-progress': {
    phase: 'feature' | 'cnn' | 'finalizing';
    status: 'running' | 'ready' | 'skipped' | 'rejected';
    progress: number;
    message: string;
    generation: number;
    trainingState: TrainingState;
  };
  'training-completed': {
    snapshot: FeatureClassifierSnapshot;
    trainingState: TrainingState;
  };
  'training-rejected': {
    reason: string;
    trainingState: TrainingState;
  };
  'artifacts-updated': {
    trainingState: TrainingState;
  };
}

export type HandwritingModuleEvent =
  { type: 'prediction'; payload: HandwritingModuleEventMap['prediction'] }
  | { type: 'sample-recorded'; payload: HandwritingModuleEventMap['sample-recorded'] }
  | { type: 'milestone-reached'; payload: HandwritingModuleEventMap['milestone-reached'] }
  | { type: 'training-started'; payload: HandwritingModuleEventMap['training-started'] }
  | { type: 'training-progress'; payload: HandwritingModuleEventMap['training-progress'] }
  | { type: 'training-completed'; payload: HandwritingModuleEventMap['training-completed'] }
  | { type: 'training-rejected'; payload: HandwritingModuleEventMap['training-rejected'] }
  | { type: 'artifacts-updated'; payload: HandwritingModuleEventMap['artifacts-updated'] };

export type HandwritingModuleListener = (event: HandwritingModuleEvent) => void;

export type TrainingTriggerReason =
  | 'milestone-50'
  | 'milestone-100'
  | 'milestone-200'
  | 'user-batch-10';

export interface TrainingTriggerDecision {
  shouldTrain: boolean;
  reason: TrainingTriggerReason | null;
  milestoneReached: number | null;
}

export interface BalancedDataset {
  training: AcceptedSampleRecord[];
  holdout: AcceptedSampleRecord[];
  readyLetters: string[];
  perLetterTarget: number;
}

export interface ExportedHandwritingBundle {
  version: 2;
  exportedAt: number;
  baselineManifest: BaselineArtifactManifest;
  coreState: SerializedRustCoreState;
  personalizedCnn: PersonalizedCnnArtifacts | null;
  trainingState: TrainingState;
  devSyncQueue: DevSyncQueueItem[];
}

export interface PersistedHandwritingState {
  baselineManifest: BaselineArtifactManifest;
  coreState: SerializedRustCoreState;
  personalizedCnn: PersonalizedCnnArtifacts | null;
  trainingState: TrainingState;
  devSyncQueue: DevSyncQueueItem[];
}

export interface KeyValueStore {
  get<T>(key: string): Promise<T | null>;
  set<T>(key: string, value: T): Promise<void>;
  clear(): Promise<void>;
}
