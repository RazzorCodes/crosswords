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
  updatedAt: number;
}

export interface BaselineArtifactManifest {
  version: string;
  labelMap: string[];
  cnn: {
    inferenceUrl: string | null;
    supportsTraining: boolean;
    trainingArtifactsUrl: string | null;
  };
  featureClassifier: FeatureClassifierSnapshot | null;
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
  lastCompletedTrainingAt: number | null;
  lastTrainingReason: TrainingTriggerReason | null;
  lastTrainingOutcome: 'idle' | 'accepted' | 'rejected';
  lastRejectedReason: string | null;
  latestMetrics: SnapshotMetrics | null;
  persistedBytes: number;
  snapshotBudgetBytes: number;
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
    cnnTrainingAvailable: boolean;
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
  snapshotBudgetBytes?: number;
  persistDebounceMs?: number;
  pendingImplicitWindowMs?: number;
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
  version: number;
  exportedAt: number;
  baseline: BaselineArtifactManifest;
  ledger: AcceptedSampleRecord[];
  classifierSnapshot: FeatureClassifierSnapshot | null;
  knnCache: KNNExampleRecord[];
  personalizedCnn: PersonalizedCnnArtifacts | null;
  trainingState: TrainingState;
}

export interface PersistedHandwritingState {
  baseline: BaselineArtifactManifest;
  ledger: AcceptedSampleRecord[];
  classifierSnapshot: FeatureClassifierSnapshot | null;
  knnCache: KNNExampleRecord[];
  personalizedCnn: PersonalizedCnnArtifacts | null;
  trainingState: TrainingState;
}

export interface KeyValueStore {
  get<T>(key: string): Promise<T | null>;
  set<T>(key: string, value: T): Promise<void>;
  clear(): Promise<void>;
}
