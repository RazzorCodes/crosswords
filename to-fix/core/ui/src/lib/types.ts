export const ALPHABET = Array.from({ length: 26 }, (_, index) => String.fromCharCode(65 + index));
export const FEATURE_COUNT = 30;

export interface Point {
  x: number;
  y: number;
  t: number;
}

export type StrokeInput = Point[][];

export interface LabSample {
  id: string;
  label: string;
  strokes: StrokeInput;
  features: number[];
  createdAt: number;
  source: 'core-lab';
}

export interface SnapshotMetrics {
  user_inputtedAccuracy: number;
  implicitAccuracy: number;
  overallAccuracy: number;
}

export interface LabSvmSnapshot {
  id: string;
  version: 'svm-rbf-v1';
  createdAt: number;
  c: number;
  gamma: number;
  labels: string[];
  biases: number[];
  starts: number[];
  counts: number[];
  coefficients: number[];
  supportVectors: number[];
  featureMean: number[];
  featureStd: number[];
  supportCount: number;
  datasetSize: number;
  readyLetters: string[];
  metrics: SnapshotMetrics;
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

export interface BaselineManifest {
  version: string;
  labelMap: string[];
  cnn: {
    inferenceUrl: string | null;
    supportsTraining: boolean;
    trainingArtifacts: CnnTrainingArtifactUrls;
    trainingRuntime: CnnTrainingRuntimeUrls;
  };
  featureClassifier?: unknown;
}

export interface LabCnnArtifacts {
  checkpoint?: ArrayBuffer | null;
  inferenceModel?: ArrayBuffer | null;
  exportMetadata?: Record<string, unknown> | null;
  metrics?: SnapshotMetrics | null;
  stage?: 'head-only' | 'partial-finetune' | null;
  updatedAt: number;
}

export interface LabState {
  ledger: LabSample[];
  svmSnapshot: LabSvmSnapshot | null;
  cnnArtifacts: LabCnnArtifacts | null;
  baselineManifest: BaselineManifest | null;
  latestMetrics: SnapshotMetrics | null;
}

export interface LabCandidate {
  label: string;
  score: number;
}

export type EngineAlgorithm = 'knn' | 'svm' | 'cnn';
export type EngineStatus = 'ready' | 'unavailable' | 'error';

export interface LabEngineResult {
  algorithm: EngineAlgorithm;
  status: EngineStatus;
  candidates: LabCandidate[];
  topLabel: string | null;
  confidence: number | null;
  detail?: string;
}

export interface LabRecognitionResult {
  aggregateCandidates: LabCandidate[];
  engineResults: LabEngineResult[];
  latenciesMs: Partial<Record<EngineAlgorithm | 'features' | 'aggregate', number>>;
  features: number[];
}

export interface DatasetEntry {
  id: string;
  group: 'regular' | 'hq' | 'inputted';
  label: string;
  url: string;
  count: number;
}

export interface DatasetManifest {
  version: string;
  datasets: DatasetEntry[];
}
