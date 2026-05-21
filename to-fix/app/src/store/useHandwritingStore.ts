import { create } from 'zustand';
import type { StrokeInput } from '../utils/recognizers/types';
import type { EngineResult, RecognitionCandidate } from '../utils/recognizers/types';
import type { HandwritingModuleEvent, TrainingState } from '../utils/handwriting';
import { resolveAppMode } from '../utils/runtimeConfig';

const TRAIN_MODE = import.meta.env.VITE_TRAIN_MODE === 'true' || resolveAppMode() === 'dev';
const DIAGNOSTIC_LOG_LIMIT = 80;

export interface PendingInkSubmission {
  cellKey: string;
  label: string;
  source: string;
  expiresAt: number;
  strokes: StrokeInput;
}

export interface DiagnosticLogEntry {
  id: string;
  createdAt: number;
  type: HandwritingModuleEvent['type'] | 'dev-board-cleared' | 'feedback-cleared';
  summary: string;
}

export interface DiagnosticPrediction {
  createdAt: number;
  topLabel: string | null;
  confidence: number;
  candidates: RecognitionCandidate[];
  engines: EngineResult[];
}

interface HandwritingState {
  trainMode: boolean;
  pendingInk: PendingInkSubmission[];
  trainingState: TrainingState | null;
  moduleReady: boolean;
  lastEvent: string | null;
  diagnosticLogs: DiagnosticLogEntry[];
  lastPrediction: DiagnosticPrediction | null;
  predictionHistory: DiagnosticPrediction[];
  setModuleReady: (ready: boolean) => void;
  setTrainingState: (state: TrainingState | null) => void;
  setLastEvent: (event: string | null) => void;
  setLastPrediction: (prediction: DiagnosticPrediction | null) => void;
  addDiagnosticLog: (entry: Omit<DiagnosticLogEntry, 'id' | 'createdAt'> & { createdAt?: number }) => void;
  upsertPendingInk: (item: PendingInkSubmission) => void;
  clearPendingInk: (cellKey: string) => void;
  clearAllPendingInk: () => void;
  clearPersonalizedModels: () => Promise<void>;
}

export const useHandwritingStore = create<HandwritingState>((set) => ({
  trainMode: TRAIN_MODE,
  pendingInk: [],
  trainingState: null,
  moduleReady: false,
  lastEvent: null,
  diagnosticLogs: [],
  lastPrediction: null,
  predictionHistory: [],
  setModuleReady: (moduleReady) => set({ moduleReady }),
  setTrainingState: (trainingState) => set({ trainingState }),
  setLastEvent: (lastEvent) => set({ lastEvent }),
  setLastPrediction: (lastPrediction) => set((state) => ({
    lastPrediction,
    predictionHistory: lastPrediction
      ? [lastPrediction, ...state.predictionHistory].slice(0, 25)
      : state.predictionHistory,
  })),
  addDiagnosticLog: (entry) => set((state) => ({
    diagnosticLogs: [
      {
        id: `log-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        createdAt: entry.createdAt ?? Date.now(),
        type: entry.type,
        summary: entry.summary,
      },
      ...state.diagnosticLogs,
    ].slice(0, DIAGNOSTIC_LOG_LIMIT),
  })),
  upsertPendingInk: (item) => set((state) => ({
    pendingInk: [
      item,
      ...state.pendingInk.filter((entry) => entry.cellKey !== item.cellKey),
    ],
  })),
  clearPendingInk: (cellKey) => set((state) => ({
    pendingInk: state.pendingInk.filter((entry) => entry.cellKey !== cellKey),
  })),
  clearAllPendingInk: () => set({ pendingInk: [] }),
  clearPersonalizedModels: async () => {
    const { handwritingModule } = await import('../utils/handwriting/module');
    await handwritingModule.clearPersonalizedModels();
  },
}));
