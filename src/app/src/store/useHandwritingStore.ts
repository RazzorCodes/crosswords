import { create } from 'zustand';
import type { StrokeInput } from '../utils/recognizers/types';
import type { TrainingState } from '../utils/handwriting';

const TRAIN_MODE = import.meta.env.VITE_TRAIN_MODE === 'true';

export interface PendingInkSubmission {
  cellKey: string;
  label: string;
  source: string;
  expiresAt: number;
  strokes: StrokeInput;
}

interface HandwritingState {
  trainMode: boolean;
  pendingInk: PendingInkSubmission[];
  trainingState: TrainingState | null;
  moduleReady: boolean;
  lastEvent: string | null;
  setModuleReady: (ready: boolean) => void;
  setTrainingState: (state: TrainingState | null) => void;
  setLastEvent: (event: string | null) => void;
  upsertPendingInk: (item: PendingInkSubmission) => void;
  clearPendingInk: (cellKey: string) => void;
  clearAllPendingInk: () => void;
}

export const useHandwritingStore = create<HandwritingState>((set) => ({
  trainMode: TRAIN_MODE,
  pendingInk: [],
  trainingState: null,
  moduleReady: false,
  lastEvent: null,
  setModuleReady: (moduleReady) => set({ moduleReady }),
  setTrainingState: (trainingState) => set({ trainingState }),
  setLastEvent: (lastEvent) => set({ lastEvent }),
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
}));
