import { create } from 'zustand';
import { GridData } from '../types';
import { RecognitionCandidate, StrokeInput } from '../utils/recognizers/types';

interface SuggestionState {
  cell: { x: number; y: number } | null;
  candidates: RecognitionCandidate[];
  sourceTrail: string[];
  strokes: StrokeInput;
}

interface ToastMessage {
  id: string;
  char: string;
  confidence: number;
  engines: { name: string; char: string; score: number }[];
}

interface GameState {
  language: 'en' | 'ro';
  showGlow: boolean;
  grid: GridData | null;
  selectedCell: { x: number; y: number } | null;
  hoveredCell: { x: number; y: number } | null;
  isGestureActive: boolean;
  suggestionState: SuggestionState | null;
  toasts: ToastMessage[];
  
  setLanguage: (lang: 'en' | 'ro') => void;
  setShowGlow: (show: boolean) => void;
  setGrid: (grid: GridData) => void;
  setSelectedCell: (cell: { x: number; y: number } | null) => void;
  setHoveredCell: (cell: { x: number; y: number } | null) => void;
  setIsGestureActive: (active: boolean) => void;
  updateCellInput: (x: number, y: number, input: string) => void;
  showSuggestions: (cell: { x: number; y: number }, candidates: RecognitionCandidate[], sourceTrail: string[], strokes: StrokeInput) => void;
  clearSuggestions: () => void;
  addToast: (char: string, confidence: number, engines: { name: string; char: string; score: number }[]) => void;
  removeToast: (id: string) => void;
}

export const useGameStore = create<GameState>((set) => ({
  language: 'en',
  showGlow: true,
  grid: null,
  selectedCell: null,
  hoveredCell: null,
  isGestureActive: false,
  suggestionState: null,
  toasts: [],

  setLanguage: (language) => set({ language }),
  setShowGlow: (showGlow) => set({ showGlow }),
  setGrid: (grid) => set({ grid }),
  setSelectedCell: (selectedCell) => set({ selectedCell, suggestionState: null }),
  setHoveredCell: (hoveredCell) => set({ hoveredCell }),
  setIsGestureActive: (isGestureActive) => set({ isGestureActive }),
  updateCellInput: (x, y, input) => set((state) => {
    if (!state.grid) return state;
    const newCells = [...state.grid.cells];
    newCells[y] = [...newCells[y]];
    newCells[y][x] = { ...newCells[y][x], userInput: input.toUpperCase() };
    return { grid: { ...state.grid, cells: newCells }, suggestionState: null };
  }),
  showSuggestions: (cell, candidates, sourceTrail, strokes) =>
    set({
      suggestionState: {
        cell,
        candidates,
        sourceTrail,
        strokes,
      },
    }),
  clearSuggestions: () => set({ suggestionState: null }),
  addToast: (char, confidence, engines) => set((state) => {
    const id = Math.random().toString(36).substring(2, 9);
    return { toasts: [{ id, char, confidence, engines }, ...state.toasts].slice(0, 3) };
  }),
  removeToast: (id) => set((state) => ({
    toasts: state.toasts.filter((t) => t.id !== id),
  })),
}));
