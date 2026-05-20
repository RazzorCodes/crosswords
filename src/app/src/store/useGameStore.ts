import { create } from 'zustand';
import { GridData, WordPlacement } from '../types';
import { EngineResult, RecognitionCandidate, StrokeInput } from '../utils/recognizers/types';
import { getWordAt, isWordPlacementCorrect } from '../utils/validation';
import type { HandwritingModuleEventMap } from '../utils/handwriting/types';

interface SuggestionState {
  cell: { x: number; y: number };
  candidates: RecognitionCandidate[];
  sourceTrail: string[];
  strokes: StrokeInput;
}

interface ToastMessage {
  id: string;
  char: string;
  confidence: number;
  engines: EngineResult[];
}

export interface TrainingToastMessage {
  id: string;
  generation: number;
  progress: number;
  feature: string;
  cnn: string;
  finalizing: string | null;
  status: 'running' | 'ready' | 'skipped' | 'rejected';
}

interface GameState {
  language: 'en' | 'ro';
  showGlow: boolean;
  grid: GridData | null;
  selectedCell: { x: number; y: number } | null;
  hoveredCell: { x: number; y: number } | null;
  isGestureActive: boolean;
  suggestions: SuggestionState[];
  toasts: ToastMessage[];
  trainingToast: TrainingToastMessage | null;
  mostNeededLetter: string | null;
  showLeftPanel: boolean;
  activeHint: WordPlacement | null;
  gameId: number;
  startTime: number | null;
  endTime: number | null;
  
  setLanguage: (lang: 'en' | 'ro') => void;
  setShowGlow: (show: boolean) => void;
  setGrid: (grid: GridData) => void;
  setSelectedCell: (cell: { x: number; y: number } | null) => void;
  setHoveredCell: (cell: { x: number; y: number } | null) => void;
  setIsGestureActive: (active: boolean) => void;
  updateCellInput: (x: number, y: number, input: string) => void;
  addSuggestion: (cell: { x: number; y: number }, candidates: RecognitionCandidate[], sourceTrail: string[], strokes: StrokeInput) => void;
  removeSuggestion: (x: number, y: number) => void;
  clearAllSuggestions: () => void;
  addToast: (char: string, confidence: number, engines: EngineResult[]) => void;
  removeToast: (id: string) => void;
  updateTrainingToast: (progress: HandwritingModuleEventMap['training-progress']) => void;
  removeTrainingToast: () => void;
  setMostNeededLetter: (letter: string | null) => void;
  setShowLeftPanel: (show: boolean) => void;
  setActiveHint: (hint: WordPlacement | null) => void;
  setGameId: (id: number) => void;
  setStartTime: (time: number | null) => void;
  setEndTime: (time: number | null) => void;
}

export const useGameStore = create<GameState>((set) => ({
  language: 'en',
  showGlow: true,
  grid: null,
  selectedCell: null,
  hoveredCell: null,
  isGestureActive: false,
  suggestions: [],
  toasts: [],
  trainingToast: null,
  mostNeededLetter: null,
  showLeftPanel: true,
  activeHint: null,
  gameId: 1,
  startTime: null,
  endTime: null,

  setLanguage: (language) => set({ language }),
  setShowGlow: (showGlow) => set({ showGlow }),
  setGrid: (grid) => set({ grid }),
  setSelectedCell: (selectedCell) => set({ selectedCell }),
  setHoveredCell: (hoveredCell) => set({ hoveredCell }),
  setIsGestureActive: (isGestureActive) => set({ isGestureActive }),
  updateCellInput: (x, y, input) => set((state) => {
    if (!state.grid) return state;
    if (y < 0 || y >= state.grid.height || x < 0 || x >= state.grid.width) return state;

    const currentCell = state.grid.cells[y][x];
    if (!currentCell || currentCell.isBlack) return state;
    
    // Check if cell is locked (part of a fully correct word)
    const currentWords = getWordAt(state.grid, x, y);
    const isLocked = currentWords.some(p => isWordPlacementCorrect(state.grid!, p));
    if (isLocked) return state;

    const newCells = [...state.grid.cells];
    newCells[y] = [...newCells[y]];
    const normalizedInput = input.slice(0, 1).toUpperCase();
    newCells[y][x] = { ...newCells[y][x], userInput: normalizedInput };
    
    const newGrid = { ...state.grid, cells: newCells };
    
    // Check for victory
    let newEndTime = state.endTime;
    const allSolved =
      newGrid.placements.length > 0 &&
      newGrid.placements.every(p => isWordPlacementCorrect(newGrid, p));
    if (allSolved && !state.endTime) {
      newEndTime = Date.now();
    }

    return { 
      grid: newGrid, 
      suggestions: state.suggestions.filter(s => s.cell.x !== x || s.cell.y !== y),
      endTime: newEndTime 
    };
  }),
  addSuggestion: (cell, candidates, sourceTrail, strokes) =>
    set((state) => ({
      suggestions: [
        ...state.suggestions.filter(s => s.cell.x !== cell.x || s.cell.y !== cell.y),
        { cell, candidates, sourceTrail, strokes },
      ],
    })),
  removeSuggestion: (x, y) =>
    set((state) => ({
      suggestions: state.suggestions.filter(s => s.cell.x !== x || s.cell.y !== y),
    })),
  clearAllSuggestions: () => set({ suggestions: [] }),
  addToast: (char, confidence, engines) => set((state) => {
    const id = Math.random().toString(36).substring(2, 9);
    return { toasts: [{ id, char, confidence, engines }, ...state.toasts].slice(0, 3) };
  }),
  removeToast: (id) => set((state) => ({
    toasts: state.toasts.filter((t) => t.id !== id),
  })),
  updateTrainingToast: (progress) => set((state) => {
    const current = state.trainingToast?.generation === progress.generation ? state.trainingToast : {
      id: 'training-progress',
      generation: progress.generation,
      progress: 0,
      feature: 'svm/feature waiting',
      cnn: 'cnn waiting',
      finalizing: null,
      status: 'running' as const,
    };
    const next: TrainingToastMessage = {
      ...current,
      generation: progress.generation,
      progress: Math.max(current.progress, progress.progress),
      status: progress.status,
    };
    if (progress.phase === 'feature') {
      next.feature = progress.message;
    } else if (progress.phase === 'cnn') {
      next.cnn = progress.message;
    } else {
      next.finalizing = progress.message;
    }
    return { trainingToast: next };
  }),
  removeTrainingToast: () => set({ trainingToast: null }),
  setMostNeededLetter: (mostNeededLetter) => set({ mostNeededLetter }),
  setShowLeftPanel: (showLeftPanel) => set({ showLeftPanel }),
  setActiveHint: (activeHint) => set({ activeHint }),
  setGameId: (gameId) => set({ gameId }),
  setStartTime: (startTime) => set({ startTime }),
  setEndTime: (endTime) => set({ endTime }),
}));
