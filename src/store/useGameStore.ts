import { create } from 'zustand';
import { GridData } from './types';

interface GameState {
  language: 'en' | 'ro';
  showGlow: boolean;
  grid: GridData | null;
  selectedCell: { x: number; y: number } | null;
  hoveredCell: { x: number; y: number } | null;
  isGestureActive: boolean;
  
  setLanguage: (lang: 'en' | 'ro') => void;
  setShowGlow: (show: boolean) => void;
  setGrid: (grid: GridData) => void;
  setSelectedCell: (cell: { x: number; y: number } | null) => void;
  setHoveredCell: (cell: { x: number; y: number } | null) => void;
  setIsGestureActive: (active: boolean) => void;
  updateCellInput: (x: number, y: number, input: string) => void;
}

export const useGameStore = create<GameState>((set) => ({
  language: 'en',
  showGlow: true,
  grid: null,
  selectedCell: null,
  hoveredCell: null,
  isGestureActive: false,

  setLanguage: (language) => set({ language }),
  setShowGlow: (showGlow) => set({ showGlow }),
  setGrid: (grid) => set({ grid }),
  setSelectedCell: (selectedCell) => set({ selectedCell }),
  setHoveredCell: (hoveredCell) => set({ hoveredCell }),
  setIsGestureActive: (isGestureActive) => set({ isGestureActive }),
  updateCellInput: (x, y, input) => set((state) => {
    if (!state.grid) return state;
    const newCells = [...state.grid.cells];
    newCells[y] = [...newCells[y]];
    newCells[y][x] = { ...newCells[y][x], userInput: input.toUpperCase() };
    return { grid: { ...state.grid, cells: newCells } };
  }),
}));
