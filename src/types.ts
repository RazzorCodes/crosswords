export interface Cell {
  x: number;
  y: number;
  char: string; // The correct character
  userInput: string; // What the user has entered/drawn
  isBlack: boolean;
  number?: number; // Clue number
}

export interface WordPlacement {
  word: string;
  clue: string;
  x: number;
  y: number;
  direction: 'across' | 'down';
  number: number;
}

export interface WordEntry {
  word: string;
  clue: string;
}

export interface GridData {
  cells: Cell[][];
  placements: WordPlacement[];
  width: number;
  height: number;
}
