import { GridData, WordPlacement } from '../types';

export function getWordAt(grid: GridData, x: number, y: number): WordPlacement[] {
  return grid.placements.filter(p => {
    if (p.direction === 'across') {
      return y === p.y && x >= p.x && x < p.x + p.word.length;
    } else {
      return x === p.x && y >= p.y && y < p.y + p.word.length;
    }
  });
}

export function isWordPlacementCorrect(grid: GridData, placement: WordPlacement): boolean {
  for (let i = 0; i < placement.word.length; i++) {
    const cx = placement.direction === 'across' ? placement.x + i : placement.x;
    const cy = placement.direction === 'across' ? placement.y : placement.y + i;
    if (grid.cells[cy][cx].userInput !== placement.word[i]) {
      return false;
    }
  }
  return true;
}
