import { Cell, GridData, WordPlacement } from '../types';

interface RawWord {
  word: string;
  clue: string;
}

export function generateCrossword(words: RawWord[], size: number = 15): GridData {
  const grid: string[][] = Array(size).fill(null).map(() => Array(size).fill(''));
  const placements: WordPlacement[] = [];
  
  // Sort words by length descending to place longer words first
  const sortedWords = [...words].sort((a, b) => b.word.length - a.word.length);
  
  // Helper to check if a word can be placed
  const canPlace = (word: string, x: number, y: number, direction: 'across' | 'down') => {
    // Bounds check
    if (x < 0 || y < 0 || x >= size || y >= size) return false;
    
    if (direction === 'across') {
      if (x + word.length > size) return false;
      if (x > 0 && grid[y][x - 1] !== '') return false; // Check before
      if (x + word.length < size && grid[y][x + word.length] !== '') return false; // Check after
      
      for (let i = 0; i < word.length; i++) {
        const current = grid[y][x + i];
        if (current !== '' && current !== word[i]) return false;
        
        // Check adjacent cells (except for intersections)
        if (current === '') {
          if (y > 0 && grid[y - 1][x + i] !== '') return false;
          if (y < size - 1 && grid[y + 1][x + i] !== '') return false;
        }
      }
    } else {
      if (y + word.length > size) return false;
      if (y > 0 && grid[y - 1][x] !== '') return false;
      if (y + word.length < size && grid[y + word.length][x] !== '') return false;
      
      for (let i = 0; i < word.length; i++) {
        const current = grid[y + i][x];
        if (current !== '' && current !== word[i]) return false;
        
        if (current === '') {
          if (x > 0 && grid[y + i][x - 1] !== '') return false;
          if (x < size - 1 && grid[y + i][x + 1] !== '') return false;
        }
      }
    }
    return true;
  };

  const placeWord = (word: RawWord, x: number, y: number, direction: 'across' | 'down', number: number) => {
    for (let i = 0; i < word.word.length; i++) {
      if (direction === 'across') {
        grid[y][x + i] = word.word[i];
      } else {
        grid[y + i][x] = word.word[i];
      }
    }
    placements.push({
      word: word.word,
      clue: word.clue,
      x,
      y,
      direction,
      number
    });
  };

  // 1. Place the first word in the middle
  const first = sortedWords[0];
  if (first) {
    placeWord(first, Math.floor((size - first.word.length) / 2), Math.floor(size / 2), 'across', 1);
  }
  
  let clueNumber = 2;
  const remaining = sortedWords.slice(1);
  
  // 2. Try to place other words by intersecting
  for (const item of remaining) {
    let placed = false;
    for (const placement of placements) {
      for (let i = 0; i < placement.word.length; i++) {
        for (let j = 0; j < item.word.length; j++) {
          if (placement.word[i] === item.word[j]) {
            const direction = placement.direction === 'across' ? 'down' : 'across';
            const x = direction === 'across' ? placement.x - j : placement.x + i;
            const y = direction === 'across' ? placement.y + i : placement.y - j;
            
            if (canPlace(item.word, x, y, direction)) {
              placeWord(item, x, y, direction, clueNumber++);
              placed = true;
              break;
            }
          }
        }
        if (placed) break;
      }
      if (placed) break;
    }
  }

  // 3. Convert grid to final Cell format
  const cells: Cell[][] = Array(size).fill(null).map((_, y) => 
    Array(size).fill(null).map((_, x) => ({
      x,
      y,
      char: grid[y][x],
      userInput: '',
      isBlack: grid[y][x] === ''
    }))
  );

  // Add clue numbers to cells
  for (const p of placements) {
    cells[p.y][p.x].number = p.number;
  }

  return { cells, placements, width: size, height: size };
}
