import { Cell, GridData, WordEntry, WordPlacement } from '../types';

export function generateCrossword(words: WordEntry[], size: number = 15, priorityLetters: string[] = []): GridData {
  const grid: string[][] = Array(size).fill(null).map(() => Array(size).fill(''));
  const placements: WordPlacement[] = [];
  
  // Create a priority set for fast lookup
  const prioritySet = new Set(priorityLetters.map(l => l.toUpperCase()));
  
  // Sort words by priority score, then by length
  const sortedWords = [...words].map(w => {
    let priorityScore = 0;
    const uniqueChars = new Set(w.word.toUpperCase().split(''));
    uniqueChars.forEach(char => {
      if (prioritySet.has(char)) {
        priorityScore += 1000;
      }
    });
    return { ...w, score: priorityScore + w.word.length };
  }).sort((a, b) => b.score - a.score);
  
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

  const placeWord = (word: WordEntry, x: number, y: number, direction: 'across' | 'down') => {
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
      number: 0 // placeholder
    });
  };

  // 1. Place the first word in the middle
  const first = sortedWords[0];
  if (first) {
    placeWord(first, Math.floor((size - first.word.length) / 2), Math.floor(size / 2), 'across');
  }
  
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
              placeWord(item, x, y, direction);
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

  // 4. Standard Crossword Numbering
  let currentNumber = 1;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (!cells[y][x].isBlack) {
        let needsNumber = false;
        
        // Start of across word
        const startsAcross = (x === 0 || cells[y][x - 1].isBlack) && (x + 1 < size && !cells[y][x + 1].isBlack);
        if (startsAcross) {
          const placement = placements.find(p => p.x === x && p.y === y && p.direction === 'across');
          if (placement) {
            placement.number = currentNumber;
            needsNumber = true;
          }
        }
        
        // Start of down word
        const startsDown = (y === 0 || cells[y - 1][x].isBlack) && (y + 1 < size && !cells[y + 1][x].isBlack);
        if (startsDown) {
          const placement = placements.find(p => p.x === x && p.y === y && p.direction === 'down');
          if (placement) {
            placement.number = currentNumber;
            needsNumber = true;
          }
        }
        
        if (needsNumber) {
          cells[y][x].number = currentNumber;
          currentNumber++;
        }
      }
    }
  }

  return { cells, placements, width: size, height: size };
}
