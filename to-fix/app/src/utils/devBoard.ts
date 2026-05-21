import type { Cell, GridData } from '../types';

const DEV_WORD = 'FEEDBACK';
const DEV_WIDTH = 10;
const DEV_HEIGHT = 3;
const DEV_X = 1;
const DEV_Y = 1;

export function createFeedbackDevGrid(): GridData {
  const cells: Cell[][] = Array.from({ length: DEV_HEIGHT }, (_, y) =>
    Array.from({ length: DEV_WIDTH }, (_, x) => ({
      x,
      y,
      char: '',
      userInput: '',
      isBlack: true,
    })),
  );

  for (let index = 0; index < DEV_WORD.length; index += 1) {
    const x = DEV_X + index;
    cells[DEV_Y][x] = {
      x,
      y: DEV_Y,
      char: DEV_WORD[index],
      userInput: '',
      isBlack: false,
      number: index === 0 ? 1 : undefined,
    };
  }

  return {
    cells,
    width: DEV_WIDTH,
    height: DEV_HEIGHT,
    placements: [
      {
        word: DEV_WORD,
        clue: 'Development handwriting feedback loop.',
        x: DEV_X,
        y: DEV_Y,
        direction: 'across',
        number: 1,
      },
    ],
  };
}
