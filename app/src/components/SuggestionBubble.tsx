import { useEffect } from 'react';
import { useGameStore } from '../store/useGameStore';
import { finalizeHandwritingSample } from '../utils/handwritingSession';

const CELL_SIZE = 40;
const GRID_GAP = 1;
const BUBBLE_TIMEOUT_MS = 5000;

export function SuggestionBubble() {
  const { suggestions } = useGameStore();

  return (
    <>
      {suggestions.map((suggestion) => (
        <SuggestionBubbleItem
          key={`${suggestion.cell.x}-${suggestion.cell.y}`}
          suggestion={suggestion}
        />
      ))}
    </>
  );
}

function SuggestionBubbleItem({ suggestion }: { suggestion: any }) {
  const { updateCellInput, removeSuggestion } = useGameStore();
  const { cell, candidates, strokes } = suggestion;

  useEffect(() => {
    const timer = setTimeout(() => {
      removeSuggestion(cell.x, cell.y);
    }, BUBBLE_TIMEOUT_MS);
    return () => clearTimeout(timer);
  }, [cell.x, cell.y, removeSuggestion]);

  const showBelow = cell.y === 0;
  const top = showBelow
    ? (cell.y * (CELL_SIZE + GRID_GAP)) + CELL_SIZE + 8
    : (cell.y * (CELL_SIZE + GRID_GAP)) - 42;
  const left = (cell.x * (CELL_SIZE + GRID_GAP)) + (CELL_SIZE / 2) + GRID_GAP;

  return (
    <div
      className="absolute z-50 pointer-events-auto"
      style={{ top, left, transform: 'translateX(-50%)' }}
    >
      <div className="rounded-full border border-slate-600 bg-slate-900/95 px-1 py-1 shadow-2xl backdrop-blur-md ring-2 ring-blue-500/30">
        <div className="flex items-center gap-1">
          {candidates.slice(0, 2).map((candidate: any) => (
            <button
              key={`${candidate.char}-${candidate.source}`}
              type="button"
              className="flex h-8 w-8 items-center justify-center rounded-full bg-slate-800 text-sm font-black text-white transition-all hover:scale-110 hover:bg-blue-600 active:scale-95"
              onClick={() => {
                updateCellInput(cell.x, cell.y, candidate.char);
                finalizeHandwritingSample({
                  x: cell.x,
                  y: cell.y,
                  label: candidate.char,
                  strokes,
                  source: 'suggestion-bubble',
                });
              }}
              title={`${candidate.char} (${Math.round(candidate.score * 100)}% via ${candidate.source})`}
            >
              {candidate.char}
            </button>
          ))}
          <button
            type="button"
            className="flex h-8 w-8 items-center justify-center rounded-full bg-slate-800 text-sm font-black text-rose-500 transition-all hover:scale-110 hover:bg-rose-600 hover:text-white active:scale-95"
            onClick={() => removeSuggestion(cell.x, cell.y)}
            title="Cancel"
          >
            ✕
          </button>
        </div>
      </div>
    </div>
  );
}
