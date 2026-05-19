import { useEffect, useRef } from 'react';
import { useGameStore } from '../store/useGameStore';
import { submitStrokeData } from '../utils/api';
import { knnRecognizer } from '../utils/recognizers/knn';

const CELL_SIZE = 40;
const GRID_GAP = 1;

export function SuggestionBubble() {
  const { suggestionState, updateCellInput, clearSuggestions } = useGameStore();
  const bubbleRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!suggestionState) return;

    const handlePointerDown = (event: PointerEvent) => {
      if (!bubbleRef.current?.contains(event.target as Node)) {
        clearSuggestions();
      }
    };

    window.addEventListener('pointerdown', handlePointerDown);
    return () => window.removeEventListener('pointerdown', handlePointerDown);
  }, [suggestionState, clearSuggestions]);

  if (!suggestionState?.cell || suggestionState.candidates.length === 0) {
    return null;
  }

  const { cell, candidates, strokes } = suggestionState;
  const showBelow = cell.y === 0;
  
  // Account for gaps and padding in the grid layout
  const top = showBelow 
    ? (cell.y * (CELL_SIZE + GRID_GAP)) + CELL_SIZE + 8 
    : (cell.y * (CELL_SIZE + GRID_GAP)) - 42;
    
  const left = (cell.x * (CELL_SIZE + GRID_GAP)) + (CELL_SIZE / 2) + GRID_GAP;

  return (
    <div
      ref={bubbleRef}
      className="absolute z-50 pointer-events-auto"
      style={{
        top,
        left,
        transform: 'translateX(-50%)',
      }}
    >
      <div className="rounded-full border border-slate-600 bg-slate-900/95 px-1 py-1 shadow-2xl backdrop-blur-md ring-2 ring-blue-500/30">
        <div className="flex items-center gap-1">
          {candidates.slice(0, 3).map((candidate) => (
            <button
              key={`${candidate.char}-${candidate.source}`}
              type="button"
              className="h-8 w-8 rounded-full bg-slate-800 text-sm font-black text-white transition-all hover:bg-blue-600 hover:scale-110 active:scale-95 flex items-center justify-center"
              onClick={() => {
                if (candidate.char !== '?') {
                  // Phase 3: Ground Truth Capture - Tier 1 or 2
                  knnRecognizer.addExample(strokes, candidate.char);
                  void submitStrokeData(candidate.char, strokes);
                }
                updateCellInput(cell.x, cell.y, candidate.char);
                clearSuggestions();
              }}
              title={`${candidate.char} (confidence: ${Math.round(candidate.score * 100)}% via ${candidate.source})`}
            >
              {candidate.char}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
