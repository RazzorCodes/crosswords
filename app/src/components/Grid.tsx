import { useCallback, useEffect, useRef, useState } from 'react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { useGameStore } from '../store/useGameStore';
import { cancelPendingSubmission, finalizeHandwritingSample } from '../utils/handwritingSession';
import { getWordAt, isWordPlacementCorrect } from '../utils/validation';
import { CellComponent } from './Cell';
import { DrawingCanvas } from './DrawingCanvas';
import { HandwritingPanel } from './HandwritingPanel';
import { SuggestionBubble } from './SuggestionBubble';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function GridComponent() {
  const {
    grid,
    selectedCell,
    setSelectedCell,
    updateCellInput,
    hoveredCell,
    setIsGestureActive,
    suggestionState,
    clearSuggestions,
    showLeftPanel,
    setActiveHint,
  } = useGameStore();

  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const lastTouchRef = useRef<{ x: number; y: number } | null>(null);
  const lastDistanceRef = useRef<number | null>(null);

  const centerOnCell = useCallback((x: number, y: number) => {
    if (!grid) return;
    const cx = (grid.width * 41) / 2;
    const cy = (grid.height * 41) / 2;
    const cellX = (x * 41) + 20.5;
    const cellY = (y * 41) + 20.5;
    setOffset({ x: cx - cellX, y: cy - cellY });
  }, [grid]);

  const findNextOpenCell = useCallback((x: number, y: number, dx: number, dy: number) => {
    if (!grid) return null;

    let nx = x + dx;
    let ny = y + dy;

    while (nx >= 0 && nx < grid.width && ny >= 0 && ny < grid.height) {
      if (!grid.cells[ny][nx].isBlack) {
        return { x: nx, y: ny };
      }
      nx += dx;
      ny += dy;
    }

    return null;
  }, [grid]);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (!grid) return;
    const targetCell = suggestionState?.cell || selectedCell || hoveredCell;
    if (!targetCell) return;
    const { x, y } = targetCell;
    const key = `${x}:${y}`;

    if (e.key === 'Backspace' || e.key === 'Delete') {
      e.preventDefault();
      cancelPendingSubmission(key);
      updateCellInput(x, y, '');
      if (suggestionState) clearSuggestions();
      return;
    }

    if (e.key.length === 1 && /[a-zA-Z]/.test(e.key)) {
      e.preventDefault();
      const value = e.key.toUpperCase();

      if (suggestionState) {
        updateCellInput(x, y, value);
        finalizeHandwritingSample({
          x,
          y,
          label: value,
          strokes: suggestionState.strokes,
          source: 'keyboard-correction',
        });
        clearSuggestions();
        return;
      }

      cancelPendingSubmission(key);
      updateCellInput(x, y, value);
      return;
    }

    if (selectedCell && e.key.startsWith('Arrow')) {
      e.preventDefault();
      const nextCell =
        e.key === 'ArrowRight' ? findNextOpenCell(x, y, 1, 0)
          : e.key === 'ArrowLeft' ? findNextOpenCell(x, y, -1, 0)
            : e.key === 'ArrowDown' ? findNextOpenCell(x, y, 0, 1)
              : findNextOpenCell(x, y, 0, -1);

      if (nextCell) {
        setSelectedCell(nextCell);
      }
    }
  }, [clearSuggestions, findNextOpenCell, grid, hoveredCell, selectedCell, setSelectedCell, suggestionState, updateCellInput]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  const handleTouchMove = (e: TouchEvent) => {
    if (e.touches.length === 2) {
      setIsGestureActive(true);
      const touch1 = e.touches[0];
      const touch2 = e.touches[1];
      const centerX = (touch1.clientX + touch2.clientX) / 2;
      const centerY = (touch1.clientY + touch2.clientY) / 2;

      if (lastTouchRef.current) {
        const dx = centerX - lastTouchRef.current.x;
        const dy = centerY - lastTouchRef.current.y;
        setOffset((current) => ({ x: current.x + dx, y: current.y + dy }));
      }
      lastTouchRef.current = { x: centerX, y: centerY };

      const distance = Math.hypot(touch1.clientX - touch2.clientX, touch1.clientY - touch2.clientY);
      if (lastDistanceRef.current !== null) {
        const delta = distance / lastDistanceRef.current;
        setScale((current) => Math.min(Math.max(current * delta, 0.2), 5));
      }
      lastDistanceRef.current = distance;
      e.preventDefault();
    }
  };

  const handleTouchStart = (e: TouchEvent) => {
    if (e.touches.length >= 2) {
      setIsGestureActive(true);
      const touch1 = e.touches[0];
      const touch2 = e.touches[1];
      const centerX = (touch1.clientX + touch2.clientX) / 2;
      const centerY = (touch1.clientY + touch2.clientY) / 2;
      lastTouchRef.current = { x: centerX, y: centerY };
      lastDistanceRef.current = Math.hypot(touch1.clientX - touch2.clientX, touch1.clientY - touch2.clientY);
    } else {
      setIsGestureActive(false);
    }
  };

  const handleTouchEnd = (e: TouchEvent) => {
    if (e.touches.length < 2) {
      lastTouchRef.current = null;
      lastDistanceRef.current = null;
      setTimeout(() => setIsGestureActive(false), 150);
    }
  };

  if (!grid) return null;

  const selectedClues = selectedCell
    ? getWordAt(grid, selectedCell.x, selectedCell.y).filter((placement) => !isWordPlacementCorrect(grid, placement))
    : [];

  return (
    <div
      className="relative flex h-full w-full flex-1 touch-none flex-col items-center justify-center overflow-hidden bg-slate-950"
      onWheel={(e) => {
        if (e.ctrlKey) {
          setScale((current) => Math.min(Math.max(current - (e.deltaY * 0.01), 0.2), 5));
        } else {
          setOffset((current) => ({ x: current.x - e.deltaX, y: current.y - e.deltaY }));
        }
      }}
      onTouchStart={(e) => handleTouchStart(e.nativeEvent)}
      onTouchMove={(e) => handleTouchMove(e.nativeEvent)}
      onTouchEnd={(e) => handleTouchEnd(e.nativeEvent)}
    >
      <div
        className="relative transition-transform duration-75 ease-out"
        style={{
          transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
          transformOrigin: 'center',
        }}
      >
        <div
          className="grid gap-px border border-slate-700 bg-slate-700 p-px shadow-2xl"
          style={{
            gridTemplateColumns: `repeat(${grid.width}, minmax(0, 1fr))`,
            width: 'fit-content',
          }}
        >
          {grid.cells.map((row, y) => row.map((_, x) => (
            <CellComponent key={`${x}-${y}`} x={x} y={y} />
          )))}
        </div>
        <SuggestionBubble />
        <DrawingCanvas />
      </div>

      <HandwritingPanel />

      {selectedClues.length > 0 && (
        <div className="absolute bottom-12 left-1/2 z-40 w-[90%] max-w-[600px] -translate-x-1/2 animate-in slide-in-from-bottom-4 duration-300 md:min-w-[400px] md:w-auto">
          <div className="flex flex-col gap-2 rounded-2xl border border-slate-700/50 bg-slate-900/90 p-4 shadow-2xl backdrop-blur-xl">
            {selectedClues.map((placement, index) => (
              <div
                key={`${placement.direction}-${placement.number}`}
                className={cn(
                  'group relative flex items-start gap-3',
                  index > 0 && 'border-t border-slate-800 pt-2',
                )}
              >
                <div
                  className={cn(
                    'mt-0.5 shrink-0 rounded px-2 py-0.5 text-[10px] font-black uppercase tracking-widest',
                    placement.direction === 'across' ? 'bg-blue-500/20 text-blue-400' : 'bg-emerald-500/20 text-emerald-400',
                  )}
                >
                  {placement.number} {placement.direction}
                </div>
                <div className="pr-8 text-sm font-medium leading-tight text-slate-100">
                  {placement.clue}
                </div>
                <button
                  className="absolute right-0 top-1/2 flex h-8 w-8 -translate-y-1/2 items-center justify-center rounded-full border border-slate-700 bg-slate-800 text-xs font-black text-slate-400 transition-all hover:bg-slate-700 hover:text-white active:scale-95 touch-none select-none"
                  onPointerDown={() => setActiveHint(placement)}
                  onPointerUp={() => setActiveHint(null)}
                  onPointerLeave={() => setActiveHint(null)}
                  onContextMenu={(e) => e.preventDefault()}
                >
                  ?
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {showLeftPanel && (
        <div className="pointer-events-auto absolute bottom-0 left-0 top-0 z-30 hidden w-64 overflow-y-auto border-r border-slate-800 bg-slate-900/90 p-4 backdrop-blur-md animate-in slide-in-from-left duration-300 md:block">
          <div className="flex flex-col gap-6 text-left">
            <div>
              <h3 className="mb-3 border-b border-blue-400/20 pb-1 text-sm font-black uppercase tracking-tighter text-blue-400">Across</h3>
              <ul className="space-y-3">
                {grid.placements.filter((placement) => placement.direction === 'across').map((placement) => {
                  const isSolved = isWordPlacementCorrect(grid, placement);
                  return (
                    <li
                      key={`across-${placement.number}`}
                      className={cn('group cursor-pointer text-xs leading-relaxed transition-all', isSolved && 'opacity-40 grayscale')}
                      onClick={() => { setSelectedCell({ x: placement.x, y: placement.y }); centerOnCell(placement.x, placement.y); }}
                    >
                      <span
                        className={cn(
                          'mr-2 rounded px-1 font-bold transition-colors',
                          isSolved
                            ? 'line-through'
                            : selectedCell?.x === placement.x && selectedCell?.y === placement.y
                              ? 'bg-blue-600 text-white'
                              : 'text-blue-400 group-hover:bg-slate-800',
                        )}
                      >
                        {placement.number}
                      </span>
                      <span className={cn('text-slate-300 group-hover:text-white', isSolved && 'line-through')}>
                        {placement.clue}
                      </span>
                    </li>
                  );
                })}
              </ul>
            </div>

            <div>
              <h3 className="mb-3 border-b border-emerald-400/20 pb-1 text-sm font-black uppercase tracking-tighter text-emerald-400">Down</h3>
              <ul className="space-y-3">
                {grid.placements.filter((placement) => placement.direction === 'down').map((placement) => {
                  const isSolved = isWordPlacementCorrect(grid, placement);
                  return (
                    <li
                      key={`down-${placement.number}`}
                      className={cn('group cursor-pointer text-xs leading-relaxed transition-all', isSolved && 'opacity-40 grayscale')}
                      onClick={() => { setSelectedCell({ x: placement.x, y: placement.y }); centerOnCell(placement.x, placement.y); }}
                    >
                      <span
                        className={cn(
                          'mr-2 rounded px-1 font-bold transition-colors',
                          isSolved
                            ? 'line-through'
                            : selectedCell?.x === placement.x && selectedCell?.y === placement.y
                              ? 'bg-emerald-600 text-white'
                              : 'text-emerald-400 group-hover:bg-slate-800',
                        )}
                      >
                        {placement.number}
                      </span>
                      <span className={cn('text-slate-300 group-hover:text-white', isSolved && 'line-through')}>
                        {placement.clue}
                      </span>
                    </li>
                  );
                })}
              </ul>
            </div>
          </div>
        </div>
      )}

      <div className="pointer-events-auto absolute bottom-0 left-0 right-0 z-30 max-h-48 overflow-y-auto border-t border-slate-800 bg-slate-900/90 p-4 backdrop-blur-md md:hidden">
        <div className="grid grid-cols-2 gap-4 text-left">
          <div>
            <h3 className="mb-2 text-[10px] font-black uppercase text-blue-400">Across</h3>
            {grid.placements.filter((placement) => placement.direction === 'across').map((placement) => {
              const isSolved = isWordPlacementCorrect(grid, placement);
              return (
                <div
                  key={`m-across-${placement.number}`}
                  className={cn('mb-1 text-[10px] transition-opacity', isSolved ? 'text-slate-500 line-through opacity-40' : 'text-slate-300')}
                  onClick={() => { setSelectedCell({ x: placement.x, y: placement.y }); centerOnCell(placement.x, placement.y); }}
                >
                  <span className="mr-1 font-bold">{placement.number}.</span>
                  {placement.clue}
                </div>
              );
            })}
          </div>
          <div>
            <h3 className="mb-2 text-[10px] font-black uppercase text-emerald-400">Down</h3>
            {grid.placements.filter((placement) => placement.direction === 'down').map((placement) => {
              const isSolved = isWordPlacementCorrect(grid, placement);
              return (
                <div
                  key={`m-down-${placement.number}`}
                  className={cn('mb-1 text-[10px] transition-opacity', isSolved ? 'text-slate-500 line-through opacity-40' : 'text-slate-300')}
                  onClick={() => { setSelectedCell({ x: placement.x, y: placement.y }); centerOnCell(placement.x, placement.y); }}
                >
                  <span className="mr-1 font-bold">{placement.number}.</span>
                  {placement.clue}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
