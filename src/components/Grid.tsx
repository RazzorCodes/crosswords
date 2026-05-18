import { useState, useRef, useEffect, useCallback } from 'react';
import { useGameStore } from '../store/useGameStore';
import { CellComponent } from './Cell';
import { DrawingCanvas } from './DrawingCanvas';
import { SuggestionBubble } from './SuggestionBubble';
import { submitStrokeData } from '../utils/api';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function GridComponent() {
  const { grid, selectedCell, setSelectedCell, updateCellInput, hoveredCell, setIsGestureActive, suggestionState, clearSuggestions } = useGameStore();
  
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const lastTouchRef = useRef<{ x: number, y: number } | null>(null);
  const lastDistanceRef = useRef<number | null>(null);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (!grid) return;
    const targetCell = suggestionState?.cell || selectedCell || hoveredCell;
    if (!targetCell) return;
    const { x, y } = targetCell;

    if (e.key === 'Backspace' || e.key === 'Delete') {
      updateCellInput(x, y, '');
      if (suggestionState) clearSuggestions();
    } else if (e.key.length === 1 && /[a-zA-Z]/.test(e.key)) {
      if (suggestionState) {
        void submitStrokeData(e.key.toUpperCase(), suggestionState.strokes);
        updateCellInput(x, y, e.key);
        clearSuggestions();
      } else {
        updateCellInput(x, y, e.key);
      }
    } else if (selectedCell && e.key.startsWith('Arrow')) {
      let nx = x, ny = y;
      if (e.key === 'ArrowRight') nx++;
      if (e.key === 'ArrowLeft') nx--;
      if (e.key === 'ArrowDown') ny++;
      if (e.key === 'ArrowUp') ny--;
      if (nx >= 0 && nx < grid.width && ny >= 0 && ny < grid.height) {
        setSelectedCell({ x: nx, y: ny });
      }
    }
  }, [grid, selectedCell, hoveredCell, updateCellInput, setSelectedCell, suggestionState, clearSuggestions]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  const handleTouchMove = (e: TouchEvent) => {
    if (e.touches.length === 2) {
      // TWO FINGER: ZOOM & PAN
      setIsGestureActive(true);
      const touch1 = e.touches[0];
      const touch2 = e.touches[1];
      
      const centerX = (touch1.clientX + touch2.clientX) / 2;
      const centerY = (touch1.clientY + touch2.clientY) / 2;
      
      if (lastTouchRef.current) {
        const dx = centerX - lastTouchRef.current.x;
        const dy = centerY - lastTouchRef.current.y;
        setOffset(o => ({ x: o.x + dx, y: o.y + dy }));
      }
      lastTouchRef.current = { x: centerX, y: centerY };

      const dist = Math.hypot(touch1.clientX - touch2.clientX, touch1.clientY - touch2.clientY);
      if (lastDistanceRef.current !== null) {
        const delta = dist / lastDistanceRef.current;
        setScale(s => Math.min(Math.max(s * delta, 0.2), 5));
      }
      lastDistanceRef.current = dist;
      
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
      // Use a timeout to keep gesture active briefly, preventing the transition from 2 fingers 
      // back to 1 finger from triggering a "draw" stroke.
      setTimeout(() => setIsGestureActive(false), 150);
    }
  };

  if (!grid) return null;

  return (
    <div 
      className="flex-1 w-full h-full overflow-hidden relative flex flex-col items-center justify-center bg-slate-950 touch-none"
      onWheel={(e) => {
        if (e.ctrlKey) {
          setScale(s => Math.min(Math.max(s - e.deltaY * 0.01, 0.2), 5));
        } else {
          setOffset(o => ({ x: o.x - e.deltaX, y: o.y - e.deltaY }));
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
          transformOrigin: 'center'
        }}
      >
        <div 
          className="grid gap-px bg-slate-700 border border-slate-700 p-px shadow-2xl"
          style={{ 
            gridTemplateColumns: `repeat(${grid.width}, minmax(0, 1fr))`,
            width: 'fit-content'
          }}
        >
          {grid.cells.map((row, y) => 
            row.map((_, x) => (
              <CellComponent key={`${x}-${y}`} x={x} y={y} />
            ))
          )}
        </div>
        <SuggestionBubble />
        <DrawingCanvas />
      </div>
      
      {/* Clues overlay panel (Desktop/Tablet) */}
      <div className="absolute top-0 bottom-0 left-0 w-64 bg-slate-900/90 backdrop-blur-md border-r border-slate-800 p-4 overflow-y-auto z-30 pointer-events-auto hidden md:block">
        <div className="flex flex-col gap-6 text-left">
          <div>
            <h3 className="text-sm font-black mb-3 text-blue-400 uppercase tracking-tighter border-b border-blue-400/20 pb-1">Across</h3>
            <ul className="space-y-3">
              {grid.placements.filter(p => p.direction === 'across').map(p => (
                <li key={`across-${p.number}`} className="text-xs leading-relaxed group cursor-pointer" onClick={() => setSelectedCell({x: p.x, y: p.y})}>
                  <span className={cn(
                    "font-bold mr-2 px-1 rounded transition-colors",
                    selectedCell?.x === p.x && selectedCell?.y === p.y ? "bg-blue-600 text-white" : "text-blue-400 group-hover:bg-slate-800"
                  )}>{p.number}</span>
                  <span className="text-slate-300 group-hover:text-white">{p.clue}</span>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-black mb-3 text-emerald-400 uppercase tracking-tighter border-b border-emerald-400/20 pb-1">Down</h3>
            <ul className="space-y-3">
              {grid.placements.filter((p: any) => p.direction === 'down').map((p: any) => (
                <li key={`down-${p.number}`} className="text-xs leading-relaxed group cursor-pointer" onClick={() => setSelectedCell({x: p.x, y: p.y})}>
                  <span className={cn(
                    "font-bold mr-2 px-1 rounded transition-colors",
                    selectedCell?.x === p.x && selectedCell?.y === p.y ? "bg-emerald-600 text-white" : "text-emerald-400 group-hover:bg-slate-800"
                  )}>{p.number}</span>
                  <span className="text-slate-300 group-hover:text-white">{p.clue}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Mobile clues panel */}
      <div className="absolute bottom-0 left-0 right-0 bg-slate-900/90 backdrop-blur-md border-t border-slate-800 p-4 max-h-48 overflow-y-auto z-30 pointer-events-auto md:hidden">
        <div className="grid grid-cols-2 gap-4 text-left">
          <div>
            <h3 className="text-[10px] font-black mb-2 text-blue-400 uppercase">Across</h3>
            {grid.placements.filter(p => p.direction === 'across').map(p => (
              <div key={`m-across-${p.number}`} className="text-[10px] text-slate-300 mb-1" onClick={() => setSelectedCell({x: p.x, y: p.y})}>
                <span className="font-bold mr-1">{p.number}.</span> {p.clue}
              </div>
            ))}
          </div>
          <div>
            <h3 className="text-[10px] font-black mb-2 text-emerald-400 uppercase">Down</h3>
            {grid.placements.filter((p: any) => p.direction === 'down').map((p: any) => (
              <div key={`m-down-${p.number}`} className="text-[10px] text-slate-300 mb-1" onClick={() => setSelectedCell({x: p.x, y: p.y})}>
                <span className="font-bold mr-1">{p.number}.</span> {p.clue}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
