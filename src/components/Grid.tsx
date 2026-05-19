import { useState, useRef, useEffect, useCallback } from 'react';
import { useGameStore } from '../store/useGameStore';
import { CellComponent } from './Cell';
import { DrawingCanvas } from './DrawingCanvas';
import { SuggestionBubble } from './SuggestionBubble';
import { submitStrokeData } from '../utils/api';
import { getWordAt, isWordPlacementCorrect } from '../utils/validation';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

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
    setActiveHint
  } = useGameStore();
  
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const lastTouchRef = useRef<{ x: number, y: number } | null>(null);
  const lastDistanceRef = useRef<number | null>(null);

  const centerOnCell = useCallback((x: number, y: number) => {
    if (!grid) return;
    // Each cell is 40px + 1px gap = 41px
    const cx = (grid.width * 41) / 2;
    const cy = (grid.height * 41) / 2;
    const cellX = x * 41 + 20.5; // Center of the cell
    const cellY = y * 41 + 20.5;
    
    setOffset({
      x: cx - cellX,
      y: cy - cellY
    });
  }, [grid]);

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

  const selectedClues = selectedCell 
    ? getWordAt(grid, selectedCell.x, selectedCell.y).filter(p => !isWordPlacementCorrect(grid, p)) 
    : [];

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
      
      {/* Selected clues floating overlay */}
      {selectedClues.length > 0 && (
        <div className="absolute bottom-12 left-1/2 -translate-x-1/2 w-[90%] md:w-auto md:min-w-[400px] max-w-[600px] z-40 animate-in fade-in slide-in-from-bottom-4 duration-300">
          <div className="bg-slate-900/90 backdrop-blur-xl border border-slate-700/50 shadow-2xl rounded-2xl p-4 flex flex-col gap-2">
            {selectedClues.map((p, idx) => (
              <div key={`${p.direction}-${p.number}`} className={cn(
                "flex items-start gap-3 relative group",
                idx > 0 && "pt-2 border-t border-slate-800"
              )}>
                <div className={cn(
                  "px-2 py-0.5 rounded text-[10px] font-black uppercase tracking-widest shrink-0 mt-0.5",
                  p.direction === 'across' ? "bg-blue-500/20 text-blue-400" : "bg-emerald-500/20 text-emerald-400"
                )}>
                  {p.number} {p.direction}
                </div>
                <div className="text-sm font-medium text-slate-100 leading-tight pr-8">
                  {p.clue}
                </div>
                
                {/* Hint Button */}
                <button
                  className="absolute right-0 top-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-xs font-black text-slate-400 hover:text-white hover:bg-slate-700 transition-all active:scale-95 touch-none select-none"
                  onPointerDown={() => setActiveHint(p)}
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

      {/* Clues overlay panel (Desktop/Tablet) */}
      {showLeftPanel && (
        <div className="absolute top-0 bottom-0 left-0 w-64 bg-slate-900/90 backdrop-blur-md border-r border-slate-800 p-4 overflow-y-auto z-30 pointer-events-auto hidden md:block animate-in slide-in-from-left duration-300">
          <div className="flex flex-col gap-6 text-left">
            <div>
              <h3 className="text-sm font-black mb-3 text-blue-400 uppercase tracking-tighter border-b border-blue-400/20 pb-1">Across</h3>
              <ul className="space-y-3">
                {grid.placements.filter(p => p.direction === 'across').map(p => {
                  const isSolved = isWordPlacementCorrect(grid, p);
                  return (
                    <li 
                      key={`across-${p.number}`} 
                      className={cn(
                        "text-xs leading-relaxed group cursor-pointer transition-all",
                        isSolved && "opacity-40 grayscale"
                      )} 
                      onClick={() => { setSelectedCell({x: p.x, y: p.y}); centerOnCell(p.x, p.y); }}
                    >
                      <span className={cn(
                        "font-bold mr-2 px-1 rounded transition-colors",
                        isSolved ? "line-through" : (selectedCell?.x === p.x && selectedCell?.y === p.y ? "bg-blue-600 text-white" : "text-blue-400 group-hover:bg-slate-800")
                      )}>{p.number}</span>
                      <span className={cn("text-slate-300 group-hover:text-white", isSolved && "line-through")}>{p.clue}</span>
                    </li>
                  );
                })}
              </ul>
            </div>
            <div>
              <h3 className="text-sm font-black mb-3 text-emerald-400 uppercase tracking-tighter border-b border-emerald-400/20 pb-1">Down</h3>
              <ul className="space-y-3">
                {grid.placements.filter((p: any) => p.direction === 'down').map((p: any) => {
                  const isSolved = isWordPlacementCorrect(grid, p);
                  return (
                    <li 
                      key={`down-${p.number}`} 
                      className={cn(
                        "text-xs leading-relaxed group cursor-pointer transition-all",
                        isSolved && "opacity-40 grayscale"
                      )} 
                      onClick={() => { setSelectedCell({x: p.x, y: p.y}); centerOnCell(p.x, p.y); }}
                    >
                      <span className={cn(
                        "font-bold mr-2 px-1 rounded transition-colors",
                        isSolved ? "line-through" : (selectedCell?.x === p.x && selectedCell?.y === p.y ? "bg-emerald-600 text-white" : "text-emerald-400 group-hover:bg-slate-800")
                      )}>{p.number}</span>
                      <span className={cn("text-slate-300 group-hover:text-white", isSolved && "line-through")}>{p.clue}</span>
                    </li>
                  );
                })}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Mobile clues panel */}
      <div className="absolute bottom-0 left-0 right-0 bg-slate-900/90 backdrop-blur-md border-t border-slate-800 p-4 max-h-48 overflow-y-auto z-30 pointer-events-auto md:hidden">
        <div className="grid grid-cols-2 gap-4 text-left">
          <div>
            <h3 className="text-[10px] font-black mb-2 text-blue-400 uppercase">Across</h3>
            {grid.placements.filter(p => p.direction === 'across').map(p => {
              const isSolved = isWordPlacementCorrect(grid, p);
              return (
                <div 
                  key={`m-across-${p.number}`} 
                  className={cn("text-[10px] mb-1 transition-opacity", isSolved ? "opacity-40 line-through text-slate-500" : "text-slate-300")} 
                  onClick={() => { setSelectedCell({x: p.x, y: p.y}); centerOnCell(p.x, p.y); }}
                >
                  <span className="font-bold mr-1">{p.number}.</span> {p.clue}
                </div>
              );
            })}
          </div>
          <div>
            <h3 className="text-[10px] font-black mb-2 text-emerald-400 uppercase">Down</h3>
            {grid.placements.filter((p: any) => p.direction === 'down').map((p: any) => {
              const isSolved = isWordPlacementCorrect(grid, p);
              return (
                <div 
                  key={`m-down-${p.number}`} 
                  className={cn("text-[10px] mb-1 transition-opacity", isSolved ? "opacity-40 line-through text-slate-500" : "text-slate-300")} 
                  onClick={() => { setSelectedCell({x: p.x, y: p.y}); centerOnCell(p.x, p.y); }}
                >
                  <span className="font-bold mr-1">{p.number}.</span> {p.clue}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
