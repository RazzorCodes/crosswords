import { useGameStore } from '../store/useGameStore';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { getWordAt, isWordPlacementCorrect } from '../utils/validation';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface CellProps {
  x: number;
  y: number;
}

export function CellComponent({ x, y }: CellProps) {
  const { grid, selectedCell, setSelectedCell, showGlow, setHoveredCell, activeHint } = useGameStore();
  const cell = grid?.cells[y][x];

  if (!cell || cell.isBlack) {
    return <div className="w-10 h-10 bg-slate-950" />;
  }

  const isSelected = selectedCell?.x === x && selectedCell?.y === y;
  const placements = grid ? getWordAt(grid, x, y) : [];
  const isCorrectWord = placements.some(p => isWordPlacementCorrect(grid!, p));
  const shouldGlow = showGlow && isCorrectWord;

  // Hint logic
  const isActiveHint = activeHint && (
    (activeHint.direction === 'across' && y === activeHint.y && x >= activeHint.x && x < activeHint.x + activeHint.word.length) ||
    (activeHint.direction === 'down' && x === activeHint.x && y >= activeHint.y && y < activeHint.y + activeHint.word.length)
  );

  let displayChar = cell.userInput;
  let charColor = "text-white";

  if (isCorrectWord) {
    charColor = "text-green-500";
  } else if (isActiveHint) {
    if (cell.userInput === cell.char) {
      charColor = "text-green-500";
    } else {
      displayChar = cell.char;
      charColor = "text-slate-500"; // Grey for the hint
    }
  }

  return (
    <div
      className={cn(
        "relative w-10 h-10 border border-slate-700 flex items-center justify-center cursor-pointer text-lg font-bold transition-all",
        isSelected ? "bg-blue-900/50 ring-2 ring-blue-500 z-10" : "bg-slate-800",
        shouldGlow && "shadow-[inset_0_0_8px_rgba(34,197,94,0.4)] border-green-500/50",
        isCorrectWord && "bg-green-950/20" // Light green tint for locked cells
      )}
      onClick={() => setSelectedCell({ x, y })}
      onMouseEnter={() => setHoveredCell({ x, y })}
      onMouseLeave={() => setHoveredCell(null)}
    >
      {cell.number && (
        <span className="absolute top-0.5 left-0.5 text-[10px] leading-none text-slate-400">
          {cell.number}
        </span>
      )}
      <span className={charColor}>
        {displayChar}
      </span>
    </div>
  );
}
