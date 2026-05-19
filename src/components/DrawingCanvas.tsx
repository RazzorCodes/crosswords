import { useEffect, useRef } from 'react';
import { useGameStore } from '../store/useGameStore';
import { recognizeHandwriting, warmRecognizers } from '../utils/recognizers/ocr';
import { submitStrokeData } from '../utils/api';

export function DrawingCanvas() {
  const { grid, updateCellInput, setSelectedCell, isGestureActive, showSuggestions, clearSuggestions } = useGameStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawingRef = useRef(false);
  const currentStrokeRef = useRef<{ x: number; y: number; t: number }[]>([]);

  const allStrokesRef = useRef<{ x: number; y: number; t: number }[][]>([]);
  const ocrTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const activeCellRef = useRef<{ cellX: number, cellY: number, cellWidth: number, cellHeight: number } | null>(null);

  useEffect(() => {
    void warmRecognizers();
    return () => {
      if (ocrTimeoutRef.current) {
        clearTimeout(ocrTimeoutRef.current);
      }
    };
  }, []);

  const getCellFromCoords = (x: number, y: number) => {
    if (!canvasRef.current || !grid) return null;
    const rect = canvasRef.current.getBoundingClientRect();
    const cellWidth = rect.width / grid.width;
    const cellHeight = rect.height / grid.height;
    const cellX = Math.floor(x / cellWidth);
    const cellY = Math.floor(y / cellHeight);
    return { cellX, cellY, cellWidth, cellHeight };
  };

  const flushOCR = async () => {
    if (ocrTimeoutRef.current) {
      clearTimeout(ocrTimeoutRef.current);
      ocrTimeoutRef.current = null;
    }
    
    if (!activeCellRef.current || allStrokesRef.current.length === 0) {
      clearCanvas();
      return;
    }

    const { cellX, cellY } = activeCellRef.current;
    const cell = grid?.cells[cellY]?.[cellX];
    const strokes = allStrokesRef.current;

    // We copy values because we are about to clear the refs
    activeCellRef.current = null;
    allStrokesRef.current = [];

    if (!cell || cell.isBlack) {
      clearCanvas();
      return;
    }

    if (strokes.length === 1) {
      const gesture = checkLineGesture(strokes[0]);
      if (gesture.isLine) {
        if (cell.userInput !== '') {
          // Strikethrough to delete
          updateCellInput(cellX, cellY, '');
          clearCanvas();
          return;
        }
      }
    }

    // Process OCR
    await processOCR(cellX, cellY, strokes);
    clearCanvas();
  };

  const handlePointerDown = (e: React.PointerEvent) => {
    if (!e.isPrimary || isGestureActive) return;
    clearSuggestions();

    const rect = canvasRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const cellInfo = getCellFromCoords(x, y);
    if (!cellInfo) return;
    if (grid?.cells[cellInfo.cellY]?.[cellInfo.cellX]?.isBlack) return;

    if (activeCellRef.current) {
      if (activeCellRef.current.cellX !== cellInfo.cellX || activeCellRef.current.cellY !== cellInfo.cellY) {
        void flushOCR();
      } else {
        if (ocrTimeoutRef.current) {
          clearTimeout(ocrTimeoutRef.current);
          ocrTimeoutRef.current = null;
        }
      }
    }

    activeCellRef.current = cellInfo;
    setSelectedCell({ x: cellInfo.cellX, y: cellInfo.cellY });

    isDrawingRef.current = true;
    currentStrokeRef.current = [{ x, y, t: Date.now() }];
    canvasRef.current?.setPointerCapture(e.pointerId);
    
    const ctx = canvasRef.current!.getContext('2d');
    if (ctx) {
      const scaleX = canvasRef.current!.width / rect.width;
      const scaleY = canvasRef.current!.height / rect.height;
      
      ctx.beginPath();
      ctx.moveTo(x * scaleX, y * scaleY);
      ctx.strokeStyle = '#60a5fa';
      ctx.lineWidth = 4;
      ctx.lineCap = 'round';
    }
  };

  const scheduleOCR = () => {
    if (ocrTimeoutRef.current) {
      clearTimeout(ocrTimeoutRef.current);
    }
    ocrTimeoutRef.current = setTimeout(() => {
      void flushOCR();
    }, 800);
  };

  const finishStroke = () => {
    if (!isDrawingRef.current) return;
    isDrawingRef.current = false;

    if (currentStrokeRef.current.length > 1) {
      allStrokesRef.current.push([...currentStrokeRef.current]);
    }
    currentStrokeRef.current = [];

    if (allStrokesRef.current.length > 0) {
      scheduleOCR();
    }
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!e.isPrimary) return;
    if (isGestureActive) {
      finishStroke();
      return;
    }
    if (!isDrawingRef.current) {
      return;
    }

    const rect = canvasRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    currentStrokeRef.current.push({ x, y, t: Date.now() });

    const ctx = canvasRef.current!.getContext('2d');
    if (ctx) {
      const scaleX = canvasRef.current!.width / rect.width;
      const scaleY = canvasRef.current!.height / rect.height;
      ctx.lineTo(x * scaleX, y * scaleY);
      ctx.stroke();
    }
  };

  const checkLineGesture = (stroke: { x: number; y: number; t: number }[]) => {
    if (stroke.length < 5) return { isLine: false, isVertical: false };
    const start = stroke[0];
    const end = stroke[stroke.length - 1];
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance < 20) return { isLine: false, isVertical: false };
    
    let deviation = 0;
    for (const p of stroke) {
      const d = Math.abs((end.y - start.y) * p.x - (end.x - start.x) * p.y + end.x * start.y - end.y * start.x) / distance;
      deviation += d;
    }
    const isLine = (deviation / stroke.length) < 5;
    const angle = Math.abs(Math.atan2(dy, dx) * 180 / Math.PI);
    const isVertical = angle > 60 && angle < 120;
    return { isLine, isVertical };
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    if (!e.isPrimary) return;
    finishStroke();
    canvasRef.current?.releasePointerCapture(e.pointerId);
  };

  const clearCanvas = () => {
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx && canvasRef.current) {
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    currentStrokeRef.current = [];
    isDrawingRef.current = false;
  };

  const processOCR = async (cx: number, cy: number, strokes: { x: number; y: number; t: number }[][]) => {
    if (strokes.length === 0) return;

    try {
      const result = await recognizeHandwriting(strokes);
      
      const bestChar = result.chosenChar || (result.candidates.length > 0 ? result.candidates[0].char : '?');
      const bestScore = result.candidates.length > 0 ? result.candidates[0].score : 0;
      
      // Trigger Toast
      useGameStore.getState().addToast(bestChar, bestScore, result.engineResults || []);

      if (result.status === 'confirmed' && result.chosenChar) {
        // Phase 3: Ground Truth Capture - Tier 3
        // Stored into disk queue (submitStrokeData) but NOT k-NN according to plan?
        // Wait, the plan says: "Tier 3 ... Goes to disk queue only — not to k-NN"
        updateCellInput(cx, cy, result.chosenChar);
        void submitStrokeData(result.chosenChar, strokes);
      } else {
        // Always show suggestions if not confirmed
        showSuggestions({ x: cx, y: cy }, result.candidates, result.sourceTrail, strokes);
      }
    } catch (err) {
      console.error('Handwriting OCR Error:', err);
    }
  };

  return (
    <canvas
      ref={canvasRef}
      className="absolute top-0 left-0 w-full h-full cursor-crosshair touch-none z-20"
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      onPointerLeave={handlePointerUp}
      width={grid?.width ? grid.width * 80 : 0}
      height={grid?.height ? grid.height * 80 : 0}
      style={{ pointerEvents: 'auto' }}
    />
  );
}
