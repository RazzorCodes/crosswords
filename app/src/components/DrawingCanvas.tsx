import { useEffect, useRef } from 'react';
import { useGameStore } from '../store/useGameStore';
import { useHandwritingStore } from '../store/useHandwritingStore';
import { cancelPendingSubmission, finalizeHandwritingSample } from '../utils/handwritingSession';
import { recognizeHandwriting, warmRecognizers } from '../utils/recognizers/ocr';

export function DrawingCanvas() {
  const { grid, updateCellInput, setSelectedCell, isGestureActive, showSuggestions, clearSuggestions } = useGameStore();
  const { trainMode } = useHandwritingStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawingRef = useRef(false);
  const currentStrokeRef = useRef<{ x: number; y: number; t: number }[]>([]);
  const allStrokesRef = useRef<{ x: number; y: number; t: number }[][]>([]);
  const ocrTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const activeCellRef = useRef<{ cellX: number; cellY: number; cellWidth: number; cellHeight: number } | null>(null);

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

  const clearCanvas = () => {
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx && canvasRef.current) {
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    currentStrokeRef.current = [];
    isDrawingRef.current = false;
  };

  const checkLineGesture = (stroke: { x: number; y: number; t: number }[]) => {
    if (stroke.length < 5) return { isLine: false };
    const start = stroke[0];
    const end = stroke[stroke.length - 1];
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const distance = Math.sqrt((dx * dx) + (dy * dy));
    if (distance < 20) return { isLine: false };

    let deviation = 0;
    for (const point of stroke) {
      const distanceToLine = Math.abs(
        ((end.y - start.y) * point.x) -
        ((end.x - start.x) * point.y) +
        (end.x * start.y) -
        (end.y * start.x),
      ) / distance;
      deviation += distanceToLine;
    }

    return { isLine: (deviation / stroke.length) < 5 };
  };

  const processOCR = async (cx: number, cy: number, strokes: { x: number; y: number; t: number }[][]) => {
    if (strokes.length === 0) return;

    try {
      const result = await recognizeHandwriting(strokes);
      const bestChar = result.chosenChar || result.candidates[0]?.char || '?';
      const bestScore = result.candidates[0]?.score || 0;

      if (trainMode) {
        useGameStore.getState().addToast(bestChar, bestScore, result.engineResults || []);
      }

      if (result.status === 'confirmed' && result.chosenChar) {
        updateCellInput(cx, cy, result.chosenChar);
        finalizeHandwritingSample({
          x: cx,
          y: cy,
          label: result.chosenChar,
          strokes,
          source: 'auto-accept',
        });
        return;
      }

      if (trainMode) {
        showSuggestions({ x: cx, y: cy }, result.candidates, result.sourceTrail, strokes);
      }
    } catch (error) {
      console.error('Handwriting OCR error:', error);
    }
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

    activeCellRef.current = null;
    allStrokesRef.current = [];

    if (!cell || cell.isBlack) {
      clearCanvas();
      return;
    }

    if (strokes.length === 1) {
      const gesture = checkLineGesture(strokes[0]);
      if (gesture.isLine && cell.userInput !== '') {
        cancelPendingSubmission(`${cellX}:${cellY}`);
        updateCellInput(cellX, cellY, '');
        clearCanvas();
        return;
      }
    }

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
      } else if (ocrTimeoutRef.current) {
        clearTimeout(ocrTimeoutRef.current);
        ocrTimeoutRef.current = null;
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

  const handlePointerUp = (e: React.PointerEvent) => {
    if (!e.isPrimary) return;
    finishStroke();
    if (canvasRef.current?.hasPointerCapture(e.pointerId)) {
      canvasRef.current.releasePointerCapture(e.pointerId);
    }
  };

  return (
    <canvas
      ref={canvasRef}
      className="absolute top-0 left-0 z-20 h-full w-full cursor-crosshair touch-none"
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
