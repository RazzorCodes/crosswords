import { useEffect, useRef, useState } from 'react';
import { useGameStore } from '../store/useGameStore';
import Tesseract from 'tesseract.js';
import { recognizePixelZoning } from '../utils/recognizers/pixelZoning';
import { recognizeHeuristic } from '../utils/recognizers/heuristic';

export function DrawingCanvas() {
  const { grid, language, updateCellInput, setSelectedCell, isGestureActive } = useGameStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentStroke, setCurrentStroke] = useState<{ x: number; y: number }[]>([]);
  const workerRef = useRef<Tesseract.Worker | null>(null);

  const allStrokesRef = useRef<{ x: number; y: number }[][]>([]);
  const ocrTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const activeCellRef = useRef<{ cellX: number, cellY: number, cellWidth: number, cellHeight: number } | null>(null);

  useEffect(() => {
    const initTesseract = async () => {
      if (workerRef.current) {
        await workerRef.current.terminate();
      }
      const tesseractLang = language === 'ro' ? 'ron' : 'eng';
      const worker = await Tesseract.createWorker(tesseractLang);
      await worker.setParameters({
        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
        tessedit_pageseg_mode: Tesseract.PSM.SINGLE_CHAR, // SINGLE_CHAR mode
      });
      workerRef.current = worker;
    };
    initTesseract();
    return () => {
      workerRef.current?.terminate();
    };
  }, [language]);

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
        } else if (gesture.isVertical) {
          // Vertical line on empty cell -> 'I'
          updateCellInput(cellX, cellY, 'I');
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

    const rect = canvasRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const cellInfo = getCellFromCoords(x, y);
    if (!cellInfo) return;

    if (activeCellRef.current) {
      if (activeCellRef.current.cellX !== cellInfo.cellX || activeCellRef.current.cellY !== cellInfo.cellY) {
        // Changed cell, flush immediately
        flushOCR();
      } else {
        // Same cell, cancel timeout to allow multi-stroke
        if (ocrTimeoutRef.current) {
          clearTimeout(ocrTimeoutRef.current);
          ocrTimeoutRef.current = null;
        }
      }
    }

    activeCellRef.current = cellInfo;
    setSelectedCell({ x: cellInfo.cellX, y: cellInfo.cellY });

    setIsDrawing(true);
    setCurrentStroke([{ x, y }]);
    
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

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!isDrawing || !e.isPrimary || isGestureActive) {
      if (isDrawing) {
        setIsDrawing(false);
        if (currentStroke.length > 0) {
          allStrokesRef.current.push(currentStroke);
          ocrTimeoutRef.current = setTimeout(flushOCR, 800);
        }
      }
      return;
    }
    const rect = canvasRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setCurrentStroke(prev => [...prev, { x, y }]);

    const ctx = canvasRef.current!.getContext('2d');
    if (ctx) {
      const scaleX = canvasRef.current!.width / rect.width;
      const scaleY = canvasRef.current!.height / rect.height;
      ctx.lineTo(x * scaleX, y * scaleY);
      ctx.stroke();
    }
  };

  const checkLineGesture = (stroke: { x: number; y: number }[]) => {
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
    if (!e.isPrimary || !isDrawing) return;
    setIsDrawing(false);
    
    if (currentStroke.length > 1) {
      allStrokesRef.current.push(currentStroke);
    }
    
    // Start timeout for multi-stroke
    ocrTimeoutRef.current = setTimeout(flushOCR, 800);
  };

  const clearCanvas = () => {
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx && canvasRef.current) {
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    setCurrentStroke([]);
  };

  const processOCR = async (cx: number, cy: number, strokes: {x: number, y: number}[][]) => {
    if (!workerRef.current || strokes.length === 0) return;

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    strokes.forEach(stroke => {
      stroke.forEach(p => {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
      });
    });

    const strokeWidth = maxX - minX;
    const strokeHeight = maxY - minY;
    
    if (strokeWidth < 5 && strokeHeight < 5) return;

    const finalCanvas = document.createElement('canvas');
    finalCanvas.width = 128;
    finalCanvas.height = 128;
    const finalCtx = finalCanvas.getContext('2d')!;
    finalCtx.fillStyle = 'white';
    finalCtx.fillRect(0, 0, finalCanvas.width, finalCanvas.height);

    const maxDim = Math.max(strokeWidth, strokeHeight);
    const scale = maxDim > 0 ? 80 / maxDim : 1;
    const offsetX = (128 - strokeWidth * scale) / 2;
    const offsetY = (128 - strokeHeight * scale) / 2;

    finalCtx.strokeStyle = 'black';
    finalCtx.lineWidth = 8;
    finalCtx.lineCap = 'round';
    finalCtx.lineJoin = 'round';

    strokes.forEach(stroke => {
      if (stroke.length === 0) return;
      finalCtx.beginPath();
      finalCtx.moveTo((stroke[0].x - minX) * scale + offsetX, (stroke[0].y - minY) * scale + offsetY);
      for (let i = 1; i < stroke.length; i++) {
        finalCtx.lineTo((stroke[i].x - minX) * scale + offsetX, (stroke[i].y - minY) * scale + offsetY);
      }
      finalCtx.stroke();
    });

    // Run all three engines in parallel
    const [tesseractResult, pixelResult, heuristicResult] = await Promise.all([
      (async () => {
        try {
          const { data: { text } } = await workerRef.current!.recognize(finalCanvas);
          let char = text.trim().charAt(0).toUpperCase();
          
          const charMap: Record<string, string> = {
            '0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'A', '5': 'S', 
            '6': 'B', '8': 'B', '9': 'G', '|': 'I', '/': 'I', 
            '\\': 'I', '+': 'T', '[': 'C', ']': 'C', '(': 'C', ')': 'C', 
            '{': 'C', '}': 'C', '<': 'C', '>': 'C', '@': 'A'
          };

          const rawChar = text.trim().charAt(0);
          if (rawChar === 't') char = 'T';
          else if (rawChar === 'f') char = 'F';
          else if (rawChar === 'b') char = 'B';
          
          if (charMap[char]) char = charMap[char];
          
          if (char && /[A-Z]/.test(char)) return char;
          return '?';
        } catch {
          return '?';
        }
      })(),
      Promise.resolve().then(() => recognizePixelZoning(strokes)),
      Promise.resolve().then(() => recognizeHeuristic(strokes))
    ]);

    console.log('OCR Quorum:', { Tesseract: tesseractResult, PixelZoning: pixelResult, Heuristic: heuristicResult });

    let finalChar = '';

    // Quorum voting logic
    const votes: Record<string, number> = {};
    if (/[A-Z]/.test(tesseractResult)) votes[tesseractResult] = (votes[tesseractResult] || 0) + 1;
    if (/[A-Z]/.test(pixelResult.char)) votes[pixelResult.char] = (votes[pixelResult.char] || 0) + 1;
    if (/[A-Z]/.test(heuristicResult.char)) votes[heuristicResult.char] = (votes[heuristicResult.char] || 0) + 1;

    let maxVotes = 0;
    for (const [char, count] of Object.entries(votes)) {
      if (count > maxVotes) {
        maxVotes = count;
        finalChar = char;
      }
    }

    // Tie-breaker:
    if (maxVotes < 2) {
       // If no agreement (maxVotes = 1 or 0), trust the one with the highest confidence score, or fallback to pixelZoning if it has a decent score
       if (pixelResult.score > 0.4 && /[A-Z]/.test(pixelResult.char)) {
         finalChar = pixelResult.char;
       } else if (heuristicResult.score > 0.6 && /[A-Z]/.test(heuristicResult.char)) {
         finalChar = heuristicResult.char;
       } else if (/[A-Z]/.test(tesseractResult)) {
         finalChar = tesseractResult;
       } else if (pixelResult.score > 0.2 && /[A-Z]/.test(pixelResult.char)) {
         finalChar = pixelResult.char;
       }
    }

    if (finalChar && /[A-Z]/.test(finalChar)) {
      updateCellInput(cx, cy, finalChar);
    }
  };

  return (
    <canvas
      ref={canvasRef}
      className="absolute top-0 left-0 w-full h-full cursor-crosshair touch-none z-20"
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      width={grid?.width ? grid.width * 80 : 0}
      height={grid?.height ? grid.height * 80 : 0}
      style={{ pointerEvents: 'auto' }}
    />
  );
}
