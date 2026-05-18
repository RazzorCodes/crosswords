import { useEffect, useRef, useState } from 'react';
import { useGameStore } from '../store/useGameStore';
import Tesseract from 'tesseract.js';

export function DrawingCanvas() {
  const { grid, language, updateCellInput, setSelectedCell, isGestureActive } = useGameStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentStroke, setCurrentStroke] = useState<{ x: number; y: number }[]>([]);
  const workerRef = useRef<Tesseract.Worker | null>(null);

  useEffect(() => {
    const initTesseract = async () => {
      if (workerRef.current) {
        await workerRef.current.terminate();
      }
      const tesseractLang = language === 'ro' ? 'ron' : 'eng';
      const worker = await Tesseract.createWorker(tesseractLang);
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

  const handlePointerDown = (e: React.PointerEvent) => {
    // Only allow one pointer for drawing.
    // Explicitly block drawing if a multi-touch gesture is active.
    if (!e.isPrimary || isGestureActive) return;

    const rect = canvasRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const cellInfo = getCellFromCoords(x, y);
    if (cellInfo) {
      setSelectedCell({ x: cellInfo.cellX, y: cellInfo.cellY });
    }

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
        clearCanvas();
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

  const isLineGesture = (stroke: { x: number; y: number }[]) => {
    if (stroke.length < 5) return false;
    const start = stroke[0];
    const end = stroke[stroke.length - 1];
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance < 20) return false;
    
    let deviation = 0;
    for (const p of stroke) {
      const d = Math.abs((end.y - start.y) * p.x - (end.x - start.x) * p.y + end.x * start.y - end.y * start.x) / distance;
      deviation += d;
    }
    return (deviation / stroke.length) < 5;
  };

  const handlePointerUp = async (e: React.PointerEvent) => {
    if (!e.isPrimary) return;
    setIsDrawing(false);
    if (currentStroke.length < 2 || isGestureActive) {
      clearCanvas();
      return;
    }

    const start = currentStroke[0];
    const cellInfo = getCellFromCoords(start.x, start.y);
    if (!cellInfo) return;
    const { cellX, cellY, cellWidth, cellHeight } = cellInfo;

    const cell = grid?.cells[cellY][cellX];
    if (!cell || cell.isBlack) {
      clearCanvas();
      return;
    }

    if (cell.userInput !== '' && isLineGesture(currentStroke)) {
      updateCellInput(cellX, cellY, '');
      clearCanvas();
      return;
    }

    await processOCR(cellX, cellY, cellWidth, cellHeight);
    clearCanvas();
  };

  const clearCanvas = () => {
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx && canvasRef.current) {
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    setCurrentStroke([]);
  };

  const processOCR = async (cx: number, cy: number, cw: number, ch: number) => {
    if (!canvasRef.current || !workerRef.current) return;

    const scaleX = canvasRef.current.width / (grid!.width * cw);
    const scaleY = canvasRef.current.height / (grid!.height * ch);
    
    const internalCW = cw * scaleX;
    const internalCH = ch * scaleY;
    const internalCX = cx * internalCW;
    const internalCY = cy * internalCH;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = internalCW;
    tempCanvas.height = internalCH;
    const tempCtx = tempCanvas.getContext('2d')!;
    tempCtx.drawImage(canvasRef.current, internalCX, internalCY, internalCW, internalCH, 0, 0, internalCW, internalCH);
    
    const finalCanvas = document.createElement('canvas');
    finalCanvas.width = internalCW + 40;
    finalCanvas.height = internalCH + 40;
    const finalCtx = finalCanvas.getContext('2d')!;
    finalCtx.fillStyle = 'white';
    finalCtx.fillRect(0, 0, finalCanvas.width, finalCanvas.height);
    finalCtx.drawImage(tempCanvas, 20, 20);

    try {
      const { data: { text } } = await workerRef.current.recognize(finalCanvas);
      const char = text.trim().charAt(0).toUpperCase();
      if (char && /[A-Z]/.test(char)) {
        updateCellInput(cx, cy, char);
      }
    } catch (err) {
      console.error('OCR Error:', err);
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
