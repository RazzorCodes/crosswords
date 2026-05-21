import { useEffect, useRef } from 'react';
import { useGameStore } from '../store/useGameStore';

export function SuggestionStrokes() {
  const { grid, suggestions } = useGameStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !grid) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    suggestions.forEach((suggestion) => {
      ctx.strokeStyle = 'rgba(96, 165, 250, 0.5)'; // Dimmed blue
      ctx.lineWidth = 4;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      suggestion.strokes.forEach((stroke: any[]) => {
        if (stroke.length < 1) return;
        ctx.beginPath();
        ctx.moveTo(stroke[0].x, stroke[0].y);
        for (let i = 1; i < stroke.length; i++) {
          ctx.lineTo(stroke[i].x, stroke[i].y);
        }
        ctx.stroke();
      });
    });
  }, [suggestions, grid]);

  if (!grid) return null;

  return (
    <canvas
      ref={canvasRef}
      className="absolute top-0 left-0 z-10 pointer-events-none"
      width={grid.width * 80}
      height={grid.height * 80}
      style={{ 
        width: '100%', 
        height: '100%',
        opacity: 0.8 
      }}
    />
  );
}
