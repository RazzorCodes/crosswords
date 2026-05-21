import type { Point, StrokeInput } from './types.ts';

export function normalizeLabel(value: string): string | null {
  const label = value.trim().slice(0, 1).toUpperCase();
  return /^[A-Z]$/.test(label) ? label : null;
}

export function renderStrokesToPixels(strokes: StrokeInput, size = 64, pad = 8, lineWidth = 3): Float32Array {
  const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(size, size) : document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Could not get 2d context.');
  }
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, size, size);
  const allPoints = strokes.flat();
  if (allPoints.length === 0) {
    return new Float32Array(size * size).fill(1);
  }
  const xs = allPoints.map((point) => point.x);
  const ys = allPoints.map((point) => point.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const drawSize = size - 2 * pad;
  const scale = drawSize / Math.max(xMax - xMin || 1e-6, yMax - yMin || 1e-6);
  const xOffset = pad + (drawSize - (xMax - xMin) * scale) / 2;
  const yOffset = pad + (drawSize - (yMax - yMin) * scale) / 2;
  const toPixel = (point: Point) => ({
    x: (point.x - xMin) * scale + xOffset,
    y: (point.y - yMin) * scale + yOffset,
  });
  ctx.strokeStyle = 'black';
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const stroke of strokes) {
    if (stroke.length === 0) {
      continue;
    }
    ctx.beginPath();
    const first = toPixel(stroke[0]);
    ctx.moveTo(first.x, first.y);
    for (const point of stroke.slice(1)) {
      const next = toPixel(point);
      ctx.lineTo(next.x, next.y);
    }
    if (stroke.length === 1) {
      ctx.lineTo(first.x + 0.01, first.y + 0.01);
    }
    ctx.stroke();
  }
  const data = ctx.getImageData(0, 0, size, size).data;
  const pixels = new Float32Array(size * size);
  for (let index = 0; index < pixels.length; index += 1) {
    pixels[index] = data[index * 4] / 255;
  }
  return pixels;
}
