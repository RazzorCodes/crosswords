export function renderStrokesToPixels(strokes: {x: number, y: number, t: number}[][], size = 64, pad = 8, lineWidth = 3): Float32Array {
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get 2d context');

    // White background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, size, size);

    const allPts = strokes.flat();
    if (allPts.length === 0) return new Float32Array(size * size).fill(1.0);

    const xs = allPts.map(p => p.x);
    const ys = allPts.map(p => p.y);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);

    const xRange = (xMax - xMin) || 1e-6;
    const yRange = (yMax - yMin) || 1e-6;

    const drawSize = size - 2 * pad;
    const scale = drawSize / Math.max(xRange, yRange);

    const xOffset = pad + (drawSize - xRange * scale) / 2;
    const yOffset = pad + (drawSize - yRange * scale) / 2;

    const toPx = (x: number, y: number) => ({
        px: (x - xMin) * scale + xOffset,
        py: (y - yMin) * scale + yOffset
    });

    ctx.strokeStyle = 'black';
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (const stroke of strokes) {
        if (stroke.length === 0) continue;
        ctx.beginPath();
        const start = toPx(stroke[0].x, stroke[0].y);
        ctx.moveTo(start.px, start.py);

        if (stroke.length === 1) {
             ctx.lineTo(start.px, start.py);
        } else {
            for (let i = 1; i < stroke.length; i++) {
                const p = toPx(stroke[i].x, stroke[i].y);
                ctx.lineTo(p.px, p.py);
            }
        }
        ctx.stroke();
    }

    const imgData = ctx.getImageData(0, 0, size, size);
    const pixels = new Float32Array(size * size);
    for (let i = 0; i < size * size; i++) {
        // Red channel is enough for grayscale
        // Normalize to [0, 1]
        pixels[i] = imgData.data[i * 4] / 255.0;
    }
    return pixels;
}
