function renderStrokesWithoutCanvas(strokes: {x: number, y: number, t: number}[][], size: number, pad: number, lineWidth: number): Float32Array {
    const pixels = new Float32Array(size * size).fill(1.0);
    const allPts = strokes.flat();
    if (allPts.length === 0) return pixels;

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
    const radius = Math.max(1, Math.round(lineWidth / 2));
    const toPx = (x: number, y: number) => ({
        px: Math.round((x - xMin) * scale + xOffset),
        py: Math.round((y - yMin) * scale + yOffset),
    });
    const mark = (x: number, y: number) => {
        for (let dy = -radius; dy <= radius; dy += 1) {
            for (let dx = -radius; dx <= radius; dx += 1) {
                const px = x + dx;
                const py = y + dy;
                if (px >= 0 && px < size && py >= 0 && py < size && dx * dx + dy * dy <= radius * radius) {
                    pixels[py * size + px] = 0;
                }
            }
        }
    };

    for (const stroke of strokes) {
        if (stroke.length === 0) continue;
        let previous = toPx(stroke[0].x, stroke[0].y);
        mark(previous.px, previous.py);
        for (let index = 1; index < stroke.length; index += 1) {
            const current = toPx(stroke[index].x, stroke[index].y);
            const steps = Math.max(Math.abs(current.px - previous.px), Math.abs(current.py - previous.py), 1);
            for (let step = 1; step <= steps; step += 1) {
                mark(
                    Math.round(previous.px + ((current.px - previous.px) * step) / steps),
                    Math.round(previous.py + ((current.py - previous.py) * step) / steps),
                );
            }
            previous = current;
        }
    }
    return pixels;
}

export function renderStrokesToPixels(strokes: {x: number, y: number, t: number}[][], size = 64, pad = 8, lineWidth = 3): Float32Array {
    if (typeof OffscreenCanvas === 'undefined' && typeof document === 'undefined') {
        return renderStrokesWithoutCanvas(strokes, size, pad, lineWidth);
    }
    const canvas = typeof OffscreenCanvas !== 'undefined'
        ? new OffscreenCanvas(size, size)
        : document.createElement('canvas');
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
