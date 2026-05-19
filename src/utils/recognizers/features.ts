import { StrokeInput } from './types';

export interface Features {
  values: number[];
}

export function extractFeatures(strokes: StrokeInput): number[] {
  // Flatten all points
  const allPts = strokes.flat();
  if (allPts.length === 0) return new Array(30).fill(0);

  const xs = allPts.map(p => p.x);
  const ys = allPts.map(p => p.y);
  // const ts = allPts.map(p => p.t);

  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);

  const xRange = (xMax - xMin) || 1e-6;
  const yRange = (yMax - yMin) || 1e-6;

  const feats: number[] = [];

  // -- 1. Global shape
  feats.push(strokes.length); // stroke count
  feats.push(yRange / xRange); // aspect ratio
  feats.push(xRange / (xRange + yRange)); // relative width

  // -- 2. Direction histogram (8 bins)
  const angles: number[] = [];
  for (const stroke of strokes) {
    if (stroke.length < 2) continue;
    for (let i = 1; i < stroke.length; i++) {
      const dx = stroke[i].x - stroke[i-1].x;
      const dy = stroke[i].y - stroke[i-1].y;
      angles.push(Math.atan2(dy, dx));
    }
  }

  const hist = new Array(8).fill(0);
  for (const angle of angles) {
    // Map [-PI, PI] to [0, 8)
    let bin = Math.floor(((angle + Math.PI) / (2 * Math.PI)) * 8);
    if (bin >= 8) bin = 7;
    if (bin < 0) bin = 0;
    hist[bin]++;
  }
  const anglesCount = angles.length || 1e-6;
  feats.push(...hist.map(h => h / anglesCount));

  // -- 3. Curvature
  if (angles.length > 1) {
    const diffs: number[] = [];
    for (let i = 1; i < angles.length; i++) {
      let d = angles[i] - angles[i-1];
      // Wrap to [-PI, PI]
      d = ((d + Math.PI) % (2 * Math.PI) + (2 * Math.PI)) % (2 * Math.PI) - Math.PI;
      diffs.push(Math.abs(d));
    }
    const mean = diffs.reduce((a, b) => a + b, 0) / diffs.length;
    const variance = diffs.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / diffs.length;
    feats.push(mean);
    feats.push(Math.sqrt(variance));
  } else {
    feats.push(0, 0);
  }

  // -- 4. Start / end points
  const firstPt = strokes[0][0];
  const lastPt = strokes[strokes.length - 1][strokes[strokes.length - 1].length - 1];
  feats.push((firstPt.x - xMin) / xRange);
  feats.push((firstPt.y - yMin) / yRange);
  feats.push((lastPt.x - xMin) / xRange);
  feats.push((lastPt.y - yMin) / yRange);

  // -- 5. Total path length
  let totalLen = 0;
  for (const stroke of strokes) {
    for (let i = 1; i < stroke.length; i++) {
      const dx = stroke[i].x - stroke[i-1].x;
      const dy = stroke[i].y - stroke[i-1].y;
      totalLen += Math.sqrt(dx * dx + dy * dy);
    }
  }
  feats.push(totalLen / (xRange + yRange));

  // -- 6. Velocity features
  const speeds: number[] = [];
  const pauses: number[] = [];
  let prevEndT: number | null = null;

  for (const stroke of strokes) {
    if (prevEndT !== null) {
      pauses.push(stroke[0].t - prevEndT);
    }
    prevEndT = stroke[stroke.length - 1].t;

    for (let i = 1; i < stroke.length; i++) {
      const dx = stroke[i].x - stroke[i-1].x;
      const dy = stroke[i].y - stroke[i-1].y;
      const dt = (stroke[i].t - stroke[i-1].t) || 1e-6;
      speeds.push(Math.sqrt(dx * dx + dy * dy) / dt);
    }
  }

  if (speeds.length > 0) {
    const mean = speeds.reduce((a, b) => a + b, 0) / speeds.length;
    const variance = speeds.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / speeds.length;
    const sortedSpeeds = [...speeds].sort((a, b) => a - b);
    const p90 = sortedSpeeds[Math.floor(sortedSpeeds.length * 0.9)];
    feats.push(mean);
    feats.push(Math.sqrt(variance));
    feats.push(p90);
  } else {
    feats.push(0, 0, 0);
  }

  if (pauses.length > 0) {
    feats.push(pauses.reduce((a, b) => a + b, 0) / pauses.length);
    feats.push(Math.max(...pauses));
  } else {
    feats.push(0, 0);
  }

  // -- 7. Centroids
  for (let i = 0; i < 3; i++) {
    if (i < strokes.length) {
      const sx = strokes[i].reduce((sum, p) => sum + p.x, 0) / strokes[i].length;
      const sy = strokes[i].reduce((sum, p) => sum + p.y, 0) / strokes[i].length;
      feats.push((sx - xMin) / xRange);
      feats.push((sy - yMin) / yRange);
    } else {
      feats.push(-1, -1);
    }
  }

  // -- 8. Crossings
  let crossings = 0;
  const midX = xMin + xRange * 0.5;
  for (const stroke of strokes) {
    for (let i = 0; i < stroke.length - 1; i++) {
      const x0 = stroke[i].x;
      const x1 = stroke[i+1].x;
      if ((x0 < midX) !== (x1 < midX)) {
        crossings++;
      }
    }
  }
  feats.push(crossings);

  return feats;
}
