export type Point = { x: number; y: number; t: number };

// Very simple directional / bounding box heuristic recognizer 
export function recognizeHeuristic(strokes: Point[][]): { char: string; score: number } {
  if (!strokes || strokes.length === 0) return { char: '', score: 0 };
  
  // Calculate bounding box
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const stroke of strokes) {
    for (const p of stroke) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
  }
  
  const height = maxY - minY;
  
  // Single vertical-ish stroke
  if (strokes.length === 1) {
    const s = strokes[0];
    const first = s[0];
    const last = s[s.length - 1];
    const dx = Math.abs(last.x - first.x);
    const dy = Math.abs(last.y - first.y);
    
    // Likely an 'I', but keep score moderate to allow other candidates/uncertainty
    if (dy > dx * 2 && height > 20) {
      return { char: 'I', score: 0.7 };
    }
  }
  
  // Cross gesture ('X' or 'T')
  if (strokes.length === 2) {
    // Check if it's an X
    const s1 = strokes[0];
    const s2 = strokes[1];
    
    const s1dx = s1[s1.length - 1].x - s1[0].x;
    const s1dy = s1[s1.length - 1].y - s1[0].y;
    const s2dx = s2[s2.length - 1].x - s2[0].x;
    const s2dy = s2[s2.length - 1].y - s2[0].y;
    
    // Diagonal lines opposite direction
    if ((s1dx * s1dy > 0 && s2dx * s2dy < 0) || (s1dx * s1dy < 0 && s2dx * s2dy > 0)) {
        // This is a rough X
        return { char: 'X', score: 0.8 };
    }
    
    // Check if T (one horizontal, one vertical)
    const isS1Horiz = Math.abs(s1dx) > Math.abs(s1dy) * 2;
    const isS2Vert = Math.abs(s2dy) > Math.abs(s2dx) * 2;
    const isS1Vert = Math.abs(s1dy) > Math.abs(s1dx) * 2;
    const isS2Horiz = Math.abs(s2dx) > Math.abs(s2dy) * 2;
    
    if ((isS1Horiz && isS2Vert) || (isS1Vert && isS2Horiz)) {
       return { char: 'T', score: 0.8 };
    }
  }

  return { char: '?', score: 0 };
}
