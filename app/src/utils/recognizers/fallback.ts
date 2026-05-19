import type { EngineRecognitionResult, RecognitionCandidate, StrokeInput } from './types';
import { recognizeHeuristic } from './heuristic';
import { recognizePixelZoning } from './pixelZoning';

export function recognizeFallback(strokes: StrokeInput): EngineRecognitionResult {
  const pixelResult = recognizePixelZoning(strokes);
  const heuristicResult = recognizeHeuristic(strokes);
  
  const candidates: RecognitionCandidate[] = [];
  
  if (pixelResult.char && /^[A-Z]$/.test(pixelResult.char)) {
    candidates.push({
      char: pixelResult.char,
      score: pixelResult.score,
      source: 'pixel-zoning'
    });
  }
  
  if (heuristicResult.char && /^[A-Z]$/.test(heuristicResult.char)) {
    candidates.push({
      char: heuristicResult.char,
      score: heuristicResult.score,
      source: 'heuristic'
    });
  }
  
  // Dedup and sort
  const merged = new Map<string, RecognitionCandidate>();
  for (const c of candidates) {
    const existing = merged.get(c.char);
    if (!existing || c.score > existing.score) {
      merged.set(c.char, c);
    }
  }
  
  const sorted = [...merged.values()].sort((a, b) => b.score - a.score).slice(0, 3);
  
  // Be more aggressive: mark as strong only if score is very high, 
  // but status 'uncertain' will trigger the pop-up
  return {
    source: 'fallback',
    status: sorted.length > 0 && sorted[0].score >= 0.85 ? 'strong' : (sorted.length > 0 ? 'uncertain' : 'failed'),
    candidates: sorted
  };
}
