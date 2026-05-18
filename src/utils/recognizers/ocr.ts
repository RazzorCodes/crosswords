import type { RecognitionResult, StrokeInput, RecognitionCandidate } from './types';
import { recognizeNativeHandwriting, warmNativeHandwritingRecognizer } from './nativeHandwriting';
import { recognizeFallback } from './fallback';
import { recognizePixelZoning } from './pixelZoning';
import { recognizeHeuristic } from './heuristic';

let isNativeApiAvailable = true;

export async function warmRecognizers() {
  try {
    await warmNativeHandwritingRecognizer();
  } catch (e) {
    isNativeApiAvailable = false;
  }
}

export async function recognizeHandwriting(strokes: StrokeInput): Promise<RecognitionResult> {
  const sourceTrail: string[] = [];
  
  // Run active engines in parallel
  const tasks: Promise<any>[] = [
    Promise.resolve(recognizePixelZoning(strokes)),
    Promise.resolve(recognizeHeuristic(strokes)),
    Promise.resolve(recognizeFallback(strokes))
  ];

  if (isNativeApiAvailable) {
    tasks.push(recognizeNativeHandwriting(strokes).catch(err => {
      console.warn('Native handwriting API failed, disabling:', err);
      isNativeApiAvailable = false;
      return { status: 'failed', candidates: [] };
    }));
  }

  const results = await Promise.all(tasks);
  
  // Results indices: 0: PixelZoning, 1: Heuristic, 2: Fallback, 3: Native (if active)
  const pixelResult = results[0];
  const heuristicResult = results[1];
  const fallbackResult = results[2];
  const nativeResult = results.length > 3 ? results[3] : { status: 'failed', candidates: [] };

  const engineResults = [
    { name: 'PixelZoning', char: pixelResult.char, score: pixelResult.score },
    { name: 'Heuristic', char: heuristicResult.char, score: heuristicResult.score },
  ];

  if (isNativeApiAvailable && nativeResult.candidates[0]) {
    engineResults.unshift({ 
      name: 'Native', 
      char: nativeResult.candidates[0].char, 
      score: nativeResult.candidates[0].score 
    });
    sourceTrail.push('native');
  }

  // Aggregate candidates from all engines (Quorum)
  const candidateMap: Record<string, RecognitionCandidate> = {};

  const addCandidate = (c: RecognitionCandidate) => {
    if (!c.char) return;
    const char = c.char.toUpperCase();
    if (!candidateMap[char]) {
      candidateMap[char] = { char, score: c.score, source: c.source };
    } else {
      // Quorum weight: boost score if multiple engines agree
      candidateMap[char].score = Math.min(1.0, candidateMap[char].score + (c.score * 0.5));
      candidateMap[char].source = `${candidateMap[char].source}+${c.source}`;
    }
  };

  // Add native candidates
  nativeResult.candidates.forEach(addCandidate);
  
  // Add heuristic/pixel/fallback
  [pixelResult, heuristicResult].forEach(r => {
    if (r.char) addCandidate({ char: r.char, score: r.score, source: r.source });
  });
  fallbackResult.candidates.forEach(addCandidate);

  let finalCandidates = Object.values(candidateMap);
  finalCandidates.sort((a, b) => b.score - a.score);

  // If no results, or very low confidence, add '?'
  if (finalCandidates.length === 0 || finalCandidates[0].score < 0.2) {
    finalCandidates.unshift({ char: '?', score: 1.0, source: 'system' });
  }

  finalCandidates = finalCandidates.slice(0, 3);
  sourceTrail.push('quorum');

  return {
    status: 'uncertain', // Always uncertain to force manual selection
    candidates: finalCandidates,
    chosenChar: null,
    sourceTrail,
    engineResults
  };
}
