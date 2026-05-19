import type { RecognitionResult, StrokeInput, RecognitionCandidate } from './types';
import { knnRecognizer } from './knn';

export async function warmRecognizers() {
  // No warming needed for the new system currently, 
  // but we keep the export for compatibility.
}

async function fetchBackendPredictions(strokes: StrokeInput) {
  try {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ strokes })
    });
    if (!response.ok) throw new Error('Backend prediction failed');
    return await response.json();
  } catch (e) {
    console.error('Backend prediction error:', e);
    return null;
  }
}

export async function recognizeHandwriting(strokes: StrokeInput): Promise<RecognitionResult> {
  const sourceTrail: string[] = [];
  
  // 1. Get k-NN predictions (instant, browser)
  const knnCandidates = knnRecognizer.predict(strokes);
  const knnProbs: Record<string, number> = {};
  for (let i = 65; i <= 90; i++) knnProbs[String.fromCharCode(i)] = 0;
  knnCandidates.forEach(c => { knnProbs[c.char] = c.score; });

  // 2. Get Backend predictions (SVM + CNN)
  const backendResult = await fetchBackendPredictions(strokes);
  const svmProbs: Record<string, number> = {};
  const cnnProbs: Record<string, number> = {};
  for (let i = 65; i <= 90; i++) {
    const char = String.fromCharCode(i);
    svmProbs[char] = backendResult?.svm?.probs?.[char] ?? (1.0 / 26);
    cnnProbs[char] = backendResult?.cnn?.probs?.[char] ?? (1.0 / 26);
  }

  // 3. Weighted voting
  // final_probs = 0.5 × knn_probs + 0.3 × svm_probs + 0.2 × cnn_probs
  const finalProbs: Record<string, number> = {};
  const candidates: RecognitionCandidate[] = [];

  for (let i = 65; i <= 90; i++) {
    const char = String.fromCharCode(i);
    const score = (0.5 * knnProbs[char]) + (0.3 * svmProbs[char]) + (0.2 * cnnProbs[char]);
    finalProbs[char] = score;
    candidates.push({ char, score, source: 'ensemble' });
  }

  candidates.sort((a, b) => b.score - a.score);
  const topCandidates = candidates.slice(0, 3);

  sourceTrail.push('knn', 'svm', 'cnn', 'weighted-vote');

  const engineResults = [
    { name: 'k-NN', char: knnCandidates[0]?.char ?? '?', score: knnCandidates[0]?.score ?? 0 },
    { name: 'SVM', char: backendResult?.svm?.label ?? '?', score: backendResult?.svm?.probs?.[backendResult?.svm?.label] ?? 0 },
    { name: 'CNN', char: backendResult?.cnn?.label ?? '?', score: backendResult?.cnn?.probs?.[backendResult?.cnn?.label] ?? 0 },
  ];

  return {
    status: backendResult?.teacher?.action === 'accept' ? 'confirmed' : 'uncertain',
    candidates: topCandidates,
    chosenChar: backendResult?.teacher?.action === 'accept' ? backendResult.teacher.label : null,
    sourceTrail,
    engineResults,
    teacher: backendResult?.teacher
  };
}
