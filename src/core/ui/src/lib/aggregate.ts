import { ALPHABET, type LabCandidate, type LabEngineResult } from './types.ts';

const ENGINE_WEIGHTS = {
  knn: 0.25,
  svm: 0.40,
  cnn: 0.35,
} as const;

export function candidatesFromScores(scores: Record<string, number>): LabCandidate[] {
  return ALPHABET
    .map((label) => ({ label, score: Number.isFinite(scores[label]) ? scores[label] : 0 }))
    .sort((left, right) => right.score - left.score);
}

export function aggregateResults(results: LabEngineResult[]): LabCandidate[] {
  const totals = Object.fromEntries(ALPHABET.map((label) => [label, 0])) as Record<string, number>;
  let usedWeight = 0;
  for (const result of results) {
    if (result.status !== 'ready' || result.candidates.length === 0) {
      continue;
    }
    const weight = ENGINE_WEIGHTS[result.algorithm];
    usedWeight += weight;
    for (const candidate of result.candidates) {
      totals[candidate.label] += candidate.score * weight;
    }
  }
  if (usedWeight === 0) {
    return [];
  }
  return candidatesFromScores(Object.fromEntries(ALPHABET.map((label) => [label, totals[label] / usedWeight])));
}
