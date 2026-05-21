import { aggregateResults, candidatesFromScores } from './aggregate.ts';
import { predictCnn } from './cnnAdapter.ts';
import type { LabRecognitionResult, LabState, StrokeInput } from './types.ts';
import { extractFeatures, predictKnn, predictSvm } from './wasmCore.ts';

function now(): number {
  return performance.now();
}

export async function recognizeLabStrokes(strokes: StrokeInput, state: LabState): Promise<LabRecognitionResult> {
  const latenciesMs: LabRecognitionResult['latenciesMs'] = {};
  const featureStart = now();
  const features = extractFeatures(strokes);
  latenciesMs.features = now() - featureStart;

  const knnStart = now();
  const knnCandidates = candidatesFromScores(predictKnn(features, state.ledger));
  latenciesMs.knn = now() - knnStart;
  const engineResults: LabRecognitionResult['engineResults'] = [
    {
      algorithm: 'knn',
      status: state.ledger.length > 0 ? 'ready' : 'unavailable',
      candidates: state.ledger.length > 0 ? knnCandidates : [],
      topLabel: state.ledger.length > 0 ? knnCandidates[0]?.label ?? null : null,
      confidence: state.ledger.length > 0 ? knnCandidates[0]?.score ?? null : null,
      detail: state.ledger.length > 0 ? `${state.ledger.length} local samples` : 'no local samples',
    },
  ];

  const svmStart = now();
  if (state.svmSnapshot) {
    const svmCandidates = candidatesFromScores(predictSvm(features, state.svmSnapshot));
    engineResults.push({
      algorithm: 'svm',
      status: 'ready',
      candidates: svmCandidates,
      topLabel: svmCandidates[0]?.label ?? null,
      confidence: svmCandidates[0]?.score ?? null,
      detail: `${state.svmSnapshot.supportCount} support vectors`,
    });
  } else {
    engineResults.push({
      algorithm: 'svm',
      status: 'unavailable',
      candidates: [],
      topLabel: null,
      confidence: null,
      detail: 'train personalized SVM first',
    });
  }
  latenciesMs.svm = now() - svmStart;

  const cnnStart = now();
  try {
    const cnn = await predictCnn(strokes, state.baselineManifest, state.cnnArtifacts);
    engineResults.push({
      algorithm: 'cnn',
      status: cnn ? 'ready' : 'unavailable',
      candidates: cnn?.candidates ?? [],
      topLabel: cnn?.candidates[0]?.label ?? null,
      confidence: cnn?.candidates[0]?.score ?? null,
      detail: cnn ? `${cnn.source} CNN` : 'no CNN model in manifest',
    });
  } catch (error) {
    engineResults.push({
      algorithm: 'cnn',
      status: 'error',
      candidates: [],
      topLabel: null,
      confidence: null,
      detail: error instanceof Error ? error.message : 'CNN inference failed',
    });
  }
  latenciesMs.cnn = now() - cnnStart;

  const aggregateStart = now();
  const aggregateCandidates = aggregateResults(engineResults);
  latenciesMs.aggregate = now() - aggregateStart;
  return { aggregateCandidates, engineResults, latenciesMs, features };
}
