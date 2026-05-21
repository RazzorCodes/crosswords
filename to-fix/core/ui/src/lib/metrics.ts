import { ALPHABET, type LabSample, type LabSvmSnapshot, type SnapshotMetrics } from './types.ts';
import { predictSvm } from './wasmCore.ts';

export function countByLetter(samples: LabSample[]): Record<string, number> {
  const counts = Object.fromEntries(ALPHABET.map((label) => [label, 0])) as Record<string, number>;
  for (const sample of samples) {
    counts[sample.label] = (counts[sample.label] ?? 0) + 1;
  }
  return counts;
}

export function readyLetters(samples: LabSample[], minSamples = 2): string[] {
  const counts = countByLetter(samples);
  return ALPHABET.filter((label) => counts[label] >= minSamples);
}

export function computeSvmMetrics(snapshot: LabSvmSnapshot | null, samples: LabSample[]): SnapshotMetrics {
  if (!snapshot || samples.length === 0) {
    return { user_inputtedAccuracy: 0, implicitAccuracy: 0, overallAccuracy: 0 };
  }
  let correct = 0;
  for (const sample of samples) {
    const scores = predictSvm(sample.features, snapshot);
    const top = Object.entries(scores).sort((left, right) => right[1] - left[1])[0]?.[0] ?? null;
    if (top === sample.label) {
      correct += 1;
    }
  }
  const overallAccuracy = correct / samples.length;
  return {
    user_inputtedAccuracy: overallAccuracy,
    implicitAccuracy: 0,
    overallAccuracy,
  };
}
