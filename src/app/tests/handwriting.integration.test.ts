import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import test from 'node:test';
import type { StrokeInput } from '../src/utils/recognizers/types';
import {
  createAcceptedSampleRecord,
  predictFeatureClassifierProbabilities,
  predictKnnFromFeatures,
  rebuildKnnCache,
  toCandidatesFromProbabilities,
  trainFeatureClassifier,
} from '../src/utils/handwriting/core';

interface HighQualityJsonlRecord {
  id: string;
  label: string;
  strokes: StrokeInput;
}

function loadHighQualityRecords(): HighQualityJsonlRecord[] {
  const here = fileURLToPath(new URL('.', import.meta.url));
  const datasetPath = resolve(here, '../../../data/high_quality/high_quality-0001.jsonl');
  const raw = readFileSync(datasetPath, 'utf8')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  return raw.map((line) => JSON.parse(line) as HighQualityJsonlRecord);
}

test('trains on 2 samples per available letter and predicts expected labels', () => {
    const records = loadHighQualityRecords();
    const byLetter = new Map<string, HighQualityJsonlRecord[]>();
    for (const record of records) {
      const label = record.label.slice(0, 1).toUpperCase();
      if (!/^[A-Z]$/.test(label)) {
        continue;
      }
      if (!byLetter.has(label)) {
        byLetter.set(label, []);
      }
      byLetter.get(label)?.push(record);
    }

    const selectedLetters = [...byLetter.keys()]
      .filter((label) => (byLetter.get(label)?.length ?? 0) >= 2)
      .sort();
  assert.ok(selectedLetters.length >= 2);

    const selected = selectedLetters
      .flatMap((label) => byLetter.get(label)!.slice(0, 2))
      .map((record, index) => createAcceptedSampleRecord({
        label: record.label,
        strokes: record.strokes,
        acceptance: 'user_inputted',
        source: 'hq-integration-test',
        createdAt: index + 1,
      }));

    const snapshot = trainFeatureClassifier(selected, 'milestone-50', selectedLetters);
  assert.ok(snapshot);

  let featureTop1Correct = 0;
  for (const sample of selected) {
    const probs = predictFeatureClassifierProbabilities(snapshot, sample.features);
    const ranked = toCandidatesFromProbabilities(probs, 'feature-classifier');
    if (ranked[0]?.char === sample.label) {
      featureTop1Correct += 1;
    }
    assert.ok(ranked.slice(0, 3).some((candidate) => candidate.char === sample.label));
  }
  assert.ok(featureTop1Correct >= Math.floor(selected.length * 0.5));

  const cache = rebuildKnnCache(selected);
  for (const sample of selected) {
    const top = predictKnnFromFeatures(sample.features, cache, 1, 100)[0];
    assert.equal(top?.char, sample.label);
  }
});
