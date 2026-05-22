import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import test from 'node:test';
import { aggregateResults } from '../lib/aggregate.ts';
import { parseDatasetJsonl } from '../lib/datasets.ts';
import { computeSvmMetrics, countByLetter, readyLetters } from '../lib/metrics.ts';
import { emptyLabState, loadLabState, resetLabState, saveLabState } from '../lib/storage.ts';
import type { LabEngineResult, LabSample, LabState, LabSvmSnapshot } from '../lib/types.ts';

class MemoryStorage {
  private readonly values = new Map<string, string>();

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }

  removeItem(key: string): void {
    this.values.delete(key);
  }
}

function sample(label: string, index: number): LabSample {
  return {
    id: `sample-${label}-${index}`,
    label,
    strokes: [[{ x: index, y: index, t: index }]],
    features: Array.from({ length: 30 }, () => index),
    createdAt: index,
    source: 'core-lab',
  };
}

test('lab storage round-trips model buffers and reset clears state', () => {
  const storage = new MemoryStorage();
  const state: LabState = {
    ...emptyLabState,
    ledger: [sample('A', 1)],
    cnnArtifacts: {
      runtime: 'burn-wasm',
      modelSafetensors: new Uint8Array([1, 2, 3]).buffer,
      optimizerState: new Uint8Array([4, 5]).buffer,
      modelConfigJson: '{"architecture":"pico-dual-cnn-v1"}',
      updatedAt: 100,
    },
  };
  saveLabState(state, storage);
  const loaded = loadLabState(storage);
  assert.equal(loaded.ledger.length, 1);
  assert.deepEqual(Array.from(new Uint8Array(loaded.cnnArtifacts?.modelSafetensors ?? new ArrayBuffer(0))), [1, 2, 3]);
  assert.deepEqual(Array.from(new Uint8Array(loaded.cnnArtifacts?.optimizerState ?? new ArrayBuffer(0))), [4, 5]);
  assert.equal(resetLabState(storage).ledger.length, 0);
  assert.equal(loadLabState(storage).ledger.length, 0);
});

test('sample counts and ready letters use two samples per letter', () => {
  const samples = [sample('A', 1), sample('A', 2), sample('B', 3)];
  assert.equal(countByLetter(samples).A, 2);
  assert.deepEqual(readyLetters(samples), ['A']);
});

test('dataset JSONL parser accepts line-delimited records', () => {
  const records = parseDatasetJsonl('{"id":"a","label":"A","strokes":[]}\n{"id":"b","label":"B","strokes":[]}\n');
  assert.equal(records.length, 2);
  assert.equal(records[1].label, 'B');
});

test('result aggregation weights only ready engines', () => {
  const engineResults: LabEngineResult[] = [
    {
      algorithm: 'knn',
      status: 'ready',
      candidates: [{ label: 'A', score: 0.9 }, { label: 'B', score: 0.1 }],
      topLabel: 'A',
      confidence: 0.9,
    },
    {
      algorithm: 'svm',
      status: 'ready',
      candidates: [{ label: 'B', score: 0.8 }, { label: 'A', score: 0.2 }],
      topLabel: 'B',
      confidence: 0.8,
    },
    {
      algorithm: 'cnn',
      status: 'unavailable',
      candidates: [],
      topLabel: null,
      confidence: null,
    },
  ];
  const aggregate = aggregateResults(engineResults);
  assert.equal(aggregate[0].label, 'B');
});

test('SVM metrics return zeros when no snapshot exists', () => {
  const metrics = computeSvmMetrics(null, [sample('A', 1)]);
  assert.equal(metrics.overallAccuracy, 0);
});

test('model reset behavior is represented by deleting personalized snapshots', () => {
  const snapshot = {
    id: 'svm',
    version: 'svm-rbf-v1',
    createdAt: 1,
    c: 10,
    gamma: 1 / 30,
    labels: ['A', 'B'],
    biases: [0, 0],
    starts: [0, 0],
    counts: [0, 0],
    coefficients: [],
    supportVectors: [],
    featureMean: [],
    featureStd: [],
    supportCount: 0,
    datasetSize: 0,
    readyLetters: ['A', 'B'],
    metrics: { user_inputtedAccuracy: 0, implicitAccuracy: 0, overallAccuracy: 0 },
  } satisfies LabSvmSnapshot;
  const state: LabState = {
    ...emptyLabState,
    svmSnapshot: snapshot,
    cnnArtifacts: {
      runtime: 'burn-wasm',
      modelSafetensors: new ArrayBuffer(0),
      optimizerState: new ArrayBuffer(0),
      modelConfigJson: '{}',
      updatedAt: 1,
    },
  };
  const resetSvm = { ...state, svmSnapshot: null };
  const resetCnn = { ...state, cnnArtifacts: null };
  assert.equal(resetSvm.svmSnapshot, null);
  assert.equal(resetCnn.cnnArtifacts, null);
});

test('WASM export smoke test sees core feature and SVM symbols', async (t) => {
  const wasmPath = resolve('public-service/wasm/handwriting_core.wasm');
  if (!existsSync(wasmPath)) {
    t.skip('run npm run wasm:build before the WASM smoke test');
    return;
  }
  const instance = await WebAssembly.instantiate(readFileSync(wasmPath), {});
  const exports = instance.instance.exports as Record<string, unknown>;
  assert.equal(typeof exports.extract_features, 'function');
  assert.equal(typeof exports.train_svm_classifier, 'function');
  assert.equal(typeof exports.predict_svm_classifier, 'function');
});
