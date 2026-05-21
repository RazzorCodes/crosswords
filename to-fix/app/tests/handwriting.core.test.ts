import assert from 'node:assert/strict';
import test from 'node:test';
import { InferenceSession } from 'onnxruntime-web';
import type {
  AcceptedSampleRecord,
  HandwritingModuleEvent,
  PersonalizedCnnArtifacts,
  SnapshotMetrics,
  TrainingState,
} from '../src/utils/handwriting/types';
import { BrowserHandwritingModule } from '../src/utils/handwriting/module';
import { CnnTrainingRuntime, getCnnTrainingAvailability, type CnnTrainingRuntimeLike } from '../src/utils/handwriting/cnnTrainingRuntime';
import { MemoryKeyValueStore } from '../src/utils/handwriting/storage';
import { createFeedbackDevGrid } from '../src/utils/devBoard';
import {
  buildBalancedDataset,
  createExportBundle,
  createDefaultBaselineManifest,
  createInitialTrainingState,
  createPersistedState,
  decideTrainingTrigger,
  sanitizeImportedState,
  shouldAcceptCandidateSnapshot,
} from '../src/utils/handwriting/core';
import { useGameStore } from '../src/store/useGameStore';

function features(value: number): number[] {
  return new Array(30).fill(value);
}

function sample(
  id: string,
  label: string,
  acceptance: 'user_inputted' | 'implicit',
  createdAt: number,
  value: number,
): AcceptedSampleRecord {
  return {
    id,
    label,
    acceptance,
    source: 'test',
    createdAt,
    strokes: [[{ x: value, y: value, t: createdAt }]],
    features: features(value),
  };
}

function stateWith(overrides: Partial<TrainingState>): TrainingState {
  const baseline = createDefaultBaselineManifest('/models');
  return {
    ...createInitialTrainingState(baseline, 2_000_000),
    ...overrides,
  };
}

function installManifestFetch(manifest = createDefaultBaselineManifest('/models')): () => void {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: string | URL | Request) => {
    const url = String(input);
    if (url.endsWith('/manifest.json')) {
      return new Response(JSON.stringify(manifest), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    }
    return new Response(null, { status: 404 });
  }) as typeof fetch;
  return () => {
    globalThis.fetch = originalFetch;
  };
}

async function recordBalancedMilestone(module: BrowserHandwritingModule): Promise<void> {
  for (let index = 0; index < 25; index += 1) {
    await module.recordAcceptedSample({
      label: 'A',
      acceptance: index === 0 ? 'user_inputted' : 'implicit',
      source: 'test',
      createdAt: index + 1,
      strokes: [[{ x: 0, y: index, t: index }]],
    });
    await module.recordAcceptedSample({
      label: 'B',
      acceptance: index === 0 ? 'user_inputted' : 'implicit',
      source: 'test',
      createdAt: index + 100,
      strokes: [[{ x: 100, y: index, t: index }]],
    });
  }
}

test('buildBalancedDataset returns balanced per-letter training for ready letters', () => {
    const ledger: AcceptedSampleRecord[] = [
      sample('a1', 'A', 'user_inputted', 1, 1),
      sample('a2', 'A', 'implicit', 2, 1.1),
      sample('a3', 'A', 'implicit', 3, 1.2),
      sample('a4', 'A', 'implicit', 4, 1.3),
      sample('a5', 'A', 'implicit', 5, 1.4),
      sample('b1', 'B', 'user_inputted', 6, 9),
      sample('b2', 'B', 'implicit', 7, 9.1),
      sample('b3', 'B', 'implicit', 8, 9.2),
      sample('b4', 'B', 'implicit', 9, 9.3),
      sample('b5', 'B', 'implicit', 10, 9.4),
    ];

    const dataset = buildBalancedDataset(ledger);
  assert.deepEqual(dataset.readyLetters, ['A', 'B']);
  assert.equal(dataset.perLetterTarget, 5);
  assert.equal(dataset.training.length, 10);
  assert.equal(dataset.holdout.length, 0);
});

test('decideTrainingTrigger handles milestones and post-200 user batch policy', () => {
    const milestone50 = decideTrainingTrigger(stateWith({
      totalAcceptedSamples: 50,
      milestonesCompleted: [],
    }));
  assert.equal(milestone50.shouldTrain, true);
  assert.equal(milestone50.reason, 'milestone-50');

    const userBatch10 = decideTrainingTrigger(stateWith({
      totalAcceptedSamples: 220,
      milestonesCompleted: [50, 100, 200],
      pendingUserInputtedSinceTraining: 10,
      readyLetters: ['A', 'B'],
    }));
  assert.equal(userBatch10.shouldTrain, true);
  assert.equal(userBatch10.reason, 'user-batch-10');
});

test('shouldAcceptCandidateSnapshot enforces regression gating', () => {
    const current: SnapshotMetrics = {
      user_inputtedAccuracy: 0.8,
      implicitAccuracy: 0.7,
      overallAccuracy: 0.75,
    };

  assert.equal(shouldAcceptCandidateSnapshot(current, {
    user_inputtedAccuracy: 0.81,
    implicitAccuracy: 0.66,
    overallAccuracy: 0.73,
  }), true);

  // Still accepts within 0.02 tolerance
  assert.equal(shouldAcceptCandidateSnapshot(current, {
    user_inputtedAccuracy: 0.78,
    implicitAccuracy: 0.8,
    overallAccuracy: 0.8,
  }), true);

  // Rejects below tolerance
  assert.equal(shouldAcceptCandidateSnapshot(current, {
    user_inputtedAccuracy: 0.77,
    implicitAccuracy: 0.8,
    overallAccuracy: 0.8,
  }), false);
});

test('sanitizeImportedState drops invalid ledger records', () => {
    const baseline = createDefaultBaselineManifest('/models');
    const sanitized = sanitizeImportedState(
      {
        baseline,
        ledger: [
          sample('ok', 'C', 'user_inputted', 123, 3),
          {
            id: 'bad',
            label: 'not-letter',
            acceptance: 'user_inputted',
            source: 'test',
            createdAt: 123,
            strokes: [],
          } as unknown as AcceptedSampleRecord,
        ],
      },
      baseline,
      2_000_000,
    );

  assert.equal(sanitized.coreState.ledger.length, 1);
  assert.equal(sanitized.coreState.ledger[0].id, 'ok');
  assert.equal(sanitized.trainingState.totalAcceptedSamples, 1);
});

test('createFeedbackDevGrid returns one FEEDBACK across placement', () => {
  const grid = createFeedbackDevGrid();
  assert.equal(grid.width, 10);
  assert.equal(grid.height, 3);
  assert.equal(grid.placements.length, 1);
  assert.equal(grid.placements[0].word, 'FEEDBACK');
  assert.equal(grid.placements[0].direction, 'across');
  assert.deepEqual(
    grid.cells[1].slice(1, 9).map((cell) => cell.char),
    ['F', 'E', 'E', 'D', 'B', 'A', 'C', 'K'],
  );
  assert.equal(grid.cells.flat().filter((cell) => !cell.isBlack).length, 8);
});

test('CNN training availability reports exact manifest blockers', () => {
  const manifest = createDefaultBaselineManifest('/models');
  const availability = getCnnTrainingAvailability(manifest);
  assert.equal(availability.available, false);
  assert.deepEqual(availability.reasons, [
    'manifest disables CNN training',
    'missing ORT Web training module',
    'missing ORT Web training WASM',
    'missing CNN training model',
    'missing CNN eval model',
    'missing CNN optimizer model',
    'missing CNN checkpoint',
  ]);

  manifest.cnn = {
    inferenceUrl: '/models/cnn.onnx',
    supportsTraining: true,
    trainingArtifacts: {
      trainUrl: '/models/ort-training/training_model.onnx',
      evalUrl: '/models/ort-training/eval_model.onnx',
      optimizerUrl: '/models/ort-training/optimizer_model.onnx',
      checkpointUrl: '/models/ort-training/checkpoint',
      exportMetadataUrl: '/models/export-metadata.json',
    },
    trainingRuntime: {
      moduleUrl: '/models/ort-training/ort-training-web.mjs',
      wasmUrl: '/models/ort-training/',
      simdWasmUrl: '/models/ort-training/ort-wasm-simd.wasm',
      threadedWasmUrl: '/models/ort-training/ort-wasm-simd-threaded.wasm',
    },
  };
  assert.equal(new CnnTrainingRuntime(manifest).getAvailability().available, true);
});

test('BrowserHandwritingModule accepts paired centroid and CNN candidates atomically', async () => {
  const manifest = createDefaultBaselineManifest('/models');
  manifest.version = 'test-artifacts';
  manifest.cnn = {
    inferenceUrl: '/models/cnn.onnx',
    supportsTraining: true,
    trainingArtifacts: {
      trainUrl: '/models/ort-training/training_model.onnx',
      evalUrl: '/models/ort-training/eval_model.onnx',
      optimizerUrl: '/models/ort-training/optimizer_model.onnx',
      checkpointUrl: '/models/ort-training/checkpoint',
      exportMetadataUrl: '/models/export-metadata.json',
    },
    trainingRuntime: {
      moduleUrl: '/models/ort-training/ort-training-web.mjs',
      wasmUrl: '/models/ort-training/',
      simdWasmUrl: '/models/ort-training/ort-wasm-simd.wasm',
      threadedWasmUrl: '/models/ort-training/ort-wasm-simd-threaded.wasm',
    },
  };

  const restoreFetch = installManifestFetch(manifest);

  const cnnArtifacts: PersonalizedCnnArtifacts = {
    checkpoint: new ArrayBuffer(4),
    inferenceModel: new ArrayBuffer(8),
    exportMetadata: { test: true },
    metrics: {
      user_inputtedAccuracy: 1,
      implicitAccuracy: 1,
      overallAccuracy: 1,
    },
    stage: 'head-only',
    updatedAt: 123,
  };
  const cnnTrainer: CnnTrainingRuntimeLike = {
    isAvailable: () => true,
    trainCandidate: async () => ({
      artifacts: cnnArtifacts,
      metrics: cnnArtifacts.metrics!,
      stage: 'head-only',
      accepted: true,
      rejectionReason: null,
    }),
  };

  try {
    const module = new BrowserHandwritingModule(new MemoryKeyValueStore());
    await module.init({
      baselineManifestUrl: '/models/manifest.json',
      cnnTrainingRuntime: cnnTrainer,
    });

    await recordBalancedMilestone(module);

    const state = module.getTrainingState();
    assert.equal(state.lastTrainingOutcome, 'accepted');
    assert.equal(state.trainerStatus.cnnTrainingStatus, 'accepted');
    assert.equal(state.trainerStatus.personalizedCnnAvailable, true);
    assert.equal(state.trainerStatus.cnnInferenceSource, 'personalized');
    assert.equal(state.trainerStatus.activeModelGeneration, 1);
    assert.deepEqual(state.personalizedCohortLabelMap, ['A', 'B']);
  } finally {
    restoreFetch();
  }
});

test('training progress events report centroid-only path with CNN unavailable reason', async () => {
  const restoreFetch = installManifestFetch();
  try {
    const module = new BrowserHandwritingModule(new MemoryKeyValueStore());
    const events: HandwritingModuleEvent[] = [];
    module.subscribe((event) => events.push(event));
    await module.init({ baselineManifestUrl: '/models/manifest.json' });

    await recordBalancedMilestone(module);

    const progress = events
      .filter((event): event is Extract<HandwritingModuleEvent, { type: 'training-progress' }> => event.type === 'training-progress')
      .map((event) => event.payload);
    assert.deepEqual(progress.map((event) => `${event.phase}:${event.status}`), [
      'feature:running',
      'feature:ready',
      'cnn:skipped',
      'finalizing:running',
      'finalizing:ready',
    ]);
    assert.equal(progress.find((event) => event.phase === 'cnn')?.message, 'cnn unavailable: manifest disables CNN training');
    assert.equal(progress[progress.length - 1]?.progress, 100);

    const state = module.getTrainingState();
    assert.equal(state.lastTrainingOutcome, 'accepted');
    assert.equal(state.personalizationGeneration, 1);
    assert.equal(state.trainerStatus.personalizedCnnAvailable, false);
    assert.equal(state.trainerStatus.cnnInferenceSource, 'baseline');
  } finally {
    restoreFetch();
  }
});

test('training progress events report paired centroid and CNN acceptance', async () => {
  const manifest = createDefaultBaselineManifest('/models');
  manifest.cnn.supportsTraining = true;
  const restoreFetch = installManifestFetch(manifest);
  const cnnArtifacts: PersonalizedCnnArtifacts = {
    checkpoint: new ArrayBuffer(4),
    inferenceModel: new ArrayBuffer(8),
    exportMetadata: null,
    metrics: { user_inputtedAccuracy: 1, implicitAccuracy: 1, overallAccuracy: 1 },
    stage: 'head-only',
    updatedAt: 456,
  };
  const cnnTrainer: CnnTrainingRuntimeLike = {
    isAvailable: () => true,
    getAvailability: () => ({ available: true, reasons: [] }),
    trainCandidate: async (_training, _holdout, _previous, options) => {
      options?.onProgress?.({
        step: 'training',
        progress: 0.5,
        message: 'cnn epoch 1/1 batch 1/1',
        stage: 'head-only',
        epoch: 1,
        epochs: 1,
        batch: 1,
        batches: 1,
      });
      return {
        artifacts: cnnArtifacts,
        metrics: cnnArtifacts.metrics!,
        stage: 'head-only',
        accepted: true,
        rejectionReason: null,
      };
    },
  };

  try {
    const module = new BrowserHandwritingModule(new MemoryKeyValueStore());
    const events: HandwritingModuleEvent[] = [];
    module.subscribe((event) => events.push(event));
    await module.init({
      baselineManifestUrl: '/models/manifest.json',
      cnnTrainingRuntime: cnnTrainer,
    });

    await recordBalancedMilestone(module);

    const progress = events
      .filter((event): event is Extract<HandwritingModuleEvent, { type: 'training-progress' }> => event.type === 'training-progress')
      .map((event) => event.payload);
    assert.deepEqual(progress.map((event) => `${event.phase}:${event.status}`), [
      'feature:running',
      'feature:ready',
      'cnn:running',
      'cnn:running',
      'cnn:ready',
      'finalizing:running',
      'finalizing:ready',
    ]);
    assert.equal(progress.find((event) => event.message === 'cnn epoch 1/1 batch 1/1')?.progress, 72.5);
    assert.equal(progress[progress.length - 1]?.message, 'ready! v1');
  } finally {
    restoreFetch();
  }
});

test('feature rejection does not block accepted CNN personalization', async () => {
  const manifest = createDefaultBaselineManifest('/models');
  manifest.cnn.supportsTraining = true;
  const restoreFetch = installManifestFetch(manifest);
  const cnnArtifacts: PersonalizedCnnArtifacts = {
    checkpoint: new ArrayBuffer(4),
    inferenceModel: new ArrayBuffer(8),
    exportMetadata: null,
    metrics: { user_inputtedAccuracy: 1, implicitAccuracy: 1, overallAccuracy: 1 },
    stage: 'head-only',
    updatedAt: 999,
  };
  const cnnTrainer: CnnTrainingRuntimeLike = {
    isAvailable: () => true,
    getAvailability: () => ({ available: true, reasons: [] }),
    trainCandidate: async () => ({
      artifacts: cnnArtifacts,
      metrics: cnnArtifacts.metrics!,
      stage: 'head-only',
      accepted: true,
      rejectionReason: null,
    }),
  };

  try {
    const module = new BrowserHandwritingModule(new MemoryKeyValueStore());
    const events: HandwritingModuleEvent[] = [];
    module.subscribe((event) => events.push(event));
    await module.init({
      baselineManifestUrl: '/models/manifest.json',
      cnnTrainingRuntime: cnnTrainer,
    });

    const baseline = createDefaultBaselineManifest('/models');
    const importedState = createInitialTrainingState(baseline, 2_000_000);
    importedState.latestMetrics = {
      user_inputtedAccuracy: 1,
      implicitAccuracy: 1,
      overallAccuracy: 1,
    };
    importedState.personalizationGeneration = 4;
    importedState.latestSnapshotId = 'existing-feature';
    await module.importDevBundle(createExportBundle(createPersistedState(
      baseline,
      [],
      null,
      null,
      importedState,
    )));

    await recordBalancedMilestone(module);

    const state = module.getTrainingState();
    assert.equal(state.lastTrainingOutcome, 'accepted');
    assert.equal(state.personalizationGeneration, 5);
    assert.equal(state.latestSnapshotId, 'existing-feature');
    assert.equal(state.trainerStatus.cnnTrainingStatus, 'accepted');
    assert.equal(state.trainerStatus.personalizedCnnAvailable, true);
    assert.equal(state.trainerStatus.activeModelGeneration, 5);

    const progress = events
      .filter((event): event is Extract<HandwritingModuleEvent, { type: 'training-progress' }> => event.type === 'training-progress')
      .map((event) => event.payload);
    assert.equal(progress.some((event) => event.phase === 'feature' && event.status === 'rejected'), true);
    assert.equal(progress.some((event) => event.phase === 'cnn' && event.status === 'ready'), true);
    assert.equal(events.some((event) => event.type === 'training-completed' && event.payload.snapshot === null), true);
  } finally {
    restoreFetch();
  }
});

test('CNN rejection does not block accepted feature personalization', async () => {
  const manifest = createDefaultBaselineManifest('/models');
  manifest.cnn.supportsTraining = true;
  const restoreFetch = installManifestFetch(manifest);
  const cnnTrainer: CnnTrainingRuntimeLike = {
    isAvailable: () => true,
    getAvailability: () => ({ available: true, reasons: [] }),
    trainCandidate: async () => ({
      artifacts: null,
      metrics: { user_inputtedAccuracy: 0, implicitAccuracy: 0, overallAccuracy: 0 },
      stage: 'head-only',
      accepted: false,
      rejectionReason: 'mock cnn rejection',
    }),
  };

  try {
    const module = new BrowserHandwritingModule(new MemoryKeyValueStore());
    const events: HandwritingModuleEvent[] = [];
    module.subscribe((event) => events.push(event));
    await module.init({
      baselineManifestUrl: '/models/manifest.json',
      cnnTrainingRuntime: cnnTrainer,
    });

    await recordBalancedMilestone(module);

    const state = module.getTrainingState();
    assert.equal(state.lastTrainingOutcome, 'accepted');
    assert.equal(state.personalizationGeneration, 1);
    assert.equal(state.latestSnapshotId !== null, true);
    assert.equal(state.trainerStatus.cnnTrainingStatus, 'rejected');
    assert.equal(state.trainerStatus.personalizedCnnAvailable, false);

    const progress = events
      .filter((event): event is Extract<HandwritingModuleEvent, { type: 'training-progress' }> => event.type === 'training-progress')
      .map((event) => event.payload);
    assert.equal(progress.some((event) => event.phase === 'cnn' && event.status === 'rejected'), true);
    assert.equal(progress.some((event) => event.phase === 'finalizing' && event.status === 'ready'), true);
  } finally {
    restoreFetch();
  }
});

test('rejected training reports progress and does not replace active models', async () => {
  const restoreFetch = installManifestFetch();
  try {
    const module = new BrowserHandwritingModule(new MemoryKeyValueStore());
    const events: HandwritingModuleEvent[] = [];
    module.subscribe((event) => events.push(event));
    await module.init({
      baselineManifestUrl: '/models/manifest.json',
      snapshotBudgetBytes: 1,
    });

    await recordBalancedMilestone(module);

    const state = module.getTrainingState();
    assert.equal(state.lastTrainingOutcome, 'rejected');
    assert.equal(state.pendingUserInputtedSinceTraining, 0);
    assert.equal(state.personalizationGeneration, 0);
    assert.equal(state.latestSnapshotId, null);
    assert.equal(module.getDiagnostics().activeModelGeneration, 0);
    assert.equal(events.some((event) => (
      event.type === 'training-progress'
      && event.payload.status === 'rejected'
      && event.payload.phase === 'finalizing'
    )), true);
  } finally {
    restoreFetch();
  }
});

test('accepted personalized CNN replaces baseline inference session on next prediction', async () => {
  const originalCreate = InferenceSession.create;
  const createSources: unknown[] = [];
  (InferenceSession as unknown as { create: (source: unknown) => Promise<unknown> }).create = async (source: unknown) => {
    createSources.push(source);
    return {
      inputNames: ['input'],
      outputNames: ['output'],
      run: async () => ({
        output: {
          data: Array.from({ length: 26 }, (_, index) => index === 0 ? 10 : 0),
        },
      }),
    };
  };

  const manifest = createDefaultBaselineManifest('/models');
  manifest.version = 'baseline-test';
  manifest.cnn.inferenceUrl = '/models/cnn.onnx';
  manifest.cnn.supportsTraining = true;
  const restoreFetch = installManifestFetch(manifest);
  const cnnArtifacts: PersonalizedCnnArtifacts = {
    checkpoint: new ArrayBuffer(4),
    inferenceModel: new ArrayBuffer(8),
    exportMetadata: null,
    metrics: { user_inputtedAccuracy: 1, implicitAccuracy: 1, overallAccuracy: 1 },
    stage: 'head-only',
    updatedAt: 789,
  };
  const cnnTrainer: CnnTrainingRuntimeLike = {
    isAvailable: () => true,
    trainCandidate: async () => ({
      artifacts: cnnArtifacts,
      metrics: cnnArtifacts.metrics!,
      stage: 'head-only',
      accepted: true,
      rejectionReason: null,
    }),
  };

  try {
    const module = new BrowserHandwritingModule(new MemoryKeyValueStore());
    await module.init({
      baselineManifestUrl: '/models/manifest.json',
      cnnTrainingRuntime: cnnTrainer,
    });
    await module.predict([[{ x: 0, y: 0, t: 0 }]]);
    assert.equal(module.getDiagnostics().cnnSessionKey, 'baseline:baseline-test');
    assert.equal(createSources[0], '/models/cnn.onnx');

    await recordBalancedMilestone(module);
    assert.equal(module.getDiagnostics().cnnSessionKey, null);
    assert.equal(module.getTrainingState().trainerStatus.personalizedCnnAvailable, true);
    assert.equal(module.getTrainingState().trainerStatus.cnnInferenceSource, 'personalized');
    assert.equal(module.getTrainingState().personalizationGeneration, 1);

    await module.predict([[{ x: 0, y: 1, t: 1 }]]);
    assert.equal(module.getDiagnostics().cnnSessionKey, 'personalized:789');
    assert.equal(createSources[1], cnnArtifacts.inferenceModel);
  } finally {
    (InferenceSession as unknown as { create: typeof originalCreate }).create = originalCreate;
    restoreFetch();
  }
});

test('predict auto-confirms strong k-NN and CNN consensus even if feature disagrees', async () => {
  const originalCreate = InferenceSession.create;
  (InferenceSession as unknown as { create: (source: unknown) => Promise<unknown> }).create = async () => ({
    inputNames: ['input'],
    outputNames: ['output'],
    run: async () => ({
      output: {
        data: Array.from({ length: 26 }, (_, index) => index === 0 ? 10 : 0),
      },
    }),
  });

  const manifest = createDefaultBaselineManifest('/models');
  manifest.version = 'consensus-test';
  manifest.cnn.inferenceUrl = '/models/cnn.onnx';
  manifest.featureClassifier = {
    id: 'baseline-feature',
    version: 'baseline',
    createdAt: 1,
    centroids: [
      { label: 'A', centroid: new Array(30).fill(100), count: 1 },
      { label: 'B', centroid: new Array(30).fill(0), count: 1 },
    ],
    labelMap: ['A', 'B'],
    metrics: { user_inputtedAccuracy: 1, implicitAccuracy: 1, overallAccuracy: 1 },
    datasetSize: 2,
    readyLetters: ['A', 'B'],
    reason: 'milestone-50',
  };
  const restoreFetch = installManifestFetch(manifest);
  const strokes = [[{ x: 0, y: 0, t: 0 }, { x: 1, y: 1, t: 1 }, { x: 2, y: 2, t: 2 }]];

  try {
    const module = new BrowserHandwritingModule(new MemoryKeyValueStore());
    await module.init({
      baselineManifestUrl: '/models/manifest.json',
    });

    await module.recordAcceptedSample({
      label: 'A',
      acceptance: 'user_inputted',
      source: 'test',
      createdAt: 10,
      strokes,
    });

    const prediction = await module.predict(strokes);
    assert.equal(prediction.status, 'confirmed');
    assert.equal(prediction.chosenLabel, 'A');
  } finally {
    (InferenceSession as unknown as { create: typeof originalCreate }).create = originalCreate;
    restoreFetch();
  }
});

test('training toast persists, mutates in place, and can be closed', () => {
  const baseline = createDefaultBaselineManifest('/models');
  const trainingState = createInitialTrainingState(baseline, 2_000_000);
  useGameStore.setState({ trainingToast: null });

  const started = {
    phase: 'feature' as const,
    status: 'running' as const,
    progress: 15,
    message: 'svm/feature training...',
    generation: 1,
    trainingState,
  };
  useGameStore.getState().updateTrainingToast(started);
  const first = useGameStore.getState().trainingToast;
  assert.equal(first?.id, 'training-progress');
  assert.equal(first?.progress, 15);
  assert.equal(first?.cnn, 'not started');

  useGameStore.getState().updateTrainingToast({
    ...started,
    status: 'rejected',
    progress: 100,
    message: 'rejected: feature regression gate',
  });
  const rejected = useGameStore.getState().trainingToast;
  assert.equal(rejected?.feature, 'rejected: feature regression gate');
  assert.equal(rejected?.cnn, 'not run');

  useGameStore.setState({ trainingToast: null });
  useGameStore.getState().updateTrainingToast(started);

  useGameStore.getState().updateTrainingToast({
    ...started,
    phase: 'finalizing',
    status: 'ready',
    progress: 100,
    message: 'ready! v1',
  });
  const completed = useGameStore.getState().trainingToast;
  assert.equal(completed?.id, first?.id);
  assert.equal(completed?.progress, 100);
  assert.equal(completed?.finalizing, 'ready! v1');

  useGameStore.getState().removeTrainingToast();
  assert.equal(useGameStore.getState().trainingToast, null);
});
