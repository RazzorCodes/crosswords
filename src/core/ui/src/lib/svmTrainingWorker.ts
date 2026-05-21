import type { LabSample, LabSvmSnapshot, SnapshotMetrics } from './types.ts';
import { initWasmCore, predictSvm, trainSvm } from './wasmCore.ts';

type RequestMessage = {
  type: 'train';
  ledger: LabSample[];
  readyLetters: string[];
  wasmUrl: string;
};

type ResponseMessage =
  | { type: 'progress'; message: string; progress: number }
  | { type: 'completed'; snapshot: LabSvmSnapshot | null; rejectionReason: string | null; elapsedMs: number }
  | { type: 'failed'; reason: string };

function post(message: ResponseMessage): void {
  self.postMessage(message);
}

function computeMetricsWithProgress(snapshot: LabSvmSnapshot, samples: LabSample[]): SnapshotMetrics {
  if (samples.length === 0) {
    return { user_inputtedAccuracy: 0, implicitAccuracy: 0, overallAccuracy: 0 };
  }

  let correct = 0;
  const progressEvery = Math.max(1, Math.floor(samples.length / 20));
  for (let index = 0; index < samples.length; index += 1) {
    const sample = samples[index];
    const scores = predictSvm(sample.features, snapshot);
    const top = Object.entries(scores).sort((left, right) => right[1] - left[1])[0]?.[0] ?? null;
    if (top === sample.label) {
      correct += 1;
    }
    if ((index + 1) % progressEvery === 0 || index + 1 === samples.length) {
      const evaluationProgress = (index + 1) / samples.length;
      post({
        type: 'progress',
        progress: 70 + Math.round(evaluationProgress * 28),
        message: `evaluating SVM snapshot ${index + 1}/${samples.length}`,
      });
    }
  }

  const overallAccuracy = correct / samples.length;
  return {
    user_inputtedAccuracy: overallAccuracy,
    implicitAccuracy: 0,
    overallAccuracy,
  };
}

self.onmessage = (event: MessageEvent<RequestMessage>) => {
  const request = event.data;
  if (request.type !== 'train') {
    return;
  }

  void (async () => {
    const startedAt = performance.now();
    try {
      post({ type: 'progress', progress: 5, message: 'initializing Rust core in worker' });
      await initWasmCore(request.wasmUrl);
      post({ type: 'progress', progress: 10, message: `training SVM on ${request.ledger.length} samples` });
      const trained = trainSvm(request.ledger, request.readyLetters);
      if (!trained) {
        post({
          type: 'completed',
          snapshot: null,
          rejectionReason: 'need at least two ready letters with enough samples',
          elapsedMs: performance.now() - startedAt,
        });
        return;
      }
      post({
        type: 'progress',
        progress: 70,
        message: `Rust training complete with ${trained.supportCount} support vectors`,
      });
      const metrics = computeMetricsWithProgress(trained, request.ledger);
      post({
        type: 'completed',
        snapshot: { ...trained, metrics },
        rejectionReason: null,
        elapsedMs: performance.now() - startedAt,
      });
    } catch (error) {
      post({ type: 'failed', reason: error instanceof Error ? error.message : 'SVM training failed' });
    }
  })();
};
