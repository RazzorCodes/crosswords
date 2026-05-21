import type {
  AcceptedSampleRecord,
  BaselineArtifactManifest,
  PersonalizedCnnArtifacts,
} from './types';
import { trainCnnCandidateDirect } from './cnnTrainingRuntime';

interface CnnTrainingWorkerRequest {
  manifest: BaselineArtifactManifest;
  training: AcceptedSampleRecord[];
  holdout: AcceptedSampleRecord[];
  previousArtifacts: PersonalizedCnnArtifacts | null;
}

self.onmessage = (event: MessageEvent<CnnTrainingWorkerRequest>) => {
  const { manifest, training, holdout, previousArtifacts } = event.data;
  void trainCnnCandidateDirect(manifest, training, holdout, previousArtifacts, {
    onProgress: (progress) => {
      self.postMessage({ type: 'progress', payload: progress });
    },
  })
    .then((result) => {
      self.postMessage({ type: 'completed', payload: result });
    })
    .catch((error) => {
      self.postMessage({
        type: 'failed',
        reason: error instanceof Error ? error.message : 'CNN training worker failed.',
      });
    });
};
