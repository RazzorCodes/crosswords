import type { LabCnnArtifacts, LabState } from './types.ts';

const STORAGE_KEY = 'handwriting-core-lab-state-v1';

interface SerializedCnnArtifacts extends Omit<LabCnnArtifacts, 'checkpoint' | 'inferenceModel'> {
  checkpoint?: number[] | null;
  inferenceModel?: number[] | null;
}

interface SerializedLabState extends Omit<LabState, 'cnnArtifacts'> {
  cnnArtifacts: SerializedCnnArtifacts | null;
}

export const emptyLabState: LabState = {
  ledger: [],
  svmSnapshot: null,
  cnnArtifacts: null,
  baselineManifest: null,
  latestMetrics: null,
};

function bufferToArray(buffer: ArrayBuffer | null | undefined): number[] | null {
  return buffer ? Array.from(new Uint8Array(buffer)) : null;
}

function arrayToBuffer(value: number[] | null | undefined): ArrayBuffer | null {
  if (!value) {
    return null;
  }
  return new Uint8Array(value).buffer;
}

function serialize(state: LabState): SerializedLabState {
  return {
    ...state,
    cnnArtifacts: state.cnnArtifacts
      ? {
          ...state.cnnArtifacts,
          checkpoint: bufferToArray(state.cnnArtifacts.checkpoint),
          inferenceModel: bufferToArray(state.cnnArtifacts.inferenceModel),
        }
      : null,
  };
}

function deserialize(value: SerializedLabState): LabState {
  return {
    ledger: Array.isArray(value.ledger) ? value.ledger : [],
    svmSnapshot: value.svmSnapshot ?? null,
    baselineManifest: value.baselineManifest ?? null,
    latestMetrics: value.latestMetrics ?? null,
    cnnArtifacts: value.cnnArtifacts
      ? {
          ...value.cnnArtifacts,
          checkpoint: arrayToBuffer(value.cnnArtifacts.checkpoint),
          inferenceModel: arrayToBuffer(value.cnnArtifacts.inferenceModel),
        }
      : null,
  };
}

export function loadLabState(storage: Pick<Storage, 'getItem'> = localStorage): LabState {
  const raw = storage.getItem(STORAGE_KEY);
  if (!raw) {
    return { ...emptyLabState };
  }
  try {
    return deserialize(JSON.parse(raw) as SerializedLabState);
  } catch {
    return { ...emptyLabState };
  }
}

export function saveLabState(state: LabState, storage: Pick<Storage, 'setItem'> = localStorage): void {
  storage.setItem(STORAGE_KEY, JSON.stringify(serialize(state)));
}

export function resetLabState(storage: Pick<Storage, 'removeItem'> = localStorage): LabState {
  storage.removeItem(STORAGE_KEY);
  return { ...emptyLabState };
}

export function estimateJsonBytes(value: unknown): number {
  return new TextEncoder().encode(JSON.stringify(value)).byteLength;
}
