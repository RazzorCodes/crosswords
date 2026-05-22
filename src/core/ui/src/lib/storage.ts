import type { LabState } from './types.ts';

const DB_NAME = 'handwriting-lab-db';
const STORE_NAME = 'state';
const STATE_KEY = 'current-state';

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export const emptyLabState: LabState = {
  ledger: [],
  svmSnapshot: null,
  cnnArtifacts: null,
  baselineManifest: null,
  latestMetrics: null,
};

export async function loadLabState(): Promise<LabState> {
  try {
    const db = await openDb();
    return new Promise((resolve) => {
      const transaction = db.transaction(STORE_NAME, 'readonly');
      const request = transaction.objectStore(STORE_NAME).get(STATE_KEY);
      request.onsuccess = () => {
        const value = request.result as LabState;
        if (!value) {
          resolve({ ...emptyLabState });
        } else {
          // Ensure ledger is always an array
          resolve({
            ...value,
            ledger: Array.isArray(value.ledger) ? value.ledger : [],
          });
        }
      };
      request.onerror = () => resolve({ ...emptyLabState });
    });
  } catch (error) {
    console.error('Failed to load state from IndexedDB:', error);
    return { ...emptyLabState };
  }
}

export async function saveLabState(state: LabState): Promise<void> {
  try {
    const db = await openDb();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, 'readwrite');
      const request = transaction.objectStore(STORE_NAME).put(state, STATE_KEY);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error('Failed to save state to IndexedDB:', error);
  }
}

export async function resetLabState(): Promise<LabState> {
  try {
    const db = await openDb();
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    transaction.objectStore(STORE_NAME).delete(STATE_KEY);
  } catch (error) {
    console.error('Failed to reset state in IndexedDB:', error);
  }
  return { ...emptyLabState };
}

export function estimateJsonBytes(value: any): number {
  if (!value) return 0;
  if (value instanceof ArrayBuffer) return value.byteLength;
  if (ArrayBuffer.isView(value)) return value.byteLength;
  if (typeof value === 'object') {
    let size = 0;
    for (const key in value) {
      if (Object.prototype.hasOwnProperty.call(value, key)) {
        size += key.length + estimateJsonBytes(value[key]);
      }
    }
    return size;
  }
  try {
    return new TextEncoder().encode(JSON.stringify(value)).byteLength;
  } catch {
    return 0;
  }
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}
