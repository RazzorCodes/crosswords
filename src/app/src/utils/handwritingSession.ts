import { useGameStore } from '../store/useGameStore';
import {
  TeacherQueueItem,
  useHandwritingStore,
} from '../store/useHandwritingStore';
import { deleteSample, submitSample } from './api';
import { knnRecognizer } from './recognizers/knn';
import { StrokeInput } from './recognizers/types';

const PENDING_WINDOW_MS = 2000;
const pendingTimers = new Map<string, ReturnType<typeof setTimeout>>();

export function cellKey(x: number, y: number) {
  return `${x}:${y}`;
}

function now() {
  return Date.now();
}

function createQueueItem(sampleId: string, label: string, source: string): TeacherQueueItem {
  return {
    localId: sampleId,
    sampleId,
    label,
    source,
    createdAt: now(),
  };
}

async function submitQueuedHighQualitySample(
  label: string,
  strokes: StrokeInput,
  source: string,
): Promise<void> {
  const response = await submitSample({
    label,
    strokes,
    storedAs: 'high_quality',
    source,
    mode: 'train',
  });
  if (!response) {
    return;
  }

  knnRecognizer.addExample(strokes, label, response.id);
  useHandwritingStore.getState().addQueueItem(createQueueItem(response.id, label, source));
}

async function submitRegularSample(cell: string, label: string, strokes: StrokeInput, source: string) {
  const grid = useGameStore.getState().grid;
  const [xText, yText] = cell.split(':');
  const x = Number(xText);
  const y = Number(yText);
  const current = grid?.cells[y]?.[x]?.userInput ?? '';

  if (current !== label) {
    useHandwritingStore.getState().clearPendingInk(cell);
    return;
  }

  const response = await submitSample({
    label,
    strokes,
    storedAs: 'regular',
    source,
    mode: 'play',
  });
  
  if (response) {
    knnRecognizer.addExample(strokes, label, response.id);
  }

  useHandwritingStore.getState().clearPendingInk(cell);
}

export function cancelPendingSubmission(cell: string) {
  const timer = pendingTimers.get(cell);
  if (timer) {
    clearTimeout(timer);
    pendingTimers.delete(cell);
  }
  useHandwritingStore.getState().clearPendingInk(cell);
}

export function cancelAllPendingSubmissions() {
  for (const timer of pendingTimers.values()) {
    clearTimeout(timer);
  }
  pendingTimers.clear();
  useHandwritingStore.getState().clearAllPendingInk();
}

export function finalizeHandwritingSample(args: {
  x: number;
  y: number;
  label: string;
  strokes: StrokeInput;
  source: string;
}) {
  const cell = cellKey(args.x, args.y);
  cancelPendingSubmission(cell);

  if (useHandwritingStore.getState().trainMode) {
    void submitQueuedHighQualitySample(args.label, args.strokes, args.source);
    return;
  }

  const expiresAt = now() + PENDING_WINDOW_MS;
  useHandwritingStore.getState().upsertPendingInk({
    cellKey: cell,
    label: args.label,
    source: args.source,
    expiresAt,
  });

  const timer = setTimeout(() => {
    pendingTimers.delete(cell);
    void submitRegularSample(cell, args.label, args.strokes, args.source);
  }, PENDING_WINDOW_MS);
  pendingTimers.set(cell, timer);
}

export async function deleteTeacherQueueItem(localId: string) {
  const store = useHandwritingStore.getState();
  const item = store.queueItems.find((entry) => entry.localId === localId);
  if (!item) {
    return;
  }

  const deleted = await deleteSample(item.sampleId);
  if (!deleted) {
    return;
  }

  store.removeQueueItem(localId);
  knnRecognizer.removeExample(item.sampleId);
}
