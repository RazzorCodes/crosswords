import { useGameStore } from '../store/useGameStore';
import { useHandwritingStore } from '../store/useHandwritingStore';
import { handwritingModule } from './handwriting';
import type { SampleAcceptance } from './handwriting';
import type { StrokeInput } from './recognizers/types';

const PENDING_WINDOW_MS = 2000;
const pendingTimers = new Map<string, ReturnType<typeof setTimeout>>();

export function cellKey(x: number, y: number) {
  return `${x}:${y}`;
}

function now() {
  return Date.now();
}

function getPendingInk(cell: string) {
  return useHandwritingStore.getState().pendingInk.find((entry) => entry.cellKey === cell) ?? null;
}

async function recordImplicitSample(
  cell: string,
  label: string,
  strokes: StrokeInput,
  source: string,
): Promise<void> {
  const grid = useGameStore.getState().grid;
  const [xText, yText] = cell.split(':');
  const x = Number(xText);
  const y = Number(yText);
  const current = grid?.cells[y]?.[x]?.userInput ?? '';

  if (current === label) {
    await handwritingModule.recordAcceptedSample({
      label,
      strokes,
      acceptance: 'implicit',
      source,
    });
  }

  useHandwritingStore.getState().clearPendingInk(cell);
}

async function recordAcceptedSample(
  label: string,
  strokes: StrokeInput,
  source: string,
  acceptance: SampleAcceptance,
): Promise<void> {
  await handwritingModule.recordAcceptedSample({
    label,
    strokes,
    acceptance,
    source,
  });
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
  acceptance?: SampleAcceptance;
}) {
  const cell = cellKey(args.x, args.y);
  cancelPendingSubmission(cell);

  if (args.acceptance === 'user_inputted') {
    void recordAcceptedSample(args.label, args.strokes, args.source, 'user_inputted');
    return;
  }

  const expiresAt = now() + PENDING_WINDOW_MS;
  useHandwritingStore.getState().upsertPendingInk({
    cellKey: cell,
    label: args.label,
    source: args.source,
    expiresAt,
    strokes: args.strokes,
  });

  const timer = setTimeout(() => {
    pendingTimers.delete(cell);
    void recordImplicitSample(cell, args.label, args.strokes, args.source);
  }, PENDING_WINDOW_MS);
  pendingTimers.set(cell, timer);
}

export function acceptPendingCorrection(args: {
  x: number;
  y: number;
  label: string;
  source: string;
}) {
  const cell = cellKey(args.x, args.y);
  const pending = getPendingInk(cell);
  cancelPendingSubmission(cell);
  if (!pending) {
    return;
  }
  void recordAcceptedSample(args.label, pending.strokes, args.source, 'user_inputted');
}
