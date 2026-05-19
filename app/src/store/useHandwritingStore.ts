import { create } from 'zustand';
import { TeacherStatus } from '../utils/api';

const QUEUE_STORAGE_KEY = 'crosswords_teacher_queue_v1';
const TRAIN_MODE = import.meta.env.VITE_TRAIN_MODE === 'true';

export interface TeacherQueueItem {
  localId: string;
  sampleId: string;
  label: string;
  source: string;
  createdAt: number;
}

export interface PendingInkSubmission {
  cellKey: string;
  label: string;
  source: string;
  expiresAt: number;
}

interface HandwritingState {
  trainMode: boolean;
  queueItems: TeacherQueueItem[];
  pendingInk: PendingInkSubmission[];
  teacherStatus: TeacherStatus | null;
  setQueueItems: (items: TeacherQueueItem[]) => void;
  addQueueItem: (item: TeacherQueueItem) => void;
  removeQueueItem: (localId: string) => void;
  upsertPendingInk: (item: PendingInkSubmission) => void;
  clearPendingInk: (cellKey: string) => void;
  clearAllPendingInk: () => void;
  setTeacherStatus: (status: TeacherStatus | null) => void;
}

function loadQueueItems(): TeacherQueueItem[] {
  if (typeof window === 'undefined') return [];
  const stored = window.localStorage.getItem(QUEUE_STORAGE_KEY);
  if (!stored) return [];

  try {
    const parsed = JSON.parse(stored);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item): item is TeacherQueueItem => (
      item &&
      typeof item.localId === 'string' &&
      typeof item.sampleId === 'string' &&
      typeof item.label === 'string' &&
      typeof item.source === 'string' &&
      typeof item.createdAt === 'number'
    ));
  } catch (error) {
    console.error('Failed to load teacher queue', error);
    return [];
  }
}

function persistQueue(items: TeacherQueueItem[]) {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(QUEUE_STORAGE_KEY, JSON.stringify(items));
}

export const useHandwritingStore = create<HandwritingState>((set) => ({
  trainMode: TRAIN_MODE,
  queueItems: loadQueueItems(),
  pendingInk: [],
  teacherStatus: null,
  setQueueItems: (items) => {
    persistQueue(items);
    set({ queueItems: items });
  },
  addQueueItem: (item) => set((state) => {
    const queueItems = [item, ...state.queueItems];
    persistQueue(queueItems);
    return { queueItems };
  }),
  removeQueueItem: (localId) => set((state) => {
    const queueItems = state.queueItems.filter((item) => item.localId !== localId);
    persistQueue(queueItems);
    return { queueItems };
  }),
  upsertPendingInk: (item) => set((state) => {
    const pendingInk = [
      item,
      ...state.pendingInk.filter((entry) => entry.cellKey !== item.cellKey),
    ];
    return { pendingInk };
  }),
  clearPendingInk: (cellKey) => set((state) => ({
    pendingInk: state.pendingInk.filter((entry) => entry.cellKey !== cellKey),
  })),
  clearAllPendingInk: () => set({ pendingInk: [] }),
  setTeacherStatus: (teacherStatus) => set({ teacherStatus }),
}));
