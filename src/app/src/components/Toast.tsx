import { useEffect } from 'react';
import { useGameStore } from '../store/useGameStore';
import type { EngineResult } from '../utils/recognizers/types';

export function ToastContainer() {
  const { toasts, trainingToast, removeToast, removeTrainingToast } = useGameStore();

  return (
    <div className="fixed top-24 right-4 z-[100] flex flex-col gap-2 pointer-events-none">
      {trainingToast && (
        <TrainingToastItem toast={trainingToast} onRemove={removeTrainingToast} />
      )}
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onRemove={removeToast} />
      ))}
    </div>
  );
}

function TrainingToastItem({
  toast,
  onRemove,
}: {
  toast: NonNullable<ReturnType<typeof useGameStore.getState>['trainingToast']>;
  onRemove: () => void;
}) {
  const isTerminal = toast.status === 'ready' || toast.status === 'rejected';

  useEffect(() => {
    if (!isTerminal) {
      return undefined;
    }
    const timer = setTimeout(onRemove, toast.status === 'ready' ? 1600 : 2800);
    return () => clearTimeout(timer);
  }, [isTerminal, onRemove, toast.status, toast.generation]);

  return (
    <div className="pointer-events-auto min-w-[240px] rounded-lg border border-slate-700 bg-slate-900/92 p-3 shadow-xl backdrop-blur-md">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs font-black uppercase tracking-widest text-slate-500">Training progress</span>
        <span className="text-xs font-bold text-emerald-300">{Math.round(toast.progress)}%</span>
      </div>
      <div className="mb-3 h-1.5 overflow-hidden rounded-full bg-slate-800">
        <div
          className="h-full rounded-full bg-emerald-400 transition-all duration-300"
          style={{ width: `${Math.min(100, Math.max(0, toast.progress))}%` }}
        />
      </div>
      <div className="space-y-1 text-xs">
        <div className="flex items-center justify-between gap-3">
          <span className="font-medium text-slate-500">svm</span>
          <span className="min-w-0 truncate text-slate-200">{toast.feature}</span>
        </div>
        <div className="flex items-center justify-between gap-3">
          <span className="font-medium text-slate-500">cnn</span>
          <span className="min-w-0 truncate text-slate-200">{toast.cnn}</span>
        </div>
        {toast.finalizing && (
          <div className="flex items-center justify-between gap-3 border-t border-slate-800 pt-1.5">
            <span className="font-medium text-slate-500">final</span>
            <span className="min-w-0 truncate text-slate-200">{toast.finalizing}</span>
          </div>
        )}
      </div>
    </div>
  );
}

function ToastItem({ toast, onRemove }: { toast: any; onRemove: (id: string) => void }) {
  useEffect(() => {
    const timer = setTimeout(() => onRemove(toast.id), 5000);
    return () => clearTimeout(timer);
  }, [toast.id, onRemove]);

  return (
    <div className="bg-slate-900/90 backdrop-blur-md border border-slate-700 rounded-lg p-3 shadow-xl pointer-events-auto min-w-[200px] animate-in slide-in-from-right-full">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-black text-slate-500 uppercase tracking-widest">Recognition</span>
        <span className="text-xs font-bold text-blue-400">{Math.round(toast.confidence * 100)}%</span>
      </div>
      
      <div className="flex items-center gap-3 mb-3">
        <div className="h-10 w-10 bg-blue-600 rounded flex items-center justify-center text-xl font-black text-white">
          {toast.char}
        </div>
        <div className="text-sm font-medium text-slate-300">
          Discovered '{toast.char}'
        </div>
      </div>

      <div className="space-y-1.5 border-t border-slate-800 pt-2">
        {toast.engines.map((eng: EngineResult) => {
          const hasPrediction = eng.char !== null && eng.score !== null;
          const score = eng.score ?? 0;
          return (
            <div key={eng.name} className="flex items-center justify-between text-[10px]">
              <span className="text-slate-500 font-medium">{eng.name}</span>
              <div className="flex items-center gap-2">
                {hasPrediction ? (
                  <>
                    <span className="text-slate-300 font-bold">{eng.char}</span>
                    <span className="text-slate-600">({Math.round(score * 100)}%)</span>
                  </>
                ) : (
                  <span className="text-slate-600">{eng.detail || eng.status}</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
