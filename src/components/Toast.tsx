import { useEffect } from 'react';
import { useGameStore } from '../store/useGameStore';

export function ToastContainer() {
  const { toasts, removeToast } = useGameStore();

  return (
    <div className="fixed top-24 right-4 z-[100] flex flex-col gap-2 pointer-events-none">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onRemove={removeToast} />
      ))}
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
        {toast.engines.map((eng: any) => (
          <div key={eng.name} className="flex items-center justify-between text-[10px]">
            <span className="text-slate-500 font-medium">{eng.name}</span>
            <div className="flex items-center gap-2">
              <span className="text-slate-300 font-bold">{eng.char || '?'}</span>
              <span className="text-slate-600">({Math.round(eng.score * 100)}%)</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
