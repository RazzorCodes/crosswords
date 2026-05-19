import { useEffect, useState } from 'react';
import { deleteTeacherQueueItem } from '../utils/handwritingSession';
import { useHandwritingStore } from '../store/useHandwritingStore';

export function HandwritingPanel() {
  const { trainMode, queueItems, pendingInk, teacherStatus } = useHandwritingStore();
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 200);
    return () => window.clearInterval(timer);
  }, []);

  if (trainMode) {
    return (
      <aside className="absolute top-24 right-4 z-[90] w-72 rounded-2xl border border-amber-500/30 bg-slate-950/92 p-4 text-left shadow-2xl backdrop-blur-md">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <div className="text-[10px] font-black uppercase tracking-[0.25em] text-amber-400">Train Mode</div>
            <h2 className="text-sm font-black text-white">Teacher Queue</h2>
          </div>
          <div className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[10px] font-bold text-amber-200">
            {queueItems.length} items
          </div>
        </div>

        <div className="max-h-80 space-y-2 overflow-y-auto pr-1">
          {queueItems.length === 0 ? (
            <div className="rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-4 text-xs text-slate-400">
              Handwriting finals will appear here and be stored as high-quality immediately.
            </div>
          ) : queueItems.map((item) => (
            <div
              key={item.localId}
              className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2"
            >
              <div>
                <div className="text-lg font-black text-white">{item.label}</div>
                <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500">{item.source}</div>
              </div>
              <button
                type="button"
                className="rounded-full border border-rose-500/30 bg-rose-500/10 px-2 py-1 text-[10px] font-bold uppercase tracking-[0.2em] text-rose-300 transition-colors hover:bg-rose-500/20"
                onClick={() => { void deleteTeacherQueueItem(item.localId); }}
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      </aside>
    );
  }

  const pendingCount = pendingInk.length;
  const nextExpiry = pendingInk.length > 0
    ? Math.max(0, Math.ceil((Math.min(...pendingInk.map((item) => item.expiresAt)) - now) / 1000))
    : 0;
  const shouldShowTeacher = teacherStatus?.enabled || pendingCount > 0;

  if (!shouldShowTeacher) {
    return null;
  }

  return (
    <aside className="absolute top-24 right-4 z-[90] w-72 rounded-2xl border border-slate-700 bg-slate-950/92 p-4 text-left shadow-2xl backdrop-blur-md">
      {pendingCount > 0 && (
        <div className="mb-3 rounded-xl border border-blue-500/25 bg-blue-500/10 px-3 py-3">
          <div className="text-[10px] font-black uppercase tracking-[0.25em] text-blue-300">Undo Window</div>
          <div className="mt-1 text-sm font-semibold text-white">
            {pendingCount} handwriting sample{pendingCount === 1 ? '' : 's'} pending for {nextExpiry}s
          </div>
          <div className="mt-1 text-xs text-slate-400">
            Delete, overwrite, or keyboard-correct the cell before the timer ends to cancel submission.
          </div>
        </div>
      )}

      {teacherStatus && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-3">
          <div className="text-[10px] font-black uppercase tracking-[0.25em] text-emerald-300">External Teacher</div>
          <div className="mt-1 text-sm font-semibold text-white">
            {teacherStatus.active ? 'Active' : 'Inactive'}
          </div>
          <div className="mt-1 text-xs text-slate-400">
            HQ share {Math.round(teacherStatus.hq_share * 100)}% / {Math.round(teacherStatus.target_share * 100)}%
          </div>
          <div className="mt-1 text-xs text-slate-500">
            {teacherStatus.active
              ? `${teacherStatus.pending_queue_size} samples waiting for LiteLLM review`
              : teacherStatus.reason.replace(/_/g, ' ')}
          </div>
        </div>
      )}
    </aside>
  );
}
