import { useHandwritingStore } from '../store/useHandwritingStore';

function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  return `${Math.round(value * 100)}%`;
}

function formatTime(timestamp: number | null) {
  if (!timestamp) {
    return 'Not yet';
  }
  return new Date(timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function HandwritingPanel() {
  const { trainMode, trainingState, moduleReady, lastEvent } = useHandwritingStore();

  if (!trainMode) {
    return null;
  }

  return (
    <aside className="absolute top-24 right-4 z-[90] w-80 rounded-2xl border border-amber-500/30 bg-slate-950/92 p-4 text-left shadow-2xl backdrop-blur-md">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <div className="text-[10px] font-black uppercase tracking-[0.25em] text-amber-400">Train Mode</div>
          <h2 className="text-sm font-black text-white">Handwriting Trainer</h2>
        </div>
        <div className={`rounded-full border px-2 py-1 text-[10px] font-bold ${moduleReady ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-200' : 'border-slate-700 bg-slate-900 text-slate-400'}`}>
          {moduleReady ? 'LOCAL READY' : 'INITIALIZING'}
        </div>
      </div>

      {!trainingState ? (
        <div className="rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-4 text-xs text-slate-400">
          Booting local handwriting storage and baseline artifacts.
        </div>
      ) : (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2">
              <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Accepted</div>
              <div className="text-lg font-black text-white">{trainingState.totalAcceptedSamples}</div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2">
              <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Ready Letters</div>
              <div className="text-lg font-black text-white">{trainingState.readyLetters.length}</div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2">
              <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Next Milestone</div>
              <div className="text-lg font-black text-white">{trainingState.nextMilestone ?? 'done'}</div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2">
              <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Pending User</div>
              <div className="text-lg font-black text-white">{trainingState.pendingUserInputtedSinceTraining}</div>
            </div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-3 text-xs text-slate-300">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Latest Snapshot</span>
              <span className={`rounded-full border px-2 py-0.5 text-[10px] font-bold ${trainingState.lastTrainingOutcome === 'accepted' ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-200' : trainingState.lastTrainingOutcome === 'rejected' ? 'border-rose-500/30 bg-rose-500/10 text-rose-200' : 'border-slate-700 bg-slate-800 text-slate-400'}`}>
                {trainingState.lastTrainingOutcome.toUpperCase()}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-500">User accuracy</span>
              <span>{formatPercent(trainingState.latestMetrics?.user_inputtedAccuracy)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-500">Implicit accuracy</span>
              <span>{formatPercent(trainingState.latestMetrics?.implicitAccuracy)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-500">Overall accuracy</span>
              <span>{formatPercent(trainingState.latestMetrics?.overallAccuracy)}</span>
            </div>
            <div className="mt-2 flex items-center justify-between border-t border-slate-800 pt-2">
              <span className="text-slate-500">Last completed</span>
              <span>{formatTime(trainingState.lastCompletedTrainingAt)}</span>
            </div>
            {trainingState.lastRejectedReason && (
              <div className="mt-2 rounded-lg border border-rose-500/20 bg-rose-500/10 px-2 py-2 text-[11px] text-rose-200">
                {trainingState.lastRejectedReason}
              </div>
            )}
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-3 text-xs text-slate-300">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Artifacts</span>
              <span className="text-[10px] font-bold text-slate-400">{trainingState.persistedBytes} bytes</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-500">Baseline</span>
              <span>{trainingState.baselineVersion}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-500">CNN ready</span>
              <span>{trainingState.trainerStatus.personalizedCnnAvailable ? 'personalized' : 'baseline'}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-500">Event</span>
              <span>{lastEvent ?? 'idle'}</span>
            </div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-3 text-xs text-slate-300">
            <div className="mb-2 text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">Recent Samples</div>
            <div className="max-h-40 space-y-2 overflow-y-auto pr-1">
              {trainingState.recentAcceptedSamples.length === 0 ? (
                <div className="text-slate-500">No accepted handwriting samples yet.</div>
              ) : trainingState.recentAcceptedSamples.map((sample) => (
                <div
                  key={sample.id}
                  className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950/70 px-2 py-2"
                >
                  <div>
                    <div className="text-base font-black text-white">{sample.label}</div>
                    <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500">{sample.source}</div>
                  </div>
                  <div className={`rounded-full border px-2 py-0.5 text-[10px] font-bold ${sample.acceptance === 'user_inputted' ? 'border-blue-500/30 bg-blue-500/10 text-blue-200' : 'border-slate-700 bg-slate-800 text-slate-300'}`}>
                    {sample.acceptance}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}
