import { useHandwritingStore, type DiagnosticPrediction } from '../store/useHandwritingStore';

const HIGH_CONFIDENCE_THRESHOLD = 0.75;

function formatValue(value: number | null): string {
  if (value === null || Number.isNaN(value)) {
    return '--';
  }
  return `${Math.round(value)}`;
}

function formatPercent(value: number | null): string {
  if (value === null || Number.isNaN(value)) {
    return '--';
  }
  return `${Math.round(value * 100)}%`;
}

function average(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function engineAverage(history: DiagnosticPrediction[], engineName: string): number | null {
  return average(
    history
      .map((prediction) => prediction.engines.find((engine) => engine.name === engineName)?.score ?? null)
      .filter((value): value is number => value !== null && Number.isFinite(value)),
  );
}

export function TrainingStatusWidget() {
  const { trainingState, predictionHistory } = useHandwritingStore();

  if (!trainingState) {
    return (
      <div className="fixed bottom-4 right-4 z-[120] rounded-full border border-slate-700 bg-slate-950/95 px-3 py-2 text-xs font-bold text-slate-300 shadow-2xl backdrop-blur-md">
        <span className="mr-2 inline-block h-2.5 w-2.5 rounded-full bg-red-500" />
        training
      </div>
    );
  }

  const generation = trainingState.personalizationGeneration ?? 0;
  const recentConfidence = average(predictionHistory.slice(0, 10).map((prediction) => prediction.confidence));
  const highConfidence = recentConfidence !== null && recentConfidence >= HIGH_CONFIDENCE_THRESHOLD;
  const status = generation >= 3
    ? highConfidence ? 'personal' : 'personal'
    : generation >= 1 ? 'optimising' : 'training';
  const dotClass = generation >= 3
    ? highConfidence
      ? 'bg-emerald-500'
      : 'bg-[linear-gradient(90deg,#10b981_0_50%,#f59e0b_50%_100%)]'
    : generation >= 1
      ? 'bg-amber-500'
      : 'bg-red-500';

  const counts = Object.values(trainingState.countsByLetter);
  const avgSamplesPerLetter = average(counts.map((count) => count.user_inputted + count.implicit));
  const featureConfidence = engineAverage(predictionHistory, 'Feature');
  const cnnConfidence = engineAverage(predictionHistory, 'CNN');

  return (
    <div className="group fixed bottom-4 right-4 z-[120]">
      <div className="rounded-full border border-slate-700 bg-slate-950/95 px-3 py-2 text-xs font-bold text-slate-200 shadow-2xl backdrop-blur-md">
        <span className={`mr-2 inline-block h-2.5 w-2.5 rounded-full align-[-1px] ${dotClass}`} />
        {status}
      </div>
      <div className="pointer-events-none absolute bottom-11 right-0 w-72 translate-y-1 rounded-lg border border-slate-700 bg-slate-950/98 p-3 text-xs text-slate-300 opacity-0 shadow-2xl backdrop-blur-md transition-all group-hover:translate-y-0 group-hover:opacity-100">
        <div className="mb-2 flex items-center justify-between border-b border-slate-800 pb-2">
          <span className="font-black uppercase tracking-[0.18em] text-slate-500">Training</span>
          <span className="text-slate-100">gen {generation}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-slate-500">knn</span>
          <span>{formatValue(avgSamplesPerLetter)} avg samples/letter</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-slate-500">svm</span>
          <span>{formatPercent(featureConfidence)} avg confidence</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-slate-500">cnn</span>
          <span>{formatPercent(cnnConfidence)} avg confidence</span>
        </div>
        <div className="mt-2 border-t border-slate-800 pt-2 text-[11px] text-slate-500">
          Last 10 confidence: {formatPercent(recentConfidence)}
        </div>
      </div>
    </div>
  );
}
