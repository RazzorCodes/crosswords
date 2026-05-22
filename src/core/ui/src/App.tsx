import { useEffect, useRef, useState } from 'react';
import type { PointerEvent } from 'react';
import { ALPHABET, type LabRecognitionResult, type LabSample, type LabState, type StrokeInput } from './lib/types';
import { normalizeLabel } from './lib/canvas';
import {
  getCnnAvailability,
  loadBaselineManifest,
  MIN_CNN_FINE_TUNE_SAMPLES,
  RECOMMENDED_CNN_SAMPLES_PER_LETTER,
  STRONG_CNN_SAMPLES_PER_LETTER,
  trainCnn,
  type CnnTrainingProgress,
} from './lib/cnnAdapter';
import { countByLetter, readyLetters } from './lib/metrics';
import {
  emptyLabState,
  estimateJsonBytes,
  formatBytes,
  loadLabState,
  resetLabState,
  saveLabState,
} from './lib/storage';
import { recognizeLabStrokes } from './lib/recognition';
import { extractFeatures, initWasmCore } from './lib/wasmCore';

type Status = 'idle' | 'busy' | 'ready' | 'error';

type SvmWorkerResponse =
  | { type: 'progress'; message: string; progress: number }
  | { type: 'completed'; snapshot: LabState['svmSnapshot']; rejectionReason: string | null; elapsedMs: number }
  | { type: 'failed'; reason: string };

function formatPercent(value: number | undefined): string {
  return value === undefined ? '-' : `${Math.round(value * 100)}%`;
}

function formatMs(value: number | undefined): string {
  return value === undefined ? '-' : `${value.toFixed(1)} ms`;
}

function downloadBlob(name: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

function activeLetters(counts: Record<string, number>): string[] {
  return ALPHABET.filter((letter) => counts[letter] > 0);
}

function fineTuneGuidance(counts: Record<string, number>, total: number): string {
  const active = activeLetters(counts);
  const target = Math.max(MIN_CNN_FINE_TUNE_SAMPLES, active.length * RECOMMENDED_CNN_SAMPLES_PER_LETTER);
  if (total < MIN_CNN_FINE_TUNE_SAMPLES) {
    return `Need ${MIN_CNN_FINE_TUNE_SAMPLES - total} more sample${MIN_CNN_FINE_TUNE_SAMPLES - total === 1 ? '' : 's'} to run a small head fine-tune.`;
  }
  const shortLetters = active.filter((letter) => counts[letter] < RECOMMENDED_CNN_SAMPLES_PER_LETTER);
  if (shortLetters.length > 0) {
    return `Runnable now; useful tuning wants about ${RECOMMENDED_CNN_SAMPLES_PER_LETTER} per active letter. Add ${Math.max(0, target - total)} more across ${shortLetters.join(', ')}.`;
  }
  return `Good head-tune set for ${active.length} active letter${active.length === 1 ? '' : 's'}. Full alphabet target: ${ALPHABET.length * RECOMMENDED_CNN_SAMPLES_PER_LETTER}-${ALPHABET.length * STRONG_CNN_SAMPLES_PER_LETTER} samples.`;
}

function Thumbnail({ strokes }: { strokes: StrokeInput }) {
  if (!strokes.length || strokes.flat().length === 0) return <div className="thumbnail-empty" />;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  strokes.flat().forEach((p) => {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  });
  const width = maxX - minX || 1;
  const height = maxY - minY || 1;
  const size = Math.max(width, height);
  const padding = size * 0.15;
  const viewBox = `${minX - padding} ${minY - padding} ${width + 2 * padding} ${height + 2 * padding}`;

  return (
    <svg viewBox={viewBox} className="thumbnail-svg">
      {strokes.map((stroke, index) => (
        <path
          key={index}
          d={`M ${stroke.map((p) => `${p.x} ${p.y}`).join(' L ')}`}
          fill="none"
          stroke="currentColor"
          strokeWidth={size / 10}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      ))}
    </svg>
  );
}

function DrawingPad({ strokes, setStrokes }: { strokes: StrokeInput; setStrokes: (strokes: StrokeInput) => void }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawingRef = useRef(false);
  const strokesRef = useRef<StrokeInput>(strokes);

  useEffect(() => {
    strokesRef.current = strokes;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#111827';
    for (const stroke of strokes) {
      if (stroke.length === 0) continue;
      ctx.beginPath();
      ctx.moveTo(stroke[0].x, stroke[0].y);
      for (const point of stroke.slice(1)) {
        ctx.lineTo(point.x, point.y);
      }
      ctx.stroke();
    }
  }, [strokes]);

  const pointFromEvent = (event: PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top, t: Date.now() };
  };

  return (
    <canvas
      ref={canvasRef}
      width={360}
      height={360}
      className="drawing-pad"
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId);
        drawingRef.current = true;
        const next = [...strokesRef.current, [pointFromEvent(event)]];
        strokesRef.current = next;
        setStrokes(next);
      }}
      onPointerMove={(event) => {
        if (!drawingRef.current) return;
        const next = strokesRef.current.map((stroke, index) =>
          index === strokesRef.current.length - 1 ? [...stroke, pointFromEvent(event)] : stroke,
        );
        strokesRef.current = next;
        setStrokes(next);
      }}
      onPointerUp={() => {
        drawingRef.current = false;
      }}
      onPointerCancel={() => {
        drawingRef.current = false;
      }}
    />
  );
}

export function App() {
  const [state, setState] = useState<LabState>(emptyLabState);
  const [strokes, setStrokes] = useState<StrokeInput>([]);
  const [label, setLabel] = useState('A');
  const [selectedLetter, setSelectedLetter] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>('idle');
  const [message, setMessage] = useState('initializing');
  const [result, setResult] = useState<LabRecognitionResult | null>(null);
  const [cnnProgress, setCnnProgress] = useState<CnnTrainingProgress | null>(null);
  const [svmProgress, setSvmProgress] = useState<{ progress: number; message: string } | null>(null);

  useEffect(() => {
    let alive = true;
    void (async () => {
      try {
        const [loadedState, , manifest] = await Promise.all([
          loadLabState(),
          initWasmCore(),
          loadBaselineManifest(),
        ]);
        if (!alive) return;
        setState({ ...loadedState, baselineManifest: manifest });
        setStatus('ready');
        setMessage(manifest ? 'scratchpad ready' : 'scratchpad ready; CNN manifest not found');
      } catch (error) {
        if (!alive) return;
        setStatus('error');
        setMessage(error instanceof Error ? error.message : 'initialization failed');
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (status === 'ready') {
      void saveLabState(state);
    }
  }, [state, status]);

  const counts = countByLetter(state.ledger);
  const ready = readyLetters(state.ledger);
  const cnnAvailability = getCnnAvailability(state.baselineManifest);
  const svmBytes = state.svmSnapshot ? estimateJsonBytes(state.svmSnapshot) : 0;
  const cnnModelBytes = state.cnnArtifacts?.modelSafetensors.byteLength ?? state.baselineManifest?.cnn.trainingArtifacts.modelSize ?? 0;
  const optimizerBytes = state.cnnArtifacts?.optimizerState.byteLength ?? 0;
  const canFineTune = status !== 'busy' && state.ledger.length >= MIN_CNN_FINE_TUNE_SAMPLES;

  const runRecognize = async () => {
    if (strokes.flat().length === 0) {
      setMessage('draw ink before recognition');
      return;
    }
    setStatus('busy');
    try {
      const next = await recognizeLabStrokes(strokes, state);
      setResult(next);
      setStatus('ready');
      setMessage(`detected ${next.aggregateCandidates[0]?.label ?? '-'}`);
    } catch (error) {
      setStatus('error');
      setMessage(error instanceof Error ? error.message : 'recognition failed');
    }
  };

  const addSample = () => {
    const normalized = normalizeLabel(label);
    if (!normalized || strokes.flat().length === 0) {
      setMessage('sample needs a label and ink');
      return;
    }
    const sample: LabSample = {
      id: `sample-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`,
      label: normalized,
      strokes,
      features: extractFeatures(strokes),
      createdAt: Date.now(),
      source: 'core-lab',
    };
    setState((current) => ({ ...current, ledger: [...current.ledger, sample] }));
    setStrokes([]);
    setMessage(`added ${normalized}; ${state.ledger.length + 1} samples total`);
  };

  const runTrainSvm = async () => {
    if (ready.length < 2) {
      setMessage('SVM needs at least two letters with two samples each');
      return;
    }
    setStatus('busy');
    setSvmProgress({ progress: 1, message: `queued ${state.ledger.length} samples` });
    const wasmUrl = new URL('wasm/handwriting_core.wasm', new URL(import.meta.env.BASE_URL, window.location.origin)).toString();
    const worker = new Worker(new URL('./lib/svmTrainingWorker.ts', import.meta.url), { type: 'module' });
    try {
      const snapshot = await new Promise<LabState['svmSnapshot']>((resolve, reject) => {
        worker.onmessage = (event: MessageEvent<SvmWorkerResponse>) => {
          if (event.data.type === 'progress') {
            setSvmProgress({ progress: event.data.progress, message: event.data.message });
            setMessage(event.data.message);
            return;
          }
          if (event.data.type === 'failed') {
            reject(new Error(event.data.reason));
            return;
          }
          if (!event.data.snapshot) {
            reject(new Error(event.data.rejectionReason ?? 'SVM training rejected'));
            return;
          }
          resolve(event.data.snapshot);
        };
        worker.onerror = (event) => reject(new Error(event.message || 'SVM worker failed'));
        worker.postMessage({ type: 'train', ledger: state.ledger, readyLetters: ready, wasmUrl });
      });
      setState((current) => ({ ...current, svmSnapshot: snapshot, latestMetrics: snapshot?.metrics ?? current.latestMetrics }));
      setStatus('ready');
      setSvmProgress({ progress: 100, message: 'SVM training complete' });
      setMessage(`SVM trained with ${snapshot?.supportCount ?? 0} support vectors`);
    } catch (error) {
      setStatus('error');
      setSvmProgress(null);
      setMessage(error instanceof Error ? error.message : 'SVM training failed');
    } finally {
      worker.terminate();
    }
  };

  const runFineTune = async () => {
    setStatus('busy');
    setCnnProgress(null);
    const training = await trainCnn(state.baselineManifest, state.ledger, state.cnnArtifacts, setCnnProgress);
    if (!training.artifacts) {
      setStatus('ready');
      setMessage(`CNN fine-tune blocked: ${training.rejectionReason}`);
      return;
    }
    setState((current) => ({ ...current, cnnArtifacts: training.artifacts, latestMetrics: training.metrics ?? current.latestMetrics }));
    setStatus('ready');
    setMessage('CNN fine-tune complete');
  };

  return (
    <main className="shell">
      <section className="toolbar">
        <div>
          <h1>Handwriting Scratchpad</h1>
          <p>{status.toUpperCase()} / {message}</p>
        </div>
        <div className="toolbar-actions">
          <button onClick={() => setStrokes([])}>Clear Ink</button>
          <button onClick={runRecognize} disabled={status === 'busy'}>Recognize</button>
          <button onClick={addSample} disabled={status === 'busy'}>Add Sample</button>
        </div>
      </section>

      <section className="workbench">
        <div className="ink-panel">
          <DrawingPad strokes={strokes} setStrokes={setStrokes} />
          <div className="label-row">
            <select value={label} onChange={(event) => setLabel(event.target.value)}>
              {ALPHABET.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
            <input value={label} maxLength={1} onChange={(event) => setLabel(event.target.value.toUpperCase())} />
          </div>
        </div>

        <div className="panel">
          <h2>Detection</h2>
          {result ? (
            <>
              <div className="aggregate">
                {result.aggregateCandidates.slice(0, 5).map((candidate) => (
                  <span key={candidate.label}>{candidate.label} {candidate.score.toFixed(2)}</span>
                ))}
              </div>
              <table>
                <thead><tr><th>Engine</th><th>Status</th><th>Top</th><th>Confidence</th><th>Latency</th><th>Detail</th></tr></thead>
                <tbody>
                  {result.engineResults.map((engine) => (
                    <tr key={engine.algorithm}>
                      <td>{engine.algorithm}</td>
                      <td>{engine.status}</td>
                      <td>{engine.topLabel ?? '-'}</td>
                      <td>{engine.confidence?.toFixed(3) ?? '-'}</td>
                      <td>{formatMs(result.latenciesMs[engine.algorithm])}</td>
                      <td>{engine.detail ?? '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          ) : <p className="muted">Draw a letter and run recognition.</p>}
        </div>

        <div className="panel">
          <h2>Fine-Tuning</h2>
          <p className="muted">{fineTuneGuidance(counts, state.ledger.length)}</p>
          <div className="controls compact">
            <button onClick={() => void runTrainSvm()} disabled={status === 'busy'}>Train SVM</button>
            <button onClick={() => void runFineTune()} disabled={!canFineTune}>Fine-Tune CNN</button>
          </div>
          {svmProgress && <p className="muted">SVM {svmProgress.message} / {Math.round(svmProgress.progress)}%</p>}
          {cnnProgress && <p className="muted">{cnnProgress.message} / {Math.round(cnnProgress.progress * 100)}%</p>}
          {!cnnAvailability.available && <p className="muted">{cnnAvailability.reasons.join('; ')}</p>}
        </div>
      </section>

      <section className="metrics-grid">
        <div className="metric"><span>Samples</span><strong>{state.ledger.length}</strong></div>
        <div className="metric"><span>SVM</span><strong>{state.svmSnapshot ? 'trained' : 'empty'}</strong><small>{formatBytes(svmBytes)}</small></div>
        <div className="metric"><span>CNN</span><strong>{state.cnnArtifacts ? 'personalized' : cnnAvailability.available ? 'baseline' : 'blocked'}</strong><small>{formatBytes(cnnModelBytes + optimizerBytes)}</small></div>
        <div className="metric"><span>Accuracy</span><strong>{formatPercent(state.latestMetrics?.overallAccuracy)}</strong></div>
        <div className="metric"><span>Total Space</span><strong>{formatBytes(svmBytes + cnnModelBytes + optimizerBytes)}</strong></div>
      </section>

      <section className="lower-grid scratchpad-lower">
        <div className="panel">
          <div className="panel-header-row">
            <h2>Ledger</h2>
            {selectedLetter && <button className="btn-small" onClick={() => setSelectedLetter(null)}>Clear Filter</button>}
          </div>
          <div className="letter-grid">
            {ALPHABET.map((item) => (
              <span
                key={item}
                className={`${counts[item] >= 2 ? 'ready-letter' : ''} ${selectedLetter === item ? 'selected-letter' : ''}`}
                onClick={() => setSelectedLetter(item === selectedLetter ? null : item)}
                style={{ cursor: 'pointer' }}
              >
                {item}:{counts[item]}
              </span>
            ))}
          </div>

          {selectedLetter && (
            <div className="sample-inspector">
              <h3>Recent {selectedLetter} Samples</h3>
              <div className="sample-list">
                {state.ledger
                  .filter((s) => s.label === selectedLetter)
                  .slice(-3)
                  .reverse()
                  .map((sample) => (
                    <div key={sample.id} className="sample-card">
                      <Thumbnail strokes={sample.strokes} />
                      <div className="sample-info">
                        <span>{new Date(sample.createdAt).toLocaleTimeString()}</span>
                        <button
                          className="btn-danger btn-small"
                          onClick={() =>
                            setState((current) => ({
                              ...current,
                              ledger: current.ledger.filter((s) => s.id !== sample.id),
                            }))
                          }
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  ))}
                {counts[selectedLetter] === 0 && <p className="muted">No samples for {selectedLetter}.</p>}
              </div>
            </div>
          )}
        </div>

        <div className="panel">
          <h2>Artifacts</h2>
          <div className="controls">
            <button onClick={() => state.cnnArtifacts?.modelSafetensors && downloadBlob('pico-dual-cnn.safetensors', new Blob([state.cnnArtifacts.modelSafetensors]))}>Download CNN</button>
            <button onClick={() => state.cnnArtifacts?.optimizerState && downloadBlob('pico-dual-cnn-optimizer.bin', new Blob([state.cnnArtifacts.optimizerState]))}>Download Optimizer</button>
            <button onClick={() => setState((current) => ({ ...current, svmSnapshot: null }))}>Reset SVM</button>
            <button onClick={() => setState((current) => ({ ...current, cnnArtifacts: null }))}>Reset CNN</button>
            <button
              onClick={async () => {
                const cleared = await resetLabState();
                setState(cleared);
                setResult(null);
                setCnnProgress(null);
                setSvmProgress(null);
              }}
            >
              Reset All
            </button>
          </div>
        </div>
      </section>
    </main>
  );
}

export default App;
