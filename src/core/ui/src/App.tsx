import { useEffect, useRef, useState } from 'react';
import { ALPHABET, type DatasetManifest, type LabRecognitionResult, type LabSample, type LabState, type StrokeInput } from './lib/types';
import { normalizeLabel } from './lib/canvas';
import { getCnnAvailability, loadBaselineManifest, trainCnn, type CnnTrainingProgress } from './lib/cnnAdapter';
import { loadDatasetManifest, loadDatasetSamples, mergeDatasetSamples } from './lib/datasets';
import { countByLetter, computeSvmMetrics, readyLetters } from './lib/metrics';
import { emptyLabState, estimateJsonBytes, loadLabState, resetLabState, saveLabState } from './lib/storage';
import { recognizeLabStrokes } from './lib/recognition';
import { extractFeatures, initWasmCore, trainSvm } from './lib/wasmCore';

type Status = 'idle' | 'busy' | 'ready' | 'error';

function formatMs(value: number | undefined): string {
  return value === undefined ? '-' : `${value.toFixed(1)} ms`;
}

function formatPercent(value: number | undefined): string {
  return value === undefined ? '-' : `${Math.round(value * 100)}%`;
}

function downloadBlob(name: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
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

  const pointFromEvent = (event: React.PointerEvent<HTMLCanvasElement>) => {
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
  const [state, setState] = useState<LabState>(() => loadLabState());
  const [strokes, setStrokes] = useState<StrokeInput>([]);
  const [label, setLabel] = useState('A');
  const [status, setStatus] = useState<Status>('idle');
  const [message, setMessage] = useState('initializing');
  const [result, setResult] = useState<LabRecognitionResult | null>(null);
  const [dump, setDump] = useState('');
  const [cnnProgress, setCnnProgress] = useState<CnnTrainingProgress | null>(null);
  const [datasetManifest, setDatasetManifest] = useState<DatasetManifest>({ version: 'loading', datasets: [] });
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);

  useEffect(() => {
    let alive = true;
    void Promise.all([initWasmCore(), loadBaselineManifest(), loadDatasetManifest()])
      .then(([, manifest, datasets]) => {
        if (!alive) return;
        setState((current) => ({ ...current, baselineManifest: manifest }));
        setDatasetManifest(datasets);
        setSelectedDatasets(datasets.datasets.filter((dataset) => dataset.group === 'hq' || dataset.group === 'inputted').map((dataset) => dataset.id));
        setStatus('ready');
        setMessage(`WASM ready; ${datasets.datasets.length} datasets indexed`);
      })
      .catch((error) => {
        if (!alive) return;
        setStatus('error');
        setMessage(error instanceof Error ? error.message : 'initialization failed');
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    saveLabState(state);
  }, [state]);

  const counts = countByLetter(state.ledger);
  const ready = readyLetters(state.ledger);
  const cnnAvailability = getCnnAvailability(state.baselineManifest);
  const svmBytes = state.svmSnapshot ? estimateJsonBytes(state.svmSnapshot) : 0;
  const cnnInferenceBytes = state.cnnArtifacts?.inferenceModel?.byteLength ?? 0;
  const cnnCheckpointBytes = state.cnnArtifacts?.checkpoint?.byteLength ?? 0;

  const runRecognize = async () => {
    if (strokes.flat().length === 0) {
      setMessage('draw a sample before recognition');
      return;
    }
    setStatus('busy');
    try {
      const next = await recognizeLabStrokes(strokes, state);
      setResult(next);
      setStatus('ready');
      setMessage(`aggregate ${next.aggregateCandidates[0]?.label ?? '-'}`);
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
    const features = extractFeatures(strokes);
    const sample: LabSample = {
      id: `lab-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`,
      label: normalized,
      strokes,
      features,
      createdAt: Date.now(),
      source: 'core-lab',
    };
    setState((current) => ({ ...current, ledger: [...current.ledger, sample] }));
    setMessage(`added ${normalized}`);
  };

  const runTrainSvm = () => {
    const trained = trainSvm(state.ledger, ready);
    if (!trained) {
      setMessage('SVM rejected: need at least two ready letters with two samples each');
      return;
    }
    const metrics = computeSvmMetrics(trained, state.ledger);
    const snapshot = { ...trained, metrics };
    setState((current) => ({ ...current, svmSnapshot: snapshot, latestMetrics: metrics }));
    setMessage(`SVM trained on ${snapshot.datasetSize} samples`);
  };

  const runTrainCnn = async () => {
    setStatus('busy');
    setCnnProgress(null);
    const training = await trainCnn(state.baselineManifest, state.ledger, state.cnnArtifacts, setCnnProgress);
    if (!training.artifacts) {
      setStatus('ready');
      setMessage(`CNN rejected: ${training.rejectionReason}`);
      return;
    }
    setState((current) => ({ ...current, cnnArtifacts: training.artifacts, latestMetrics: training.metrics ?? current.latestMetrics }));
    setStatus('ready');
    setMessage('CNN artifacts accepted');
  };

  const loadSelectedDatasets = async (mode: 'merge' | 'replace') => {
    const entries = datasetManifest.datasets.filter((dataset) => selectedDatasets.includes(dataset.id));
    if (entries.length === 0) {
      setMessage('select at least one dataset');
      return;
    }
    setStatus('busy');
    try {
      const samples = await loadDatasetSamples(entries);
      setState((current) => ({
        ...current,
        ledger: mode === 'replace' ? samples : mergeDatasetSamples(current.ledger, samples),
        svmSnapshot: mode === 'replace' ? null : current.svmSnapshot,
        latestMetrics: mode === 'replace' ? null : current.latestMetrics,
      }));
      setStatus('ready');
      setMessage(`${mode === 'replace' ? 'loaded' : 'merged'} ${samples.length} dataset samples`);
    } catch (error) {
      setStatus('error');
      setMessage(error instanceof Error ? error.message : 'dataset load failed');
    }
  };

  const selectDatasetGroup = (group: 'all' | 'regular' | 'hq' | 'inputted') => {
    const ids = group === 'all'
      ? datasetManifest.datasets.map((dataset) => dataset.id)
      : datasetManifest.datasets.filter((dataset) => dataset.group === group).map((dataset) => dataset.id);
    setSelectedDatasets(ids);
    setMessage(`selected ${ids.length} ${group} dataset${ids.length === 1 ? '' : 's'}`);
  };

  const dumpSvm = () => {
    const text = JSON.stringify(state.svmSnapshot, null, 2);
    setDump(text);
    navigator.clipboard?.writeText(text).catch(() => undefined);
  };

  const dumpCnn = () => {
    const metadata = {
      updatedAt: state.cnnArtifacts?.updatedAt ?? null,
      stage: state.cnnArtifacts?.stage ?? null,
      metrics: state.cnnArtifacts?.metrics ?? null,
      inferenceModelBytes: cnnInferenceBytes,
      checkpointBytes: cnnCheckpointBytes,
      exportMetadata: state.cnnArtifacts?.exportMetadata ?? null,
    };
    const text = JSON.stringify(metadata, null, 2);
    setDump(text);
    navigator.clipboard?.writeText(text).catch(() => undefined);
  };

  return (
    <main className="shell">
      <section className="toolbar">
        <div>
          <h1>Handwriting Core Lab</h1>
          <p>{status.toUpperCase()} / {message}</p>
        </div>
        <div className="toolbar-actions">
          <button onClick={() => setStrokes([])}>Clear Ink</button>
          <button onClick={runRecognize} disabled={status === 'busy'}>Recognize</button>
          <button onClick={addSample}>Add Sample</button>
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

        <div className="metrics-grid">
          <div className="metric"><span>Samples</span><strong>{state.ledger.length}</strong></div>
          <div className="metric"><span>Ready Letters</span><strong>{ready.length}</strong></div>
          <div className="metric"><span>SVM</span><strong>{state.svmSnapshot ? 'trained' : 'empty'}</strong></div>
          <div className="metric"><span>CNN</span><strong>{state.cnnArtifacts ? 'personalized' : state.baselineManifest?.cnn.inferenceUrl ? 'baseline' : 'empty'}</strong></div>
          <div className="metric"><span>SVM Size</span><strong>{svmBytes} B</strong></div>
          <div className="metric"><span>CNN Size</span><strong>{cnnInferenceBytes + cnnCheckpointBytes} B</strong></div>
          <div className="metric"><span>Accuracy</span><strong>{formatPercent(state.latestMetrics?.overallAccuracy)}</strong></div>
          <div className="metric"><span>CNN Train</span><strong>{cnnAvailability.available ? 'ready' : 'blocked'}</strong></div>
        </div>

        <div className="controls">
          <button onClick={runTrainSvm}>Train SVM</button>
          <button onClick={runTrainCnn} disabled={status === 'busy'}>Train CNN</button>
          <button onClick={dumpSvm}>Dump SVM</button>
          <button onClick={dumpCnn}>Dump CNN</button>
          <button onClick={() => state.cnnArtifacts?.inferenceModel && downloadBlob('core-lab-cnn-inference.onnx', new Blob([state.cnnArtifacts.inferenceModel]))}>Download CNN Model</button>
          <button onClick={() => state.cnnArtifacts?.checkpoint && downloadBlob('core-lab-cnn-checkpoint.bin', new Blob([state.cnnArtifacts.checkpoint]))}>Download CNN Checkpoint</button>
          <button onClick={() => setState((current) => ({ ...current, svmSnapshot: null }))}>Reset SVM</button>
          <button onClick={() => setState((current) => ({ ...current, cnnArtifacts: null }))}>Reset CNN</button>
          <button onClick={() => { setState(resetLabState()); setResult(null); setDump(''); }}>Reset All</button>
        </div>
      </section>

      <section className="lower-grid">
        <div className="panel dataset-panel">
          <h2>Datasets</h2>
          <div className="dataset-actions">
            <button onClick={() => selectDatasetGroup('all')}>Select All</button>
            <button onClick={() => selectDatasetGroup('regular')}>Select Regular</button>
            <button onClick={() => selectDatasetGroup('hq')}>Select HQ</button>
            <button onClick={() => selectDatasetGroup('inputted')}>Select Inputted</button>
          </div>
          <div className="dataset-list">
            {datasetManifest.datasets.map((dataset) => (
              <label key={dataset.id} className="dataset-row">
                <input
                  type="checkbox"
                  checked={selectedDatasets.includes(dataset.id)}
                  onChange={(event) => {
                    setSelectedDatasets((current) => event.target.checked
                      ? [...current, dataset.id]
                      : current.filter((id) => id !== dataset.id));
                  }}
                />
                <span>{dataset.label}</span>
                <em>{dataset.count}</em>
              </label>
            ))}
          </div>
          <div className="dataset-actions">
            <button onClick={() => void loadSelectedDatasets('merge')} disabled={status === 'busy'}>Merge Selected</button>
            <button onClick={() => void loadSelectedDatasets('replace')} disabled={status === 'busy'}>Replace Ledger</button>
          </div>
        </div>

        <div className="panel">
          <h2>Recognition</h2>
          {result ? (
            <>
              <div className="aggregate">{result.aggregateCandidates.slice(0, 5).map((candidate) => (
                <span key={candidate.label}>{candidate.label} {candidate.score.toFixed(2)}</span>
              ))}</div>
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
          ) : <p className="muted">No recognition result yet.</p>}
        </div>

        <div className="panel">
          <h2>Ledger</h2>
          <div className="letter-grid">
            {ALPHABET.map((item) => <span key={item} className={counts[item] >= 2 ? 'ready-letter' : ''}>{item}:{counts[item]}</span>)}
          </div>
          {cnnProgress && <p className="muted">{cnnProgress.message} / {Math.round(cnnProgress.progress * 100)}%</p>}
          {!cnnAvailability.available && <p className="muted">{cnnAvailability.reasons.join('; ')}</p>}
        </div>

        <div className="panel dump-panel">
          <h2>Dump</h2>
          <pre>{dump || 'Dumped model JSON and CNN metadata appear here. Dump actions also copy text to the clipboard when permitted.'}</pre>
        </div>
      </section>
    </main>
  );
}

export default App;
