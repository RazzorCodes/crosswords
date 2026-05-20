import { useEffect, useRef, useState } from 'react';
import { useGameStore } from './store/useGameStore';
import { useHandwritingStore } from './store/useHandwritingStore';
import { generateCrossword } from './utils/generator';
import { loadRomanianDictionaryEntries } from './utils/dictionary';
import { getMostNeededLetter, getPriorityLetters, handwritingModule, initHandwritingModule, type HandwritingModuleEvent } from './utils/handwriting';
import { createFeedbackDevGrid } from './utils/devBoard';
import { resolveAppMode } from './utils/runtimeConfig';
import { GridComponent } from './components/Grid';
import { HandwritingPanel } from './components/HandwritingPanel';
import { SettingsComponent } from './components/Settings';
import { TrainingStatusWidget } from './components/TrainingStatusWidget';
import { ToastContainer } from './components/Toast';
import { cancelAllPendingSubmissions } from './utils/handwritingSession';
import enWords from './data/en.json';
import roWords from './data/ro.json';

function summarizeHandwritingEvent(event: HandwritingModuleEvent): string {
  switch (event.type) {
    case 'prediction': {
      const top = event.payload.candidates[0];
      return top ? `Prediction ${top.char} ${Math.round(top.score * 100)}%` : 'Prediction unavailable';
    }
    case 'sample-recorded':
      return `Sample ${event.payload.sample.label} ${event.payload.sample.acceptance}`;
    case 'milestone-reached':
      return `Milestone ${event.payload.milestone} reached`;
    case 'training-started':
      return `Training started: ${event.payload.reason}`;
    case 'training-progress':
      return `Training progress: ${event.payload.message}`;
    case 'training-completed':
      return event.payload.snapshot
        ? `Training accepted: ${event.payload.snapshot.id}`
        : 'Training accepted: CNN updated';
    case 'training-rejected':
      return `Training rejected: ${event.payload.reason}`;
    case 'artifacts-updated':
      return 'Artifacts updated';
    default:
      return 'Unknown event';
  }
}

function DevFeedbackBox() {
  const [feedback, setFeedback] = useState('');
  const { addDiagnosticLog } = useHandwritingStore();

  return (
    <form
      className="mx-auto flex w-full max-w-xl items-center gap-2 rounded-lg border border-slate-800 bg-slate-950/90 p-2 shadow-2xl backdrop-blur-md"
      onSubmit={(event) => {
        event.preventDefault();
        setFeedback('');
        addDiagnosticLog({
          type: 'feedback-cleared',
          summary: 'Feedback textbox cleared',
        });
      }}
    >
      <input
        value={feedback}
        onChange={(event) => setFeedback(event.target.value)}
        placeholder="Feedback note"
        className="min-w-0 flex-1 rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white outline-none focus:border-blue-500"
      />
      <button
        type="submit"
        className="rounded-md border border-blue-500/40 bg-blue-500/15 px-4 py-2 text-xs font-black uppercase tracking-widest text-blue-100 transition-colors hover:bg-blue-500/25"
      >
        Send
      </button>
    </form>
  );
}

function App() {
  const appMode = resolveAppMode();
  const isDevMode = appMode === 'dev';
  const { 
    language, 
    setGrid, 
    mostNeededLetter, 
    setMostNeededLetter,
    updateTrainingToast,
    gameId,
    setGameId,
    startTime,
    setStartTime,
    endTime,
    setEndTime
  } = useGameStore();
  const {
    trainMode,
    trainingState,
    setTrainingState,
    setModuleReady,
    setLastEvent,
    setLastPrediction,
    addDiagnosticLog,
  } = useHandwritingStore();

  const [isVictoryVisible, setIsVictoryVisible] = useState(false);
  const trainingStateRef = useRef(trainingState);

  useEffect(() => {
    trainingStateRef.current = trainingState;
  }, [trainingState]);

  useEffect(() => {
    let cancelled = false;

    void initHandwritingModule()
      .then((result) => {
        if (cancelled) {
          return;
        }
        setTrainingState(result.trainingState);
        setModuleReady(true);
      })
      .catch((error) => {
        console.error('Failed to initialize handwriting module', error);
        if (!cancelled) {
          setModuleReady(false);
        }
      });

    const unsubscribe = handwritingModule.subscribe((event) => {
      if ('trainingState' in event.payload) {
        setTrainingState(event.payload.trainingState);
      }
      if (event.type === 'prediction') {
        setLastPrediction({
          createdAt: Date.now(),
          topLabel: event.payload.candidates[0]?.char ?? null,
          confidence: event.payload.confidence,
          candidates: event.payload.candidates,
          engines: event.payload.engineResults,
        });
      }
      if (event.type === 'training-progress') {
        updateTrainingToast(event.payload);
      }
      setLastEvent(event.type);
      addDiagnosticLog({
        type: event.type,
        summary: summarizeHandwritingEvent(event),
      });
    });

    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [addDiagnosticLog, setLastEvent, setLastPrediction, setModuleReady, setTrainingState, updateTrainingToast]);

  useEffect(() => {
    setMostNeededLetter(trainingState ? getMostNeededLetter(trainingState.countsByLetter) : null);
  }, [setMostNeededLetter, trainingState]);

  useEffect(() => {
    let cancelled = false;

    const updateGrid = async () => {
      if (isDevMode) {
        cancelAllPendingSubmissions();
        setGrid(createFeedbackDevGrid());
        setStartTime(Date.now());
        setEndTime(null);
        setIsVictoryVisible(false);
        return;
      }

      const currentTrainingState = trainingStateRef.current;
      const priorityLetters = currentTrainingState
        ? getPriorityLetters(currentTrainingState.countsByLetter)
        : [];
      let words = enWords;
      if (language === 'ro') {
        const importedWords = await loadRomanianDictionaryEntries();
        words = importedWords.length > 0 ? importedWords : roWords;
      }

      if (cancelled) return;

      const newGrid = generateCrossword(words, 15, priorityLetters);
      if (cancelled) return;
      cancelAllPendingSubmissions();
      setGrid(newGrid);
      setStartTime(Date.now());
      setEndTime(null);
      setIsVictoryVisible(false);
    };

    void updateGrid();

    return () => {
      cancelled = true;
    };
  }, [gameId, isDevMode, language, setEndTime, setGrid, setStartTime]);

  // Victory effect
  useEffect(() => {
    if (isDevMode && endTime) {
      setIsVictoryVisible(true);
      const timer = setTimeout(() => {
        cancelAllPendingSubmissions();
        setGrid(createFeedbackDevGrid());
        setStartTime(Date.now());
        setEndTime(null);
        setIsVictoryVisible(false);
        addDiagnosticLog({
          type: 'dev-board-cleared',
          summary: 'FEEDBACK board solved and cleared',
        });
      }, 1200);
      return () => clearTimeout(timer);
    }

    if (endTime) {
      setIsVictoryVisible(true);
      const timer = setTimeout(() => {
        // We keep it visible so they can see the time, 
        // but "flashes briefly" could mean it fades out.
        // Let's keep it until they click or reset.
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [addDiagnosticLog, endTime, isDevMode, setEndTime, setGrid, setStartTime]);

  const formatTime = (ms: number) => {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}m ${seconds}s`;
  };

  return (
    <div className="w-screen h-screen bg-slate-950 text-slate-200 flex flex-col overflow-hidden relative">
      <header className="p-4 flex flex-col items-center shrink-0 z-40 bg-slate-950/50 backdrop-blur-md border-b border-slate-800">
        <div className="w-full flex justify-between items-center mb-2 px-4">
          <button 
            onClick={() => {
              if (isDevMode) {
                cancelAllPendingSubmissions();
                setGrid(createFeedbackDevGrid());
                setStartTime(Date.now());
                setEndTime(null);
                return;
              }
              setGameId(gameId + 1);
            }}
            className="text-[10px] font-bold text-slate-500 hover:text-white border border-slate-800 hover:border-slate-600 px-3 py-1 rounded transition-all uppercase tracking-widest"
          >
            {isDevMode ? 'Clear Word' : 'Reset Game'}
          </button>
          <h1 className="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
            {isDevMode ? 'FEEDBACK DEV' : 'CROSSWORD'}
          </h1>
          <div className="w-20" /> {/* Spacer */}
        </div>
        
        {trainMode && mostNeededLetter && (
          <div className="mb-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-[10px] font-bold text-blue-400 uppercase tracking-widest">
            Most needed: <span className="text-white text-xs">{mostNeededLetter}</span>
          </div>
        )}
        {!isDevMode && <SettingsComponent />}
      </header>
      
      {isDevMode ? (
        <main className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_440px] overflow-hidden">
          <section className="relative flex min-w-0 flex-col overflow-hidden">
            <div className="relative min-h-0 flex-1 overflow-hidden">
              <GridComponent appMode={appMode} />
            </div>
            <div className="shrink-0 border-t border-slate-800 bg-slate-950/95 p-4">
              <DevFeedbackBox />
            </div>
          </section>
          <aside className="min-h-0 overflow-y-auto border-l border-slate-800 bg-slate-950/90">
            <HandwritingPanel docked />
          </aside>
        </main>
      ) : (
        <main className="flex-1 relative overflow-hidden">
          <GridComponent appMode={appMode} />
        </main>
      )}
      <ToastContainer />

      {/* Victory Overlay */}
      {!isDevMode && endTime && isVictoryVisible && (
        <div 
          className="absolute inset-0 z-[100] flex items-center justify-center bg-slate-950/80 backdrop-blur-xl animate-in fade-in duration-500"
          onClick={() => setIsVictoryVisible(false)}
        >
          <div className="text-center p-8 rounded-3xl bg-slate-900 border border-slate-800 shadow-2xl scale-in-center animate-in zoom-in duration-300">
            <div className="text-6xl mb-4">🏆</div>
            <h2 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 to-amber-600 mb-2">
              VICTORY!
            </h2>
            <p className="text-slate-400 mb-6 uppercase tracking-widest font-bold">
              Time taken: <span className="text-white">{formatTime(endTime - (startTime || endTime))}</span>
            </p>
            <button
              onClick={(e) => { e.stopPropagation(); setGameId(gameId + 1); }}
              className="px-8 py-3 bg-gradient-to-r from-blue-600 to-emerald-600 text-white font-black rounded-full hover:shadow-[0_0_20px_rgba(59,130,246,0.5)] transition-all uppercase tracking-widest"
            >
              Play Again
            </button>
          </div>
        </div>
      )}

      <TrainingStatusWidget />

      <footer className="absolute bottom-4 left-4 max-w-[calc(100%-10rem)] text-slate-500 text-xs italic pointer-events-none z-50">
        {isDevMode
          ? 'Dev mode: write FEEDBACK repeatedly to inspect recognition, triggers, and training status.'
          : trainMode
          ? 'Train mode keeps handwriting artifacts local, tracks milestones, and surfaces trainer state inline.'
          : 'Use keyboard, mouse, or pen to solve. Pinch to zoom, drag to pan.'}
      </footer>
    </div>
  );
}

export default App;
