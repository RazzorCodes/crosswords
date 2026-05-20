import { useEffect, useRef, useState } from 'react';
import { useGameStore } from './store/useGameStore';
import { useHandwritingStore } from './store/useHandwritingStore';
import { generateCrossword } from './utils/generator';
import { loadRomanianDictionaryEntries } from './utils/dictionary';
import { getMostNeededLetter, getPriorityLetters, handwritingModule, initHandwritingModule } from './utils/handwriting';
import { GridComponent } from './components/Grid';
import { SettingsComponent } from './components/Settings';
import { ToastContainer } from './components/Toast';
import { cancelAllPendingSubmissions } from './utils/handwritingSession';
import enWords from './data/en.json';
import roWords from './data/ro.json';

function App() {
  const { 
    language, 
    setGrid, 
    mostNeededLetter, 
    setMostNeededLetter,
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
      setLastEvent(event.type);
    });

    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [setLastEvent, setModuleReady, setTrainingState]);

  useEffect(() => {
    setMostNeededLetter(trainingState ? getMostNeededLetter(trainingState.countsByLetter) : null);
  }, [setMostNeededLetter, trainingState]);

  useEffect(() => {
    let cancelled = false;

    const updateGrid = async () => {
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
  }, [gameId, language, setEndTime, setGrid, setStartTime]);

  // Victory effect
  useEffect(() => {
    if (endTime) {
      setIsVictoryVisible(true);
      const timer = setTimeout(() => {
        // We keep it visible so they can see the time, 
        // but "flashes briefly" could mean it fades out.
        // Let's keep it until they click or reset.
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [endTime]);

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
            onClick={() => setGameId(gameId + 1)}
            className="text-[10px] font-bold text-slate-500 hover:text-white border border-slate-800 hover:border-slate-600 px-3 py-1 rounded transition-all uppercase tracking-widest"
          >
            Reset Game
          </button>
          <h1 className="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
            CROSSWORD
          </h1>
          <div className="w-20" /> {/* Spacer */}
        </div>
        
        {trainMode && mostNeededLetter && (
          <div className="mb-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-[10px] font-bold text-blue-400 uppercase tracking-widest">
            Most needed: <span className="text-white text-xs">{mostNeededLetter}</span>
          </div>
        )}
        <SettingsComponent />
      </header>
      
      <main className="flex-1 relative overflow-hidden">
        <GridComponent />
      </main>
      <ToastContainer />

      {/* Victory Overlay */}
      {endTime && isVictoryVisible && (
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

      <footer className="absolute bottom-4 right-4 text-slate-500 text-xs italic pointer-events-none z-50">
        {trainMode
          ? 'Train mode keeps handwriting artifacts local, tracks milestones, and surfaces trainer state inline.'
          : 'Use keyboard, mouse, or pen to solve. Pinch to zoom, drag to pan.'}
      </footer>
    </div>
  );
}

export default App;
