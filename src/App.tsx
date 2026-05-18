import { useEffect } from 'react';
import { useGameStore } from './store/useGameStore';
import { generateCrossword } from './utils/generator';
import { loadRomanianDictionaryEntries } from './utils/dictionary';
import { GridComponent } from './components/Grid';
import { SettingsComponent } from './components/Settings';
import { ToastContainer } from './components/Toast';
import enWords from './data/en.json';
import roWords from './data/ro.json';

function App() {
  const { language, setGrid } = useGameStore();

  useEffect(() => {
    let cancelled = false;

    const updateGrid = async () => {
      let words = enWords;
      if (language === 'ro') {
        const importedWords = await loadRomanianDictionaryEntries();
        words = importedWords.length > 0 ? importedWords : roWords;
      }

      if (cancelled) return;

      const newGrid = generateCrossword(words, 15);
      setGrid(newGrid);
    };

    void updateGrid();

    return () => {
      cancelled = true;
    };
  }, [language, setGrid]);

  return (
    <div className="w-screen h-screen bg-slate-950 text-slate-200 flex flex-col overflow-hidden">
      <ToastContainer />
      <header className="p-4 flex flex-col items-center shrink-0 z-40 bg-slate-950/50 backdrop-blur-md border-b border-slate-800">
        <h1 className="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400 mb-2">
          CROSSWORD
        </h1>
        <SettingsComponent />
      </header>
      
      <main className="flex-1 relative overflow-hidden">
        <GridComponent />
      </main>

      <footer className="absolute bottom-4 right-4 text-slate-500 text-xs italic pointer-events-none z-50">
        Use keyboard, mouse, or pen to solve. Pinch to zoom, drag to pan.
      </footer>
    </div>
  );
}

export default App;
