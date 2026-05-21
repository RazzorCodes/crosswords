import { useGameStore } from '../store/useGameStore';
import { useHandwritingStore } from '../store/useHandwritingStore';

export function SettingsComponent() {
  const { language, setLanguage, showGlow, setShowGlow, suggestions, showLeftPanel, setShowLeftPanel } = useGameStore();
  const { trainMode, moduleReady, trainingState } = useHandwritingStore();

  const hasSuggestions = suggestions.length > 0;

  return (
    <div className="flex gap-4 items-center">
      <div className="hidden md:flex items-center gap-2">
        <span className="text-[10px] text-slate-500 font-bold uppercase">Side Panel</span>
        <button
          onClick={() => setShowLeftPanel(!showLeftPanel)}
          className={`px-2 py-0.5 text-xs rounded border transition-colors ${showLeftPanel ? 'bg-blue-600/20 text-blue-400 border-blue-600/50' : 'bg-slate-800 text-slate-400 border-slate-700'}`}
        >
          {showLeftPanel ? 'CLUES ON' : 'CLUES OFF'}
        </button>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-[10px] text-slate-500 font-bold uppercase">Language</span>
        <div className="flex bg-slate-800 p-0.5 rounded border border-slate-700">
          <button
            onClick={() => setLanguage('en')}
            className={`px-2 py-0.5 text-xs rounded ${language === 'en' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:bg-slate-700'}`}
          >
            EN
          </button>
          <button
            onClick={() => setLanguage('ro')}
            className={`px-2 py-0.5 text-xs rounded ${language === 'ro' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:bg-slate-700'}`}
          >
            RO
          </button>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-[10px] text-slate-500 font-bold uppercase">Feedback</span>
        <button
          onClick={() => setShowGlow(!showGlow)}
          className={`px-2 py-0.5 text-xs rounded border transition-colors ${showGlow ? 'bg-green-600/20 text-green-400 border-green-600/50' : 'bg-slate-800 text-slate-400 border-slate-700'}`}
        >
          {showGlow ? 'GLOW ON' : 'GLOW OFF'}
        </button>
      </div>

      <div className="hidden md:flex items-center gap-2">
        <span className="text-[10px] text-slate-500 font-bold uppercase">Ink</span>
        <span className={`px-2 py-0.5 text-xs rounded border ${hasSuggestions ? 'bg-amber-600/20 text-amber-300 border-amber-600/40' : 'bg-slate-800 text-slate-400 border-slate-700'}`}>
          {hasSuggestions ? 'CHOICES' : 'AUTO'}
        </span>
      </div>

      <div className="hidden md:flex items-center gap-2">
        <span className="text-[10px] text-slate-500 font-bold uppercase">Mode</span>
        <span className={`px-2 py-0.5 text-xs rounded border ${trainMode ? 'bg-amber-600/20 text-amber-300 border-amber-600/40' : 'bg-slate-800 text-slate-400 border-slate-700'}`}>
          {trainMode ? 'TRAIN' : 'PLAY'}
        </span>
      </div>

      <div className="hidden md:flex items-center gap-2">
        <span className="text-[10px] text-slate-500 font-bold uppercase">Trainer</span>
        <span className={`px-2 py-0.5 text-xs rounded border ${moduleReady ? 'bg-emerald-600/20 text-emerald-300 border-emerald-600/40' : 'bg-slate-800 text-slate-400 border-slate-700'}`}>
          {moduleReady ? `${trainingState?.totalAcceptedSamples ?? 0} LOCAL` : 'LOADING'}
        </span>
      </div>
    </div>
  );
}
