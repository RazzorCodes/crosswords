import { WordEntry } from '../types';

const DICTIONARY_URL = '/data/loc-reduse-6.0.txt';
const SAMPLE_SIZE = 220;
const MIN_WORD_LENGTH = 3;
const MAX_WORD_LENGTH = 12;

let dictionaryPromise: Promise<string[]> | null = null;

function normalizeWord(word: string): string {
  return word
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-zA-Z]/g, '')
    .toUpperCase();
}

function shuffle<T>(items: T[]): T[] {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

async function loadDictionaryWords(): Promise<string[]> {
  if (!dictionaryPromise) {
    dictionaryPromise = fetch(DICTIONARY_URL)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Failed to load dictionary: ${response.status}`);
        }

        const text = await response.text();
        const seen = new Set<string>();

        return text
          .split(/\r?\n/)
          .map((line) => normalizeWord(line))
          .filter((word) => {
            if (word.length < MIN_WORD_LENGTH || word.length > MAX_WORD_LENGTH) return false;
            if (!/^[A-Z]+$/.test(word)) return false;
            if (seen.has(word)) return false;
            seen.add(word);
            return true;
          });
      })
      .catch((error) => {
        console.warn('Failed to load imported dictionary:', error);
        return [];
      });
  }

  return dictionaryPromise;
}

export async function loadRomanianDictionaryEntries(): Promise<WordEntry[]> {
  const words = await loadDictionaryWords();
  if (words.length === 0) return [];

  return shuffle(words)
    .slice(0, SAMPLE_SIZE)
    .map((word) => ({
      word,
      clue: `Cuvant din dictionar (${word.length} litere).`,
    }));
}
