import { RecognitionCandidate, StrokeInput } from './types';
import { extractFeatures } from './features';

interface KNNExample {
  features: number[];
  label: string;
}

const STORAGE_KEY = 'knn_handwriting_data';
const K = 5;

export class KNNRecognizer {
  private examples: KNNExample[] = [];

  constructor() {
    this.load();
  }

  private load() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        this.examples = JSON.parse(stored);
      } catch (e) {
        console.error('Failed to load k-NN data', e);
        this.examples = [];
      }
    }
  }

  private save() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(this.examples));
  }

  addExample(strokes: StrokeInput, label: string) {
    const features = extractFeatures(strokes);
    this.examples.push({ features, label: label.toUpperCase() });
    this.save();
  }

  predict(strokes: StrokeInput): RecognitionCandidate[] {
    if (this.examples.length === 0) return [];

    const features = extractFeatures(strokes);
    const distances = this.examples.map(ex => ({
      label: ex.label,
      dist: this.euclideanDistance(features, ex.features)
    }));

    distances.sort((a, b) => a.dist - b.dist);
    const nearest = distances.slice(0, K);

    const counts: Record<string, number> = {};
    for (const n of nearest) {
      counts[n.label] = (counts[n.label] || 0) + 1;
    }

    return Object.entries(counts)
      .map(([label, count]) => ({
        char: label,
        score: count / K,
        source: 'knn'
      }))
      .sort((a, b) => b.score - a.score);
  }

  private euclideanDistance(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += Math.pow(a[i] - b[i], 2);
    }
    return Math.sqrt(sum);
  }
}

export const knnRecognizer = new KNNRecognizer();
