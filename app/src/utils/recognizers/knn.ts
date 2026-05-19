import { extractFeatures } from './features';
import { RecognitionCandidate, StrokeInput } from './types';

interface KNNExample {
  id: string;
  features: number[];
  label: string;
}

const STORAGE_KEY = 'knn_handwriting_data_v2';
const K = 5;
const FAR_NEIGHBOR_DISTANCE = 4.5;

function makeLocalId() {
  return `local-${Math.random().toString(36).slice(2, 10)}`;
}

export class KNNRecognizer {
  private examples: KNNExample[] = [];

  constructor() {
    this.load();
  }

  private load() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      this.examples = [];
      return;
    }

    try {
      const parsed = JSON.parse(stored);
      if (!Array.isArray(parsed)) {
        this.examples = [];
        return;
      }
      this.examples = parsed.filter((entry): entry is KNNExample => (
        entry &&
        typeof entry.id === 'string' &&
        Array.isArray(entry.features) &&
        typeof entry.label === 'string'
      ));
    } catch (error) {
      console.error('Failed to load k-NN data', error);
      this.examples = [];
    }
  }

  private save() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(this.examples));
  }

  addExample(strokes: StrokeInput, label: string, id?: string) {
    const features = extractFeatures(strokes);
    this.examples.push({
      id: id ?? makeLocalId(),
      features,
      label: label.toUpperCase(),
    });
    this.save();
  }

  removeExample(id: string) {
    this.examples = this.examples.filter((entry) => entry.id !== id);
    this.save();
  }

  getExampleCount() {
    return this.examples.length;
  }

  predict(strokes: StrokeInput): RecognitionCandidate[] {
    if (this.examples.length === 0) {
      return [];
    }

    const features = extractFeatures(strokes);
    const distances = this.examples.map((entry) => ({
      id: entry.id,
      label: entry.label,
      distance: this.euclideanDistance(features, entry.features),
    }));
    distances.sort((a, b) => a.distance - b.distance);

    const nearest = distances.slice(0, K);
    if (nearest.length === 0 || nearest[0].distance > FAR_NEIGHBOR_DISTANCE) {
      return [];
    }

    let totalWeight = 0;
    const scores: Record<string, number> = {};
    for (const neighbor of nearest) {
      const weight = 1 / Math.max(neighbor.distance, 1e-6);
      totalWeight += weight;
      scores[neighbor.label] = (scores[neighbor.label] || 0) + weight;
    }

    return Object.entries(scores)
      .map(([label, score]) => ({
        char: label,
        score: totalWeight > 0 ? score / totalWeight : 0,
        source: 'knn',
      }))
      .sort((a, b) => b.score - a.score);
  }

  private euclideanDistance(a: number[], b: number[]) {
    let sum = 0;
    for (let index = 0; index < a.length; index += 1) {
      sum += (a[index] - b[index]) ** 2;
    }
    return Math.sqrt(sum);
  }
}

export const knnRecognizer = new KNNRecognizer();
