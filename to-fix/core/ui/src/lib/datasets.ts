import { normalizeLabel } from './canvas.ts';
import type { DatasetEntry, DatasetManifest, LabSample, Point, StrokeInput } from './types.ts';
import { extractFeatures } from './wasmCore.ts';

function assetUrl(path: string): string {
  return new URL(path.replace(/^\/+/, ''), new URL(import.meta.env.BASE_URL, window.location.origin)).toString();
}

interface RawDatasetRecord {
  id?: string;
  label?: string;
  strokes?: unknown;
  created_at?: string;
  createdAt?: number;
}

function isPoint(value: unknown): value is Point {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const point = value as Record<string, unknown>;
  return typeof point.x === 'number' && typeof point.y === 'number' && typeof point.t === 'number';
}

export function parseDatasetJsonl(text: string): RawDatasetRecord[] {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as RawDatasetRecord);
}

function normalizeStrokes(value: unknown): StrokeInput | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const strokes = value
    .filter(Array.isArray)
    .map((stroke) => stroke.filter(isPoint).map((point) => ({ x: point.x, y: point.y, t: point.t })))
    .filter((stroke) => stroke.length > 0);
  return strokes.length > 0 ? strokes : null;
}

function createdAt(record: RawDatasetRecord): number {
  if (typeof record.createdAt === 'number' && Number.isFinite(record.createdAt)) {
    return record.createdAt;
  }
  if (record.created_at) {
    const parsed = Date.parse(record.created_at);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return Date.now();
}

export async function loadDatasetManifest(): Promise<DatasetManifest> {
  const response = await fetch(assetUrl('datasets/manifest.json'));
  if (!response.ok) {
    return { version: 'missing', datasets: [] };
  }
  const contentType = response.headers.get('content-type') ?? '';
  if (contentType.includes('text/html')) {
    throw new Error('Dataset manifest resolved to HTML; make sure the standalone core-lab server is serving public/datasets.');
  }
  return (await response.json()) as DatasetManifest;
}

export async function loadDatasetSamples(entries: DatasetEntry[]): Promise<LabSample[]> {
  const samples: LabSample[] = [];
  for (const entry of entries) {
    const response = await fetch(assetUrl(entry.url));
    if (!response.ok) {
      throw new Error(`Failed to load ${entry.label}: HTTP ${response.status}`);
    }
    const records = parseDatasetJsonl(await response.text());
    for (const record of records) {
      const label = normalizeLabel(record.label ?? '');
      const strokes = normalizeStrokes(record.strokes);
      if (!label || !strokes) {
        continue;
      }
      samples.push({
        id: `${entry.id}:${record.id ?? samples.length}`,
        label,
        strokes,
        features: extractFeatures(strokes),
        createdAt: createdAt(record),
        source: 'core-lab',
      });
    }
  }
  return samples;
}

export function mergeDatasetSamples(existing: LabSample[], incoming: LabSample[]): LabSample[] {
  const seen = new Set(existing.map((sample) => sample.id));
  const merged = [...existing];
  for (const sample of incoming) {
    if (!seen.has(sample.id)) {
      merged.push(sample);
      seen.add(sample.id);
    }
  }
  return merged;
}
