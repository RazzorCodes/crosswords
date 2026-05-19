import { StrokeInput } from './recognizers/types';

const defaultSrvUrl =
  typeof window === 'undefined'
    ? 'http://localhost:8000'
    : `${window.location.protocol}//${window.location.hostname}:8000`;

const SRV_URL = import.meta.env.VITE_SRV_URL || defaultSrvUrl;

export async function submitStrokeData(label: string, strokes: StrokeInput) {
  try {
    const response = await fetch(SRV_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ label, strokes }),
    });
    
    if (!response.ok) {
      console.error('Failed to submit stroke data:', response.statusText);
    }
  } catch (err) {
    console.error('Error submitting stroke data:', err);
  }
}

export async function fetchLetterStats(): Promise<{counts: Record<string, number>, total: number} | null> {
  try {
    const response = await fetch(`${SRV_URL}/stats`);
    if (!response.ok) return null;
    return await response.json();
  } catch (err) {
    console.warn('Could not fetch letter stats from local server:', err);
    return null;
  }
}
