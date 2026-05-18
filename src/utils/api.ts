import { StrokeInput } from './recognizers/types';

export async function submitStrokeData(label: string, strokes: StrokeInput) {
  try {
    const response = await fetch('http://localhost:8000', {
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
