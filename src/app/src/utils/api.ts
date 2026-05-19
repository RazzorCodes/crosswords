import { StrokeInput } from './recognizers/types';

function resolveSrvUrl(): string {
  if (import.meta.env.VITE_SRV_URL !== undefined) {
    return import.meta.env.VITE_SRV_URL;
  }

  if (typeof window !== 'undefined') {
    const runtimeSrvUrl = window.CROSSWORDS_CONFIG?.SRV_URL;
    if (runtimeSrvUrl !== undefined) {
      return runtimeSrvUrl;
    }
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }

  return import.meta.env.VITE_SRV_URL || 'http://localhost:8000';
}

const SRV_URL = resolveSrvUrl();

export interface SubmitSampleRequest {
  label: string;
  strokes: StrokeInput;
  storedAs: 'regular' | 'high_quality';
  source: string;
  mode: 'train' | 'play';
  metadata?: Record<string, unknown>;
}

export interface SubmitSampleResponse {
  id: string;
  stored_as: 'regular' | 'high_quality';
  queued_for_llm: boolean;
}

export interface TeacherStatus {
  enabled: boolean;
  configured: boolean;
  health_ok: boolean;
  model_supports_multimodal: boolean;
  active: boolean;
  reason: string;
  checked_at: string | null;
  hq_share: number;
  target_share: number;
  pending_queue_size: number;
  model: string;
}

export async function submitSample(request: SubmitSampleRequest): Promise<SubmitSampleResponse | null> {
  if (!SRV_URL) {
    console.warn('Sample submission is disabled because no SRV_URL is configured.');
    return null;
  }

  try {
    const response = await fetch(`${SRV_URL}/samples`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        label: request.label,
        strokes: request.strokes,
        stored_as: request.storedAs,
        source: request.source,
        mode: request.mode,
        metadata: request.metadata ?? {},
      }),
    });

    if (!response.ok) {
      console.error('Failed to submit sample:', response.statusText);
      return null;
    }

    return await response.json();
  } catch (err) {
    console.error('Error submitting sample:', err);
    return null;
  }
}

export async function deleteSample(sampleId: string): Promise<boolean> {
  if (!SRV_URL) {
    console.warn('Sample deletion is disabled because no SRV_URL is configured.');
    return false;
  }

  try {
    const response = await fetch(`${SRV_URL}/samples/${sampleId}`, {
      method: 'DELETE',
    });
    return response.ok;
  } catch (err) {
    console.error('Error deleting sample:', err);
    return false;
  }
}

export async function fetchLetterStats(): Promise<{ counts: Record<string, number>; total: number } | null> {
  if (!SRV_URL) {
    return null;
  }

  try {
    const response = await fetch(`${SRV_URL}/stats`);
    if (!response.ok) return null;
    return await response.json();
  } catch (err) {
    console.warn('Could not fetch letter stats from local server:', err);
    return null;
  }
}

export async function fetchTeacherStatus(): Promise<TeacherStatus | null> {
  if (!SRV_URL) {
    return null;
  }

  try {
    const response = await fetch(`${SRV_URL}/teacher/status`);
    if (!response.ok) return null;
    return await response.json();
  } catch (err) {
    console.warn('Could not fetch teacher status:', err);
    return null;
  }
}
