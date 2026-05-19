import type { EngineRecognitionResult, RecognitionCandidate, StrokeInput } from './types';

let recognizerPromise: Promise<{ getPrediction: (strokes: unknown) => Promise<any[]> } | null> | null = null;

function normalizeChar(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  const char = value.trim().charAt(0).toUpperCase();
  return /^[A-Z]$/.test(char) ? char : null;
}

function parsePredictionResult(predictions: any[]): RecognitionCandidate[] {
  console.log('[Native OCR] Raw Predictions:', predictions);
  
  if (!Array.isArray(predictions)) return [];

  const candidates: RecognitionCandidate[] = [];

  for (let i = 0; i < predictions.length; i++) {
    const pred = predictions[i];
    const text = typeof pred === 'string' ? pred : pred.text;
    const char = normalizeChar(text);
    
    if (char) {
      // The API doesn't give scores, so we assign based on rank
      // 1.0 for first, 0.8 for second, 0.6 for third
      const score = Math.max(0.4, 1.0 - i * 0.2);
      candidates.push({
        char,
        score,
        source: 'native',
      });
    }

    // Stop at 3 unique candidates
    if (candidates.length >= 3) break;
  }

  // Deduping (sometimes API returns same char for different segmentations)
  const deduped = new Map<string, RecognitionCandidate>();
  for (const c of candidates) {
    if (!deduped.has(c.char)) {
      deduped.set(c.char, c);
    }
  }

  return [...deduped.values()].slice(0, 3);
}

function getStatus(candidates: RecognitionCandidate[]): EngineRecognitionResult['status'] {
  if (candidates.length === 0) return 'failed';
  
  const top = candidates[0];

  // Since we don't have real scores, we rely on the fact that the API 
  // returns them in order of confidence. 
  // If we have multiple candidates, it's "uncertain" unless the first one is significantly better (which we can't know).
  // However, we'll mark single results as 'strong' for better UX.
  if (candidates.length === 1 && top.score >= 0.9) {
    return 'strong';
  }

  // If we have alternatives, it's uncertain by definition
  return 'uncertain';
}

async function getNativeRecognizer() {
  const nav = navigator as any;

  if (!window.isSecureContext) {
    console.error('[Native OCR] Handwriting Recognition requires a Secure Context (HTTPS or localhost).');
    return null;
  }

  if (!nav.createHandwritingRecognizer || typeof nav.createHandwritingRecognizer !== 'function') {
    console.error('[Native OCR] Handwriting Recognition API is NOT supported in this browser. If you are in Chrome, ensure "Experimental Web Platform features" is enabled in chrome://flags if the API is not yet standard in your version.');
    return null;
  }

  if (!recognizerPromise) {
    // Check for "en" language support first
    try {
      if (nav.queryHandwritingRecognizer) {
        const support = await nav.queryHandwritingRecognizer({ languages: ['en'] });
        if (!support) {
          console.error('[Native OCR] Language "en" is NOT supported by the native recognizer.');
          return null;
        }
      }

      recognizerPromise = nav.createHandwritingRecognizer({ languages: ['en'] }).catch((error: any) => {
        console.error('[Native OCR] Initialization failed:', error);
        recognizerPromise = null;
        return null;
      });
    } catch (e) {
      console.warn('[Native OCR] queryHandwritingRecognizer failed, attempting direct creation');
      recognizerPromise = nav.createHandwritingRecognizer({ languages: ['en'] }).catch(() => null);
    }
  }

  return recognizerPromise;
}

export async function warmNativeHandwritingRecognizer() {
  await getNativeRecognizer();
}

export async function recognizeNativeHandwriting(strokes: StrokeInput): Promise<EngineRecognitionResult> {
  const recognizer = await getNativeRecognizer();
  if (!recognizer) {
    return { source: 'native', status: 'failed', candidates: [] };
  }

  try {
    // Map to the format expected by the API: { x, y, t }
    const payload = strokes.map(stroke => stroke.map(p => ({ x: p.x, y: p.y, t: p.t })));
    
    // The API uses a "HandwritingDrawing" object in some versions, 
    // but often accepts the raw stroke array in getPrediction.
    // If it requires an object, we'd need:
    // const drawing = recognizer.createDrawing();
    // strokes.forEach(s => drawing.addStroke(s));
    // const prediction = await drawing.getPrediction();
    
    let prediction;
    if ((recognizer as any).createDrawing) {
      const drawing = (recognizer as any).createDrawing();
      for (const stroke of payload) {
        drawing.addStroke(stroke);
      }
      prediction = await drawing.getPrediction();
    } else {
      prediction = await (recognizer as any).getPrediction(payload);
    }

    const candidates = parsePredictionResult(prediction);
    
    return {
      source: 'native',
      status: getStatus(candidates),
      candidates,
    };
  } catch (error) {
    console.error('[Native OCR] Prediction failed:', error);
    return { source: 'native', status: 'failed', candidates: [] };
  }
}
