export type Point = { x: number; y: number; t: number };

export type RecognizerStatus = 'strong' | 'confirmed' | 'uncertain' | 'failed';

export interface RecognitionCandidate {
  char: string;
  score: number;
  source: string;
}

export interface EngineResult {
  name: string;
  char: string | null;
  score: number | null;
  status: 'ready' | 'unavailable' | 'error';
  detail?: string;
}

export interface EngineRecognitionResult {
  source: string;
  status: RecognizerStatus;
  candidates: RecognitionCandidate[];
}

export interface RecognitionResult {
  status: RecognizerStatus;
  candidates: RecognitionCandidate[];
  chosenChar: string | null;
  sourceTrail: string[];
  engineResults?: EngineResult[];
  teacher?: {
    label: string;
    confidence: number;
    action: 'accept' | 'prompt' | 'discard' | 'none';
  };
}

export type StrokeInput = Point[][];
