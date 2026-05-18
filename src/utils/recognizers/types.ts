export type Point = { x: number; y: number; t: number };

export type RecognizerStatus = 'strong' | 'uncertain' | 'failed';

export interface RecognitionCandidate {
  char: string;
  score: number;
  source: string;
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
  engineResults?: { name: string; char: string; score: number }[];
}

export type StrokeInput = Point[][];

