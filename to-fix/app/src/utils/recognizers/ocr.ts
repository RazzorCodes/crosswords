import type { RecognitionResult, StrokeInput } from './types';
import { handwritingModule, initHandwritingModule } from '../handwriting/module';

export async function warmRecognizers(): Promise<void> {
  await initHandwritingModule();
}

export async function recognizeHandwriting(strokes: StrokeInput): Promise<RecognitionResult> {
  return await handwritingModule.asRecognitionResult(strokes);
}
