import { InferenceSession, Tensor } from 'onnxruntime-web';
import { knnRecognizer } from './knn';
import { extractFeatures } from './features';
import { renderStrokesToPixels } from './rasterizer';
import type { RecognitionResult, StrokeInput, RecognitionCandidate } from './types';

let svmSession: InferenceSession | null = null;
let cnnSession: InferenceSession | null = null;

export async function warmRecognizers() {
  try {
    if (!svmSession) {
      svmSession = await InferenceSession.create('/models/svm.onnx');
    }
    if (!cnnSession) {
      cnnSession = await InferenceSession.create('/models/cnn.onnx');
    }
  } catch (e) {
    console.warn('Failed to load ONNX models (this is expected if they haven\'t been exported yet):', e);
  }
}

export async function recognizeHandwriting(strokes: StrokeInput): Promise<RecognitionResult> {
  const sourceTrail: string[] = [];
  await warmRecognizers();
  
  // 1. Get k-NN predictions (instant, browser)
  const knnCandidates = knnRecognizer.predict(strokes);
  const knnProbs: Record<string, number> = {};
  for (let i = 65; i <= 90; i++) knnProbs[String.fromCharCode(i)] = 0;
  knnCandidates.forEach(c => { knnProbs[c.char] = c.score; });

  // 2. SVM Prediction (ONNX)
  const svmProbs: Record<string, number> = {};
  let svmBestLabel = '?';
  if (svmSession) {
    try {
        const features = extractFeatures(strokes);
        const input = new Tensor('float32', new Float32Array(features), [1, 30]);
        const results = await svmSession.run({ float_input: input });
        
        // Handle probabilities output
        const probsData = results.probabilities.data;
        if (probsData instanceof Float32Array) {
            for (let i = 0; i < 26; i++) {
                svmProbs[String.fromCharCode(65 + i)] = probsData[i];
            }
            const labelIdx = Number(results.label.data[0]);
            svmBestLabel = String.fromCharCode(65 + labelIdx);
        } else {
            // Sequence of maps or other format
            const map = (probsData as any)[0];
            if (map instanceof Map) {
                map.forEach((prob, char) => {
                    svmProbs[char.toUpperCase()] = prob as number;
                });
            } else {
                for (const [char, prob] of Object.entries(map)) {
                    svmProbs[char.toUpperCase()] = prob as number;
                }
            }
            svmBestLabel = String(results.label.data[0]).toUpperCase();
        }
    } catch (e) {
        console.error('SVM Inference error:', e);
    }
  }
  
  if (Object.keys(svmProbs).length === 0) {
      for (let i = 65; i <= 90; i++) svmProbs[String.fromCharCode(i)] = 1.0 / 26;
  }

  // 3. CNN Prediction (ONNX)
  const cnnProbs: Record<string, number> = {};
  let cnnBestLabel = '?';
  if (cnnSession) {
    try {
        const pixels = renderStrokesToPixels(strokes);
        const input = new Tensor('float32', pixels, [1, 1, 64, 64]);
        const results = await cnnSession.run({ input });
        const probs = results.output.data as Float32Array;
        for (let i = 0; i < 26; i++) {
            cnnProbs[String.fromCharCode(65 + i)] = probs[i];
        }
        let maxProb = -1;
        let maxIdx = 0;
        for (let i = 0; i < 26; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIdx = i;
            }
        }
        cnnBestLabel = String.fromCharCode(65 + maxIdx);
    } catch (e) {
        console.error('CNN Inference error:', e);
    }
  }
  
  if (Object.keys(cnnProbs).length === 0) {
      for (let i = 65; i <= 90; i++) cnnProbs[String.fromCharCode(i)] = 1.0 / 26;
  }

  // 4. Weighted voting
  const candidates: RecognitionCandidate[] = [];
  for (let i = 65; i <= 90; i++) {
    const char = String.fromCharCode(i);
    const score = (0.5 * knnProbs[char]) + (0.3 * (svmProbs[char] || 0)) + (0.2 * (cnnProbs[char] || 0));
    candidates.push({ char, score, source: 'ensemble' });
  }

  candidates.sort((a, b) => b.score - a.score);
  const topCandidates = candidates.slice(0, 3);
  
  sourceTrail.push('knn', 'svm', 'cnn', 'weighted-vote');

  const bestLabel = topCandidates[0].char;
  const bestScore = topCandidates[0].score;

  // Local Teacher Logic
  let action: 'accept' | 'prompt' | 'discard' = 'discard';
  if (bestScore > 0.85) action = 'accept';
  else if (bestScore > 0.60) action = 'prompt';

  return {
    status: action === 'accept' ? 'confirmed' : 'uncertain',
    candidates: topCandidates,
    chosenChar: action === 'accept' ? bestLabel : null,
    sourceTrail,
    engineResults: [
        { name: 'k-NN', char: knnCandidates[0]?.char ?? '?', score: knnCandidates[0]?.score ?? 0 },
        { name: 'SVM', char: svmBestLabel, score: svmProbs[svmBestLabel] ?? 0 },
        { name: 'CNN', char: cnnBestLabel, score: cnnProbs[cnnBestLabel] ?? 0 },
    ],
    teacher: {
        label: bestLabel,
        confidence: bestScore,
        action
    }
  };
}
