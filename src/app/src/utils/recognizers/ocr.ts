import { InferenceSession, Tensor } from 'onnxruntime-web';
import { knnRecognizer } from './knn';
import { extractFeatures } from './features';
import { renderStrokesToPixels } from './rasterizer';
import type { EngineResult, RecognitionResult, StrokeInput, RecognitionCandidate } from './types';

let svmSession: InferenceSession | null = null;
let cnnSession: InferenceSession | null = null;
let warmPromise: Promise<void> | null = null;
let lastWarmFailureAt = 0;

const RECOGNIZER_RETRY_DELAY_MS = 30_000;
const AUTO_ACCEPT_SCORE = 0.92;
const AUTO_ACCEPT_MARGIN = 0.12;
type TensorSequence = readonly unknown[] | (ArrayBufferView & { readonly length: number; [index: number]: unknown });

function alphabetMap(defaultValue: number) {
  const values: Record<string, number> = {};
  for (let code = 65; code <= 90; code += 1) {
    values[String.fromCharCode(code)] = defaultValue;
  }
  return values;
}

function isLetterLabel(value: unknown): value is string {
  return typeof value === 'string' && /^[A-Z]$/i.test(value);
}

function normalizeLabelValue(value: unknown): string | null {
  if (typeof value === 'number' && Number.isFinite(value) && value >= 0 && value < 26) {
    return String.fromCharCode(65 + value);
  }
  if (isLetterLabel(value)) {
    return value.toUpperCase();
  }
  return null;
}

function getSessionValue(
  session: InferenceSession,
  results: Record<string, unknown>,
  preferredNames: string[],
): unknown {
  for (const name of preferredNames) {
    if (name in results) {
      return results[name];
    }
  }
  for (const name of session.outputNames) {
    if (name in results) {
      return results[name];
    }
  }
  return Object.values(results)[0];
}

function getOutputName(session: InferenceSession, preferredNames: string[]): string | null {
  for (const name of preferredNames) {
    if (session.outputNames.includes(name)) {
      return name;
    }
  }
  return session.outputNames[0] ?? null;
}

function isTensorOutput(session: InferenceSession, outputName: string): boolean {
  const metadata = session.outputMetadata.find((item) => item.name === outputName);
  if (!metadata) {
    return true;
  }
  return metadata.isTensor;
}

function getTensorData(value: unknown): TensorSequence | null {
  if (!value || typeof value !== 'object') {
    return null;
  }
  if ('data' in value) {
    const data = (value as { data?: unknown }).data;
    if (Array.isArray(data) || ArrayBuffer.isView(data)) {
      return data as TensorSequence;
    }
  }
  return null;
}

function fillProbabilitiesFromSequence(target: Record<string, number>, sequence: TensorSequence): boolean {
  let updated = false;
  const values = Array.from(sequence);
  for (let index = 0; index < Math.min(values.length, 26); index += 1) {
    const value = values[index];
    if (typeof value === 'number' && Number.isFinite(value)) {
      target[String.fromCharCode(65 + index)] = value;
      updated = true;
    }
  }
  return updated;
}

function fillProbabilitiesFromMap(target: Record<string, number>, value: unknown): boolean {
  let updated = false;
  if (value instanceof Map) {
    for (const [label, score] of value.entries()) {
      const normalized = normalizeLabelValue(label);
      if (normalized && typeof score === 'number' && Number.isFinite(score)) {
        target[normalized] = score;
        updated = true;
      }
    }
    return updated;
  }

  if (!value || typeof value !== 'object') {
    return false;
  }

  for (const [label, score] of Object.entries(value as Record<string, unknown>)) {
    const normalized = normalizeLabelValue(label);
    if (normalized && typeof score === 'number' && Number.isFinite(score)) {
      target[normalized] = score;
      updated = true;
    }
  }
  return updated;
}

function getBestProbabilityLabel(probabilities: Record<string, number>): string | null {
  let bestLabel: string | null = null;
  let bestScore = -Infinity;
  for (const [label, score] of Object.entries(probabilities)) {
    if (score > bestScore) {
      bestLabel = label;
      bestScore = score;
    }
  }
  return bestLabel;
}

export async function warmRecognizers() {
  if (svmSession && cnnSession) {
    return;
  }

  if (warmPromise) {
    return warmPromise;
  }

  if (lastWarmFailureAt > 0 && Date.now() - lastWarmFailureAt < RECOGNIZER_RETRY_DELAY_MS) {
    return;
  }

  warmPromise = (async () => {
    try {
      const config = (window as any).CROSSWORDS_CONFIG || {};
      const baseUrl = (config.MODEL_BASE_URL || '/models').replace(/\/$/, '');
      
      if (!svmSession) {
        svmSession = await InferenceSession.create(`${baseUrl}/svm.onnx`);
      }
      if (!cnnSession) {
        cnnSession = await InferenceSession.create(`${baseUrl}/cnn.onnx`);
      }
      lastWarmFailureAt = 0;
    } catch (error) {
      lastWarmFailureAt = Date.now();
      console.warn('Failed to load ONNX models:', error);
    } finally {
      warmPromise = null;
    }
  })();

  return warmPromise;
}

export async function recognizeHandwriting(strokes: StrokeInput): Promise<RecognitionResult> {
  await warmRecognizers();

  const sourceTrail: string[] = ['knn', 'svm', 'cnn', 'weighted-vote'];

  const knnCandidates = knnRecognizer.predict(strokes);
  const knnProbs = alphabetMap(0);
  knnCandidates.forEach((candidate) => {
    knnProbs[candidate.char] = candidate.score;
  });

  const svmProbs = alphabetMap(1 / 26);
  let svmBestLabel = '?';
  let svmHasProbabilities = false;
  let svmStatus: EngineResult['status'] = svmSession ? 'ready' : 'unavailable';
  let svmDetail: string | undefined = svmSession ? undefined : 'Model not loaded';
  if (svmSession) {
    try {
      const features = extractFeatures(strokes);
      const input = new Tensor('float32', new Float32Array(features), [1, 30]);
      const inputName = svmSession.inputNames[0] ?? 'float_input';
      const labelOutputName = getOutputName(svmSession, ['label', 'output_label']);
      const probabilitiesOutputName = getOutputName(svmSession, ['probabilities', 'output_probability']);
      const fetches: string[] = [];

      if (labelOutputName) {
        fetches.push(labelOutputName);
      }
      if (probabilitiesOutputName && isTensorOutput(svmSession, probabilitiesOutputName)) {
        fetches.push(probabilitiesOutputName);
      } else if (probabilitiesOutputName) {
        svmDetail = 'Label-only model output';
      }

      const results = fetches.length > 0
        ? await svmSession.run({ [inputName]: input }, fetches)
        : await svmSession.run({ [inputName]: input });
      const labelValue = labelOutputName ? (results as Record<string, unknown>)[labelOutputName] : null;
      const probabilitiesValue = probabilitiesOutputName ? (results as Record<string, unknown>)[probabilitiesOutputName] : null;

      const labelData = getTensorData(labelValue);
      const normalizedLabel = labelData ? normalizeLabelValue(Array.from(labelData)[0]) : normalizeLabelValue(labelValue);
      if (normalizedLabel) {
        svmBestLabel = normalizedLabel;
      }

      const probsData = getTensorData(probabilitiesValue);
      if (probsData) {
        svmHasProbabilities = fillProbabilitiesFromSequence(svmProbs, probsData);
      } else if (Array.isArray(probabilitiesValue)) {
        const first = probabilitiesValue[0];
        if (first && typeof first === 'object') {
          svmHasProbabilities = fillProbabilitiesFromMap(svmProbs, first);
        } else {
          svmHasProbabilities = fillProbabilitiesFromSequence(svmProbs, probabilitiesValue);
        }
      } else {
        svmHasProbabilities = fillProbabilitiesFromMap(svmProbs, probabilitiesValue);
      }

      if (svmBestLabel === '?') {
        const bestFromProbabilities = getBestProbabilityLabel(svmProbs);
        if (bestFromProbabilities) {
          svmBestLabel = bestFromProbabilities;
        }
      } else if (!svmHasProbabilities) {
        for (let code = 65; code <= 90; code += 1) {
          const char = String.fromCharCode(code);
          svmProbs[char] = char === svmBestLabel ? 1 : 0;
        }
      }
    } catch (error) {
      svmStatus = 'error';
      svmDetail = 'Inference failed';
      console.error('SVM inference error:', error);
    }
  }

  const cnnProbs = alphabetMap(1 / 26);
  let cnnBestLabel = '?';
  let cnnStatus: EngineResult['status'] = cnnSession ? 'ready' : 'unavailable';
  let cnnDetail: string | undefined = cnnSession ? undefined : 'Model not loaded';
  if (cnnSession) {
    try {
      const pixels = renderStrokesToPixels(strokes);
      const input = new Tensor('float32', pixels, [1, 1, 64, 64]);
      const results = await cnnSession.run({ [cnnSession.inputNames[0]]: input });
      const outputValue = getSessionValue(cnnSession, results as Record<string, unknown>, ['output']);
      const probsData = getTensorData(outputValue);
      const probs = probsData ? Array.from(probsData as ArrayLike<number>) : [];
      let maxProb = -1;
      let maxIndex = 0;
      for (let index = 0; index < Math.min(probs.length, 26); index += 1) {
        const char = String.fromCharCode(65 + index);
        cnnProbs[char] = probs[index];
        if (probs[index] > maxProb) {
          maxProb = probs[index];
          maxIndex = index;
        }
      }
      cnnBestLabel = String.fromCharCode(65 + maxIndex);
    } catch (error) {
      cnnStatus = 'error';
      cnnDetail = 'Inference failed';
      console.error('CNN inference error:', error);
    }
  }

  const candidates: RecognitionCandidate[] = [];
  for (let code = 65; code <= 90; code += 1) {
    const char = String.fromCharCode(code);
    const score = (0.50 * knnProbs[char]) + (0.25 * svmProbs[char]) + (0.25 * cnnProbs[char]);
    candidates.push({ char, score, source: 'ensemble' });
  }
  candidates.sort((a, b) => b.score - a.score);

  const topCandidates = candidates.slice(0, 3);
  const bestLabel = topCandidates[0]?.char ?? '?';
  const bestScore = topCandidates[0]?.score ?? 0;
  const secondScore = topCandidates[1]?.score ?? 0;
  const margin = bestScore - secondScore;

  const engineVotes = [
    knnCandidates[0]?.char ?? '?',
    svmBestLabel,
    cnnBestLabel,
  ];
  const agreementCount = engineVotes.filter((char) => char === bestLabel).length;
  const autoAccepted = (
    bestScore >= AUTO_ACCEPT_SCORE &&
    margin >= AUTO_ACCEPT_MARGIN &&
    agreementCount >= 2
  );

  return {
    status: autoAccepted ? 'confirmed' : 'uncertain',
    candidates: topCandidates,
    chosenChar: autoAccepted ? bestLabel : null,
    sourceTrail,
    engineResults: [
      knnRecognizer.getExampleCount() > 0
        ? {
            name: 'k-NN',
            char: knnCandidates[0]?.char ?? '?',
            score: knnCandidates[0]?.score ?? 0,
            status: 'ready',
          }
        : {
            name: 'k-NN',
            char: null,
            score: null,
            status: 'unavailable',
            detail: 'No local samples yet',
          },
      {
        name: 'SVM',
        char: svmBestLabel === '?' ? null : svmBestLabel,
        score: svmBestLabel === '?' || !svmHasProbabilities ? null : (svmProbs[svmBestLabel] ?? null),
        status: svmBestLabel === '?' && svmStatus === 'ready' ? 'error' : svmStatus,
        detail: svmBestLabel === '?' && svmStatus === 'ready'
          ? 'No label returned'
          : (!svmHasProbabilities && svmStatus === 'ready' ? (svmDetail ?? 'Label-only output') : svmDetail),
      },
      {
        name: 'CNN',
        char: cnnBestLabel === '?' ? null : cnnBestLabel,
        score: cnnBestLabel === '?' ? null : (cnnProbs[cnnBestLabel] ?? null),
        status: cnnBestLabel === '?' && cnnStatus === 'ready' ? 'error' : cnnStatus,
        detail: cnnBestLabel === '?' && cnnStatus === 'ready' ? 'No label returned' : cnnDetail,
      },
    ],
    teacher: {
      label: bestLabel,
      confidence: bestScore,
      action: autoAccepted ? 'accept' : 'prompt',
    },
  };
}
