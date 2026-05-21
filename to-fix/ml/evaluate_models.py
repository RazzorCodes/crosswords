from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from cnn_builder import render_stroke_entry
from cnn_model import get_cnn_model
from trainer_1 import extract_features
from training_data import load_training_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
KNN_K = 5
KNN_FAR_DISTANCE = float(os.getenv("KNN_FAR_DISTANCE", "4.5"))


def build_probability_map(values: np.ndarray) -> dict[str, float]:
    return {chr(65 + index): float(values[index]) for index in range(26)}


def prepare_knn(train_samples):
    if not train_samples:
        return None
    features = np.array([extract_features(sample) for sample in train_samples], dtype=np.float32)
    labels = np.array([sample["label"].upper() for sample in train_samples])
    return features, labels


def knn_probabilities(prepared, sample) -> dict[str, float]:
    if prepared is None:
        return {chr(65 + index): 0.0 for index in range(26)}

    train_features, train_labels = prepared
    feature = extract_features(sample).astype(np.float32)
    distances = np.linalg.norm(train_features - feature, axis=1)
    nearest_indices = np.argsort(distances)[:KNN_K]
    nearest_distances = distances[nearest_indices]
    nearest_labels = train_labels[nearest_indices]

    if len(nearest_distances) == 0 or float(nearest_distances[0]) > KNN_FAR_DISTANCE:
        return {chr(65 + index): 0.0 for index in range(26)}

    weights = 1.0 / np.maximum(nearest_distances, 1e-6)
    scores = {chr(65 + index): 0.0 for index in range(26)}
    total = float(weights.sum())
    for label, weight in zip(nearest_labels, weights):
        scores[str(label)] += float(weight)

    if total > 0:
        for key in scores:
            scores[key] /= total
    return scores


def load_svm(models_dir: Path):
    path = models_dir / "svm.onnx"
    if not path.exists():
        return None
    return ort.InferenceSession(str(path))


def load_cnn(models_dir: Path):
    path = models_dir / "cnn.onnx"
    if not path.exists():
        return None
    return ort.InferenceSession(str(path))


def cnn_probabilities(session, sample) -> dict[str, float]:
    if session is None:
        return {chr(65 + index): 1.0 / 26 for index in range(26)}

    image = render_stroke_entry(sample, size=64)
    # Match browser rendering and tensor shape [1, 1, 64, 64]
    tensor = image.astype(np.float32).reshape(1, 1, 64, 64) / 255.0
    
    inputs = {session.get_inputs()[0].name: tensor}
    outputs = session.run(None, inputs)
    probs = outputs[0][0]
    
    return build_probability_map(probs)


def svm_probabilities(session, sample) -> dict[str, float]:
    if session is None:
        return {chr(65 + index): 1.0 / 26 for index in range(26)}

    features = extract_features(sample).reshape(1, -1).astype(np.float32)
    # svm.onnx input name is 'float_input'
    # output is [label, probabilities]
    inputs = {session.get_inputs()[0].name: features}
    outputs = session.run(None, inputs)
    # outputs[1] is a list of dicts or a numpy array depending on zipmap
    # since we used zipmap=False in skl2onnx, it's a numpy array of shape (1, num_classes)
    probs = outputs[1][0]
    
    # We need to map these probabilities to letters.
    # skl2onnx usually preserves label order or provides it in classes.
    # In our case, the labels are uppercase A-Z.
    # We assume alphabetical order since they are single chars.
    return {chr(65 + index): float(probs[index]) for index in range(26)}


def top_label(probabilities: dict[str, float]) -> str:
    return max(probabilities.items(), key=lambda item: item[1])[0]


def evaluate() -> None:
    models_dir = Path(os.getenv("MODELS_DIR", str(DEFAULT_MODELS_DIR))).expanduser()
    split = load_training_split()
    eval_samples = split["high_quality_eval"]

    metrics = {
        "eval_sample_count": len(eval_samples),
        "knn_top1": None,
        "svm_top1": None,
        "cnn_top1": None,
        "ensemble_top1": None,
    }

    if not eval_samples:
        print(json.dumps(metrics))
        return

    knn = prepare_knn(split["train_samples"])
    svm = load_svm(models_dir)
    cnn = load_cnn(models_dir)

    knn_hits = 0
    svm_hits = 0
    cnn_hits = 0
    ensemble_hits = 0

    for sample in eval_samples:
        label = sample["label"].upper()
        knn_probs = knn_probabilities(knn, sample)
        svm_probs = svm_probabilities(svm, sample)
        cnn_probs = cnn_probabilities(cnn, sample)
        ensemble_probs = {
            char: (0.50 * knn_probs[char]) + (0.25 * svm_probs[char]) + (0.25 * cnn_probs[char])
            for char in knn_probs
        }

        if top_label(knn_probs) == label:
            knn_hits += 1
        if top_label(svm_probs) == label:
            svm_hits += 1
        if top_label(cnn_probs) == label:
            cnn_hits += 1
        if top_label(ensemble_probs) == label:
            ensemble_hits += 1

    total = len(eval_samples)
    metrics["knn_top1"] = knn_hits / total
    metrics["svm_top1"] = svm_hits / total
    metrics["cnn_top1"] = cnn_hits / total
    metrics["ensemble_top1"] = ensemble_hits / total
    print(json.dumps(metrics))

    # Basic tests / sanity checks
    min_cnn_acc = float(os.getenv("MIN_CNN_ACCURACY", "0.60"))
    min_ensemble_acc = float(os.getenv("MIN_ENSEMBLE_ACCURACY", "0.75"))

    if metrics["cnn_top1"] < min_cnn_acc:
        print(f"FAILED: CNN accuracy {metrics['cnn_top1']:.2f} < {min_cnn_acc:.2f}", file=sys.stderr)
        sys.exit(1)

    if metrics["ensemble_top1"] < min_ensemble_acc:
        print(f"FAILED: Ensemble accuracy {metrics['ensemble_top1']:.2f} < {min_ensemble_acc:.2f}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    evaluate()
