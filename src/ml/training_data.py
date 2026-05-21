from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sample_store import load_snapshot, normalize_label  # noqa: E402


def stable_sample_key(sample: dict[str, Any]) -> str:
    raw = f"{sample.get('id', '')}:{sample.get('label', '')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def group_samples_by_label(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        label = normalize_label(sample.get("label"))
        if label is None:
            continue
        grouped.setdefault(label, []).append(sample)

    for label_samples in grouped.values():
        label_samples.sort(key=stable_sample_key)

    return grouped


def split_high_quality_samples(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = group_samples_by_label(samples)
    train_split: list[dict[str, Any]] = []
    eval_split: list[dict[str, Any]] = []

    for label in sorted(grouped):
        label_samples = grouped[label]
        if len(label_samples) == 1:
            train_split.extend(label_samples)
            continue

        eval_count = max(1, round(len(label_samples) * 0.25))
        eval_count = min(eval_count, len(label_samples) - 1)
        eval_split.extend(label_samples[:eval_count])
        train_split.extend(label_samples[eval_count:])

    return train_split, eval_split


def load_training_split() -> dict[str, Any]:
    snapshot = load_snapshot(include_legacy=False)
    high_quality_train, high_quality_eval = split_high_quality_samples(snapshot.usable_high_quality)
    train_samples = snapshot.usable_regular + high_quality_train
    return {
        "snapshot": snapshot,
        "train_samples": train_samples,
        "regular_train": snapshot.usable_regular,
        "high_quality_train": high_quality_train,
        "high_quality_eval": high_quality_eval,
    }
