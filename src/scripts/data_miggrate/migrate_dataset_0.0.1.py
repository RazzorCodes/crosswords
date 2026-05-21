from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Legacy JSONL file")
    parser.add_argument("--dest", required=True, help="New JSON output file")
    return parser.parse_args()


def normalize_extra_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower().replace("_", "-").replace(" ", "-")
    filtered = "".join(char for char in text if char.isalnum() or char == "-")
    collapsed = "-".join(part for part in filtered.split("-") if part)
    if not collapsed:
        return None
    if collapsed in {"regular", "high-quality", "highquality"}:
        return None
    return collapsed


def has_usable_strokes(strokes: Any) -> bool:
    if not isinstance(strokes, list) or not strokes:
        return False
    for stroke in strokes:
        if not isinstance(stroke, list) or not stroke:
            return False
        for point in stroke:
            if not isinstance(point, dict):
                return False
            for key in ("x", "y", "t"):
                value = point.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    return False
    return True


def fallback_sample_id(source_path: Path, line_number: int) -> str:
    return f"{source_path.stem}-{line_number:06d}"


def dataset_key_for(dest_path: Path) -> str:
    return dest_path.stem


def migrate_records(source_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    migrated: list[dict[str, Any]] = []
    errors: list[str] = []

    try:
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    errors.append(f"{source_path}:{line_number}: empty line")
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{source_path}:{line_number}: invalid json: {exc.msg}")
                    continue

                if not isinstance(record, dict):
                    errors.append(f"{source_path}:{line_number}: record is not an object")
                    continue

                strokes = record.get("strokes")
                if not has_usable_strokes(strokes):
                    errors.append(f"{source_path}:{line_number}: missing usable strokes")
                    continue

                sample_id = record.get("id")
                if not isinstance(sample_id, str) or not sample_id.strip():
                    sample_id = fallback_sample_id(source_path, line_number)
                else:
                    sample_id = sample_id.strip()

                sample_label = record.get("label")
                if isinstance(sample_label, str):
                    sample_label = sample_label.strip()
                else:
                    sample_label = ""

                extra_labels: list[str] = ["legacy"]
                for raw_label in (record.get("source"), record.get("mode")):
                    normalized = normalize_extra_label(raw_label)
                    if normalized and normalized not in extra_labels:
                        extra_labels.append(normalized)

                migrated.append(
                    {
                        "sample_id": sample_id,
                        "sample_label": sample_label,
                        "extra_labels": extra_labels,
                        "strokes": strokes,
                    }
                )
    except OSError as exc:
        raise RuntimeError(f"failed to read {source_path}: {exc}") from exc

    return migrated, errors


def write_dataset(dest_path: Path, samples: list[dict[str, Any]]) -> None:
    payload = {dataset_key_for(dest_path): samples}
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise RuntimeError(f"failed to write {dest_path}: {exc}") from exc


def main() -> int:
    args = parse_args()
    source_path = Path(args.source)
    dest_path = Path(args.dest)

    try:
        samples, errors = migrate_records(source_path)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for error in errors:
        print(error, file=sys.stderr)

    if not samples:
        print(f"{source_path}: no valid samples produced", file=sys.stderr)
        return 1

    try:
        write_dataset(dest_path, samples)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"migrated {len(samples)} samples from {source_path} to {dest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
