from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

TILE_SIZE = 64
BATCH_SIZE = 25
GRID_SIZE = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Input dataset JSON file")
    parser.add_argument("--dest", required=True, help="Output dataset JSON file")
    parser.add_argument(
        "--address", default="localhost:4000", help="OpenAI-compatible host:port"
    )
    parser.add_argument("--model", default="fast", help="Model name")
    parser.add_argument("--progress-phase", default="annotate", help=argparse.SUPPRESS)
    parser.add_argument("--file-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--file-total", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--file-name", default="", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def progress_percent(
    file_index: int,
    file_total: int,
    batch_index: int | None = None,
    batch_total: int | None = None,
) -> int:
    if file_total <= 0 or file_index <= 0:
        return 0
    completed_files = max(file_index - 1, 0)
    if batch_index is not None and batch_total and batch_total > 0:
        progress = (completed_files + (batch_index / batch_total)) / file_total
    else:
        progress = file_index / file_total
    return max(0, min(100, int(progress * 100)))


def log_progress(
    phase: str,
    file_index: int,
    file_total: int,
    file_name: str,
    *,
    batch_index: int | None = None,
    batch_total: int | None = None,
    status: str | None = None,
) -> None:
    percent = progress_percent(file_index, file_total, batch_index, batch_total)
    parts = [
        f"[{timestamp()}]",
        phase,
        f"[{percent}%]",
        f"{file_index}/{file_total}"
        if file_total > 0 and file_index > 0
        else file_name,
        file_name,
    ]
    if batch_index is not None and batch_total is not None:
        parts.append(f"- batch {batch_index}/{batch_total}")
    if status:
        parts.append(f"- {status}")
    print(" ".join(part for part in parts if part))


def load_dataset(source_path: Path) -> tuple[str, list[dict[str, Any]]]:
    try:
        with source_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to load {source_path}: {exc}") from exc

    if not isinstance(payload, dict) or len(payload) != 1:
        raise RuntimeError(f"{source_path}: expected one dataset key")

    dataset_key = next(iter(payload))
    samples = payload[dataset_key]
    if not isinstance(samples, list):
        raise RuntimeError(f"{source_path}: dataset value must be a list")
    return dataset_key, samples


def save_dataset(
    dest_path: Path, dataset_key: str, samples: list[dict[str, Any]]
) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest_path.open("w", encoding="utf-8") as handle:
            json.dump({dataset_key: samples}, handle, ensure_ascii=True, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise RuntimeError(f"failed to write {dest_path}: {exc}") from exc


def is_annotated(sample: dict[str, Any]) -> bool:
    labels = sample.get("extra_labels")
    return isinstance(labels, list) and any(label == "annotated" for label in labels)


def ensure_extra_labels(sample: dict[str, Any]) -> list[str]:
    extra_labels = sample.get("extra_labels")
    if not isinstance(extra_labels, list):
        extra_labels = []
        sample["extra_labels"] = extra_labels
    return extra_labels


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


def normalize_strokes_to_tile(
    strokes: list[list[dict[str, float]]], size: int = TILE_SIZE
) -> Image.Image:
    image = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(image)

    xs = [point["x"] for stroke in strokes for point in stroke]
    ys = [point["y"] for stroke in strokes for point in stroke]
    if not xs or not ys:
        return image

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    scale = min((size - 12) / width, (size - 12) / height)
    offset_x = (size - width * scale) / 2
    offset_y = (size - height * scale) / 2

    for stroke in strokes:
        if len(stroke) == 1:
            point = stroke[0]
            x = offset_x + (point["x"] - min_x) * scale
            y = offset_y + (point["y"] - min_y) * scale
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=0)
            continue

        points = [
            (
                offset_x + (point["x"] - min_x) * scale,
                offset_y + (point["y"] - min_y) * scale,
            )
            for point in stroke
        ]
        draw.line(points, fill=0, width=4, joint="curve")

    return image


def render_contact_sheet(samples: list[dict[str, Any]]) -> str:
    sheet = Image.new("L", (GRID_SIZE * TILE_SIZE, GRID_SIZE * TILE_SIZE), 255)
    for index, sample in enumerate(samples):
        tile = normalize_strokes_to_tile(sample["strokes"])
        row = index // GRID_SIZE
        col = index % GRID_SIZE
        sheet.paste(tile, (col * TILE_SIZE, row * TILE_SIZE))

    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def extract_message_text(response: dict[str, Any]) -> str | None:
    content = response.get("choices", [{}])[0].get("message", {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                parts.append(item["text"])
        return "\n".join(parts) if parts else None
    return None


def request_labels(
    address: str, model: str, samples: list[dict[str, Any]]
) -> list[str]:
    image_b64 = render_contact_sheet(samples)
    prompt = (
        "Classify the 25 cells in the 5x5 image grid in row-major order, left-to-right and top-to-bottom.\n"
        "Output rules:\n"
        "1. Return exactly 5 lines.\n"
        "2. Each line must contain exactly 5 items.\n"
        "3. Separate items with commas.\n"
        "4. Each item must be exactly one character: A-Z or ?.\n"
        f"5. Only the first {len(samples)} cells contain drawings. Every remaining blank cell must be ?.\n"
        "6. Use ? unless the letter is clearly identifiable.\n"
        "7. If there is any plausible confusion with another uppercase letter, output ?.\n"
        "8. For C versus G: a glyph that looks like C but also has an inward horizontal bar or hook should be labeled G; if that is still unclear, output ?.\n"
        "9. Do not guess.\n"
        "10. Do not return words, explanations, numbering, spaces, or extra punctuation.\n"
        "11. The final response must have exactly 5 lines and exactly 4 commas on each line.\n"
        "Output template:\n"
        "A,B,C,D,E\n"
        "F,G,H,I,J\n"
        "K,L,M,N,O\n"
        "P,Q,R,S,T\n"
        "U,V,W,X,Y"
    )

    def build_payload(privileged_role: str) -> dict[str, Any]:
        return {
            "model": model,
            "temperature": 0,
            "max_tokens": 64,
            "messages": [
                {
                    "role": privileged_role,
                    "content": (
                        "Context: this request comes from an internal pipeline that annotates "
                        "handwritten uppercase letters from images."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ],
        }

    body: dict[str, Any] | None = None
    last_error: Exception | None = None
    for privileged_role in ("admin", "system"):
        request = urllib.request.Request(
            f"http://{address.rstrip('/')}/chat/completions",
            data=json.dumps(build_payload(privileged_role)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            response_body = exc.read().decode("utf-8", errors="replace")
            if (
                exc.code == 400
                and privileged_role == "admin"
                and "messages[0]" in response_body
            ):
                continue
            raise

    if body is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("annotation request did not produce a response")

    content = extract_message_text(body)
    if not isinstance(content, str):
        raise RuntimeError("missing text response content")

    tokens = [token.strip().upper() for token in content.replace("\n", ",").split(",")]
    tokens = [token for token in tokens if token]
    if len(tokens) != BATCH_SIZE:
        raise RuntimeError(f"expected {BATCH_SIZE} labels, got {len(tokens)}")
    return tokens


def apply_label(sample: dict[str, Any], raw_label: str) -> bool:
    extra_labels = ensure_extra_labels(sample)
    expected_label = sample.get("sample_label")
    if isinstance(expected_label, str):
        expected_label = expected_label.strip().upper()
    else:
        expected_label = ""

    if raw_label == "?":
        if "unreliable" not in extra_labels:
            extra_labels.append("unreliable")
    elif len(raw_label) == 1 and "A" <= raw_label <= "Z":
        if raw_label != expected_label and "unreliable" not in extra_labels:
            extra_labels.append("unreliable")
    else:
        return False

    for label in ("annotated", "auto"):
        if label not in extra_labels:
            extra_labels.append(label)
    return True


def annotate_samples(
    samples: list[dict[str, Any]],
    address: str,
    model: str,
    *,
    phase: str,
    file_index: int,
    file_total: int,
    file_name: str,
) -> tuple[int, int]:
    pending: list[tuple[int, dict[str, Any]]] = [
        (index, sample)
        for index, sample in enumerate(samples)
        if isinstance(sample, dict) and not is_annotated(sample)
    ]

    processed = 0
    failures = 0
    total_batches = max(1, math.ceil(len(pending) / BATCH_SIZE)) if pending else 0

    for batch_number, batch_start in enumerate(
        range(0, len(pending), BATCH_SIZE), start=1
    ):
        batch = pending[batch_start : batch_start + BATCH_SIZE]
        log_progress(
            phase,
            file_index,
            file_total,
            file_name,
            batch_index=batch_number,
            batch_total=total_batches,
        )
        annotatable: list[tuple[int, dict[str, Any]]] = []
        for index, sample in batch:
            if has_usable_strokes(sample.get("strokes")):
                annotatable.append((index, sample))
            else:
                failures += 1
                print(f"sample {index}: missing usable strokes", file=sys.stderr)

        if not annotatable:
            continue

        try:
            labels = request_labels(
                address, model, [sample for _, sample in annotatable]
            )
        except (
            RuntimeError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            failures += len(annotatable)
            print(
                f"batch starting at sample {annotatable[0][0]} failed: {exc}",
                file=sys.stderr,
            )
            log_progress(
                phase,
                file_index,
                file_total,
                file_name,
                batch_index=batch_number,
                batch_total=total_batches,
                status="failed",
            )
            continue

        for position, (index, sample) in enumerate(annotatable):
            if apply_label(sample, labels[position]):
                processed += 1
            else:
                failures += 1
                print(
                    f"sample {index}: invalid label {labels[position]!r}",
                    file=sys.stderr,
                )

    return processed, failures


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_path = Path(args.source)
    dest_path = Path(args.dest)

    try:
        dataset_key, samples = load_dataset(source_path)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    file_name = args.file_name or source_path.name
    processed, failures = annotate_samples(
        samples,
        args.address,
        args.model,
        phase=args.progress_phase,
        file_index=args.file_index,
        file_total=args.file_total,
        file_name=file_name,
    )

    try:
        save_dataset(dest_path, dataset_key, samples)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.file_index > 0 and args.file_total > 0:
        log_progress(
            args.progress_phase,
            args.file_index,
            args.file_total,
            file_name,
            status=f"processed {processed} samples with {failures} failures",
        )
    else:
        print(
            f"processed {processed} samples from {source_path} with {failures} failures"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
