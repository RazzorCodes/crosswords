from __future__ import annotations

import base64
import http.server
import io
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sample_store import (  # noqa: E402
    HIGH_QUALITY,
    REGULAR,
    append_audit_event,
    create_sample,
    get_label_counts,
    load_snapshot,
    normalize_label,
    promote_sample_to_high_quality,
    set_pending_status,
    tombstone_sample,
)

MAX_REQUEST_BYTES = 512 * 1024
MAX_STROKES = 32
MAX_POINTS_PER_STROKE = 4096
TARGET_HIGH_QUALITY_SHARE = 0.20
TEACHER_POLL_SECONDS = max(5, int(os.getenv("TEACHER_POLL_SECONDS", "15")))
TILE_SIZE = 64
BATCH_SIZE = 20

teacher_state_lock = threading.Lock()
teacher_state: dict[str, Any] = {
    "enabled": False,
    "configured": False,
    "health_ok": False,
    "model_supports_multimodal": False,
    "active": False,
    "reason": "disabled",
    "checked_at": None,
    "hq_share": 0.0,
    "target_share": TARGET_HIGH_QUALITY_SHARE,
    "pending_queue_size": 0,
    "model": os.getenv("LITELLM_MODEL", "fast"),
}


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _normalized_strokes(strokes: Any) -> list[list[dict[str, float]]] | None:
    if not isinstance(strokes, list) or len(strokes) == 0 or len(strokes) > MAX_STROKES:
        return None

    normalized_strokes: list[list[dict[str, float]]] = []
    for stroke in strokes:
        if not isinstance(stroke, list) or len(stroke) == 0 or len(stroke) > MAX_POINTS_PER_STROKE:
            return None
        normalized_points: list[dict[str, float]] = []
        for point in stroke:
            if not isinstance(point, dict):
                return None
            x = point.get("x")
            y = point.get("y")
            t = point.get("t")
            if not (is_finite_number(x) and is_finite_number(y) and is_finite_number(t)):
                return None
            normalized_points.append({"x": float(x), "y": float(y), "t": float(t)})
        normalized_strokes.append(normalized_points)
    return normalized_strokes


def normalize_payload(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None

    label = normalize_label(data.get("label"))
    strokes = _normalized_strokes(data.get("strokes"))
    stored_as = data.get("stored_as")
    source = data.get("source")
    mode = data.get("mode")
    metadata = data.get("metadata")

    if label is None or strokes is None:
        return None
    if stored_as not in {REGULAR, HIGH_QUALITY}:
        return None
    if not isinstance(source, str) or not source.strip():
        return None
    if not isinstance(mode, str) or not mode.strip():
        return None
    if metadata is not None and not isinstance(metadata, dict):
        return None

    return {
        "label": label,
        "strokes": strokes,
        "stored_as": stored_as,
        "source": source.strip(),
        "mode": mode.strip(),
        "metadata": metadata or {},
    }


def send_json(handler: http.server.BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_request_json(handler: http.server.BaseHTTPRequestHandler) -> dict[str, Any] | None:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        send_json(handler, 400, {"error": "Empty request body"})
        return None
    if content_length > MAX_REQUEST_BYTES:
        send_json(handler, 413, {"error": "Request body too large"})
        return None

    raw_body = handler.rfile.read(content_length)
    try:
        value = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError:
        send_json(handler, 400, {"error": "Invalid JSON"})
        return None

    if not isinstance(value, dict):
        send_json(handler, 400, {"error": "Expected JSON object"})
        return None

    return value


def get_litellm_base_url() -> str:
    return os.getenv("LITELLM_BASE_URL", "").strip().rstrip("/")


def get_litellm_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("LITELLM_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def fetch_json(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None, timeout: int = 15) -> Any:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method=method, headers=get_litellm_headers())
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def record_supports_images(record: dict[str, Any]) -> bool:
    for key in (
        "supports_vision",
        "supports_multimodal",
        "supports_image_input",
        "supports_media_input",
        "vision",
        "images_enabled",
    ):
        if record.get(key) is True:
            return True

    for key in ("input_modalities", "modalities", "supported_input_modalities", "input_types"):
        values = record.get(key)
        if isinstance(values, list) and any(str(value).lower() == "image" for value in values):
            return True
        if isinstance(values, str) and "image" in values.lower():
            return True

    return False


def model_record_matches(record: dict[str, Any], model_name: str) -> bool:
    lowered = model_name.lower()
    for key in ("model", "model_name", "model_group", "group", "alias", "litellm_model_name", "name"):
        value = record.get(key)
        if isinstance(value, str) and value.lower() == lowered:
            return True
    return False


def find_multimodal_model_info(value: Any, model_name: str) -> bool:
    if isinstance(value, dict):
        if model_record_matches(value, model_name) and record_supports_images(value):
            return True
        return any(find_multimodal_model_info(child, model_name) for child in value.values())
    if isinstance(value, list):
        return any(find_multimodal_model_info(item, model_name) for item in value)
    return False


def refresh_teacher_state() -> dict[str, Any]:
    snapshot = load_snapshot(include_legacy=False)
    base_state = {
        "enabled": os.getenv("EXTERNAL_TEACHER_ENABLED", "0") == "1",
        "configured": False,
        "health_ok": False,
        "model_supports_multimodal": False,
        "active": False,
        "reason": "disabled",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hq_share": snapshot.high_quality_share,
        "target_share": TARGET_HIGH_QUALITY_SHARE,
        "pending_queue_size": len(snapshot.pending_batch_candidates()),
        "model": os.getenv("TEACHER_MODEL_NAME") or os.getenv("LITELLM_MODEL", "fast"),
    }

    if not base_state["enabled"]:
        with teacher_state_lock:
            teacher_state.update(base_state)
        return dict(base_state)

    base_url = (os.getenv("TEACHER_OPENAPI_ENDPOINT") or get_litellm_base_url()).strip().rstrip("/")
    model_name = base_state["model"]
    if not base_url or not model_name:
        base_state["reason"] = "missing_litellm_configuration"
        with teacher_state_lock:
            teacher_state.update(base_state)
        return dict(base_state)

    base_state["configured"] = True

    try:
        fetch_json(f"{base_url}/health", timeout=10)
        base_state["health_ok"] = True
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        base_state["reason"] = f"health_check_failed:{exc.__class__.__name__}"
        with teacher_state_lock:
            teacher_state.update(base_state)
        return dict(base_state)

    try:
        model_info = fetch_json(f"{base_url}/model_group/info", timeout=10)
        base_state["model_supports_multimodal"] = find_multimodal_model_info(model_info, model_name)
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        base_state["reason"] = f"model_info_failed:{exc.__class__.__name__}"
        with teacher_state_lock:
            teacher_state.update(base_state)
        return dict(base_state)

    if not base_state["model_supports_multimodal"]:
        base_state["reason"] = "model_not_multimodal"
        with teacher_state_lock:
            teacher_state.update(base_state)
        return dict(base_state)

    base_state["active"] = True
    base_state["reason"] = "ready"
    with teacher_state_lock:
        teacher_state.update(base_state)
    return dict(base_state)


def normalize_strokes_to_tile(strokes: list[list[dict[str, float]]], size: int = TILE_SIZE) -> Image.Image:
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


def render_batch_image(samples: list[dict[str, Any]]) -> str:
    sheet = Image.new("L", (5 * TILE_SIZE, 4 * TILE_SIZE), 255)
    for index, sample in enumerate(samples):
        tile = normalize_strokes_to_tile(sample["regular"]["strokes"])
        row = index // 5
        col = index % 5
        sheet.paste(tile, (col * TILE_SIZE, row * TILE_SIZE))

    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_teacher_csv(raw_text: str) -> list[str] | None:
    tokens = [token.strip().upper() for token in raw_text.replace("\n", ",").split(",")]
    tokens = [token for token in tokens if token]
    if len(tokens) != BATCH_SIZE:
        return None
    labels: list[str] = []
    for token in tokens:
        if token == "?":
            labels.append(token)
            continue
        label = normalize_label(token)
        if label is None:
            return None
        labels.append(label)
    return labels


def classify_pending_batch(batch: list[dict[str, Any]]) -> list[str] | None:
    base_url = (os.getenv("TEACHER_OPENAPI_ENDPOINT") or get_litellm_base_url()).strip().rstrip("/")
    model_name = os.getenv("TEACHER_MODEL_NAME") or os.getenv("LITELLM_MODEL", "fast")
    image_b64 = render_batch_image(batch)
    prompt = (
        "Return exactly 20 comma-separated labels in row-major order. "
        "Each label must be a single uppercase letter A-Z or ?. No extra text."
    )
    payload = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
    }
    response = fetch_json(f"{base_url}/chat/completions", method="POST", payload=payload, timeout=60)
    content = (
        response.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        content = "\n".join(parts)
    if not isinstance(content, str):
        return None
    return parse_teacher_csv(content)


def external_teacher_worker() -> None:
    while True:
        try:
            state = refresh_teacher_state()
            if not state.get("active"):
                time.sleep(TEACHER_POLL_SECONDS)
                continue

            snapshot = load_snapshot(include_legacy=False)
            if snapshot.high_quality_share >= TARGET_HIGH_QUALITY_SHARE:
                time.sleep(TEACHER_POLL_SECONDS)
                continue

            pending = snapshot.pending_batch_candidates()
            if len(pending) < BATCH_SIZE:
                time.sleep(TEACHER_POLL_SECONDS)
                continue

            batch = pending[:BATCH_SIZE]
            labels = classify_pending_batch(batch)
            if labels is None:
                append_audit_event("teacher", "teacher_batch_parse_failed", {"size": len(batch)})
                time.sleep(TEACHER_POLL_SECONDS)
                continue

            for item, label in zip(batch, labels):
                sample_id = item["sample_id"]
                if label == "?":
                    set_pending_status(sample_id, "rejected", {"provider": "litellm", "reason": "unknown"})
                    continue
                promote_sample_to_high_quality(
                    sample_id,
                    label,
                    provider="litellm",
                    metadata={"original_label": item["regular"].get("label")},
                )
        except Exception as exc:
            with teacher_state_lock:
                teacher_state["reason"] = f"worker_error:{exc.__class__.__name__}"
        time.sleep(TEACHER_POLL_SECONDS)


def current_teacher_state() -> dict[str, Any]:
    snapshot = load_snapshot(include_legacy=False)
    with teacher_state_lock:
        state = dict(teacher_state)
    state["hq_share"] = snapshot.high_quality_share
    state["pending_queue_size"] = len(snapshot.pending_batch_candidates())
    return state


class DataCollectionHandler(http.server.BaseHTTPRequestHandler):
    server_version = "CrosswordsDataServer/2.0"

    def log_message(self, format: str, *args: Any) -> None:
        print(format % args)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/health":
            send_json(self, 200, {"status": "ok"})
            return

        if parsed.path == "/stats":
            snapshot = load_snapshot(include_legacy=False)
            send_json(
                self,
                200,
                {
                    "counts": get_label_counts(snapshot.usable_samples),
                    "total": snapshot.total_usable_count,
                },
            )
            return

        if parsed.path == "/teacher/status":
            send_json(self, 200, current_teacher_state())
            return

        send_json(self, 404, {"error": "Not found"})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/samples":
            send_json(self, 404, {"error": "Not found"})
            return

        payload = read_request_json(self)
        if payload is None:
            return

        normalized = normalize_payload(payload)
        if normalized is None:
            send_json(self, 400, {"error": "Invalid sample payload"})
            return

        snapshot = load_snapshot(include_legacy=False)
        teacher = current_teacher_state()
        queue_for_llm = (
            normalized["stored_as"] == REGULAR
            and normalized["mode"] != "train"
            and teacher.get("active") is True
            and snapshot.high_quality_share < TARGET_HIGH_QUALITY_SHARE
        )

        sample = create_sample(
            label=normalized["label"],
            strokes=normalized["strokes"],
            stored_as=normalized["stored_as"],
            source=normalized["source"],
            mode=normalized["mode"],
            queue_for_llm=queue_for_llm,
            metadata=normalized["metadata"],
        )
        send_json(
            self,
            200,
            {
                "id": sample["id"],
                "stored_as": normalized["stored_as"],
                "queued_for_llm": queue_for_llm,
            },
        )

    def do_DELETE(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) != 2 or path_parts[0] != "samples":
            send_json(self, 404, {"error": "Not found"})
            return

        sample_id = path_parts[1]
        if not sample_id:
            send_json(self, 400, {"error": "Missing sample id"})
            return

        deleted = tombstone_sample(sample_id)
        if not deleted:
            send_json(self, 404, {"error": "Sample not found"})
            return

        send_json(self, 200, {"status": "ok", "id": sample_id})


def run(port: int | None = None) -> None:
    if port is None:
        port = int(os.getenv("PORT", "8000"))

    worker = threading.Thread(target=external_teacher_worker, daemon=True)
    worker.start()

    server_address = ("", port)
    httpd = http.server.ThreadingHTTPServer(server_address, DataCollectionHandler)
    print(f"Starting data collection server on port {port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
