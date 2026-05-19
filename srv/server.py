import http.server
import json
import math
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset.jsonl"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_AI_DIR = PROJECT_ROOT / "ai"
MAX_REQUEST_BYTES = 512 * 1024
MAX_STROKES = 32
MAX_POINTS_PER_STROKE = 4096
MAX_ITEMS_PER_FILE = 250


def get_dataset_path() -> Path:
    return Path(os.getenv("DATA_PATH", str(DEFAULT_DATASET_PATH))).expanduser()


def get_dataset_dir() -> Path:
    return Path(os.getenv("DATA_DIR", str(DEFAULT_DATASET_DIR))).expanduser()


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0

    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def iter_dataset_files():
    dataset_path = get_dataset_path()
    dataset_dir = get_dataset_dir()
    files = []

    if dataset_path.exists() and dataset_path.is_file():
      files.append(dataset_path)

    if dataset_dir.exists():
        files.extend(sorted(
            path for path in dataset_dir.glob("*.jsonl")
            if path.is_file()
        ))

    return files


def get_active_dataset_file() -> Path:
    dataset_dir = get_dataset_dir()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    chunk_files = sorted(
        path for path in dataset_dir.glob("*.jsonl")
        if path.is_file()
    )

    if not chunk_files:
        return dataset_dir / "dataset-0001.jsonl"

    current = chunk_files[-1]
    if count_lines(current) < MAX_ITEMS_PER_FILE:
        return current

    next_index = len(chunk_files) + 1
    return dataset_dir / f"dataset-{next_index:04d}.jsonl"


def get_total_sample_count() -> int:
    return sum(count_lines(path) for path in iter_dataset_files())


def maybe_trigger_retrain(sample_count: int) -> None:
    if sample_count <= 0 or sample_count % 50 != 0:
        return

    ai_dir = Path(os.getenv("AI_DIR", str(DEFAULT_AI_DIR))).expanduser()
    if not ai_dir.exists():
        print(f"Skipping retrain at {sample_count} samples: AI dir not found at {ai_dir}")
        return

    print(f"Triggering retrain at {sample_count} samples...")
    subprocess.Popen([sys.executable, "train_svm.py"], cwd=ai_dir, start_new_session=True)
    subprocess.Popen([sys.executable, "train_cnn.py"], cwd=ai_dir, start_new_session=True)


def is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def normalize_payload(data):
    if not isinstance(data, dict):
        return None

    label = data.get("label")
    strokes = data.get("strokes")

    if not isinstance(label, str):
        return None

    label = label.strip().upper()
    if len(label) != 1 or not ("A" <= label <= "Z"):
        return None

    if not isinstance(strokes, list) or len(strokes) == 0 or len(strokes) > MAX_STROKES:
        return None

    normalized_strokes = []
    for stroke in strokes:
        if not isinstance(stroke, list) or len(stroke) == 0 or len(stroke) > MAX_POINTS_PER_STROKE:
            return None

        normalized_points = []
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

    return {"label": label, "strokes": normalized_strokes}

class DataCollectionHandler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/stats':
            counts = {chr(i): 0 for i in range(65, 91)}
            total = 0
            for dataset_path in iter_dataset_files():
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            label = data.get('label', '').upper()
                            if label in counts:
                                counts[label] += 1
                                total += 1
                        except Exception:
                            pass
            
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'counts': counts, 'total': total}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', '0'))
        if content_length <= 0:
            self.send_error(400, "Empty request body")
            return
        if content_length > MAX_REQUEST_BYTES:
            self.send_error(413, "Request body too large")
            return
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            normalized = normalize_payload(data)
            
            if normalized is not None:
                dataset_path = get_active_dataset_file()
                with open(dataset_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(normalized) + '\n')
                
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
                print(f"Saved sample for label: {normalized['label']}")

                # Check for retrain trigger
                try:
                    count = get_total_sample_count()
                    maybe_trigger_retrain(count)
                except Exception as exc:
                    print(f"Retrain trigger failed: {exc}")
            else:
                self.send_error(400, "Invalid label or strokes payload")
        except Exception as e:
            print(f"Request handling failed: {e}")
            self.send_error(500, "Failed to process request")

def run(server_class=http.server.HTTPServer, handler_class=DataCollectionHandler, port=None):
    if port is None:
        port = int(os.getenv("PORT", 8000))
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting data collection server on port {port}...")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
