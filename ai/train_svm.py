import json
import pickle
import os
import sys
from pathlib import Path
from trainer_1 import train
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset.jsonl"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


def iter_dataset_files():
    dataset_path = Path(os.getenv("DATA_PATH", str(DEFAULT_DATASET_PATH))).expanduser()
    dataset_dir = Path(os.getenv("DATA_DIR", str(DEFAULT_DATASET_DIR))).expanduser()
    files = []

    if dataset_path.exists() and dataset_path.is_file():
        files.append(dataset_path)

    if dataset_dir.exists():
        files.extend(sorted(
            path for path in dataset_dir.glob("*.jsonl")
            if path.is_file()
        ))

    return files


def load_dataset(paths):
    data = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    pass
    return data

if __name__ == "__main__":
    models_dir = os.getenv("MODELS_DIR", str(DEFAULT_MODELS_DIR))
    os.makedirs(models_dir, exist_ok=True)

    data = load_dataset(iter_dataset_files())

    if len(data) < 10:
        print("Not enough data to train SVM.")
        sys.exit(0)

    clf = train(data)

    # Save Pickle
    pkl_path = os.path.join(models_dir, "letter_clf.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(clf, f)
    os.chmod(pkl_path, 0o644)
    
    # Save ONNX
    # Our extractFeatures returns 30 values
    initial_type = [('float_input', FloatTensorType([None, 30]))]
    onx = convert_sklearn(clf, initial_types=initial_type)
    onnx_path = os.path.join(models_dir, "svm.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())
    os.chmod(onnx_path, 0o644)

    print(f"SVM retrained and exported to ONNX with {len(data)} samples.")
