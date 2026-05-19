import os
import subprocess
import time
from pathlib import Path

DEFAULT_DATASET_PATH = Path("/data/legacy-dataset.jsonl")
DEFAULT_DATASET_DIR = Path("/data/dataset")


def get_dataset_path() -> Path:
    return Path(os.getenv("DATA_PATH", str(DEFAULT_DATASET_PATH))).expanduser()


def get_dataset_dir() -> Path:
    return Path(os.getenv("DATA_DIR", str(DEFAULT_DATASET_DIR))).expanduser()


def iter_dataset_files():
    files = []
    dataset_path = get_dataset_path()
    dataset_dir = get_dataset_dir()

    if dataset_path.exists() and dataset_path.is_file():
        files.append(dataset_path)

    if dataset_dir.exists():
        files.extend(sorted(
            path for path in dataset_dir.glob("*.jsonl")
            if path.is_file()
        ))

    return files


def dataset_signature():
    return tuple(
        (str(path), path.stat().st_size, int(path.stat().st_mtime))
        for path in iter_dataset_files()
    )


def total_samples():
    count = 0
    for path in iter_dataset_files():
        with open(path, "r", encoding="utf-8") as f:
            count += sum(1 for line in f if line.strip())
    return count


def run_trainers():
    subprocess.run(["python", "train_svm.py"], check=False)
    subprocess.run(["python", "train_cnn.py"], check=False)


def main():
    poll_seconds = max(1, int(os.getenv("TRAIN_POLL_SECONDS", "10")))
    last_signature = None

    print(f"Starting training loop with {poll_seconds}s polling.")

    while True:
        try:
            signature = dataset_signature()
            if signature != last_signature:
                sample_count = total_samples()
                if sample_count > 0:
                    print(f"Dataset changed; retraining on {sample_count} samples.")
                    run_trainers()
                else:
                    print("Dataset changed but is empty; skipping training.")
                last_signature = signature
        except Exception as exc:
            print(f"Training loop iteration failed: {exc}")

        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
