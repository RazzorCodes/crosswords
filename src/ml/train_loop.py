import json
import os
import subprocess
import time
from pathlib import Path

from training_data import load_training_split

ML_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ML_ROOT.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


def dataset_signature():
    split = load_training_split()
    samples = split["train_samples"] + split["high_quality_eval"]
    signature = []
    for sample in samples:
        signature.append(
            (
                sample.get("id", ""),
                sample.get("label", ""),
                sample.get("updated_at", sample.get("created_at", "")),
                sample.get("status", ""),
            )
        )
    signature.sort()
    return tuple(signature)


def total_trainable_samples() -> int:
    split = load_training_split()
    return len(split["train_samples"])


def write_metrics(models_dir: Path) -> None:
    result = subprocess.run(
        ["python", str(ML_ROOT / "evaluate_models.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Model evaluation failed: {result.stderr.strip()}")
        return

    stdout = result.stdout.strip()
    if not stdout:
        return

    try:
        metrics = json.loads(stdout)
    except json.JSONDecodeError:
        print(f"Unexpected evaluation output: {stdout}")
        return

    metrics_path = models_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle)
    os.chmod(metrics_path, 0o644)
    print(f"HQ-eval metrics: {json.dumps(metrics)}")


def run_trainers() -> None:
    models_dir = Path(os.getenv("MODELS_DIR", str(DEFAULT_MODELS_DIR))).expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["python", str(ML_ROOT / "train_svm.py")], check=False)
    subprocess.run(["python", str(ML_ROOT / "train_cnn.py")], check=False)
    write_metrics(models_dir)


def main() -> None:
    poll_seconds = max(1, int(os.getenv("TRAIN_POLL_SECONDS", "10")))
    min_train_interval_seconds = max(1, int(os.getenv("TRAIN_MIN_INTERVAL_SECONDS", "60")))
    min_samples_delta = max(1, int(os.getenv("TRAIN_MIN_SAMPLES_DELTA", "10")))
    last_signature = None
    last_trained_sample_count = 0
    last_train_at = 0.0

    print(
        "Starting training loop with "
        f"{poll_seconds}s polling, {min_train_interval_seconds}s min interval, "
        f"and {min_samples_delta} new samples per retrain."
    )

    while True:
        try:
            signature = dataset_signature()
            if signature != last_signature:
                sample_count = total_trainable_samples()
                if sample_count > 0:
                    now = time.time()
                    enough_new_samples = (
                        sample_count >= 10
                        and (
                            last_trained_sample_count == 0
                            or sample_count - last_trained_sample_count >= min_samples_delta
                        )
                    )
                    waited_long_enough = (now - last_train_at) >= min_train_interval_seconds

                    if enough_new_samples and waited_long_enough:
                        print(f"Training data changed; retraining on {sample_count} samples.")
                        run_trainers()
                        last_trained_sample_count = sample_count
                        last_train_at = now
                    else:
                        print(
                            "Training data changed; skipping retrain for now "
                            f"(samples={sample_count}, last_trained={last_trained_sample_count})."
                        )
                else:
                    print("Training data changed but is empty; skipping training.")
                last_signature = signature
        except Exception as exc:
            print(f"Training loop iteration failed: {exc}")

        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
