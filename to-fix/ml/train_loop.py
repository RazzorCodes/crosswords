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


def run_trainers() -> bool:
    base_models_dir = Path(os.getenv("MODELS_DIR", str(DEFAULT_MODELS_DIR))).expanduser()
    base_models_dir.mkdir(parents=True, exist_ok=True)
    
    version = time.strftime("%Y%m%d-%H%M%S")
    version_dir = base_models_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting training for version {version} in {version_dir}...")

    env = {**os.environ, "MODELS_DIR": str(version_dir)}

    # 1. Train models
    try:
        subprocess.run(["python", str(ML_ROOT / "train_svm.py")], env=env, check=True)
        subprocess.run(["python", str(ML_ROOT / "train_cnn.py")], env=env, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Training FAILED for version {version}: {exc}")
        return False

    # 2. Evaluate and test
    result = subprocess.run(
        ["python", str(ML_ROOT / "evaluate_models.py")],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"Model verification FAILED for version {version}:")
        print(result.stderr.strip())
        # Keep the failed version dir for debugging but don't link it as latest
        return False

    print(f"Model verification PASSED for version {version}.")
    stdout = result.stdout.strip()
    if stdout:
        try:
            metrics = json.loads(stdout)
            metrics_path = version_dir / "metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(metrics, handle)
            os.chmod(metrics_path, 0o644)
        except json.JSONDecodeError:
            pass

    # 3. Generate artifacts and update manifest
    public_base_url = os.getenv("PUBLIC_MODELS_BASE_URL", "/models").rstrip("/")
    subprocess.run([
        "python", 
        str(ML_ROOT / "generate_handwriting_artifacts.py"),
        "--models-dir", str(version_dir),
        "--public-base-url", f"{public_base_url}/{version}",
        "--version", version,
        "--require-training"
    ], check=True)
    
    # 4. Link this version as the latest in the root models dir
    # This assumes the app loads /models/manifest.json
    root_manifest = base_models_dir / "manifest.json"
    version_manifest = version_dir / "manifest.json"
    if root_manifest.exists() or root_manifest.is_symlink():
        root_manifest.unlink()
    
    # Symlink the manifest so the app always finds the latest one
    try:
        root_manifest.symlink_to(f"{version}/manifest.json")
    except OSError:
        # Fallback to copy if symlink fails (e.g. on some filesystems)
        import shutil
        shutil.copy2(version_manifest, root_manifest)

    print(f"Version {version} successfully deployed and linked as latest.")
    return True


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
                        if run_trainers():
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
