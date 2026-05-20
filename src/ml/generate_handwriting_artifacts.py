import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from cnn_model import get_cnn_model


LABEL_MAP = [chr(ord("A") + index) for index in range(26)]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
ML_ROOT = Path(__file__).resolve().parent


def _asset_url(base_url: str, name: str) -> str:
    return f"{base_url.rstrip('/')}/{name}"


def export_baseline_onnx(models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    model = get_cnn_model(num_classes=len(LABEL_MAP))
    weights_path = models_dir / "cnn_model.pth"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    onnx_path = models_dir / "cnn.onnx"
    dummy_input = torch.randn(1, 1, 64, 64)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    os.chmod(onnx_path, 0o644)
    return onnx_path


def train_local_baselines(models_dir: Path) -> None:
    env = {
        **os.environ,
        "MODELS_DIR": str(models_dir),
    }
    for script in ("train_svm.py", "train_cnn.py"):
        result = subprocess.run(
            [sys.executable, str(ML_ROOT / script)],
            env=env,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{script} failed with exit code {result.returncode}")


def generate_training_artifacts(models_dir: Path, baseline_onnx: Path) -> dict[str, bool]:
    try:
        import onnx
        from onnxruntime.training import artifacts
    except Exception as exc:
        print(f"Skipping ORT training artifacts: {exc}")
        return {"supported": False}

    model = onnx.load(str(baseline_onnx))
    artifact_dir = models_dir / "ort-training"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    requires_grad = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
    frozen_params = [
        "conv1.weight",
        "conv1.bias",
        "conv2.weight",
        "conv2.bias",
    ]

    artifacts.generate_artifacts(
        model,
        requires_grad=requires_grad,
        frozen_params=frozen_params,
        loss=artifacts.LossType.CrossEntropyLoss,
        optimizer=artifacts.OptimType.AdamW,
        artifact_directory=str(artifact_dir),
    )

    expected = [
        "training_model.onnx",
        "eval_model.onnx",
        "optimizer_model.onnx",
        "checkpoint",
    ]
    missing = [name for name in expected if not (artifact_dir / name).exists()]
    if missing:
        raise RuntimeError(f"ORT artifact generation did not create: {', '.join(missing)}")

    metadata_path = models_dir / "export-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "inputName": "input",
                "labelName": "labels",
                "outputName": "output",
                "labelMap": LABEL_MAP,
                "stages": {
                    "head-only": {"trainable": requires_grad, "frozen": frozen_params},
                    "partial-finetune": {"trainable": "runtime-selected"},
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"supported": True}


def copy_required_web_training_runtime(runtime_dir: Path, models_dir: Path) -> None:
    required = [
        "ort-training-web.mjs",
        "ort-wasm-simd.wasm",
        "ort-wasm-simd-threaded.wasm",
    ]
    missing = [name for name in required if not (runtime_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            "Missing ORT Web training runtime assets in "
            f"{runtime_dir}: {', '.join(missing)}. "
            "Build ONNX Runtime Web with training APIs and place the files there."
        )

    target_dir = models_dir / "ort-training"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in required:
        source = runtime_dir / name
        target = target_dir / name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        os.chmod(target, 0o644)


def try_copy_web_training_runtime(runtime_dir_text: str, models_dir: Path, require_training: bool) -> bool:
    if not runtime_dir_text:
        if require_training:
            raise RuntimeError("--ort-web-runtime-dir is required when --require-training is set.")
        print("ORT Web training runtime directory not configured; CNN browser training will be disabled.")
        return False

    runtime_dir = Path(runtime_dir_text).expanduser().resolve()
    try:
        copy_required_web_training_runtime(runtime_dir, models_dir)
        return True
    except RuntimeError:
        if require_training:
            raise
        print(
            "ORT Web training runtime assets are missing; generated baseline and training model "
            "artifacts, but manifest will mark CNN browser training unavailable."
        )
        return False


def write_manifest(models_dir: Path, public_base_url: str, version: str, supports_training: bool) -> Path:
    manifest = {
        "version": version,
        "labelMap": LABEL_MAP,
        "cnn": {
            "inferenceUrl": _asset_url(public_base_url, "cnn.onnx"),
            "supportsTraining": supports_training,
            "trainingArtifacts": {
                "trainUrl": _asset_url(public_base_url, "ort-training/training_model.onnx") if supports_training else None,
                "evalUrl": _asset_url(public_base_url, "ort-training/eval_model.onnx") if supports_training else None,
                "optimizerUrl": _asset_url(public_base_url, "ort-training/optimizer_model.onnx") if supports_training else None,
                "checkpointUrl": _asset_url(public_base_url, "ort-training/checkpoint") if supports_training else None,
                "exportMetadataUrl": _asset_url(public_base_url, "export-metadata.json") if supports_training else None,
            },
            "trainingRuntime": {
                "moduleUrl": _asset_url(public_base_url, "ort-training/ort-training-web.mjs") if supports_training else None,
                "wasmUrl": _asset_url(public_base_url, "ort-training/") if supports_training else None,
                "simdWasmUrl": _asset_url(public_base_url, "ort-training/ort-wasm-simd.wasm") if supports_training else None,
                "threadedWasmUrl": _asset_url(public_base_url, "ort-training/ort-wasm-simd-threaded.wasm") if supports_training else None,
            },
        },
        "featureClassifier": None,
    }

    path = models_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.chmod(path, 0o644)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate browser handwriting release artifacts.")
    parser.add_argument("--models-dir", default=os.getenv("MODELS_DIR", str(DEFAULT_MODELS_DIR)))
    parser.add_argument("--public-base-url", default="/models")
    parser.add_argument("--version", default=os.getenv("HANDWRITING_ARTIFACT_VERSION", "local"))
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--train-baseline", action="store_true")
    parser.add_argument("--require-training", action="store_true")
    parser.add_argument("--ort-web-runtime-dir", default=os.getenv("ORT_WEB_RUNTIME_DIR", ""))
    args = parser.parse_args()

    models_dir = Path(args.models_dir).expanduser().resolve()
    if args.train_baseline:
        train_local_baselines(models_dir)
    baseline_onnx = export_baseline_onnx(models_dir)
    result = {"supported": False} if args.skip_training else generate_training_artifacts(models_dir, baseline_onnx)
    if args.require_training and not result["supported"]:
        raise RuntimeError("ORT training artifact generation is required but unavailable.")
    supports_training = False
    if result["supported"]:
        supports_training = try_copy_web_training_runtime(
            args.ort_web_runtime_dir,
            models_dir,
            args.require_training,
        )
    manifest_path = write_manifest(models_dir, args.public_base_url, args.version, supports_training)
    print(f"Wrote {baseline_onnx}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
