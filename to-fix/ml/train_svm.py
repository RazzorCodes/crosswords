import os
import pickle
import sys
from pathlib import Path

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from trainer_1 import train
from training_data import load_training_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


if __name__ == "__main__":
    models_dir = Path(os.getenv("MODELS_DIR", str(DEFAULT_MODELS_DIR))).expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)

    split = load_training_split()
    train_samples = split["train_samples"]

    if len(train_samples) < 10:
        print("Not enough training data to train SVM.")
        sys.exit(0)

    clf = train(train_samples)

    pkl_path = models_dir / "letter_clf.pkl"
    with open(pkl_path, "wb") as handle:
        pickle.dump(clf, handle)
    os.chmod(pkl_path, 0o644)

    initial_type = [("float_input", FloatTensorType([None, 30]))]
    onx = convert_sklearn(
        clf,
        initial_types=initial_type,
        options={id(clf): {"zipmap": False}},
    )
    onnx_path = models_dir / "svm.onnx"
    with open(onnx_path, "wb") as handle:
        handle.write(onx.SerializeToString())
    os.chmod(onnx_path, 0o644)

    print(
        "SVM retrained and exported to ONNX "
        f"with {len(train_samples)} train samples and {len(split['high_quality_eval'])} HQ-eval samples."
    )
