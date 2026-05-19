# AI & Training Pipeline (`ml/`)

The `ml/` directory contains the core logic for training, evaluating, and exporting the machine learning models used for handwriting recognition.

## Models & Export

The system uses a hybrid approach, training models in Python (PyTorch/Scikit-Learn) and exporting them to **ONNX** for high-performance, cross-platform inference in the browser.

- **SVM (Support Vector Machine)**: Trained using `scikit-learn`.
  - `train_svm.py`: Extracts 30 geometric features and trains a radial basis function (RBF) SVC.
  - Export: Converted via `skl2onnx` to `svm.onnx`.
- **CNN (Convolutional Neural Network)**: Trained using `PyTorch`.
  - `cnn_model.py`: Simple 2-layer CNN architecture.
  - `train_cnn.py`: Renders strokes to 64x64 images and trains for 10 epochs.
  - Export: Exported via `torch.onnx.export` to `cnn.onnx`.
- **k-NN (k-Nearest Neighbors)**: Implemented directly in TypeScript in the frontend for real-time adaptability to a user's style.

## Training Orchestration

- **`train_loop.py`**: A continuous background process that:
  - Polls the data store for changes using a **dataset signature** (hash of IDs/timestamps).
  - Triggers a retrain if enough new samples are detected (configurable via `TRAIN_MIN_SAMPLES_DELTA`).
  - Ensures a minimum interval between runs (`TRAIN_MIN_INTERVAL_SECONDS`).
- **`training_data.py`**: Handles loading and splitting data into training and high-quality evaluation sets.
- **`evaluate_models.py`**: Runs a benchmark of the trained models against the high-quality eval set, outputting a `metrics.json` file.

## Feature Extraction

Geometric features used by the SVM include:
- Aspect ratio and relative width.
- Direction histogram (8 bins).
- Mean and variance of curvature.
- Start/end points and total path length.
- Speed and pause durations (temporal features).
- Stroke centroids and crossing counts.
