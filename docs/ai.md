# AI & Training Pipeline (`ai/`)

The `ai/` directory contains the core logic for training, evaluating, and managing the machine learning models used for handwriting recognition.

## Models Supported

- **CNN (Convolutional Neural Network)**: Deep learning model for image-based letter recognition.
  - `cnn_model.py`: Model architecture.
  - `train_cnn.py`: Training script for the CNN.
- **SVM (Support Vector Machine)**: Traditional ML model for feature-based classification.
  - `train_svm.py`: Training script for the SVM.
- **k-NN (k-Nearest Neighbors)**: Used for instance-based learning, often part of the ensemble recognition.

## Key Scripts

- **`train_loop.py`**: Orchestrates the continuous training process, checking for new data and retraining models as needed.
- **`evaluate_models.py`**: Benchmarks the accuracy of all models (and their ensemble) against a high-quality evaluation dataset.
- **`cnn_builder.py`**: Prepares the image-based datasets required for CNN training.
- **`training_data.py`**: Utilities for loading and preprocessing samples from the data store.

## Ensemble Recognition

The system typically uses an ensemble approach, weighting predictions from multiple models (e.g., `k-NN 0.50`, `SVM 0.25`, `CNN 0.25`) to achieve higher overall accuracy.
