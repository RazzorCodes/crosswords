# System Architecture

The Crossword game system is a handwriting-recognition-enabled application designed for continuous learning and model improvement. It integrates a frontend game interface with a backend API and an AI training pipeline.

## High-Level Flow

1.  **Frontend (`app/`)**: Users play the crossword game. Handwriting input (strokes) is processed locally using an ensemble of recognizers (k-NN, SVM, CNN) via **ONNX Runtime Web**.
2.  **API Server (`srv/`)**: Receives handwriting samples and labels. It manages sample lifecycle, including storage, promotion, and deletion.
3.  **Data Store (`data/`)**: Organizes samples into structured **JSONL chunks** categorized by status:
    -   `regular`: Normal gameplay samples.
    -   `high_quality`: Verified samples (human or LLM promoted).
    -   `pending_llm`: Samples queued for automated review.
    -   `tombstones`: Soft-deleted records.
    -   `audit`: Comprehensive event logs.
4.  **AI Pipeline (`ml/`)**: 
    -   `train_loop.py` monitors the data store for changes using a dataset signature.
    -   Retrains **SVM** (scikit-learn) and **CNN** (PyTorch) models when enough new data is available.
    -   Exports models to **ONNX** format for frontend consumption.
5.  **External Teacher**: A background worker in the server uses **LiteLLM** to automatically review "Regular" samples and promote them to "High Quality" if they meet confidence criteria.

## Component Overview

| Component | Description |
| :--- | :--- |
| **App** | Vite/React/Tailwind frontend. Performs local inference using ONNX models. |
| **Server** | Python API for sample management. Orchestrates the LiteLLM-based External Teacher. |
| **AI (ML)** | Python scripts for training SVM and CNN models and exporting them to ONNX. |
| **Data** | Structured JSONL storage with audit logs and categorical separation. |
| **Infrastructure** | Docker Compose for orchestration and Nix for the development environment. |
