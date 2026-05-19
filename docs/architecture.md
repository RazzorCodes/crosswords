# System Architecture

The Crossword game system is a handwriting-recognition-enabled application designed for continuous learning and model improvement. It integrates a frontend game interface with a backend API and an AI training pipeline.

## High-Level Flow

1.  **Frontend (`app/`)**: Users play the crossword game and provide handwriting input for letters.
2.  **API Server (`srv/`)**: Receives handwriting samples (strokes and labels) and stores them.
3.  **Data Store (`data/`)**: Organizes samples into structured categories (Regular, High Quality, Pending LLM).
4.  **AI Pipeline (`ai/`)**: Periodically trains and evaluates models (CNN, SVM, k-NN) using the stored data.
5.  **External Teacher**: Uses LiteLLM to automatically review and promote "Regular" samples to "High Quality" if they meet certain criteria.
6.  **Inference**: The frontend uses the trained models to recognize handwriting in real-time.

## Component Overview

| Component | Description |
| :--- | :--- |
| **App** | Vite/React/Tailwind frontend for gameplay and training. |
| **Server** | Python/Docker API for sample management and LLM promotion. |
| **AI** | Training scripts for CNN, SVM, and k-NN models. |
| **Data** | Structured JSONL storage with audit logs and tombstones. |
| **Infrastructure** | Docker Compose for orchestration and Nix for dev environment. |
