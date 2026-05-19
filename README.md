# Crossword Handwriting Game

A handwriting-enabled crossword game with a continuous learning pipeline for letter recognition.

## 🚀 Quick Start

1.  **Environment Setup**: Ensure you have [Docker](https://www.docker.com/) and [Nix](https://nixos.org/) installed.
2.  **Start Dev Environment**:
    ```bash
    docker compose --profile dev up -d
    ```
3.  **Train Mode**: To start as a human teacher:
    ```bash
    VITE_TRAIN_MODE=true docker compose --profile dev up -d
    ```

## 📂 Repository Structure

-   `src/app/`: Vite/React frontend with **ONNX Runtime Web** for local inference.
-   `src/srv/`: Python API server for sample management and **LiteLLM** integration.
-   `src/ml/`: Model training scripts for **CNN** (PyTorch) and **SVM** (Scikit-Learn).
-   `deploy/`: Deployment assets (Helm chart, GHA workflows, release scripts).
-   `data/`: Categorical **JSONL** storage for handwriting samples.
-   `models/`: Shared volume for exported ONNX models.
-   `docs/`: Detailed documentation for each component.

## 🚢 Release & Deployment

The project is configured for automated releases via GitHub Actions.
- **Release Mode**: Use `--profile release`. It runs a optimized build that fetches models from release assets.
- **Dev Mode**: Use `--profile dev`. Runs the full stack (App, Srv, ML) for local training.
- **Helm Chart**: Located in `deploy/helm`, supports both modes.

## 📖 Documentation

For detailed information on each logical component, see the [docs/](./docs) folder:

-   [Architecture Overview](./docs/architecture.md)
-   [Frontend App](./docs/app.md)
-   [Backend Server](./docs/server.md)
-   [AI & Training](./docs/ai.md)
-   [Data Management](./docs/data.md)
-   [Infrastructure](./docs/infrastructure.md)

## 🛠️ Features

-   **Local Ensemble Inference**: Real-time recognition using k-NN, SVM, and CNN via ONNX.
-   **Continuous Retraining**: `train_loop.py` automatically updates models as data grows.
-   **External Teacher**: Optional LiteLLM integration for automated sample promotion.
-   **Reproducible Env**: Full development setup via Nix Flake.

---
*Created with the help of Gemini CLI.*
