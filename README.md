# Crossword Handwriting Game

A handwriting-enabled crossword game with a continuous learning pipeline for letter recognition.

## 🚀 Quick Start

1.  **Environment Setup**: Ensure you have [Docker](https://www.docker.com/) and [Nix](https://nixos.org/) installed.
2.  **Start the App**:
    ```bash
    docker compose up -d
    ```
3.  **Train Mode**: To start as a human teacher:
    ```bash
    VITE_TRAIN_MODE=true docker compose up -d
    ```

## 📂 Repository Structure

-   `src/app/`: Vite/React frontend application.
-   `src/srv/`: Python API server for sample management.
-   `src/ml/`: Model training and evaluation scripts (CNN, SVM, k-NN).
-   `deploy/`: Deployment assets (Helm chart, GHA workflows, release scripts).
-   `data/`: Structured storage for handwriting samples.
-   `models/`: Source of truth for exported ONNX models.
-   `docs/`: Detailed documentation for each component.

## 🚢 Release & Deployment

The project is configured for automated releases via GitHub Actions.
- **Release Mode**: A single-container app that streams ONNX models from GitHub Release assets.
- **Dev Mode**: Full stack (App, Srv, ML) for local development and training.
- **Helm Chart**: Located in `deploy/helm`, supports both modes.
- **Dynamic Models**: Models are fetched at runtime based on `MODEL_RELEASE_TAG` or `MODEL_BASE_URL`.

## 📖 Documentation

For detailed information on each logical component, see the [docs/](./docs) folder:

-   [Architecture Overview](./docs/architecture.md)
-   [Frontend App](./docs/app.md)
-   [Backend Server](./docs/server.md)
-   [AI & Training](./docs/ai.md)
-   [Data Management](./docs/data.md)
-   [Infrastructure](./docs/infrastructure.md)

## 🛠️ Features

-   **Handwriting Input**: Capture and recognize letters from stroke data.
-   **Continuous Learning**: Models retrain as new high-quality data becomes available.
-   **Ensemble Recognition**: Combined predictions from CNN, SVM, and k-NN.
-   **External Teacher**: Optional LiteLLM integration for automated sample promotion.

---
*Created with the help of Gemini CLI.*
