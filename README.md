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

-   `app/`: Vite/React frontend application.
-   `srv/`: Python API server for sample management.
-   `ai/`: Model training and evaluation scripts (CNN, SVM, k-NN).
-   `data/`: Structured storage for handwriting samples.
-   `docs/`: Detailed documentation for each component.
-   `models/`: Exported model files for inference.
-   `litellm/`: Configuration for External Teacher (LLM promotion).

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
