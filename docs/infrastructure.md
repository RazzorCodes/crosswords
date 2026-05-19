# Infrastructure & Orchestration

The project uses modern infrastructure tools to ensure reproducibility and ease of deployment.

## Docker Compose

The system is orchestrated using Docker Compose (`compose.yaml`).

- **Profiles**:
  - `dev`: Runs the `app` (Vite dev server), `srv` (API), and `ml` (Training loop) services.
  - `release`: Runs a optimized `app-release` service.
- **Services**:
  - `app`: Frontend with HMR. Depends on `srv`.
  - `srv`: Sample management API with LiteLLM background worker.
  - `ml`: Continuous training pipeline.
- **Volumes**:
  - `./data`: Host-mounted directory for JSONL sample storage.
  - `models-data`: A named volume (can be backed by NFS or local disk) for sharing `.onnx` and `.pth` models between the training pipeline and the frontend.

## Nix & Flake

The `flake.nix` file provides a reproducible development environment.
- **Dependencies**: Includes Python 3.x (with PyTorch/scikit-learn), Node.js, and essential build tools.
- **Environment**: Automatically sets up `PYTHONPATH` and other variables needed for local development.

## Data Persistence

Host directories are mounted into containers to ensure data survives restarts:
- `./data` -> `/data`
- Shared volume -> `/models` (Training writes here, Frontend reads from here).

## Environment Configuration

A `.env` file is used to manage sensitive and environment-specific variables like `LITELLM_API_KEY`, `TEACHER_MODEL_NAME`, and `EXTERNAL_TEACHER_ENABLED`.
