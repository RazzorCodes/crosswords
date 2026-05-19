# Infrastructure & Orchestration

The project uses modern infrastructure tools to ensure reproducibility and ease of deployment.

## Docker Compose

The system is orchestrated using Docker Compose (`compose.yaml`).

- **Default Services**: Runs the `app`, `srv`, and potentially the `ai` training loop.
- **External Teacher Profile**: `compose.external-teacher.yaml` enables the LiteLLM-based promotion path by injecting necessary environment variables and configuration.

## Nix & Flake

The `flake.nix` file provides a reproducible development environment. It defines:
- Required dependencies (Python, Node.js, etc.).
- Shell hooks for setting up the environment.
- Any necessary system-level libraries for ML (like those needed for PyTorch or TensorFlow if used by the CNN).

## Data Persistence

Host directories are mounted into containers to ensure data and model persistence:
- `./data` -> `/data`
- `./models` -> `/models`

## Migration

The `migrate_legacy_dataset.py` script is provided to transition from the legacy `dataset.jsonl` format to the new structured directory layout.
