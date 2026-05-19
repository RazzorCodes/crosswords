# Data Storage & Management

The system uses a flat-file database approach based on **JSON Lines (JSONL)** chunks, managed by `src/srv/sample_store.py`. This ensures high performance for append-heavy workloads and easy versioning/backup.

## Data Directory Structure (`data/`)

Samples are grouped into subdirectories by their current status. Each subdirectory contains multiple `.jsonl` files, each limited to 250 records for easier management.

- **`regular/`**: Default landing spot for samples from normal gameplay.
- **`high_quality/`**: Verified samples used for final model training. These come from:
  - Human Teachers (Train Mode).
  - External Teacher (LiteLLM promotion).
- **`pending_llm/`**: Queue entries for the External Teacher. Tracks the "queued," "promoted," or "rejected" status of individual samples.
- **`tombstones/`**: Records of soft-deleted samples. Any sample ID found here is ignored during snapshot loading.
- **`audit/`**: A comprehensive event log. Every sample creation, promotion, and deletion is recorded here with a timestamp.

## `sample_store.py` Architecture

- **Snapshots**: The server loads a point-in-time "Snapshot" by reading all relevant JSONL files and resolving the current state (latest record wins, tombstones subtract).
- **Chunking**: When an active `.jsonl` file reaches 250 lines, a new file (e.g., `regular-0002.jsonl`) is automatically created.
- **Legacy Support**: The store can optionally import older `dataset.jsonl` files for backward compatibility.

## Audit Events

Each audit event includes:
- `sample_id`: The UUID of the sample.
- `event`: The type (e.g., `sample_created`, `sample_promoted_to_high_quality`).
- `created_at`: ISO UTC timestamp.
- `payload`: Context-specific data (e.g., who promoted it, original labels).

## Persistence & Volumes

In production (Docker), the `./data` directory is typically mounted as a persistent volume. This allows the AI pipeline and Server containers to share access to the same dataset while ensuring data survives container upgrades.
