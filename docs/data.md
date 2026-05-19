# Data Storage & Management

Data management is a critical component, handled primarily by `sample_store.py` in the root and organized within the `data/` directory.

## Data Directory Structure (`data/`)

Samples are stored in structured JSONL (JSON Lines) files, grouped into categories:

- **`regular/`**: Standard samples collected during normal gameplay.
- **`high_quality/`**: Samples verified by a Human Teacher (Train Mode) or promoted by the External Teacher (LiteLLM).
- **`pending_llm/`**: Samples queued for review by the External Teacher.
- **`tombstones/`**: Records of deleted samples (soft deletion).
- **`audit/`**: Event logs tracking the lifecycle of every sample (creation, promotion, deletion).

## `sample_store.py`

This module provides the core API for data operations:

- **`Snapshot`**: A point-in-time view of all usable samples, handling deduplication and status filtering.
- **`append_record()`**: Writes new records to the active JSONL chunk for a given category.
- **`promote_sample_to_high_quality()`**: Moves a sample from `regular` to `high_quality`.
- **`tombstone_sample()`**: Soft-deletes a sample by adding it to the `tombstones` category.

## Persistence

Data is persisted on the host via Docker volume mounts (typically mapping `./data` to `/data` in the containers). This ensures that training data survives container restarts and recreations.
