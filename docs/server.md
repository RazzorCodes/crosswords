# Backend API Server (`srv/`)

The backend is a Python-based server that manages the lifecycle of handwriting samples and coordinates the "External Teacher" logic.

## Responsibilities

- **Sample Management**: Provides RESTful endpoints (`POST /samples`, `DELETE /samples/:id`) to store and manage handwriting data.
- **External Teacher Integration**: When enabled via the `external-teacher` profile, it uses LiteLLM to automatically promote samples to "High Quality" status.
- **Data Integrity**: Uses `sample_store.py` to ensure consistent data reads/writes and maintains audit logs for all sample events.
- **Model Serving Support**: Provides endpoints or data access for the AI pipeline to retrieve training data and export models.

## External Teacher Logic

The server enables LiteLLM-based promotion only if:
1. `EXTERNAL_TEACHER_ENABLED=1` is set.
2. LiteLLM is correctly configured and healthy.
3. The model `fast` is available and supports multimodal input.

It promotes samples by batching them into images and asking the LLM for labels, comparing them against the original user labels.

## Implementation

- **File**: `srv/server.py`
- **Infrastructure**: Runs in a Docker container (see `srv/Containerfile`).
