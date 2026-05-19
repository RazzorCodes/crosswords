# Backend API Server (`srv/`)

The backend is a Python-based server that manages the lifecycle of handwriting samples and coordinates the "External Teacher" logic.

## Responsibilities

- **Sample Management**: Endpoints for creating (`POST /samples`), deleting (`DELETE /samples/:id`), and monitoring statistics (`GET /stats`).
- **External Teacher Integration**: Automatically promotes "Regular" samples to "High Quality" using multimodal LLMs.
- **Data Integrity**: Uses `sample_store.py` for structured JSONL storage, audit logging, and soft deletions (tombstones).
- **Inference Support**: Serves as the source of truth for labels and data used by the ML training pipeline.

## External Teacher (LiteLLM)

The server runs a background worker that polls the data store for "Pending" samples. If enough samples are available and the high-quality share is below the target (20%), it triggers a promotion cycle.

### Configuration
- `EXTERNAL_TEACHER_ENABLED=1`: Main toggle.
- `TEACHER_OPENAPI_ENDPOINT` (or `LITELLM_BASE_URL`): The LiteLLM proxy address.
- `TEACHER_MODEL_NAME` (or `LITELLM_MODEL`): The multimodal model to use (e.g., `gpt-4o`, `fast`).
- `LITELLM_API_KEY`: Authentication for the LiteLLM proxy.
- `TEACHER_POLL_SECONDS`: How often to check for new work (default: 15s).

### Logic Flow
1. **Batching**: 20 samples are rendered into a single 5x4 grid image.
2. **Classification**: The image is sent to LiteLLM with a prompt asking for 20 comma-separated labels.
3. **Promotion**: If the LLM identifies a letter clearly, the sample is moved to the `high_quality` category with a `litellm` provider tag.
4. **Rejection**: If the LLM returns `?`, the sample is marked as rejected for LLM promotion.

## Implementation Details

- **Server**: `srv/server.py` using `http.server.ThreadingHTTPServer`.
- **Worker**: A daemon thread running `external_teacher_worker()`.
- **Infrastructure**: Containerized via `srv/Containerfile`, typically exposed on port 8000.
