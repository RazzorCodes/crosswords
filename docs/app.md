# Frontend Application (`app/`)

The frontend is a modern web application built with **Vite**, **React**, and **Tailwind CSS**. It provides the crossword game interface and facilitates handwriting data collection.

## Key Features

- **Crossword Gameplay**: Standard crossword interface with handwriting-first input.
- **Handwriting Input**: Captures strokes as sequences of `{x, y, t}` points.
- **Ensemble Inference**: Uses **ONNX Runtime Web** to run multiple models locally:
  - **k-NN**: Instance-based learning from local session samples.
  - **SVM**: Support Vector Machine trained on high-quality features.
  - **CNN**: Convolutional Neural Network for image-based recognition.
  - These are weighted (50% k-NN, 25% SVM, 25% CNN) to produce a final prediction.
- **Train Mode**: Enabled via `VITE_TRAIN_MODE=true`. This mode allows users to act as "Human Teachers," explicitly labeling samples which are then enqueued for high-quality storage.
- **Local Teacher Queue**: In Train Mode, samples are kept in local storage and synced with the server to ensure high-quality data collection even with intermittent connectivity.

## Tech Stack

- **Framework**: React (TypeScript)
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **ML Runtime**: `onnxruntime-web` for local SVM/CNN inference.
- **State Management**: Zustand for UI and handwriting state.

## Environment Variables

- `VITE_TRAIN_MODE`: Set to `true` to enable the human-teacher training interface.
- `VITE_API_BASE_URL`: The endpoint for the backend server (e.g., `http://localhost:8000`).
- `CROSSWORDS_CONFIG`: A global configuration object (often injected via `env-config.js`) that can specify `MODEL_BASE_URL` for ONNX models.

## Recognition Pipeline

1. **Rasterization**: Strokes are rendered to a 64x64 pixel grid for the CNN.
2. **Feature Extraction**: 30 distinct geometric and temporal features are extracted for the SVM.
3. **Inference**:
   - CNN and SVM run in ONNX sessions.
   - k-NN compares against recent successful inputs.
4. **Voting**: Scores are combined, and if confidence exceeds a threshold (0.92) with strong agreement, the letter is "auto-accepted."
