# Frontend Application (`app/`)

The frontend is a modern web application built with **Vite**, **React**, and **Tailwind CSS**. It serves two primary purposes: providing the crossword game interface and facilitating the collection of handwriting training data.

## Key Features

- **Crossword Gameplay**: A standard crossword interface where players can enter letters.
- **Handwriting Input**: Instead of keyboard input, players can draw letters, which are captured as stroke sequences.
- **Train Mode**: Activated via `VITE_TRAIN_MODE=true`, this mode allows users to act as "Human Teachers," explicitly labeling and enqueuing high-quality samples.
- **Inference Integration**: Uses trained models (e.g., k-NN, CNN, SVM) to predict letters from strokes in real-time.

## Tech Stack

- **Framework**: React (TypeScript)
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **State Management**: Browser-local state for UI, with server synchronization for samples.

## Environment Variables

- `VITE_TRAIN_MODE`: Set to `true` to enable the dedicated human-teacher training session.
- `VITE_API_BASE_URL`: (Optional) Custom API endpoint for the backend server.
