# ASL Translator

A real-time **American Sign Language (ASL) translator** built with a **Next.js frontend** and a **FastAPI backend**, using **MediaPipe hand landmarks** and a **TensorFlow neural network** for sign classification.

The app captures webcam frames in the browser, sends them to the backend through a WebSocket connection, extracts hand landmarks, predicts the signed letter, and builds a sentence in real time.

---

## Overview

This project translates hand signs into text using a full pipeline:

- **Frontend** captures webcam frames and displays predictions in real time
- **Backend** receives frames over WebSocket and runs inference
- **MediaPipe Hand Landmarker** extracts 3D hand landmarks
- **TensorFlow neural network** predicts the ASL sign from landmark coordinates
- **Hold logic + buffering** reduce flickering and accidental predictions

This project also includes scripts for:

- extracting landmarks from ASL image datasets
- training a Random Forest baseline
- training the final neural network model
- testing inference locally with OpenCV

---

## Features

- Real-time ASL sign recognition
- Browser-based webcam interface
- WebSocket communication between frontend and backend
- Live hand landmark overlay
- Confidence score display
- Sentence builder with support for:
  - normal letters
  - `space`
  - `del`
  - `nothing`
- Prediction smoothing using:
  - majority-vote buffer
  - hold-to-confirm logic
- Separate local OpenCV inference script for model testing

---

## Tech Stack

### Frontend
- Next.js
- React
- TypeScript
- Tailwind CSS

### Backend
- FastAPI
- WebSockets
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- Pickle

### Training / Data Processing
- Pandas
- Scikit-learn
- TensorFlow
- Matplotlib
- tqdm

---

## Project Structure

```bash
asl-translator/
├── data/
├── models/
├── src/
│   ├── app.py
│   ├── inference.py
│   ├── extract_landmarks.py
│   ├── train.py
│   ├── train_nn.py
│   └── debug.py
├── frontend/
│   ├── app/
│   │   └── page.tsx
│   ├── public/
│   ├── package.json
│   └── ...
├── .gitignore
└── README.md
```

---

## How It Works

### 1. Landmark Extraction

Images from ASL datasets are passed through MediaPipe Hand Landmarker. For each image, 21 hand landmarks are extracted, each with `x`, `y`, and `z` coordinates.

That gives:

- `21 landmarks × 3 values = 63 features`

These are stored in CSV files and used for training.

### 2. Model Training

Two approaches were tested:

- **Random Forest** as a baseline
- **Neural Network** as the final model

The neural network performed significantly better and is the model used in the real-time application.

### 3. Real-Time Prediction

In the browser version:

- frontend captures frames from webcam
- frames are sent to FastAPI through WebSocket
- backend extracts landmarks and predicts the sign
- frontend displays the predicted sign and builds the sentence

### 4. Stabilization

To avoid noisy predictions, the system uses:

- **prediction buffer** for majority voting
- **hold logic** so the same sign must appear multiple times before being added

This makes the app much more stable and usable.

---

## Dataset Preparation

The project supports extraction from ASL image folders using `extract_landmarks.py`.

Example outputs:

- `landmarks_train_orig.csv`
- `landmarks_train_synthetic.csv`
- `landmarks_test_orig.csv`
- `landmarks_test_synthetic.csv`

The final working model was trained using the original train landmarks file.

---

## Model Training

### Random Forest baseline

```bash
python src/train.py
```

### Neural Network training

```bash
python src/train_nn.py
```

This saves:

- `models/asl_nn_model.keras`
- `models/label_encoder.pkl`

---

## Running the Project

### 1. Backend

From the project root:

```bash
uvicorn src.app:app --reload --port 8000
```

Backend health check:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### 2. Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Then open:

```text
http://localhost:3000
```

### 3. Local OpenCV Test (Optional)

If you want to test the model without the frontend:

```bash
python src/inference.py
```

This opens a local webcam window using OpenCV and performs live prediction directly.

> Do not run this at the same time as the browser version if your webcam is already in use.

---

## Important Files

### `src/app.py`

FastAPI backend for the web app. Handles WebSocket connections, receives frames, runs inference, and returns predictions.

### `frontend/app/page.tsx`

Frontend interface. Handles webcam access, WebSocket communication, prediction display, landmark overlay, and sentence building.

### `src/inference.py`

Standalone local inference script using OpenCV. Useful for testing the model independently from the web app.

### `src/extract_landmarks.py`

Extracts MediaPipe hand landmarks from dataset images and writes them into CSV files.

### `src/train.py`

Trains a Random Forest classifier as a baseline.

### `src/train_nn.py`

Trains the final TensorFlow neural network used by the app.

---

## Current Limitations

- Designed for **single-hand detection**
- Prediction quality depends on:
  - hand visibility
  - lighting
  - camera quality
  - consistency of signing
- Some labels such as `space`, `del`, and `nothing` require careful stabilization
- TensorFlow GPU support is limited on native Windows for newer versions

---

## Future Improvements

- better dataset balancing
- support for full words and sentence-level decoding
- improved temporal modeling across frames
- stronger handling of repeated letters
- deployment to cloud / containerized setup
- model evaluation on broader real-world signing conditions
- support for dynamic ASL gestures

---

## Why This Project

This project was built to explore a practical computer vision + machine learning pipeline that connects:

- dataset preprocessing
- landmark extraction
- model training
- backend inference
- frontend real-time interaction

It is both a machine learning project and a full-stack AI application.

---

## Author

**Aid Aliu**  
GitHub: [AidAliu](https://github.com/AidAliu)

---

## License

This project is for educational and portfolio purposes.
