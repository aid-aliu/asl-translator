import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
import numpy as np
import pickle
import tensorflow as tf
from collections import Counter
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import base64
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("📂 Loading model...")
nn_model = tf.keras.models.load_model("../models/asl_nn_model.keras")
with open("../models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
print("✅ Model loaded")

MODEL_PATH = "../data/hand_landmarker.task"
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1
)
detector = HandLandmarker.create_from_options(options)

BUFFER_SIZE = 5

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Client connected")

    prediction_buffer = []

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)

            img_data = base64.b64decode(payload["frame"])
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]

                coords = []
                for lm in hand:
                    coords.extend([lm.x, lm.y, lm.z])

                coords = np.array(coords).reshape(1, -1)
                max_val = np.max(np.abs(coords))
                if max_val > 0:
                    coords = coords / max_val

                proba = nn_model.predict(coords, verbose=0)[0]
                pred_index = np.argmax(proba)
                confidence = float(proba[pred_index] * 100)
                prediction = le.inverse_transform([pred_index])[0]

                prediction_buffer.append(prediction)
                if len(prediction_buffer) > BUFFER_SIZE:
                    prediction_buffer.pop(0)

                stable_prediction = Counter(prediction_buffer).most_common(1)[0][0]
                landmarks = [{"x": lm.x, "y": lm.y} for lm in hand]

                await websocket.send_text(json.dumps({
                    "prediction": stable_prediction,
                    "confidence": round(confidence, 1),
                    "landmarks": landmarks,
                    "hand_detected": True
                }))
            else:
                prediction_buffer = []
                await websocket.send_text(json.dumps({
                    "prediction": "",
                    "confidence": 0,
                    "landmarks": [],
                    "hand_detected": False
                }))

    except Exception as e:
        print(f"❌ Connection closed: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}