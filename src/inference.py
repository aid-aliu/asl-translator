import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
import numpy as np
import pickle
import tensorflow as tf
from collections import Counter

print("📂 Loading model...")
model = tf.keras.models.load_model("../models/asl_nn_model.keras")
with open("../models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
print("✅ Model loaded")

MODEL_PATH = "../data/hand_landmarker.task"
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1
)
detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
print("🎥 Webcam started — press Q to quit")

sentence = ""
frame_count = 0
PREDICTION_INTERVAL = 15
HOLD_REQUIRED = 3

current_prediction = ""
current_confidence = 0.0
prediction_buffer = []
BUFFER_SIZE = 5

hold_count = 0
hold_candidate = ""
last_committed = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    display = frame.copy()

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        h, w, _ = frame.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

        for point in points:
            cv2.circle(display, point, 5, (0, 255, 0), -1)

        if frame_count % PREDICTION_INTERVAL == 0:
            coords = []
            for lm in hand:
                coords.extend([lm.x, lm.y, lm.z])

            coords = np.array(coords).reshape(1, -1)
            max_val = np.max(np.abs(coords))
            if max_val > 0:
                coords = coords / max_val

            proba = model.predict(coords, verbose=0)[0]
            pred_index = np.argmax(proba)
            current_confidence = float(proba[pred_index] * 100)
            current_prediction = le.inverse_transform([pred_index])[0]

            prediction_buffer.append(current_prediction)
            if len(prediction_buffer) > BUFFER_SIZE:
                prediction_buffer.pop(0)

            current_prediction = Counter(prediction_buffer).most_common(1)[0][0]

            if current_confidence > 70:
                if current_prediction == "nothing":
                    hold_count = 0
                    hold_candidate = ""
                    last_committed = ""
                else:
                    if current_prediction != hold_candidate:
                        hold_candidate = current_prediction
                        hold_count = 1
                    else:
                        hold_count += 1

                        if hold_count >= HOLD_REQUIRED:
                            if current_prediction != last_committed:
                                if current_prediction == "space":
                                    sentence += " "
                                elif current_prediction == "del":
                                    sentence = sentence[:-1]
                                else:
                                    sentence += current_prediction

                                last_committed = current_prediction

                            hold_count = 0
            else:
                hold_count = 0
                hold_candidate = ""
    else:
        current_prediction = ""
        current_confidence = 0.0
        prediction_buffer = []
        hold_count = 0
        hold_candidate = ""
        last_committed = ""

    cv2.rectangle(display, (0, 0), (640, 60), (0, 0, 0), -1)
    cv2.putText(
        display,
        f"Sign: {current_prediction} ({current_confidence:.0f}%)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.rectangle(display, (0, frame.shape[0] - 60), (640, frame.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        display,
        f"Sentence: {sentence}",
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("ASL Translator", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
print(f"\n📝 Final sentence: {sentence}")