import os
import csv
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
import cv2
import urllib.request
from tqdm import tqdm

MODEL_PATH = "../data/hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("✅ Model downloaded")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1
)
detector = HandLandmarker.create_from_options(options)

# TRAIN VERSION FOR LATER
# DATASET_PATH = "../data/asl_alphabet_train"
# OUTPUT_FILE = "../data/landmarks.csv"

DATASET_PATH = "../data/asl_alphabet_test"
OUTPUT_FILE = "../data/landmarks_test.csv"


def extract_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        coords = []
        for lm in result.hand_landmarks[0]:
            coords.extend([lm.x, lm.y, lm.z])
        return coords

    return None


def extract_test_set():
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"x{i}" for i in range(63)] + ["label"]
        writer.writerow(header)

        images = sorted(os.listdir(DATASET_PATH))

        for img_file in tqdm(images):
            img_path = os.path.join(DATASET_PATH, img_file)

            if not os.path.isfile(img_path):
                continue

            label = os.path.splitext(img_file)[0].replace("_test", "")

            landmarks = extract_landmarks(img_path)
            if landmarks:
                writer.writerow(landmarks + [label])

    print("✅ Done! Test landmarks saved to", OUTPUT_FILE)


extract_test_set()
detector.close()