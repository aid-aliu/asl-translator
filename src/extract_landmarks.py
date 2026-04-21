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

def extract_folder(dataset_path, output_file, mode="subfolders"):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"x{i}" for i in range(63)] + ["label"])

        if mode == "subfolders":
            labels = sorted(os.listdir(dataset_path))
            for label in labels:
                label_path = os.path.join(dataset_path, label)
                if not os.path.isdir(label_path):
                    continue
                images = os.listdir(label_path)
                print(f"  Processing {label} ({len(images)} images)...")
                for img_file in tqdm(images):
                    img_path = os.path.join(label_path, img_file)
                    landmarks = extract_landmarks(img_path)
                    if landmarks:
                        writer.writerow(landmarks + [label])

        elif mode == "files":
            images = sorted(os.listdir(dataset_path))
            for img_file in tqdm(images):
                img_path = os.path.join(dataset_path, img_file)
                if not os.path.isfile(img_path):
                    continue
                label = os.path.splitext(img_file)[0].replace("_test", "")
                landmarks = extract_landmarks(img_path)
                if landmarks:
                    writer.writerow(landmarks + [label])

    print(f"✅ Done! Saved to {output_file}")

# ── DONE ──────────────────────────────────────────
# ✅ [1/4] Original ASL train set — DONE
# extract_folder("../data/asl_alphabet_train", "../data/landmarks_train_orig.csv", mode="subfolders")

# ── RUNNING ──────────────────────────────────────────
if not os.path.exists("../data/landmarks_train_synthetic.csv"):
    print("\n📂 [2/4] Synthetic train set...")
    extract_folder(
        "../data/Train_Alphabet",
        "../data/landmarks_train_synthetic.csv",
        mode="subfolders"
    )
else:
    print("⏭️ Skipping [2/4] — already done")

if not os.path.exists("../data/landmarks_test_orig.csv"):
    print("\n📂 [3/4] Original ASL test set...")
    extract_folder(
        "../data/asl_alphabet_test",
        "../data/landmarks_test_orig.csv",
        mode="files"
    )
else:
    print("⏭️ Skipping [3/4] — already done")

if not os.path.exists("../data/landmarks_test_synthetic.csv"):
    print("\n📂 [4/4] Synthetic test set...")
    extract_folder(
        "../data/Test_Alphabet",
        "../data/landmarks_test_synthetic.csv",
        mode="subfolders"
    )
else:
    print("⏭️ Skipping [4/4] — already done")

detector.close()
print("\n🎉 All extractions complete!")