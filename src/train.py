import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import os

print("📂 Loading landmarks...")

# Use only original train data — synthetic train was empty
df = pd.read_csv("../data/landmarks_train_orig.csv")

# Drop "nothing" — only 1 sample, useless
df = df[df["label"] != "nothing"]

print(f"✅ Loaded {len(df)} samples across {len(df['label'].unique())} classes")

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("label", axis=1).values,
    df["label"].values,
    test_size=0.2,
    random_state=42
)

print(f"✅ Training: {len(X_train)} | Test: {len(X_test)}")

# Train
print("\n🏋️ Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
print("📊 Evaluating...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Save
os.makedirs("../models", exist_ok=True)
with open("../models/asl_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model saved to ../models/asl_model.pkl")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=45)
plt.tight_layout()
plt.savefig("../models/confusion_matrix.png")
print("✅ Confusion matrix saved")