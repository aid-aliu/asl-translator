import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import os

print("📂 Loading training landmarks...")
df = pd.read_csv("../data/landmarks.csv")
X = df.drop("label", axis=1).values
y = df["label"].values
print(f"✅ Loaded {len(X)} samples across {len(set(y))} classes")

# Proper 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"🏋️ Training on {len(X_train)}, testing on {len(X_test)}...")

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))


os.makedirs("../models", exist_ok=True)
with open("../models/asl_model.pkl", "wb") as f:
    pickle.dump(model, f)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=45)
plt.tight_layout()
plt.savefig("../models/confusion_matrix.png")
print("✅ Confusion matrix saved")

print("\n✅ Model saved to ../models/asl_model.pkl")