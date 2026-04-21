import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os

print("📂 Loading landmarks...")
df = pd.read_csv("../data/landmarks_train_orig.csv")
df = df[df["label"] != "nothing"]
print(f"✅ Loaded {len(df)} samples across {len(df['label'].unique())} classes")

# Encode labels to numbers
le = LabelEncoder()
y = le.fit_transform(df["label"].values)
X = df.drop("label", axis=1).values

# Normalize features
X = X / np.max(np.abs(X))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Training: {len(X_train)} | Test: {len(X_test)}")

# Build model
model = Sequential([
    Dense(256, activation='relu', input_shape=(63,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and label encoder
os.makedirs("../models", exist_ok=True)
model.save("../models/asl_nn_model.keras")
with open("../models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Neural network saved to ../models/asl_nn_model.keras")
print("✅ Label encoder saved to ../models/label_encoder.pkl")