import pandas as pd

print("=== TRAIN ORIG ===")
df = pd.read_csv("../data/landmarks_train_orig.csv")
print(f"Rows: {len(df)}")
print(df["label"].value_counts())

print("\n=== TRAIN SYNTHETIC ===")
df2 = pd.read_csv("../data/landmarks_train_synthetic.csv")
print(f"Rows: {len(df2)}")
print(df2["label"].value_counts())

print("\n=== TEST SYNTHETIC ===")
df3 = pd.read_csv("../data/landmarks_test_synthetic.csv")
print(f"Rows: {len(df3)}")
print(df3["label"].value_counts())