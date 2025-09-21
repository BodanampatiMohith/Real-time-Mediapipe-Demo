import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA_DIR = "sign_data_both_hands"
MODEL_PATH = "sign_language_model.pkl"
files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
all_data, all_labels = [], []

for file in files:
    label = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)
    all_data.append(df.values)
    all_labels.extend([label] * len(df))

X = np.vstack(all_data)
y = np.array(all_labels)

print("Total samples:", X.shape[0], " | Features per sample:", X.shape[1])
le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("✅ Model accuracy:", round(acc * 100, 2), "%")
joblib.dump({"model": model, "scaler": scaler, "encoder": le}, MODEL_PATH)
print(f"💾 Model saved as {MODEL_PATH}")
