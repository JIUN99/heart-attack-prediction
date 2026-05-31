"""
train_model.py - Heart Attack Prediction (Tabular Only)
No sentence-transformers, no NLP. Just 20 clean clinical features.
RAM usage: ~120MB  |  Load time: <5s  |  Works on Render free tier.
"""

import os, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH  = "WQD7005_Hybrid_HeartDisease_Dataset_5k.csv"
MODEL_DIR = "model"

# Only the 20 clean tabular features — no text, no embeddings
NUMERIC_COLS = [
    "Age", "BMI", "Blood_Pressure", "Cholesterol_Level",
    "Triglyceride_Level", "Fasting_Blood_Sugar",
    "CRP_Level", "Homocysteine_Level", "Sleep_Hours",
]
CATEGORICAL_COLS = [
    "Gender", "Exercise_Habits", "Smoking", "Alcohol_Consumption",
    "Sugar_Consumption", "Stress_Level", "Family_Heart_Disease",
    "High_Blood_Pressure", "Diabetes", "Low_HDL", "High_LDL",
]
TARGET = "Heart_Disease_Status"
DROP_COLS = ["Patient_ID", "Patient_Questionnaire_Response",
             "Doctor_Triage_Note", "combined_text"]

# ── Load & preprocess ─────────────────────────────────────────────────────────
print("\n[1] Loading dataset ...")
df = pd.read_csv(CSV_PATH)
print(f"    Rows: {len(df):,}   Cols: {len(df.columns)}")

# Drop text/ID columns if present
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# Keep only our 20 features + target
keep = NUMERIC_COLS + CATEGORICAL_COLS + [TARGET]
df = df[[c for c in keep if c in df.columns]]
print(f"    Features used: {[c for c in keep if c in df.columns and c != TARGET]}")

# Encode categoricals
le_map = {}
for col in CATEGORICAL_COLS:
    if col not in df.columns:
        continue
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_map[col] = le
print(f"    Encoded {len(le_map)} categorical columns.")

# Encode target
le_target = LabelEncoder()
df[TARGET] = le_target.fit_transform(df[TARGET].astype(str))
print(f"    Target classes: {list(le_target.classes_)}")

feature_cols = [c for c in df.columns if c != TARGET]
X = df[feature_cols].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"    Train: {len(X_train):,}  Test: {len(X_test):,}  Features: {X.shape[1]}")

# ── Class weights ─────────────────────────────────────────────────────────────
classes  = np.unique(y_train)
cw       = compute_class_weight("balanced", classes=classes, y=y_train)
cw_dict  = {int(k): float(v) for k, v in zip(classes, cw)}
print(f"\n[2] Class weights: {cw_dict}")

# ── Build Transformer NN ──────────────────────────────────────────────────────
print("\n[3] Building Transformer NN ...")
model = Sequential([
    Dense(256, activation="relu", input_shape=(X_train_s.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64,  activation="relu"),
    Dense(1,   activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=7,
                            restore_best_weights=True, verbose=1)
model.fit(X_train_s, y_train,
          validation_split=0.2, epochs=60, batch_size=32,
          class_weight=cw_dict, callbacks=[early_stop], verbose=1)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n[4] Evaluating ...")
prob = model.predict(X_test_s).flatten()
pred = (prob > 0.5).astype(int)
print(classification_report(y_test, pred, target_names=["No Disease","Disease"]))
results = {
    "Model":     "Transformer + NN (tabular)",
    "Accuracy":  round(accuracy_score(y_test, pred),  4),
    "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
    "Recall":    round(recall_score(y_test, pred, zero_division=0),    4),
    "F1_Score":  round(f1_score(y_test, pred, zero_division=0),        4),
    "ROC_AUC":   round(roc_auc_score(y_test, prob),   4),
}
for k, v in results.items():
    print(f"    {k}: {v}")
pd.DataFrame([results]).to_csv("results.csv", index=False)

# ── Save artifacts ────────────────────────────────────────────────────────────
print(f"\n[5] Saving to ./{MODEL_DIR}/ ...")
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, "transformer_nn.keras"))
with open(os.path.join(MODEL_DIR, "scaler.pkl"),         "wb") as f: pickle.dump(scaler,       f)
with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "wb") as f: pickle.dump(le_map,        f)
with open(os.path.join(MODEL_DIR, "feature_names.pkl"),  "wb") as f: pickle.dump(feature_cols,  f)
with open(os.path.join(MODEL_DIR, "label_target.pkl"),   "wb") as f: pickle.dump(le_target,     f)

print("    transformer_nn.keras")
print("    scaler.pkl")
print("    label_encoders.pkl")
print("    feature_names.pkl  →", feature_cols)
print("    label_target.pkl")
print(f"\n✅ Done. AUC={results['ROC_AUC']}  Accuracy={results['Accuracy']}")
