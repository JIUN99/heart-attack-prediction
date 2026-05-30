"""
train_model.py
Heart Attack Prediction — starts from Cell 6 (dataset already generated)

Pipeline:
  Cell 6  — Load CSV → LabelEncoder + StandardScaler + train/test split
  Cell 7  — Class weights + evaluation helper
  Cell 11 — Transformer + NN (only model trained)
  Save    — model artifacts to ./model/ for app.py
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── Sklearn ───────────────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CSV_PATH  = "WQD7005_Hybrid_HeartDisease_Dataset_5k.csv"
MODEL_DIR = "model"

CATEGORICAL_COLS = [
    "Gender", "Exercise_Habits", "Smoking", "Alcohol_Consumption",
    "Sugar_Consumption", "Stress_Level", "Family_Heart_Disease",
    "Heart_Disease_Status", "High_Blood_Pressure", "Diabetes",
    "Low_HDL", "High_LDL",
]
DROP_COLS = [
    "Patient_ID", "Patient_Questionnaire_Response",
    "Doctor_Triage_Note", "combined_text", "Heart_Disease_Status",
]

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — LOAD CSV + PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(csv_path: str):
    print(f"\n[Cell 6] Loading dataset from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}   Columns: {len(df.columns):,}")

    # Encode each categorical with its own LabelEncoder (saved for app.py)
    le_map = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_map[col] = le
    print(f"  ✅ Encoded {len(CATEGORICAL_COLS)} categorical columns.")

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df["Heart_Disease_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")

    scaler       = StandardScaler()
    X_train_s    = scaler.fit_transform(X_train)
    X_test_s     = scaler.transform(X_test)
    feature_names = list(X.columns)

    print(f"  ✅ Scaling done. Feature count: {len(feature_names)}")
    return X_train_s, X_test_s, y_train, y_test, scaler, le_map, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — CLASS WEIGHTS + EVALUATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def get_class_weights(y_train):
    print("\n[Cell 7] Computing class weights ...")
    classes = np.unique(y_train)
    cw      = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict = dict(zip(classes, cw))
    print(f"  Class weight dict: {cw_dict}")
    return cw_dict


def evaluate_model(y_true, y_pred, y_prob):
    return {
        "Model":     "Transformer + NN",
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1_Score":  f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC":   roc_auc_score(y_true, y_prob),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 — TRANSFORMER + NN
# ─────────────────────────────────────────────────────────────────────────────

def train_transformer_nn(X_train_s, y_train, X_test_s, y_test, cw_dict):
    print("\n[Cell 11] Building Transformer + NN ...")
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
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss", patience=7,
        restore_best_weights=True, verbose=1,
    )

    print("\n  Training …")
    model.fit(
        X_train_s, y_train,
        validation_split=0.2,
        epochs=60,
        batch_size=32,
        class_weight=cw_dict,
        callbacks=[early_stop],
        verbose=1,
    )

    prob = model.predict(X_test_s).flatten()
    pred = (prob > 0.5).astype(int)

    print("\n── Evaluation ──────────────────────────────────────────")
    print(classification_report(y_test, pred, target_names=["No Disease", "Disease"]))

    results = evaluate_model(y_test, pred, prob)
    print(f"  Accuracy : {results['Accuracy']:.4f}")
    print(f"  Precision: {results['Precision']:.4f}")
    print(f"  Recall   : {results['Recall']:.4f}")
    print(f"  F1 Score : {results['F1_Score']:.4f}")
    print(f"  ROC-AUC  : {results['ROC_AUC']:.4f}")

    # Save results to CSV
    pd.DataFrame([results]).round(4).to_csv("results.csv", index=False)
    print("  ✅ results.csv saved.")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# SAVE ARTIFACTS (for app.py)
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(model, scaler, le_map, feature_names, out_dir=MODEL_DIR):
    print(f"\n[Save] Writing artifacts to ./{out_dir}/ ...")
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "transformer_nn.keras"))
    with open(os.path.join(out_dir, "scaler.pkl"),          "wb") as f: pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "label_encoders.pkl"),  "wb") as f: pickle.dump(le_map, f)
    with open(os.path.join(out_dir, "feature_names.pkl"),   "wb") as f: pickle.dump(feature_names, f)
    print("  ✅ transformer_nn.keras + scaler + label_encoders + feature_names saved.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Cell 6 — load existing dataset, skip data/text generation entirely
    X_train_s, X_test_s, y_train, y_test, scaler, le_map, feature_names = \
        load_and_preprocess(CSV_PATH)

    # Cell 7 — class weights
    cw_dict = get_class_weights(y_train)

    # Cell 11 — Transformer + NN only
    model = train_transformer_nn(X_train_s, y_train, X_test_s, y_test, cw_dict)

    # Save for app.py
    save_artifacts(model, scaler, le_map, feature_names)
