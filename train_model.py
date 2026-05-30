"""
train_model.py
Heart Attack Prediction — Transformer + NN (best model)
Trains on synthetic hybrid data and saves model artifacts to /model/
"""

import os
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Sentence Embeddings ──────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

# ── Sklearn ──────────────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)

# ── Keras / TensorFlow ───────────────────────────────────────────────────────
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# ── Transformers (DistilBERT sentiment) ─────────────────────────────────────
from transformers import pipeline

# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATE SYNTHETIC TABULAR DATA
# ─────────────────────────────────────────────────────────────────────────────

def generate_tabular_data(total_records: int = 5000) -> pd.DataFrame:
    print(f"[1/6] Generating {total_records} synthetic patient records …")
    np.random.seed(42)
    random.seed(42)

    n = total_records
    ages            = np.random.randint(30, 85, n)
    genders         = np.random.choice(["Male", "Female"], n)
    exercise        = np.random.choice(["Low", "Medium", "High"], n, p=[0.4, 0.4, 0.2])
    smoking         = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])
    alcohol         = np.random.choice(["None", "Low", "Medium", "High"], n)
    stress          = np.random.choice(["Low", "Medium", "High"], n)
    sleep_hours     = np.random.randint(4, 10, n)
    sugar_intake    = np.random.choice(["Low", "Medium", "High"], n)
    bmi             = np.round(np.random.uniform(18.5, 40.0, n), 1)
    systolic_bp     = np.random.randint(100, 180, n) + np.random.normal(0, 8, n)
    cholesterol     = np.random.randint(150, 300, n) + np.random.normal(0, 15, n)
    triglycerides   = np.random.randint(100, 250, n)
    fasting_sugar   = np.random.randint(70, 160, n)
    crp             = np.round(np.random.uniform(0.5, 15.0, n), 1)
    homocysteine    = np.round(np.random.uniform(5.0, 20.0, n), 1)
    high_bp         = ["Yes" if bp >= 140 else "No" for bp in systolic_bp]
    diabetes        = ["Yes" if s >= 126 else "No" for s in fasting_sugar]
    family_history  = np.random.choice(["Yes", "No"], n, p=[0.25, 0.75])
    low_hdl         = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])
    high_ldl        = ["Yes" if c >= 200 else "No" for c in cholesterol]

    heart_disease = []
    for i in range(n):
        score = 0
        score += 2 if ages[i] > 60 and smoking[i] == "Yes" else (1 if ages[i] > 60 else 0)
        score += float(np.random.uniform(1, 3)) if smoking[i] == "Yes" else 0
        score += 2 if high_bp[i] == "Yes" else 0
        score += 2 if diabetes[i] == "Yes" else 0
        score += 2 if family_history[i] == "Yes" else 0
        score += 2 if bmi[i] > 30 and diabetes[i] == "Yes" else (1 if bmi[i] > 30 else 0)
        has = "Yes" if score >= 5 and random.random() < 0.8 else ("Yes" if random.random() < 0.1 else "No")
        heart_disease.append(has)

    return pd.DataFrame({
        "Patient_ID": [f"HD_{i:04d}" for i in range(1, n + 1)],
        "Age": ages, "Gender": genders, "BMI": bmi,
        "Blood_Pressure": systolic_bp, "High_Blood_Pressure": high_bp,
        "Cholesterol_Level": cholesterol, "Low_HDL": low_hdl, "High_LDL": high_ldl,
        "Triglyceride_Level": triglycerides, "Fasting_Blood_Sugar": fasting_sugar,
        "Diabetes": diabetes, "CRP_Level": crp, "Homocysteine_Level": homocysteine,
        "Exercise_Habits": exercise, "Smoking": smoking, "Alcohol_Consumption": alcohol,
        "Sugar_Consumption": sugar_intake, "Stress_Level": stress, "Sleep_Hours": sleep_hours,
        "Family_Heart_Disease": family_history, "Heart_Disease_Status": heart_disease,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERATE TEXT FEATURES (rule-based fallback — no GPU needed at train time)
# ─────────────────────────────────────────────────────────────────────────────

_QUESTIONNAIRE_YES = [
    "I occasionally feel tired after physical activity and sometimes notice mild discomfort when stressed.",
    "I sometimes feel breathless when climbing stairs, but overall my symptoms are inconsistent.",
    "I may experience fatigue, dizziness, or poor sleep without it sounding overly serious.",
    "I occasionally notice chest tightness or reduced stamina, but symptoms are subtle.",
    "I mostly feel normal but notice general tiredness or low energy at times.",
    "I occasionally feel my heart racing or discomfort after exertion, but not all the time.",
]
_QUESTIONNAIRE_NO = [
    "I mainly feel work-related stress and occasional tiredness after long days.",
    "I may mention poor sleep, anxiety, or muscle soreness from daily activities.",
    "I generally feel healthy but occasionally experience fatigue or low energy.",
    "I sometimes feel mild chest discomfort related to stress or posture, but nothing severe.",
    "I occasionally feel dizzy or tired due to lack of rest or a busy schedule.",
    "I describe normal daily routines with minor health complaints.",
]
_TRIAGE_YES = [
    "Pt presents with slightly elevated cardiovascular risk factors; no definitive conclusions at this time.",
    "Patient appears stable overall while noting mild symptoms worth monitoring.",
    "Occasional exertional discomfort or fatigue noted; no explicit cardiac diagnosis made.",
    "Follow-up observation may be beneficial if symptoms persist.",
    "Clinically neutral assessment with only subtle concern regarding cardiovascular health.",
    "ECG and vitals mostly stable; minor abnormalities observed.",
]
_TRIAGE_NO = [
    "Pt appears generally stable with no acute distress observed.",
    "Routine monitoring indicated; stable cardiovascular presentation noted.",
    "Vitals within acceptable range; no major abnormalities detected.",
    "Mild fatigue or stress-related symptoms noted without serious concern.",
    "Clinically neutral; no urgent findings at this time.",
    "Normal ECG rhythm and stable overall condition documented.",
]

def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Adding rule-based text features …")
    qs, tn = [], []
    for _, row in df.iterrows():
        pool_q = _QUESTIONNAIRE_YES if row["Heart_Disease_Status"] == "Yes" else _QUESTIONNAIRE_NO
        pool_t = _TRIAGE_YES        if row["Heart_Disease_Status"] == "Yes" else _TRIAGE_NO
        qs.append(random.choice(pool_q))
        tn.append(random.choice(pool_t))
    df = df.copy()
    df["Patient_Questionnaire_Response"] = qs
    df["Doctor_Triage_Note"] = tn
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. SENTENCE EMBEDDINGS + SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────

def add_embeddings_and_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/6] Generating sentence embeddings (all-MiniLM-L6-v2) …")
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    combined  = (df["Patient_Questionnaire_Response"] + " " + df["Doctor_Triage_Note"]).tolist()
    embeddings = emb_model.encode(combined, show_progress_bar=True, batch_size=64)
    emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])

    print("[3/6] Running sentiment analysis (distilbert-base-uncased-finetuned-sst-2) …")
    sent_pipe = pipeline("sentiment-analysis",
                         model="distilbert-base-uncased-finetuned-sst-2-english",
                         truncation=True, max_length=512)
    scores = [sent_pipe(t[:512])[0]["score"] for t in tqdm(df["Patient_Questionnaire_Response"])]

    df = df.reset_index(drop=True)
    df = pd.concat([df, emb_df], axis=1)
    df["Sentiment_Score"] = scores
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "Gender", "Exercise_Habits", "Smoking", "Alcohol_Consumption",
    "Sugar_Consumption", "Stress_Level", "Family_Heart_Disease",
    "Heart_Disease_Status", "High_Blood_Pressure", "Diabetes", "Low_HDL", "High_LDL",
]
DROP_COLS = ["Patient_ID", "Patient_Questionnaire_Response",
             "Doctor_Triage_Note", "combined_text", "Heart_Disease_Status"]

def preprocess(df: pd.DataFrame):
    print("[4/6] Preprocessing …")
    le_map: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_map[col] = le

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df["Heart_Disease_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, cw))

    return X_train_s, X_test_s, y_train, y_test, scaler, le_map, class_weight_dict


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRANSFORMER + NN (best model)
# ─────────────────────────────────────────────────────────────────────────────

def build_transformer_nn(input_dim: int) -> Sequential:
    model = Sequential([
        Dense(256, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train(X_train, y_train, class_weight_dict: dict) -> Sequential:
    print("[5/6] Training Transformer + NN …")
    model = build_transformer_nn(X_train.shape[1])
    early_stop = EarlyStopping(patience=7, restore_best_weights=True, verbose=1)
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=60,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATE & SAVE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test):
    prob = model.predict(X_test).flatten()
    pred = (prob > 0.5).astype(int)
    metrics = {
        "Accuracy":  accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall":    recall_score(y_test, pred, zero_division=0),
        "F1":        f1_score(y_test, pred, zero_division=0),
        "ROC_AUC":   roc_auc_score(y_test, prob),
    }
    print("\n── Evaluation ──────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<12} {v:.4f}")
    print(classification_report(y_test, pred, target_names=["No Disease", "Disease"]))
    return metrics


def save_artifacts(model, scaler, le_map, feature_names: list, out_dir: str = "model"):
    print(f"[6/6] Saving artifacts to ./{out_dir}/ …")
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "transformer_nn.keras"))
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "label_encoders.pkl"), "wb") as f:
        pickle.dump(le_map, f)
    with open(os.path.join(out_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)
    print("  ✅  All artifacts saved.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_tabular_data(5000)
    df = add_text_features(df)
    df = add_embeddings_and_sentiment(df)

    X_train, X_test, y_train, y_test, scaler, le_map, cw = preprocess(df)

    model = train(X_train, y_train, cw)
    evaluate(model, X_test, y_test)

    # Reconstruct feature names after preprocessing
    drop = {"Patient_ID", "Patient_Questionnaire_Response",
            "Doctor_Triage_Note", "combined_text", "Heart_Disease_Status"}
    feature_names = [c for c in df.columns if c not in drop]
    save_artifacts(model, scaler, le_map, feature_names)
