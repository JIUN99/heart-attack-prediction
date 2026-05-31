"""
train_model.py - Heart Attack Prediction
Uses scikit-learn only — no TensorFlow, no NLP.
RAM: ~80MB | Load time: <2s | Works perfectly on Render free tier.
"""
import os, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

CSV_PATH  = "WQD7005_Hybrid_HeartDisease_Dataset_5k.csv"
MODEL_DIR = "model"

NUMERIC_COLS = [
    "Age","BMI","Blood_Pressure","Cholesterol_Level",
    "Triglyceride_Level","Fasting_Blood_Sugar",
    "CRP_Level","Homocysteine_Level","Sleep_Hours",
]
CATEGORICAL_COLS = [
    "Gender","Exercise_Habits","Smoking","Alcohol_Consumption",
    "Sugar_Consumption","Stress_Level","Family_Heart_Disease",
    "High_Blood_Pressure","Diabetes","Low_HDL","High_LDL",
]
TARGET   = "Heart_Disease_Status"
DROP_COLS = ["Patient_ID","Patient_Questionnaire_Response",
             "Doctor_Triage_Note","combined_text"]

print("\n[1] Loading dataset ...")
df = pd.read_csv(CSV_PATH)
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
keep = NUMERIC_COLS + CATEGORICAL_COLS + [TARGET]
df = df[[c for c in keep if c in df.columns]]
print(f"    Shape: {df.shape}")

le_map = {}
for col in CATEGORICAL_COLS:
    if col not in df.columns: continue
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_map[col] = le

le_target = LabelEncoder()
df[TARGET] = le_target.fit_transform(df[TARGET].astype(str))
print(f"    Target classes: {list(le_target.classes_)}")

feature_cols = [c for c in df.columns if c != TARGET]
X = df[feature_cols].values.astype(np.float32)
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"    Train: {len(X_train):,}  Test: {len(X_test):,}  Features: {X.shape[1]}")

cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
cw_dict = dict(zip(np.unique(y_train), cw))

print("\n[2] Training models ...")
models = {
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=42, n_jobs=-1),
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42),
}

best_auc, best_name, best_model = 0, None, None
results = []
for name, m in models.items():
    if name == "LogisticRegression":
        m.fit(X_train_s, y_train)
        prob = m.predict_proba(X_test_s)[:,1]
        pred = m.predict(X_test_s)
    else:
        m.fit(X_train, y_train)
        prob = m.predict_proba(X_test)[:,1]
        pred = m.predict(X_test)
    auc = roc_auc_score(y_test, prob)
    acc = accuracy_score(y_test, pred)
    print(f"    {name}: AUC={auc:.4f}  Acc={acc:.4f}")
    results.append({"Model":name,"AUC":round(auc,4),"Accuracy":round(acc,4)})
    if auc > best_auc:
        best_auc, best_name, best_model = auc, name, m

print(f"\n    ✅ Best: {best_name} (AUC={best_auc:.4f})")
print(classification_report(y_test,
    best_model.predict(X_test_s if best_name=="LogisticRegression" else X_test),
    target_names=["No Disease","Disease"]))

pd.DataFrame(results).to_csv("results.csv", index=False)

print(f"\n[3] Saving to ./{MODEL_DIR}/ ...")
os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR,"best_model.pkl"),      "wb") as f: pickle.dump(best_model,   f)
with open(os.path.join(MODEL_DIR,"scaler.pkl"),          "wb") as f: pickle.dump(scaler,       f)
with open(os.path.join(MODEL_DIR,"label_encoders.pkl"),  "wb") as f: pickle.dump(le_map,       f)
with open(os.path.join(MODEL_DIR,"feature_names.pkl"),   "wb") as f: pickle.dump(feature_cols, f)
with open(os.path.join(MODEL_DIR,"meta.pkl"),            "wb") as f:
    pickle.dump({"best_name":best_name,"best_auc":best_auc}, f)
print(f"    Saved: best_model={best_name}, AUC={best_auc:.4f}")
print("✅ Done!")
