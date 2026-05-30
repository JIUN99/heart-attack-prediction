"""
train_model.py
Heart Attack Prediction — Full Pipeline (mirrors GP__1_.ipynb exactly)

Steps:
  Cell 2  — Tabular data generation with risk scoring + 3% label noise
  Cell 3  — SLM text generation (Mistral-7B-Instruct) with rule-based fallback
  Cell 4  — Sentence embeddings (all-MiniLM-L6-v2)
  Cell 5  — Sentiment analysis (distilbert-base-uncased-finetuned-sst-2-english)
  Cell 6  — Preprocessing: LabelEncoder + StandardScaler + train/test split
  Cell 7  — Class weights + evaluation helper
  Cell 8  — Random Forest
  Cell 9  — XGBoost
  Cell 10 — Weighted Neural Network
  Cell 11 — Transformer + NN  ← best model (saved for app.py)
  Cell 12 — Model comparison → results.csv
"""

import os
import random
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Sklearn ──────────────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

# ── XGBoost ──────────────────────────────────────────────────────────────────
from xgboost import XGBClassifier

# ── TensorFlow / Keras ───────────────────────────────────────────────────────
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ── Transformers ─────────────────────────────────────────────────────────────
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — TABULAR DATA GENERATION (with 3% label noise)
# ─────────────────────────────────────────────────────────────────────────────

def generate_tabular_data(total_records=5000):
    print(f"\n[Cell 2] Generating {total_records} structured records...")
    np.random.seed(42)

    patient_ids = [f"HD_{i:04d}" for i in range(1, total_records + 1)]
    ages        = np.random.randint(30, 85, total_records)
    genders     = np.random.choice(["Male", "Female"], total_records)

    exercise    = np.random.choice(["Low", "Medium", "High"], total_records, p=[0.4, 0.4, 0.2])
    smoking     = np.random.choice(["Yes", "No"], total_records, p=[0.3, 0.7])
    alcohol     = np.random.choice(["None", "Low", "Medium", "High"], total_records)
    stress      = np.random.choice(["Low", "Medium", "High"], total_records)
    sleep_hours = np.random.randint(4, 10, total_records)
    sugar_intake = np.random.choice(["Low", "Medium", "High"], total_records)

    bmi          = np.round(np.random.uniform(18.5, 40.0, total_records), 1)
    systolic_bp  = np.random.randint(100, 180, total_records) + np.random.normal(0, 8, total_records)
    cholesterol  = np.random.randint(150, 300, total_records) + np.random.normal(0, 15, total_records)
    triglycerides = np.random.randint(100, 250, total_records)
    fasting_sugar = np.random.randint(70, 160, total_records)
    crp          = np.round(np.random.uniform(0.5, 15.0, total_records), 1)
    homocysteine = np.round(np.random.uniform(5.0, 20.0, total_records), 1)

    high_bp       = ["Yes" if bp >= 140 else "No" for bp in systolic_bp]
    diabetes      = ["Yes" if s  >= 126 else "No" for s  in fasting_sugar]
    family_history = np.random.choice(["Yes", "No"], total_records, p=[0.25, 0.75])
    low_hdl       = np.random.choice(["Yes", "No"], total_records, p=[0.3, 0.7])
    high_ldl      = ["Yes" if c >= 200 else "No" for c in cholesterol]

    heart_disease = []
    for i in range(total_records):
        risk_score = 0
        if ages[i] > 60 and smoking[i] == "No":  risk_score += 1
        if ages[i] > 60 and smoking[i] == "Yes": risk_score += 2
        if smoking[i] == "Yes": risk_score += np.random.uniform(1, 3)   # stochastic noise
        if high_bp[i]        == "Yes": risk_score += 2
        if diabetes[i]       == "Yes": risk_score += 2
        if family_history[i] == "Yes": risk_score += 2
        if bmi[i] > 30 and diabetes[i] == "No":  risk_score += 1
        if bmi[i] > 30 and diabetes[i] == "Yes": risk_score += 2

        has_disease = (
            "Yes" if risk_score >= 5 and random.random() < 0.8
            else ("Yes" if random.random() < 0.1 else "No")
        )
        # 3% label noise (exactly as in notebook)
        if random.random() < 0.03:
            has_disease = "No" if has_disease == "Yes" else "Yes"
        heart_disease.append(has_disease)

    df = pd.DataFrame({
        "Patient_ID": patient_ids, "Age": ages, "Gender": genders, "BMI": bmi,
        "Blood_Pressure": systolic_bp, "High_Blood_Pressure": high_bp,
        "Cholesterol_Level": cholesterol, "Low_HDL": low_hdl, "High_LDL": high_ldl,
        "Triglyceride_Level": triglycerides, "Fasting_Blood_Sugar": fasting_sugar,
        "Diabetes": diabetes, "CRP_Level": crp, "Homocysteine_Level": homocysteine,
        "Exercise_Habits": exercise, "Smoking": smoking, "Alcohol_Consumption": alcohol,
        "Sugar_Consumption": sugar_intake, "Stress_Level": stress, "Sleep_Hours": sleep_hours,
        "Family_Heart_Disease": family_history, "Heart_Disease_Status": heart_disease,
    })
    print(f"✅ {total_records} tabular rows generated.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — SLM TEXT GENERATION (Mistral-7B with rule-based fallback)
# ─────────────────────────────────────────────────────────────────────────────

# Rule-based pools (used when Mistral is unavailable / as fallback)
_Q_YES = [
    "I occasionally feel tired after physical activity and sometimes notice mild discomfort when stressed.",
    "I sometimes feel breathless when climbing stairs, but overall my symptoms are inconsistent.",
    "I may describe fatigue, dizziness, or poor sleep without sounding overly serious.",
    "I occasionally notice chest tightness, stress, or reduced stamina, but symptoms are subtle.",
    "I may appear mostly normal and only mention general tiredness or low energy.",
    "I occasionally feel heart racing or discomfort after exertion, but not all the time.",
]
_Q_NO = [
    "I mainly feel work-related stress and occasional tiredness after long days.",
    "I may mention poor sleep, anxiety, or muscle soreness from daily activities.",
    "I generally feel healthy but occasionally experience fatigue or low energy.",
    "I sometimes feel mild chest discomfort related to stress or posture, but nothing severe.",
    "I occasionally feel dizzy or tired due to lack of rest or busy schedules.",
    "I describe normal daily routines with minor health complaints.",
]
_T_YES = [
    "Mention slightly elevated cardiovascular risk factors but avoid definitive conclusions.",
    "Describe the patient as stable overall, while noting mild symptoms worth monitoring.",
    "Mention occasional exertional discomfort or fatigue without explicitly diagnosing heart disease.",
    "State that follow-up observation may be beneficial if symptoms persist.",
    "Keep the note clinically neutral with only subtle concern regarding cardiovascular health.",
    "Mention that ECG and vitals are mostly stable with minor abnormalities observed.",
]
_T_NO = [
    "Describe the patient as generally stable with no acute distress observed.",
    "Mention routine monitoring and stable cardiovascular presentation.",
    "State that vitals are within acceptable range with no major abnormalities.",
    "Mention mild fatigue or stress-related symptoms without serious concern.",
    "Keep the note clinically neutral with no urgent findings.",
    "Document normal ECG rhythm and stable condition overall.",
]


def _try_load_mistral():
    """Attempt to load Mistral-7B-Instruct. Returns generator or None."""
    try:
        import torch
        print("[Cell 3] Loading Mistral-7B-Instruct-v0.1 ...")
        gen = hf_pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.1",
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print("✅ Mistral loaded.")
        return gen
    except Exception as e:
        print(f"⚠️  Mistral unavailable ({e}). Using rule-based fallback.")
        return None


def _generate_with_mistral(generator, row, patient_condition, triage_condition):
    prompt_q = f"""
<s>[INST]
Act as a {row['Age']}-year-old {row['Gender']} patient.
Your lifestyle profile is: Exercise: {row['Exercise_Habits']}, Smoking: {row['Smoking']}, Alcohol: {row['Alcohol_Consumption']}, Sleep: {row['Sleep_Hours']} hours/night, Stress: {row['Stress_Level']}.

{patient_condition}

The doctor asks: "Can you describe your daily habits lately? How are you sleeping, eating, exercising, and managing stress?"

Write a natural, 3-to-4 sentence conversational response.
You MUST explicitly mention your sleep hours, smoking habits, drinking frequency, and exercise routine in your own words, alongside your symptoms.
Naturally describe your lifestyle, sleep, stress levels, daily habits, and any physical discomfort in your own words. Not every detail needs to be mentioned explicitly.
Speak naturally as a patient. Do not use medical jargon.
[/INST]
"""
    prompt_t = f"""
<s>[INST]
Act as a triage nurse. Write a brief 2-sentence clinical note for a patient with:
Blood Pressure {row['Blood_Pressure']} (High BP: {row['High_Blood_Pressure']}), BMI {row['BMI']}, Diabetes: {row['Diabetes']}. Family history of heart disease: {row['Family_Heart_Disease']}.

{triage_condition}

Keep it professional, highly objective, and use standard medical abbreviations (e.g., pt, BP, HR, ECG).
[/INST]
"""
    res_q = generator(
        prompt_q, max_new_tokens=300, min_new_tokens=60, do_sample=True,
        temperature=0.7, top_p=0.9, repetition_penalty=1.1,
        eos_token_id=generator.tokenizer.eos_token_id,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    q_text = res_q[0]["generated_text"].split("[/INST]")[-1].strip().replace("\n", " ")
    if "." in q_text:
        q_text = ".".join(q_text.split(".")[:-1]) + "."

    res_t = generator(
        prompt_t, max_new_tokens=300, min_new_tokens=40, do_sample=True,
        temperature=0.6, top_p=0.9, repetition_penalty=1.1,
        eos_token_id=generator.tokenizer.eos_token_id,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    t_text = res_t[0]["generated_text"].split("[/INST]")[-1].strip().replace("\n", " ")
    if "." in t_text:
        t_text = ".".join(t_text.split(".")[:-1]) + "."

    return q_text, t_text


def generate_text_features_safely(df, save_interval=500):
    print("\n[Cell 3] Starting SLM predictive text generation...")
    generator = _try_load_mistral()

    questionnaires, triage_notes = [], []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating Text"):
        is_positive = row["Heart_Disease_Status"] == "Yes"
        patient_condition = random.choice(_Q_YES if is_positive else _Q_NO)
        triage_condition  = random.choice(_T_YES if is_positive else _T_NO)

        try:
            if generator:
                q_text, t_text = _generate_with_mistral(
                    generator, row, patient_condition, triage_condition)
            else:
                # Rule-based fallback — construct a sentence using patient stats
                q_text = (
                    f"I'm a {row['Age']}-year-old {row['Gender'].lower()} and I "
                    f"{'smoke regularly' if row['Smoking']=='Yes' else 'do not smoke'}. "
                    f"I exercise {row['Exercise_Habits'].lower()}ly, sleep about "
                    f"{row['Sleep_Hours']} hours a night, and my stress is "
                    f"{row['Stress_Level'].lower()}. {patient_condition}"
                )
                t_text = (
                    f"Pt presents with BP {row['Blood_Pressure']:.0f}, BMI {row['BMI']}, "
                    f"DM: {row['Diabetes']}, FHx heart disease: {row['Family_Heart_Disease']}. "
                    f"{triage_condition}"
                )
        except Exception as e:
            print(f"\n❌ ERROR at row {index}: {e}")
            q_text = "Error generating response."
            t_text = "Error generating note."

        questionnaires.append(q_text)
        triage_notes.append(t_text)

        # Incremental CSV backup (same interval as notebook)
        current_row = index + 1
        if current_row % save_interval == 0:
            temp_df = df.iloc[:current_row].copy()
            temp_df["Patient_Questionnaire_Response"] = questionnaires
            temp_df["Doctor_Triage_Note"] = triage_notes
            temp_df.to_csv(f"WQD7005_Backup_Row_{current_row}.csv", index=False)
            print(f"  💾 Backup saved at row {current_row}")

    df = df.copy()
    df["Patient_Questionnaire_Response"] = questionnaires
    df["Doctor_Triage_Note"] = triage_notes
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — SENTENCE EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

def add_sentence_embeddings(df):
    print("\n[Cell 4] Generating sentence embeddings (all-MiniLM-L6-v2) ...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    df["combined_text"] = (
        df["Patient_Questionnaire_Response"] + " " + df["Doctor_Triage_Note"]
    )
    embeddings = embedding_model.encode(
        df["combined_text"].tolist(), show_progress_bar=True
    )
    embedding_df = pd.DataFrame(
        embeddings,
        columns=[f"embedding_{i}" for i in range(embeddings.shape[1])]
    )
    df_final = pd.concat([df.reset_index(drop=True), embedding_df], axis=1)
    print("✅ Embeddings added.")
    return df_final


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — SENTIMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def add_sentiment(df_final):
    print("\n[Cell 5] Running sentiment analysis ...")
    sentiment_pipeline = hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512,
    )
    scores = []
    for text in tqdm(df_final["Patient_Questionnaire_Response"],
                     total=len(df_final), desc="Sentiment"):
        result = sentiment_pipeline(text[:512])[0]
        scores.append(result["score"])
    df_final = df_final.copy()
    df_final["Sentiment_Score"] = scores
    df_final.to_csv("WQD7005_Hybrid_HeartDisease_Dataset_5k.csv", index=False)
    print("✅ Sentiment done. Full dataset saved to WQD7005_Hybrid_HeartDisease_Dataset_5k.csv")
    return df_final


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "Gender", "Exercise_Habits", "Smoking", "Alcohol_Consumption",
    "Sugar_Consumption", "Stress_Level", "Family_Heart_Disease",
    "Heart_Disease_Status", "High_Blood_Pressure", "Diabetes", "Low_HDL", "High_LDL",
]
DROP_COLS = [
    "Patient_ID", "Patient_Questionnaire_Response",
    "Doctor_Triage_Note", "combined_text", "Heart_Disease_Status",
]

def preprocess(df_final):
    print("\n[Cell 6] Preprocessing ...")
    le_map = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_final[col] = le.fit_transform(df_final[col])
        le_map[col] = le

    X = df_final.drop(columns=[c for c in DROP_COLS if c in df_final.columns])
    y = df_final["Heart_Disease_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("✅ Preprocessing done.")
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, le_map


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — CLASS WEIGHTS + EVALUATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def get_class_weights(y_train):
    print("\n[Cell 7] Computing class weights ...")
    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict = dict(zip(classes, cw))
    print("  Class weight dict:", cw_dict)
    return cw_dict


def evaluate_model(model_name, y_true, y_pred, y_prob):
    return {
        "Model":     model_name,
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1_Score":  f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC":   roc_auc_score(y_true, y_prob),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train, X_test, y_test, cw_dict):
    print("\n[Cell 8] Training Random Forest ...")
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=10,
        class_weight=cw_dict, random_state=42,
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, rf_pred))
    return rf_model, evaluate_model("Random Forest", y_test, rf_pred, rf_prob)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — XGBOOST
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n[Cell 9] Training XGBoost ...")
    neg = sum(y_train == 0)
    pos = sum(y_train == 1)
    spw = neg / pos
    print(f"  scale_pos_weight = {spw:.3f}")
    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=10, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, random_state=42,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, xgb_pred))
    return xgb_model, evaluate_model("XGBoost", y_test, xgb_pred, xgb_prob)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — WEIGHTED NEURAL NETWORK
# ─────────────────────────────────────────────────────────────────────────────

def train_weighted_nn(X_train_scaled, y_train, X_test_scaled, y_test, cw_dict):
    print("\n[Cell 10] Training Weighted Neural Network ...")
    nn_model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
        Dense(64,  activation="relu"),
        Dense(1,   activation="sigmoid"),
    ])
    nn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    nn_model.fit(
        X_train_scaled, y_train,
        validation_split=0.2, epochs=50, batch_size=32,
        class_weight=cw_dict, callbacks=[early_stop], verbose=0,
    )
    nn_prob = nn_model.predict(X_test_scaled)
    nn_pred = (nn_prob > 0.5).astype(int)
    print(classification_report(y_test, nn_pred))
    return nn_model, evaluate_model("Weighted Neural Network", y_test, nn_pred, nn_prob)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 — TRANSFORMER + NN (best model → saved to disk)
# ─────────────────────────────────────────────────────────────────────────────

def train_transformer_nn(X_train_scaled, y_train, X_test_scaled, y_test, cw_dict):
    print("\n[Cell 11] Training Transformer + NN ...")
    transformer_nn = Sequential([
        Dense(256, activation="relu", input_shape=(X_train_scaled.shape[1],)),
        Dense(128, activation="relu"),
        Dense(1,   activation="sigmoid"),
    ])
    transformer_nn.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    transformer_nn.fit(
        X_train_scaled, y_train,
        epochs=50, batch_size=32,
        class_weight=cw_dict,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1,
    )
    prob = transformer_nn.predict(X_test_scaled)
    pred = (prob > 0.5).astype(int)
    print(classification_report(y_test, pred))
    return transformer_nn, evaluate_model("Transformer + NN", y_test, pred, prob)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 12 — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(rf_r, xgb_r, nn_r, tnn_r):
    print("\n[Cell 12] Model comparison:")
    results_df = pd.DataFrame([rf_r, xgb_r, nn_r, tnn_r]).round(4)
    print(results_df.to_string(index=False))
    results_df.to_csv("results.csv", index=False)
    print("✅ results.csv saved.")
    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# SAVE BEST-MODEL ARTIFACTS (for app.py)
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(model, scaler, le_map, feature_names, out_dir="model"):
    print(f"\n[Save] Writing artifacts to ./{out_dir}/ ...")
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "transformer_nn.keras"))
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "label_encoders.pkl"), "wb") as f:
        pickle.dump(le_map, f)
    with open(os.path.join(out_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)
    print("✅ All artifacts saved.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Cell 2
    df_tabular = generate_tabular_data(total_records=5000)

    # Cell 3
    df_hybrid = generate_text_features_safely(df_tabular.copy(), save_interval=500)

    # Cell 4
    df_final = add_sentence_embeddings(df_hybrid)

    # Cell 5
    df_final = add_sentiment(df_final)

    # Cell 6
    X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler, le_map = preprocess(df_final)

    # Cell 7
    cw_dict = get_class_weights(y_train)

    # Cell 8
    _, rf_r  = train_random_forest(X_train, y_train, X_test, y_test, cw_dict)

    # Cell 9
    _, xgb_r = train_xgboost(X_train, y_train, X_test, y_test)

    # Cell 10
    _, nn_r  = train_weighted_nn(X_train_s, y_train, X_test_s, y_test, cw_dict)

    # Cell 11
    best_model, tnn_r = train_transformer_nn(X_train_s, y_train, X_test_s, y_test, cw_dict)

    # Cell 12
    compare_models(rf_r, xgb_r, nn_r, tnn_r)

    # Save best model for app.py
    drop_set = {
        "Patient_ID", "Patient_Questionnaire_Response",
        "Doctor_Triage_Note", "combined_text", "Heart_Disease_Status",
    }
    feature_names = [c for c in df_final.columns if c not in drop_set]
    save_artifacts(best_model, scaler, le_map, feature_names)
