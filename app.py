"""
app.py - Heart Attack Prediction (scikit-learn only)
No TensorFlow. RAM ~80MB. Loads in <2s. Render free tier compatible.
"""
import os, pickle, time, json
import urllib.request, urllib.error
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

MODEL_DIR      = os.path.join(os.path.dirname(__file__), "model")
_ready         = False
_load_error    = None
_model         = None
_scaler        = None
_le_map        = None
_feature_names = None
_meta          = {}

CATEGORICAL_DEFAULTS = {
    "Gender":"Male","Exercise_Habits":"Medium","Smoking":"No",
    "Alcohol_Consumption":"Low","Sugar_Consumption":"Medium",
    "Stress_Level":"Medium","Family_Heart_Disease":"No",
    "High_Blood_Pressure":"No","Diabetes":"No","Low_HDL":"No","High_LDL":"No",
}
NUMERIC_DEFAULTS = {
    "Age":50,"BMI":25.0,"Blood_Pressure":120.0,"Cholesterol_Level":200.0,
    "Triglyceride_Level":150,"Fasting_Blood_Sugar":95,
    "CRP_Level":3.0,"Homocysteine_Level":10.0,"Sleep_Hours":7,
}

def load_models():
    global _ready,_load_error,_model,_scaler,_le_map,_feature_names,_meta
    t0 = time.time()
    try:
        print("[startup] Loading best_model.pkl ...", flush=True)
        with open(os.path.join(MODEL_DIR,"best_model.pkl"),     "rb") as f: _model         = pickle.load(f)
        print("[startup] Loading scaler.pkl ...", flush=True)
        with open(os.path.join(MODEL_DIR,"scaler.pkl"),         "rb") as f: _scaler        = pickle.load(f)
        print("[startup] Loading label_encoders.pkl ...", flush=True)
        with open(os.path.join(MODEL_DIR,"label_encoders.pkl"), "rb") as f: _le_map        = pickle.load(f)
        print("[startup] Loading feature_names.pkl ...", flush=True)
        with open(os.path.join(MODEL_DIR,"feature_names.pkl"),  "rb") as f: _feature_names = pickle.load(f)
        print("[startup] Loading meta.pkl ...", flush=True)
        with open(os.path.join(MODEL_DIR,"meta.pkl"),           "rb") as f: _meta          = pickle.load(f)
        print(f"[startup] All files loaded in {time.time()-t0:.1f}s, running warm-up ...", flush=True)
        dummy = np.zeros((1, len(_feature_names)), dtype=np.float32)
        _model.predict_proba(dummy)
        _ready = True
        print(f"[startup] ✅ Ready in {time.time()-t0:.1f}s — model={_meta.get('best_name')} AUC={_meta.get('best_auc')}", flush=True)
    except Exception:
        import traceback
        _load_error = traceback.format_exc()
        print(f"[startup] ❌ FAILED:\n{_load_error}", flush=True)

# Load synchronously — sklearn loads in <3s so port opens fast enough for Render
load_models()

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
# Phi-3-mini: fast, free, no cold-start issues on HF serverless
HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"

def _build_prompt(data: dict, label: str, pct: float) -> str:
    return (
        f"<|user|>\n"
        f"You are a compassionate cardiac health advisor. "
        f"A patient received a heart attack risk result: {label} ({pct}% probability).\n"
        f"Full clinical profile: "
        f"Age {data.get('Age',50)}, {data.get('Gender','Unknown')}, BMI {data.get('BMI',25)}, "
        f"BP {data.get('Blood_Pressure',120)} mmHg, Cholesterol {data.get('Cholesterol_Level',200)} mg/dL, "
        f"Triglycerides {data.get('Triglyceride_Level',150)} mg/dL, "
        f"Fasting Sugar {data.get('Fasting_Blood_Sugar',95)} mg/dL, "
        f"CRP {data.get('CRP_Level',3)} mg/L, Homocysteine {data.get('Homocysteine_Level',10)} umol/L, "
        f"Sleep {data.get('Sleep_Hours',7)} hrs/night, "
        f"High BP: {data.get('High_Blood_Pressure','No')}, Diabetes: {data.get('Diabetes','No')}, "
        f"Low HDL: {data.get('Low_HDL','No')}, High LDL: {data.get('High_LDL','No')}, "
        f"Family History: {data.get('Family_Heart_Disease','No')}, "
        f"Smoking: {data.get('Smoking','No')}, Alcohol: {data.get('Alcohol_Consumption','Low')}, "
        f"Exercise: {data.get('Exercise_Habits','Medium')}, Stress: {data.get('Stress_Level','Medium')}, "
        f"Sugar: {data.get('Sugar_Consumption','Medium')}.\n\n"
        f"Respond ONLY with a valid JSON object, no extra text, no markdown, exactly this structure:\n"
        f"{{\n"
        f"  \"explanation\": \"2 sentences explaining the top 2-3 specific factors from their data driving this risk level. Mention actual numbers.\",\n"
        f"  \"advice\": \"3-4 sentences of warm personalised action plan addressing their specific conditions. Mention exact values. No bullet points.\",\n"
        f"  \"followup\": \"One engaging follow-up question offering to help them go deeper, e.g. a 7-day plan or diet guide.\"\n"
        f"}}\n"
        f"Return only the JSON. No preamble.<|end|>\n<|assistant|>"
    )

def _call_hf(prompt: str, retries: int = 2) -> str | None:
    payload = json.dumps({
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 220,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False,
        }
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    for attempt in range(retries):
        req = urllib.request.Request(HF_API_URL, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if isinstance(result, list) and result:
                    text = result[0].get("generated_text", "").strip()
                    # Clean up any leftover prompt artifacts
                    for tag in ["<|assistant|>", "<|end|>", "<|user|>"]:
                        text = text.replace(tag, "")
                    return text.strip()
                if isinstance(result, dict):
                    err = result.get("error", "")
                    # Model loading — wait and retry
                    if "loading" in err.lower() or "currently loading" in err.lower():
                        print(f"[advice] Model loading, waiting 8s (attempt {attempt+1})...", flush=True)
                        time.sleep(8)
                        continue
                    print(f"[advice] HF error: {err}", flush=True)
                    return None
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            print(f"[advice] HTTP {e.code}: {body[:200]}", flush=True)
            if e.code == 503:
                time.sleep(8); continue
        except Exception as e:
            print(f"[advice] Exception: {e}", flush=True)
            return None
    return None

def _genai_advice(data: dict, label: str, pct: float) -> dict | None:
    prompt = _build_prompt(data, label, pct)
    raw = _call_hf(prompt)
    if not raw:
        return None
    # Parse JSON from response
    try:
        # Strip any markdown fences
        clean = raw.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"): clean = clean[4:]
        # Find JSON object
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(clean[start:end])
    except Exception as e:
        print(f"[advice] JSON parse error: {e}\nRaw: {raw[:300]}", flush=True)
    # If JSON parse fails, return raw text as advice
    return {"explanation": "", "advice": raw.strip(), "followup": "Would you like a personalised 7-day lifestyle improvement plan?"}

def _encode(values):
    encoded = {}
    for col, le in _le_map.items():
        val = values.get(col, CATEGORICAL_DEFAULTS.get(col, le.classes_[0]))
        if val not in le.classes_: val = le.classes_[0]
        encoded[col] = int(le.transform([val])[0])
    return encoded

def _predict(data):
    t0  = time.time()
    num = {k: float(data.get(k, v)) for k, v in NUMERIC_DEFAULTS.items()}
    cat = _encode(data)
    row = {**num, **cat}
    vec = np.array([row.get(f, 0.0) for f in _feature_names], dtype=np.float32)
    # Use scaler only for models that need it (LR), tree models use raw
    model_name = _meta.get("best_name","")
    if "Logistic" in model_name:
        vec = _scaler.transform(vec.reshape(1,-1))
    else:
        vec = vec.reshape(1,-1)
    prob = float(_model.predict_proba(vec)[0][1])
    ms   = round((time.time()-t0)*1000)
    return {"probability":round(prob,4),"pct":round(prob*100,1),
            "label":"High Risk" if prob>0.5 else "Low Risk",
            "model":model_name,"ms":ms}

@app.route("/")
def index(): return render_template_string(HTML)

@app.route("/health")
def health(): return jsonify({"ready":_ready,"error":_load_error})

@app.route("/debug")
def debug():
    return jsonify({
        "ready":_ready,"load_error":_load_error,"meta":_meta,
        "features":_feature_names or [],
        "files": os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else [],
    })

@app.route("/reload")
def reload_route():
    global _ready,_load_error
    if _ready: return jsonify({"status":"already ready"})
    _ready=False; _load_error=None
    load_models()
    return jsonify({"status":"reloaded","ready":_ready,"error":_load_error})

@app.route("/advice", methods=["POST"])
def advice():
    """Generate GenAI structured advice: explanation + personalised advice + follow-up."""
    data  = request.get_json(force=True) or {}
    label = data.pop("label", "Low Risk")
    pct   = float(data.pop("pct", 0))
    hi    = label == "High Risk"

    result = _genai_advice(data, label, pct)
    if result:
        return jsonify({
            "source":      "genai",
            "explanation": result.get("explanation", ""),
            "advice":      result.get("advice", ""),
            "followup":    result.get("followup", "Would you like a personalised 7-day lifestyle plan?"),
        })

    # Fallback — personalised by key values even without AI
    age    = int(float(data.get("Age", 50)))
    bmi    = float(data.get("BMI", 25))
    bp     = int(float(data.get("Blood_Pressure", 120)))
    chol   = int(float(data.get("Cholesterol_Level", 200)))
    smoke  = data.get("Smoking", "No") == "Yes"
    diabt  = data.get("Diabetes", "No") == "Yes"
    stress = data.get("Stress_Level", "Medium")
    sleep  = float(data.get("Sleep_Hours", 7))
    exer   = data.get("Exercise_Habits", "Medium")

    # Build explanation from actual values
    top_factors = []
    if smoke:                  top_factors.append("smoking")
    if bp >= 140:              top_factors.append(f"high blood pressure ({bp} mmHg)")
    if chol >= 240:            top_factors.append(f"high cholesterol ({chol} mg/dL)")
    if bmi >= 30:              top_factors.append(f"BMI of {bmi:.1f}")
    if diabt:                  top_factors.append("diabetes")
    if stress == "High":       top_factors.append("high stress levels")
    if sleep < 6:              top_factors.append(f"insufficient sleep ({sleep} hrs)")
    if exer == "Low":          top_factors.append("low physical activity")
    if data.get("Family_Heart_Disease") == "Yes": top_factors.append("family history")

    explanation = (
        f"Your {label.lower()} is primarily driven by {', '.join(top_factors[:3]) if top_factors else 'your overall health profile'}."
        if top_factors else
        f"Your overall health profile contributed to this {label.lower()} result."
    )

    if hi:
        advice = (
            f"With a BP of {bp} mmHg and cholesterol of {chol} mg/dL, "
            f"it is important to consult a cardiologist promptly. "
            + ("Quitting smoking would significantly reduce your risk. " if smoke else "")
            + (f"Your BMI of {bmi:.1f} suggests weight management through diet and exercise would help. " if bmi >= 28 else "")
            + (f"Increasing your sleep from {sleep} hrs to at least 7 hrs per night supports heart health. " if sleep < 7 else "")
            + "Small daily changes compound into major long-term improvements."
        )
    else:
        advice = (
            f"Your BP of {bp} mmHg and cholesterol of {chol} mg/dL are within a manageable range — keep monitoring them. "
            + (f"Increasing exercise from {exer.lower()} to at least moderate levels will further protect your heart. " if exer == "Low" else "")
            + (f"Managing your stress levels and improving sleep to 7-8 hrs will strengthen your cardiovascular health. " if stress == "High" or sleep < 7 else "")
            + "Continue your healthy habits and schedule an annual check-up to stay on track."
        )

    return jsonify({
        "source":      "fallback",
        "explanation": explanation,
        "advice":      advice,
        "followup":    "Would you like a personalised 7-day lifestyle improvement plan based on your profile?",
    })


@app.route("/plan", methods=["POST"])
def plan():
    """Generate a personalised 7-day lifestyle improvement plan."""
    data  = request.get_json(force=True) or {}
    label = data.pop("label", "Low Risk")
    pct   = float(data.pop("pct", 0))
    data.pop("request_type", None)

    prompt = (
        f"<|user|>\n"
        f"You are a cardiac health coach. A patient has {label} ({pct}% probability) of heart attack.\n"
        f"Profile: Age {data.get('Age',50)}, BMI {data.get('BMI',25)}, "
        f"BP {data.get('Blood_Pressure',120)} mmHg, Cholesterol {data.get('Cholesterol_Level',200)} mg/dL, "
        f"Sleep {data.get('Sleep_Hours',7)} hrs, Exercise: {data.get('Exercise_Habits','Medium')}, "
        f"Stress: {data.get('Stress_Level','Medium')}, Smoking: {data.get('Smoking','No')}, "
        f"Diabetes: {data.get('Diabetes','No')}, Diet sugar: {data.get('Sugar_Consumption','Medium')}.\n\n"
        f"Create a practical 7-day heart health improvement plan. "
        f"For each day write one specific action tailored to this patient's actual values. "
        f"Format exactly as:\nDay 1: ...\nDay 2: ...\nDay 3: ...\nDay 4: ...\nDay 5: ...\nDay 6: ...\nDay 7: ...\n"
        f"Be specific to their numbers. Keep each day to one sentence.<|end|>\n<|assistant|>"
    )
    raw = _call_hf(prompt)
    if raw:
        # Clean up
        for tag in ["<|assistant|>","<|end|>","<|user|>"]:
            raw = raw.replace(tag,"")
        return jsonify({"plan": raw.strip(), "source": "genai"})

    # Fallback plan based on their worst factors
    lines = []
    smoke = data.get("Smoking","No")=="Yes"
    exer  = data.get("Exercise_Habits","Medium")
    sleep = float(data.get("Sleep_Hours",7))
    stress= data.get("Stress_Level","Medium")
    bp    = int(float(data.get("Blood_Pressure",120)))
    chol  = int(float(data.get("Cholesterol_Level",200)))
    sugar = data.get("Sugar_Consumption","Medium")

    lines.append(f"Day 1: Track your blood pressure ({bp} mmHg) — measure morning and evening, log the readings.")
    lines.append(f"Day 2: {'Replace one cigarette with a 5-minute walk outside.' if smoke else 'Walk briskly for 20 minutes — even a short walk lowers BP.'}")
    lines.append(f"Day 3: Swap one high-cholesterol meal for oats, nuts, or salmon to address your {chol} mg/dL cholesterol.")
    lines.append(f"Day 4: {'Aim to be in bed by 10pm to reach 7 hrs sleep — your current '+str(sleep)+' hrs increases cardiac risk.' if sleep<7 else 'Add a 10-minute stretching routine before bed to maintain good sleep quality.'}")
    lines.append(f"Day 5: {'Practice 10 minutes of deep breathing or meditation to reduce your high stress levels.' if stress=='High' else 'Do 30 minutes of moderate cardio — cycling, swimming, or jogging.'}")
    lines.append(f"Day 6: {'Cut one sugary drink or snack — high sugar intake raises triglycerides and cardiac risk.' if sugar in ['Medium','High'] else 'Prepare a heart-healthy meal: grilled fish, leafy greens, and olive oil.'}")
    lines.append(f"Day 7: Review your week — note what improved, schedule a GP appointment to discuss your {'high' if label=='High Risk' else 'borderline'} risk result.")
    return jsonify({"plan": "\n".join(lines), "source": "fallback"})


@app.route("/predict", methods=["POST"])
def predict():
    if not _ready:
        return jsonify({"error":"loading","detail":_load_error}), 503
    try:
        data = request.get_json(force=True) or {}
        for k in NUMERIC_DEFAULTS:
            if k in data:
                try: data[k]=float(data[k])
                except: data[k]=NUMERIC_DEFAULTS[k]
        return jsonify(_predict(data))
    except Exception:
        import traceback; err=traceback.format_exc()
        print(f"[predict ERROR]\n{err}", flush=True)
        return jsonify({"error":err}), 500

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>HeartGuard AI</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#f7f3ee;--card:#fff;--ink:#1a1208;--muted:#7a6e62;--accent:#c0392b;--border:#e2dbd2;--gold:#b5831a;--radius:14px}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font-family:'DM Sans',sans-serif;min-height:100vh}
.hero{background:var(--ink);color:#fff;text-align:center;padding:48px 24px 40px;position:relative;overflow:hidden}
.hero::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse 80% 60% at 50% 120%,#c0392b44 0%,transparent 70%)}
.hero h1{font-family:'Playfair Display',serif;font-size:clamp(2rem,5vw,3rem);font-weight:900;position:relative}
.hero h1 span{color:#e74c3c}
.hero p{margin-top:8px;color:#b0a89e;font-size:.95rem;position:relative}
#warm-banner{display:none;background:#fff3cd;color:#856404;text-align:center;padding:10px;font-size:.88rem;border-bottom:1px solid #ffc107}
#err-banner{display:none;background:#fde8e8;color:#a93226;text-align:center;padding:10px;font-size:.88rem;border-bottom:1px solid #e74c3c}
.container{max-width:780px;margin:0 auto;padding:32px 20px 80px}
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:28px 32px;margin-bottom:18px;box-shadow:0 2px 10px #1a120806}
.section-label{font-family:'Playfair Display',serif;font-size:.72rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--gold);margin-bottom:18px;display:flex;align-items:center;gap:8px}
.section-label::after{content:'';flex:1;height:1px;background:var(--border)}
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px 24px}
.full{grid-column:1/-1}
@media(max-width:540px){.form-grid{grid-template-columns:1fr}}
.field label{display:block;font-size:.8rem;font-weight:600;color:var(--muted);margin-bottom:6px}
.field input,.field select{width:100%;background:#faf8f5;border:1.5px solid var(--border);border-radius:9px;padding:10px 13px;font-family:'DM Sans',sans-serif;font-size:.93rem;color:var(--ink);outline:none;transition:border-color .15s;appearance:none;-webkit-appearance:none}
.field input:focus,.field select:focus{border-color:var(--accent);box-shadow:0 0 0 3px #c0392b12}
.field input[readonly]{background:#f0ede8;color:var(--muted)}
.select-wrap{position:relative}.select-wrap::after{content:'▾';position:absolute;right:12px;top:50%;transform:translateY(-50%);color:var(--muted);pointer-events:none}
.slider-field label{font-size:.8rem;font-weight:600;color:var(--muted);display:flex;justify-content:space-between;margin-bottom:8px}
.slider-field label span{font-family:'Playfair Display',serif;font-size:.95rem;color:var(--accent);font-weight:700}
input[type=range]{width:100%;-webkit-appearance:none;height:5px;background:linear-gradient(to right,var(--accent) var(--pct,50%),var(--border) var(--pct,50%));border-radius:4px;outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:17px;height:17px;background:var(--accent);border-radius:50%;border:2px solid #fff;box-shadow:0 1px 4px #0002}
.pill-group{display:flex;gap:7px;flex-wrap:wrap;margin-top:4px}
.pill{padding:7px 16px;border-radius:30px;font-size:.83rem;font-weight:600;border:1.5px solid var(--border);background:#faf8f5;color:var(--muted);cursor:pointer;transition:all .13s;user-select:none}
.pill.active{background:var(--accent);border-color:var(--accent);color:#fff}
.pill-label{font-size:.8rem;font-weight:600;color:var(--muted);margin-bottom:8px}
.predict-btn{display:block;width:100%;background:var(--accent);color:#fff;border:none;border-radius:12px;padding:16px;font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;cursor:pointer;transition:transform .12s,opacity .12s;box-shadow:0 4px 18px #c0392b30;margin-top:6px}
.predict-btn:hover{opacity:.88;transform:translateY(-1px)}
.predict-btn:disabled{background:#bbb;box-shadow:none;cursor:not-allowed;transform:none;opacity:1}
#loading{display:none;text-align:center;padding:18px}
.spinner{width:34px;height:34px;border:3px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .7s linear infinite;margin:0 auto 8px}
@keyframes spin{to{transform:rotate(360deg)}}
#result{display:none;margin-top:20px}
.result-card{border-radius:var(--radius);padding:32px;text-align:center;border:1.5px solid;animation:pop .3s cubic-bezier(.34,1.56,.64,1)}
@keyframes pop{from{opacity:0;transform:scale(.94)}to{opacity:1;transform:scale(1)}}
.result-card.high{background:#fff5f5;border-color:#e74c3c}
.result-card.low{background:#f0faf3;border-color:#27ae60}
.result-icon{font-size:2.6rem;margin-bottom:8px}
.result-label{font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:900;margin-bottom:4px}
.result-card.high .result-label{color:#c0392b}
.result-card.low  .result-label{color:#27ae60}
.result-prob{font-size:.95rem;color:var(--muted);margin-bottom:16px}
.result-bar-wrap{background:#e8e0d8;border-radius:30px;height:9px;overflow:hidden;margin:0 auto 18px;max-width:320px}
.result-bar{height:100%;border-radius:30px;transition:width .8s cubic-bezier(.34,1.2,.64,1)}
.result-card.high .result-bar{background:linear-gradient(90deg,#e74c3c,#c0392b)}
.result-card.low  .result-bar{background:linear-gradient(90deg,#27ae60,#1e8449)}
.result-advice{font-size:.88rem;line-height:1.65;color:var(--ink);background:#fff;border-radius:9px;padding:14px 18px;text-align:left;margin-top:4px;border:1px solid var(--border)}
.timing{display:inline-block;margin-top:10px;background:#f0ede8;border-radius:20px;padding:2px 11px;font-size:.74rem;color:var(--muted)}
.disclaimer{font-size:.76rem;color:var(--muted);margin-top:12px;font-style:italic}
</style>
</head>
<body>
<div class="hero">
  <h1>Heart<span>Guard</span> AI 🫀</h1>
  <p>Clinical heart attack risk assessment · Machine Learning</p>
</div>
<div id="warm-banner">⏳ <span id="warm-txt">AI model loading...</span> &nbsp;Predict button will unlock automatically.</div>
<div id="err-banner">❌ <span id="err-txt"></span> &nbsp;<a href="/reload" style="color:inherit;font-weight:600">Retry →</a></div>
<div class="container">
  <div class="card">
    <div class="section-label">01 — Basic Information</div>
    <div class="form-grid">
      <div class="field"><label>Age (years)</label><input type="number" id="Age" min="18" max="100" placeholder="e.g. 45"/></div>
      <div class="field"><label>Gender</label><div class="select-wrap"><select id="Gender"><option value="">Select</option><option>Male</option><option>Female</option></select></div></div>
      <div class="field"><label>Height (cm)</label><input type="number" id="height_cm" min="100" max="220" placeholder="e.g. 170" oninput="calcBMI()"/></div>
      <div class="field"><label>Weight (kg)</label><input type="number" id="weight_kg" min="30" max="250" placeholder="e.g. 70" oninput="calcBMI()"/></div>
      <div class="field"><label>BMI (auto-calculated)</label><input type="text" id="BMI" readonly placeholder="Fill height &amp; weight"/></div>
    </div>
  </div>
  <div class="card">
    <div class="section-label">02 — Clinical Vitals</div>
    <div class="form-grid">
      <div class="field"><label>Blood Pressure (mmHg)</label><input type="number" id="Blood_Pressure" min="60" max="250" placeholder="e.g. 120"/></div>
      <div class="field"><label>Cholesterol (mg/dL)</label><input type="number" id="Cholesterol_Level" min="50" max="700" placeholder="e.g. 200"/></div>
      <div class="field"><label>Triglycerides (mg/dL)</label><input type="number" id="Triglyceride_Level" min="20" max="1000" placeholder="e.g. 150"/></div>
      <div class="field"><label>Fasting Blood Sugar (mg/dL)</label><input type="number" id="Fasting_Blood_Sugar" min="50" max="400" placeholder="e.g. 95"/></div>
      <div class="field"><label>CRP Level (mg/L)</label><input type="number" id="CRP_Level" min="0" max="50" step="0.1" placeholder="e.g. 3.0"/></div>
      <div class="field"><label>Homocysteine (µmol/L)</label><input type="number" id="Homocysteine_Level" min="0" max="50" step="0.1" placeholder="e.g. 10.0"/></div>
    </div>
  </div>
  <div class="card">
    <div class="section-label">03 — Medical Conditions</div>
    <div class="form-grid">
      <div class="field full"><div class="pill-label">High Blood Pressure?</div><div class="pill-group" data-f="High_Blood_Pressure"><div class="pill active" data-v="No">No</div><div class="pill" data-v="Yes">Yes</div></div></div>
      <div class="field full"><div class="pill-label">Diabetes?</div><div class="pill-group" data-f="Diabetes"><div class="pill active" data-v="No">No</div><div class="pill" data-v="Yes">Yes</div></div></div>
      <div class="field full"><div class="pill-label">Low HDL Cholesterol?</div><div class="pill-group" data-f="Low_HDL"><div class="pill active" data-v="No">No</div><div class="pill" data-v="Yes">Yes</div></div></div>
      <div class="field full"><div class="pill-label">High LDL Cholesterol?</div><div class="pill-group" data-f="High_LDL"><div class="pill active" data-v="No">No</div><div class="pill" data-v="Yes">Yes</div></div></div>
      <div class="field full"><div class="pill-label">Family History of Heart Disease?</div><div class="pill-group" data-f="Family_Heart_Disease"><div class="pill active" data-v="No">No</div><div class="pill" data-v="Yes">Yes</div></div></div>
    </div>
  </div>
  <div class="card">
    <div class="section-label">04 — Lifestyle</div>
    <div class="slider-field" style="margin-bottom:20px">
      <label>Sleep Hours / Night <span id="sleep_val">7</span> hrs</label>
      <input type="range" id="Sleep_Hours" min="3" max="12" step="0.5" value="7" oninput="document.getElementById('sleep_val').textContent=this.value;updateSlider(this)"/>
    </div>
    <div class="form-grid">
      <div class="field"><label>Exercise Habits</label><div class="select-wrap"><select id="Exercise_Habits"><option value="">Select</option><option>Low</option><option selected>Medium</option><option>High</option></select></div></div>
      <div class="field"><label>Stress Level</label><div class="select-wrap"><select id="Stress_Level"><option value="">Select</option><option>Low</option><option selected>Medium</option><option>High</option></select></div></div>
      <div class="field"><label>Alcohol Consumption</label><div class="select-wrap"><select id="Alcohol_Consumption"><option value="">Select</option><option selected>Low</option><option>Medium</option><option>High</option></select></div></div>
      <div class="field"><label>Sugar Consumption</label><div class="select-wrap"><select id="Sugar_Consumption"><option value="">Select</option><option>Low</option><option selected>Medium</option><option>High</option></select></div></div>
    </div>
    <div style="margin-top:20px">
      <div class="pill-label">Smoking?</div>
      <div class="pill-group" data-f="Smoking"><div class="pill active" data-v="No">No</div><div class="pill" data-v="Yes">Yes</div></div>
    </div>
  </div>
  <button class="predict-btn" id="predictBtn" onclick="submitForm()">Predict My Heart Attack Risk</button>
  <div id="loading"><div class="spinner"></div><p style="color:var(--muted);font-size:.88rem">Analysing...</p></div>
  <div id="result">
    <div id="resultCard" class="result-card">
      <div id="rIcon"   class="result-icon"></div>
      <div id="rLabel"  class="result-label"></div>
      <div id="rProb"   class="result-prob"></div>
      <div class="result-bar-wrap"><div id="rBar" class="result-bar" style="width:0%"></div></div>
      <!-- Explanation -->
      <div id="rExplain" style="display:none;background:#f8f4f0;border-radius:9px;padding:13px 16px;margin-top:14px;text-align:left;border:1px solid var(--border)">
        <div style="font-size:.72rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--gold);margin-bottom:6px">📊 Why this result</div>
        <div id="rExplainText" style="font-size:.88rem;line-height:1.65;color:var(--ink)"></div>
      </div>
      <!-- Advice -->
      <div id="rAdvice" class="result-advice" style="margin-top:12px"></div>
      <!-- Follow-up -->
      <div id="rFollowup" style="display:none;margin-top:14px;background:linear-gradient(135deg,#1a1208,#2d1f0e);border-radius:10px;padding:14px 18px;text-align:left">
        <div style="font-size:.72rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#e8c97a;margin-bottom:6px">💬 Want to go further?</div>
        <div id="rFollowupText" style="font-size:.88rem;color:#e8ddd0;line-height:1.6"></div>
        <button onclick="generatePlan()" style="margin-top:10px;background:#c0392b;color:#fff;border:none;border-radius:8px;padding:8px 18px;font-size:.83rem;font-weight:600;cursor:pointer">Generate 7-Day Plan</button>
      </div>
      <!-- 7-day plan output -->
      <div id="rPlan" style="display:none;margin-top:14px;background:#fff;border:1px solid var(--border);border-radius:10px;padding:16px 18px;text-align:left">
        <div style="font-size:.72rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#0d7a5f;margin-bottom:8px">📅 Your 7-Day Lifestyle Plan</div>
        <div id="rPlanText" style="font-size:.87rem;line-height:1.75;color:var(--ink);white-space:pre-wrap"></div>
      </div>
      <div id="rMs" class="timing" style="margin-top:12px"></div>
      <!-- Download report -->
      <button id="downloadBtn" onclick="downloadReport()" style="display:none;margin-top:12px;width:100%;background:#1a1208;color:#fff;border:none;border-radius:9px;padding:11px;font-size:.88rem;font-weight:600;cursor:pointer;letter-spacing:.03em">⬇️ Download Risk Summary Report</button>
      <div class="disclaimer">⚕️ For informational purposes only. Not a substitute for professional medical advice.</div>
    </div>
  </div>
</div>
<script>
function calcBMI(){
  const h=parseFloat(document.getElementById('height_cm').value);
  const w=parseFloat(document.getElementById('weight_kg').value);
  document.getElementById('BMI').value=(h>0&&w>0)?(w/((h/100)**2)).toFixed(1):'';
}
function updateSlider(el){
  const pct=((parseFloat(el.value)-parseFloat(el.min))/(parseFloat(el.max)-parseFloat(el.min))*100).toFixed(1)+'%';
  el.style.setProperty('--pct',pct);
}
document.querySelectorAll('input[type=range]').forEach(updateSlider);
document.querySelectorAll('.pill-group').forEach(g=>{
  g.querySelectorAll('.pill').forEach(p=>{
    p.addEventListener('click',()=>{
      g.querySelectorAll('.pill').forEach(x=>x.classList.remove('active'));
      p.classList.add('active');
    });
  });
});
const gp=f=>document.querySelector(`.pill-group[data-f="${f}"]`)?.querySelector('.pill.active')?.dataset.v||null;

(async()=>{
  const btn=document.getElementById('predictBtn');
  const wb=document.getElementById('warm-banner');
  const eb=document.getElementById('err-banner');
  const poll=async()=>{
    try{ return await fetch('/health').then(r=>r.json()); }
    catch(e){ return {}; }
  };
  let s=await poll();
  if(!s.ready){
    wb.style.display='block'; btn.disabled=true; btn.textContent='Loading model...';
    let secs=0;
    const iv=setInterval(async()=>{
      secs+=2;
      document.getElementById('warm-txt').textContent='Model loading... '+secs+'s';
      s=await poll();
      if(s.ready){
        clearInterval(iv); wb.style.display='none';
        btn.disabled=false; btn.textContent='Predict My Heart Attack Risk';
      } else if(s.error){
        clearInterval(iv); wb.style.display='none';
        document.getElementById('err-txt').textContent='Load failed — check /debug for details';
        eb.style.display='block';
        btn.disabled=false; btn.textContent='Predict My Heart Attack Risk';
      }
    },2000);
  }
})();

async function generatePlan(){
  const r=window._lastResult;
  if(!r) return;
  const planEl=document.getElementById('rPlan');
  const planText=document.getElementById('rPlanText');
  planEl.style.display='block';
  planText.textContent='Generating your personalised 7-day plan...';
  try{
    const payload={...r.data, label:r.label, pct:r.pct, request_type:'7day_plan'};
    const res=await fetch('/plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const d=await res.json();
    planText.textContent=d.plan||'Could not generate plan. Please try again.';
    if(window._lastResult) window._lastResult.plan=d.plan;
  }catch(e){
    planText.textContent='Could not generate plan right now. Please try again shortly.';
  }
}

function downloadReport(){
  const r=window._lastResult;
  if(!r) return;
  const d=r.data;
  const a=r.advice||{};
  const now=new Date().toLocaleDateString('en-MY',{year:'numeric',month:'long',day:'numeric'});
  const lines=[
    '╔══════════════════════════════════════════════════════╗',
    '║          HEARTGUARD AI — RISK SUMMARY REPORT         ║',
    '╚══════════════════════════════════════════════════════╝',
    '',
    'Generated: '+now,
    'Prediction Time: '+r.ms+'ms',
    '',
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    '  RISK RESULT',
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    '  Prediction  : '+r.label,
    '  Probability : '+r.pct+'%',
    '',
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    '  PATIENT PROFILE',
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    '  Age              : '+d.Age+' years',
    '  Gender           : '+d.Gender,
    '  BMI              : '+d.BMI,
    '  Blood Pressure   : '+d.Blood_Pressure+' mmHg',
    '  Cholesterol      : '+d.Cholesterol_Level+' mg/dL',
    '  Triglycerides    : '+d.Triglyceride_Level+' mg/dL',
    '  Fasting Sugar    : '+d.Fasting_Blood_Sugar+' mg/dL',
    '  CRP Level        : '+d.CRP_Level+' mg/L',
    '  Homocysteine     : '+d.Homocysteine_Level+' umol/L',
    '  Sleep            : '+d.Sleep_Hours+' hrs/night',
    '  High BP          : '+d.High_Blood_Pressure,
    '  Diabetes         : '+d.Diabetes,
    '  Low HDL          : '+d.Low_HDL,
    '  High LDL         : '+d.High_LDL,
    '  Family History   : '+d.Family_Heart_Disease,
    '  Smoking          : '+d.Smoking,
    '  Alcohol          : '+d.Alcohol_Consumption,
    '  Exercise         : '+d.Exercise_Habits,
    '  Stress Level     : '+d.Stress_Level,
    '  Sugar Intake     : '+d.Sugar_Consumption,
    '',
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    '  WHY THIS RESULT',
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    '  '+(a.explanation||'Based on your overall clinical profile.'),
    '',
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    '  PERSONALISED ADVICE',
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    '  '+(a.advice||'').replace(/\n/g,'\n  '),
  ];
  if(r.plan){
    lines.push('','━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    lines.push('  7-DAY LIFESTYLE PLAN');
    lines.push('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    lines.push('  '+r.plan.replace(/\n/g,'\n  '));
  }
  lines.push('','━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  lines.push('  ⚕ For informational purposes only.');
  lines.push('    Not a substitute for professional medical advice.');
  lines.push('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

  const blob=new Blob([lines.join('\n')],{type:'text/plain'});
  const url=URL.createObjectURL(blob);
  const el=document.createElement('a');
  el.href=url; el.download='HeartGuard_Report_'+now.replace(/ /g,'_')+'.txt';
  el.click(); URL.revokeObjectURL(url);
}

async function submitForm(){
  const btn=document.getElementById('predictBtn');
  const data={
    Age:parseFloat(document.getElementById('Age').value)||50,
    Gender:document.getElementById('Gender').value||'Male',
    BMI:parseFloat(document.getElementById('BMI').value)||25,
    Blood_Pressure:parseFloat(document.getElementById('Blood_Pressure').value)||120,
    Cholesterol_Level:parseFloat(document.getElementById('Cholesterol_Level').value)||200,
    Triglyceride_Level:parseFloat(document.getElementById('Triglyceride_Level').value)||150,
    Fasting_Blood_Sugar:parseFloat(document.getElementById('Fasting_Blood_Sugar').value)||95,
    CRP_Level:parseFloat(document.getElementById('CRP_Level').value)||3,
    Homocysteine_Level:parseFloat(document.getElementById('Homocysteine_Level').value)||10,
    Sleep_Hours:parseFloat(document.getElementById('Sleep_Hours').value)||7,
    High_Blood_Pressure:gp('High_Blood_Pressure')||'No',
    Diabetes:gp('Diabetes')||'No',
    Low_HDL:gp('Low_HDL')||'No',
    High_LDL:gp('High_LDL')||'No',
    Family_Heart_Disease:gp('Family_Heart_Disease')||'No',
    Exercise_Habits:document.getElementById('Exercise_Habits').value||'Medium',
    Stress_Level:document.getElementById('Stress_Level').value||'Medium',
    Alcohol_Consumption:document.getElementById('Alcohol_Consumption').value||'Low',
    Sugar_Consumption:document.getElementById('Sugar_Consumption').value||'Medium',
    Smoking:gp('Smoking')||'No',
  };
  btn.disabled=true;
  document.getElementById('loading').style.display='block';
  document.getElementById('result').style.display='none';
  try{
    const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const d=await r.json().catch(()=>null);
    document.getElementById('loading').style.display='none'; btn.disabled=false;
    if(!d||r.status===503){alert('Model still loading, please wait a moment.');return;}
    if(d.error){alert('Error:\n'+String(d.error).split('\n').slice(-3).join('\n'));return;}
    const hi=d.label==='High Risk';
    document.getElementById('resultCard').className='result-card '+(hi?'high':'low');
    document.getElementById('rIcon').textContent=hi?'⚠️':'✅';
    document.getElementById('rLabel').textContent=d.label;
    document.getElementById('rProb').textContent='Risk Probability: '+d.pct+'%';
    document.getElementById('rBar').style.width=d.pct+'%';
    document.getElementById('rMs').textContent='⚡ '+d.ms+'ms';
    // Show result card immediately with loading advice
    document.getElementById('rAdvice').innerHTML='<span style="color:var(--muted);font-style:italic" id="adviceLoading">🤖 Generating personalised advice<span id="adviceDots">.</span></span>';
    // Animate dots while waiting
    let dotCount=1;
    const dotIv=setInterval(()=>{
      dotCount=(dotCount%3)+1;
      const el=document.getElementById('adviceDots');
      if(el) el.textContent='.'.repeat(dotCount);
    },600);
    document.getElementById('result').style.display='block';
    document.getElementById('result').scrollIntoView({behavior:'smooth',block:'start'});
    // Fetch GenAI advice in background
    const advicePayload = {...data, label: d.label, pct: d.pct};
    fetch('/advice',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(advicePayload)})
      .then(r=>r.json())
      .then(a=>{
        clearInterval(dotIv);
        const isGenAI = a.source==='genai';
        const src = isGenAI
          ? '<span style="color:#0d7a5f">🤖 Personalised AI advice</span>'
          : '<span style="color:var(--muted)">📋 Evidence-based advice</span>';
        // Explanation
        if(a.explanation){
          document.getElementById('rExplainText').textContent=a.explanation;
          document.getElementById('rExplain').style.display='block';
        }
        // Advice
        document.getElementById('rAdvice').innerHTML=
          '<div style="font-size:.72rem;font-weight:600;letter-spacing:.04em;margin-bottom:8px">'+src+'</div>'+
          '<div style="line-height:1.7">'+a.advice.replace(/\n/g,'<br/>')+'</div>';
        // Follow-up
        if(a.followup){
          document.getElementById('rFollowupText').textContent=a.followup;
          document.getElementById('rFollowup').style.display='block';
        }
        document.getElementById('downloadBtn').style.display='block';
        // Store for report & plan generation
        window._lastResult={label:d.label,pct:d.pct,ms:d.ms,data,advice:a};
      })
      .catch(()=>{
        clearInterval(dotIv);
        document.getElementById('rAdvice').textContent=hi
          ?'Please consult a cardiologist promptly and monitor your blood pressure and cholesterol regularly.'
          :'Keep up your healthy lifestyle with regular exercise, balanced diet, and annual check-ups.';
        document.getElementById('downloadBtn').style.display='block';
        window._lastResult={label:d.label,pct:d.pct,ms:d.ms,data,advice:null};
      });
  }catch(e){
    document.getElementById('loading').style.display='none'; btn.disabled=false;
    alert('Request failed: '+e);
  }
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
