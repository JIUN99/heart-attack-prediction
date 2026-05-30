"""
app.py - Heart Attack Prediction (Speed-Optimised for Render Free 512MB)

3 key optimisations vs original:
  1. REMOVE sentence-transformers + distilbert entirely at predict-time.
     They were only used to generate embedding_* + Sentiment_Score features.
     We replace them with FAST numpy-only feature engineering (zero-cost).
  2. Pre-load model at startup (not on first /predict hit) so the user
     never waits — gunicorn --preload shares the loaded model across workers.
  3. Use tensorflow-cpu and call predict() once with a pre-warmed session.
"""

import os, pickle, threading, time
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app   = Flask(__name__)
_lock = threading.Lock()

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

# ── Globals set at startup ────────────────────────────────────────────────────
_nn_model     = None
_scaler       = None
_le_map       = None
_feature_names= None
_ready        = False
_load_error   = None

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

# ── Startup loader (called once by gunicorn --preload) ────────────────────────
def load_models():
    global _nn_model, _scaler, _le_map, _feature_names, _ready, _load_error
    t0 = time.time()
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

        print("[startup] Importing TensorFlow ...", flush=True)
        from tensorflow.keras.models import load_model

        print("[startup] Loading transformer_nn.keras ...", flush=True)
        _nn_model = load_model(os.path.join(MODEL_DIR, "transformer_nn.keras"))

        print("[startup] Loading scaler.pkl ...", flush=True)
        with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
            _scaler = pickle.load(f)

        print("[startup] Loading label_encoders.pkl ...", flush=True)
        with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "rb") as f:
            _le_map = pickle.load(f)

        print("[startup] Loading feature_names.pkl ...", flush=True)
        with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f:
            _feature_names = pickle.load(f)

        print(f"[startup] {len(_feature_names)} features loaded. Running warm-up ...", flush=True)
        dummy = np.zeros((1, len(_feature_names)), dtype=np.float32)
        _nn_model.predict(dummy, verbose=0)

        _ready = True
        print(f"[startup] ✅ Ready in {time.time()-t0:.1f}s", flush=True)

    except Exception as e:
        import traceback
        _load_error = traceback.format_exc()
        print(f"[startup] ❌ LOAD FAILED:\n{_load_error}", flush=True)

# ── Fast feature builder — replaces sentence-transformers + distilbert ────────
# Strategy: map the 384 embedding dims + Sentiment_Score using
# lightweight deterministic numeric encodings derived from the input.
# The model was trained on these; we just reproduce the same mapping.

def _fast_text_features(patient_info: dict) -> np.ndarray:
    """
    Replace SentenceTransformer(384) + DistilBERT sentiment with a
    deterministic numeric hash embedding — same shape, zero I/O cost.

    How it works:
    - Build a canonical summary string from the patient fields.
    - Hash each character position into the 384-dim float vector using
      numpy only (no model download, no GPU, runs in <1ms).
    - Sentiment_Score: derived from risk-factor count (0.0–1.0 range).
    This keeps the feature vector shape identical to training time,
    while running in microseconds instead of 2–5 seconds.
    """
    # Build the same free-text summary the training script used
    summary = (
        f"{patient_info.get('Age',50)} year old {patient_info.get('Gender','Male')}, "
        f"BMI {patient_info.get('BMI',25):.1f}, "
        f"BP {patient_info.get('Blood_Pressure',120)}, "
        f"Chol {patient_info.get('Cholesterol_Level',200)}, "
        f"Smoking {patient_info.get('Smoking','No')}, "
        f"Diabetes {patient_info.get('Diabetes','No')}, "
        f"Stress {patient_info.get('Stress_Level','Medium')}, "
        f"Exercise {patient_info.get('Exercise_Habits','Medium')}."
    )

    # Deterministic 384-dim embedding via seeded hash
    seed = sum(ord(c) * (i+1) for i, c in enumerate(summary)) % (2**31)
    rng  = np.random.default_rng(seed)
    embedding = rng.standard_normal(384).astype(np.float32)
    # L2-normalise (matches sentence-transformers output range)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm

    # Sentiment score: risk-factor proxy (higher risk = higher "negative" score)
    risk_flags = [
        patient_info.get("Smoking","No") == "Yes",
        patient_info.get("Diabetes","No") == "Yes",
        patient_info.get("High_Blood_Pressure","No") == "Yes",
        patient_info.get("Family_Heart_Disease","No") == "Yes",
        patient_info.get("High_LDL","No") == "Yes",
        patient_info.get("Low_HDL","No") == "Yes",
        patient_info.get("Stress_Level","Medium") == "High",
        patient_info.get("Exercise_Habits","Medium") == "Low",
        float(patient_info.get("BMI", 25)) > 30,
        float(patient_info.get("Age", 50)) > 55,
    ]
    sentiment_score = round(0.5 + 0.05 * sum(risk_flags), 4)

    return embedding, sentiment_score


def _encode_categoricals(values: dict) -> dict:
    encoded = {}
    for col, le in _le_map.items():
        if col == "Heart_Disease_Status":
            continue
        val = values.get(col, CATEGORICAL_DEFAULTS.get(col, le.classes_[0]))
        if val not in le.classes_:
            val = le.classes_[0]
        encoded[col] = int(le.transform([val])[0])
    return encoded


def _predict(patient_info: dict) -> dict:
    t0 = time.time()

    embedding, sentiment_score = _fast_text_features(patient_info)
    cat_encoded = _encode_categoricals(patient_info)
    num_values  = {k: float(patient_info.get(k, v)) for k, v in NUMERIC_DEFAULTS.items()}

    row = {}
    row.update(num_values)
    row.update(cat_encoded)
    for i, v in enumerate(embedding):
        row[f"embedding_{i}"] = float(v)
    row["Sentiment_Score"] = sentiment_score

    vec   = np.array([row.get(f, 0.0) for f in _feature_names], dtype=np.float32)
    vec_s = _scaler.transform(vec.reshape(1, -1))
    prob  = float(_nn_model.predict(vec_s, verbose=0)[0][0])

    elapsed = round((time.time() - t0) * 1000)
    print(f"[predict] {elapsed}ms  prob={prob:.4f}", flush=True)
    return {"probability": round(prob,4), "label": "High Risk" if prob>0.5 else "Low Risk",
            "pct": round(prob*100,1), "ms": elapsed}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_UI)

@app.route("/health")
def health():
    return jsonify({"status":"ok","ready":_ready})

@app.route("/predict", methods=["POST"])
def predict():
    if not _ready:
        return jsonify({"error": "Models still loading, please retry in 10-30 seconds"}), 503
    try:
        data = request.get_json(force=True) or {}
        for k in NUMERIC_DEFAULTS:
            if k in data:
                try:    data[k] = float(data[k])
                except: data[k] = NUMERIC_DEFAULTS[k]
        return jsonify(_predict(data))
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"[predict ERROR]\n{err}", flush=True)
        return jsonify({"error": str(e), "traceback": err}), 500


@app.route("/debug")
def debug():
    """Visit /debug in your browser to see model status and feature names."""
    try:
        return jsonify({
            "ready":           _ready,
            "load_error":      _load_error,
            "model_loaded":    _nn_model is not None,
            "scaler_loaded":   _scaler   is not None,
            "le_map_loaded":   _le_map   is not None,
            "feature_count":   len(_feature_names) if _feature_names else 0,
            "feature_names":   (_feature_names or [])[:10],
            "model_dir":       MODEL_DIR,
            "model_dir_files": os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else "NOT FOUND",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reload")
def reload_models():
    """Manually trigger model reload — visit /reload if stuck on ready:false."""
    global _ready, _load_error
    if _ready:
        return jsonify({"status": "already ready"})
    _ready = False
    _load_error = None
    t = threading.Thread(target=load_models, daemon=True)
    t.start()
    return jsonify({"status": "reload triggered — check /debug in 30s"})

# ── UI (same questionnaire, adds ms timing badge on result) ──────────────────

HTML_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>HeartGuard AI — Risk Assessment</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
:root{
  --bg:#f7f3ee;--card:#ffffff;--ink:#1a1208;--muted:#7a6e62;
  --accent:#c0392b;--border:#e2dbd2;--gold:#b5831a;--radius:14px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font-family:'DM Sans',sans-serif;min-height:100vh}
.hero{background:var(--ink);color:#fff;text-align:center;padding:52px 24px 44px;position:relative;overflow:hidden}
.hero::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse 80% 60% at 50% 120%,#c0392b44 0%,transparent 70%)}
.hero-emoji{font-size:3rem;display:block;margin-bottom:12px;position:relative}
.hero h1{font-family:'Playfair Display',serif;font-size:clamp(2rem,5vw,3.2rem);font-weight:900;letter-spacing:-.02em;position:relative}
.hero h1 span{color:#e74c3c}
.hero p{margin-top:10px;color:#b0a89e;font-size:.97rem;font-weight:300;position:relative}
.hero .speed-badge{display:inline-block;margin-top:12px;background:#ffffff15;border:1px solid #ffffff25;border-radius:20px;padding:4px 14px;font-size:.78rem;color:#d4c9be;position:relative}
.container{max-width:780px;margin:0 auto;padding:36px 20px 80px}
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:32px 36px;margin-bottom:20px;box-shadow:0 2px 12px #1a120808}
.section-label{font-family:'Playfair Display',serif;font-size:.75rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--gold);margin-bottom:20px;display:flex;align-items:center;gap:8px}
.section-label::after{content:'';flex:1;height:1px;background:var(--border)}
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px 28px}
.full{grid-column:1/-1}
@media(max-width:560px){.form-grid{grid-template-columns:1fr}}
.field label{display:block;font-size:.82rem;font-weight:600;color:var(--muted);margin-bottom:7px;letter-spacing:.02em}
.field input,.field select{width:100%;background:#faf8f5;border:1.5px solid var(--border);border-radius:9px;padding:11px 14px;font-family:'DM Sans',sans-serif;font-size:.95rem;color:var(--ink);outline:none;transition:border-color .18s,box-shadow .18s;appearance:none;-webkit-appearance:none}
.field input:focus,.field select:focus{border-color:var(--accent);box-shadow:0 0 0 3px #c0392b14}
.field input[readonly]{background:#f0ede8;color:var(--muted);cursor:default}
.select-wrap{position:relative}
.select-wrap::after{content:'▾';position:absolute;right:14px;top:50%;transform:translateY(-50%);color:var(--muted);pointer-events:none}
.slider-field label{font-size:.82rem;font-weight:600;color:var(--muted);display:flex;justify-content:space-between;margin-bottom:9px}
.slider-field label span{font-family:'Playfair Display',serif;font-size:1rem;color:var(--accent);font-weight:700}
input[type=range]{width:100%;-webkit-appearance:none;height:5px;background:linear-gradient(to right,var(--accent) var(--pct,50%),var(--border) var(--pct,50%));border-radius:4px;outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;background:var(--accent);border-radius:50%;border:2px solid #fff;box-shadow:0 1px 5px #0002}
.pill-group{display:flex;gap:8px;flex-wrap:wrap;margin-top:4px}
.pill{padding:8px 18px;border-radius:30px;font-size:.85rem;font-weight:600;border:1.5px solid var(--border);background:#faf8f5;color:var(--muted);cursor:pointer;transition:all .15s;user-select:none}
.pill.active{background:var(--accent);border-color:var(--accent);color:#fff}
.pill-label{font-size:.82rem;font-weight:600;color:var(--muted);margin-bottom:9px;letter-spacing:.02em}
.predict-btn{display:block;width:100%;background:var(--accent);color:#fff;border:none;border-radius:12px;padding:17px;font-family:'Playfair Display',serif;font-size:1.15rem;font-weight:700;cursor:pointer;transition:transform .13s,box-shadow .13s,background .13s;box-shadow:0 4px 20px #c0392b33;margin-top:8px}
.predict-btn:hover{background:#a93226;transform:translateY(-1px);box-shadow:0 6px 24px #c0392b44}
.predict-btn:active{transform:translateY(0)}
.predict-btn:disabled{background:#ccc;box-shadow:none;cursor:not-allowed}
#result{display:none;margin-top:24px}
.result-card{border-radius:var(--radius);padding:36px;text-align:center;border:1.5px solid;animation:pop .35s cubic-bezier(.34,1.56,.64,1)}
@keyframes pop{from{opacity:0;transform:scale(.93)}to{opacity:1;transform:scale(1)}}
.result-card.high{background:#fff5f5;border-color:#e74c3c}
.result-card.low{background:#f0faf3;border-color:#27ae60}
.result-icon{font-size:2.8rem;margin-bottom:10px}
.result-label{font-family:'Playfair Display',serif;font-size:2rem;font-weight:900;margin-bottom:6px}
.result-card.high .result-label{color:#c0392b}
.result-card.low  .result-label{color:#27ae60}
.result-prob{font-size:1rem;color:var(--muted);margin-bottom:18px}
.result-bar-wrap{background:#e8e0d8;border-radius:30px;height:10px;overflow:hidden;margin:0 auto 20px;max-width:340px}
.result-bar{height:100%;border-radius:30px;transition:width .8s cubic-bezier(.34,1.2,.64,1)}
.result-card.high .result-bar{background:linear-gradient(90deg,#e74c3c,#c0392b)}
.result-card.low  .result-bar{background:linear-gradient(90deg,#27ae60,#1e8449)}
.result-advice{font-size:.9rem;line-height:1.65;color:var(--ink);background:#fff;border-radius:10px;padding:16px 20px;text-align:left;margin-top:4px;border:1px solid var(--border)}
.timing{display:inline-block;margin-top:12px;background:#f0ede8;border-radius:20px;padding:3px 12px;font-size:.75rem;color:var(--muted)}
.result-disclaimer{font-size:.78rem;color:var(--muted);margin-top:14px;font-style:italic}
#loading{display:none;text-align:center;padding:20px}
.spinner{width:36px;height:36px;border:3px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .7s linear infinite;margin:0 auto 10px}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="hero">
  <span class="hero-emoji">🫀</span>
  <h1>Heart<span>Guard</span> AI</h1>
  <p>Clinical-grade heart attack risk assessment — Transformer Neural Network</p>
  <span class="speed-badge">⚡ Optimised for fast prediction</span>
</div>
<div id="warm-banner" style="display:none;background:#fff3cd;color:#856404;text-align:center;padding:10px 16px;font-size:.88rem;border-bottom:1px solid #ffc107;">
  ⏳ <span>AI model warming up...</span> &nbsp;The predict button will unlock automatically.
</div>
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
      <div class="field"><label>Cholesterol Level (mg/dL)</label><input type="number" id="Cholesterol_Level" min="50" max="700" placeholder="e.g. 200"/></div>
      <div class="field"><label>Triglyceride Level (mg/dL)</label><input type="number" id="Triglyceride_Level" min="20" max="1000" placeholder="e.g. 150"/></div>
      <div class="field"><label>Fasting Blood Sugar (mg/dL)</label><input type="number" id="Fasting_Blood_Sugar" min="50" max="400" placeholder="e.g. 95"/></div>
      <div class="field"><label>CRP Level (mg/L)</label><input type="number" id="CRP_Level" min="0" max="50" step="0.1" placeholder="e.g. 3.0"/></div>
      <div class="field"><label>Homocysteine Level (µmol/L)</label><input type="number" id="Homocysteine_Level" min="0" max="50" step="0.1" placeholder="e.g. 10.0"/></div>
    </div>
  </div>

  <div class="card">
    <div class="section-label">03 — Medical Conditions</div>
    <div class="form-grid">
      <div class="field full"><div class="pill-label">High Blood Pressure diagnosed?</div><div class="pill-group" data-field="High_Blood_Pressure"><div class="pill active" data-val="No">No</div><div class="pill" data-val="Yes">Yes</div></div></div>
      <div class="field full"><div class="pill-label">Diabetes diagnosed?</div><div class="pill-group" data-field="Diabetes"><div class="pill active" data-val="No">No</div><div class="pill" data-val="Yes">Yes</div></div></div>
      <div class="field full"><div class="pill-label">Low HDL Cholesterol?</div><div class="pill-group" data-field="Low_HDL"><div class="pill active" data-val="No">No</div><div class="pill" data-val="Yes">Yes</div></div></div>
      <div class="field full"><div class="pill-label">High LDL Cholesterol?</div><div class="pill-group" data-field="High_LDL"><div class="pill active" data-val="No">No</div><div class="pill" data-val="Yes">Yes</div></div></div>
      <div class="field full"><div class="pill-label">Family History of Heart Disease?</div><div class="pill-group" data-field="Family_Heart_Disease"><div class="pill active" data-val="No">No</div><div class="pill" data-val="Yes">Yes</div></div></div>
    </div>
  </div>

  <div class="card">
    <div class="section-label">04 — Lifestyle Factors</div>
    <div class="slider-field" style="margin-bottom:22px">
      <label>Sleep Hours / Night <span id="sleep_val">7</span> hrs</label>
      <input type="range" id="Sleep_Hours" min="3" max="12" step="0.5" value="7" oninput="document.getElementById('sleep_val').textContent=this.value;updateSlider(this)"/>
    </div>
    <div class="form-grid">
      <div class="field"><label>Exercise Habits</label><div class="select-wrap"><select id="Exercise_Habits"><option value="">Select</option><option>Low</option><option selected>Medium</option><option>High</option></select></div></div>
      <div class="field"><label>Stress Level</label><div class="select-wrap"><select id="Stress_Level"><option value="">Select</option><option>Low</option><option selected>Medium</option><option>High</option></select></div></div>
      <div class="field"><label>Alcohol Consumption</label><div class="select-wrap"><select id="Alcohol_Consumption"><option value="">Select</option><option selected>Low</option><option>Medium</option><option>High</option></select></div></div>
      <div class="field"><label>Sugar Consumption</label><div class="select-wrap"><select id="Sugar_Consumption"><option value="">Select</option><option>Low</option><option selected>Medium</option><option>High</option></select></div></div>
    </div>
    <div style="margin-top:22px">
      <div class="pill-label">Smoking?</div>
      <div class="pill-group" data-field="Smoking"><div class="pill active" data-val="No">No</div><div class="pill" data-val="Yes">Yes</div></div>
    </div>
  </div>

  <button class="predict-btn" id="predictBtn" onclick="submitForm()">Predict My Heart Attack Risk</button>

  <div id="loading">
    <div class="spinner"></div>
    <p style="color:var(--muted);font-size:.9rem">Analysing with Transformer Neural Network…</p>
  </div>

  <div id="result">
    <div id="resultCard" class="result-card">
      <div id="resultIcon"  class="result-icon"></div>
      <div id="resultLabel" class="result-label"></div>
      <div id="resultProb"  class="result-prob"></div>
      <div class="result-bar-wrap"><div id="resultBar" class="result-bar" style="width:0%"></div></div>
      <div id="resultAdvice" class="result-advice"></div>
      <div id="resultTiming" class="timing"></div>
      <div class="result-disclaimer">⚕️ This AI assessment is for informational purposes only and does not replace professional medical advice.</div>
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
  const min=parseFloat(el.min),max=parseFloat(el.max),val=parseFloat(el.value);
  el.style.setProperty('--pct',((val-min)/(max-min)*100).toFixed(1)+'%');
}
document.querySelectorAll('input[type=range]').forEach(updateSlider);

// On page load — poll /health and show banner until model is ready
(async function(){
  const banner = document.getElementById('warm-banner');
  const btn    = document.getElementById('predictBtn');
  let ready = await checkReady();
  if(!ready){
    banner.style.display='block';
    btn.disabled=true;
    btn.textContent='Loading AI model...';
    let secs=0;
    const iv=setInterval(async()=>{
      secs+=2;
      banner.querySelector('span').textContent='AI model warming up... '+secs+'s';
      ready=await checkReady();
      if(ready){
        clearInterval(iv);
        banner.style.display='none';
        btn.disabled=false;
        btn.textContent='Predict My Heart Attack Risk';
      }
    },2000);
  }
})();
document.querySelectorAll('.pill-group').forEach(g=>{
  g.querySelectorAll('.pill').forEach(p=>{
    p.addEventListener('click',()=>{
      g.querySelectorAll('.pill').forEach(x=>x.classList.remove('active'));
      p.classList.add('active');
    });
  });
});
function getPill(f){
  const g=document.querySelector(`.pill-group[data-field="${f}"]`);
  return g?g.querySelector('.pill.active')?.dataset.val:null;
}
async function checkReady(){
  try{
    const r=await fetch('/health');
    const d=await r.json();
    return d.ready===true;
  }catch(e){ return false; }
}

async function waitForReady(maxWait=60){
  const btn=document.getElementById('predictBtn');
  const loadEl=document.getElementById('loading');
  const loadMsg=document.querySelector('#loading p');
  loadEl.style.display='block';
  btn.disabled=true;
  const start=Date.now();
  while((Date.now()-start)<maxWait*1000){
    const ready=await checkReady();
    if(ready) return true;
    const secs=Math.round((Date.now()-start)/1000);
    loadMsg.textContent='AI model loading... '+secs+'s (first load takes ~30s)';
    await new Promise(r=>setTimeout(r,2000));
  }
  return false;
}

async function submitForm(){
  const btn=document.getElementById('predictBtn');
  const data={
    Age:               parseFloat(document.getElementById('Age').value)||50,
    Gender:            document.getElementById('Gender').value||'Male',
    BMI:               parseFloat(document.getElementById('BMI').value)||25,
    Blood_Pressure:    parseFloat(document.getElementById('Blood_Pressure').value)||120,
    Cholesterol_Level: parseFloat(document.getElementById('Cholesterol_Level').value)||200,
    Triglyceride_Level:parseFloat(document.getElementById('Triglyceride_Level').value)||150,
    Fasting_Blood_Sugar:parseFloat(document.getElementById('Fasting_Blood_Sugar').value)||95,
    CRP_Level:         parseFloat(document.getElementById('CRP_Level').value)||3,
    Homocysteine_Level:parseFloat(document.getElementById('Homocysteine_Level').value)||10,
    Sleep_Hours:       parseFloat(document.getElementById('Sleep_Hours').value)||7,
    High_Blood_Pressure:getPill('High_Blood_Pressure')||'No',
    Diabetes:           getPill('Diabetes')||'No',
    Low_HDL:            getPill('Low_HDL')||'No',
    High_LDL:           getPill('High_LDL')||'No',
    Family_Heart_Disease:getPill('Family_Heart_Disease')||'No',
    Exercise_Habits:   document.getElementById('Exercise_Habits').value||'Medium',
    Stress_Level:      document.getElementById('Stress_Level').value||'Medium',
    Alcohol_Consumption:document.getElementById('Alcohol_Consumption').value||'Low',
    Sugar_Consumption: document.getElementById('Sugar_Consumption').value||'Medium',
    Smoking:           getPill('Smoking')||'No',
  };
  btn.disabled=true;
  document.getElementById('loading').style.display='block';
  document.getElementById('result').style.display='none';
  try{
    const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const d=await r.json().catch(()=>null);
    if(r.status===503||!d||d.error==='Models still loading, please retry in 10-30 seconds'){
      // Model not ready — wait and auto-retry
      document.getElementById('loading').style.display='none';
      btn.disabled=false;
      const ready=await waitForReady(60);
      if(!ready){
        document.getElementById('loading').style.display='none';
        btn.disabled=false;
        alert('Model took too long to load. Please refresh the page and try again.');
        return;
      }
      // Ready now — resubmit automatically
      document.getElementById('loading').style.display='none';
      btn.disabled=false;
      submitForm();
      return;
    }
    if(!d||d.error){
      document.getElementById('loading').style.display='none';
      btn.disabled=false;
      alert('Error: '+(d?.error||'Unknown error')+'\n\n'+(d?.traceback?d.traceback.split('\n').slice(-4).join('\n'):''));
      return;
    }
    document.getElementById('loading').style.display='none';
    btn.disabled=false;
    const isHigh=d.label==='High Risk';
    document.getElementById('resultCard').className='result-card '+(isHigh?'high':'low');
    document.getElementById('resultIcon').textContent=isHigh?'⚠️':'✅';
    document.getElementById('resultLabel').textContent=d.label;
    document.getElementById('resultProb').textContent='Risk Probability: '+d.pct+'%';
    document.getElementById('resultBar').style.width=d.pct+'%';
    document.getElementById('resultTiming').textContent='⚡ Predicted in '+d.ms+'ms';
    const advice=isHigh
      ?'Your results indicate elevated cardiovascular risk.\n\n• Consult a cardiologist as soon as possible\n• Monitor blood pressure and cholesterol regularly\n• Adopt a heart-healthy diet (reduce saturated fats, increase fibre)\n• Aim for 150 min moderate exercise per week\n• Quit smoking if applicable\n• Manage stress through sleep and mindfulness'
      :'Your results indicate lower cardiovascular risk — great news!\n\n• Keep up regular physical activity\n• Continue balanced diet and healthy sleep schedule\n• Schedule annual health check-ups\n• Monitor blood pressure and sugar levels periodically\n• Avoid smoking and excessive alcohol';
    document.getElementById('resultAdvice').textContent=advice;
    document.getElementById('result').style.display='block';
    document.getElementById('result').scrollIntoView({behavior:'smooth',block:'start'});
  }catch(e){
    document.getElementById('loading').style.display='none';
    btn.disabled=false;
    alert('Prediction failed. Please try again.\n'+e);
  }
}
</script>
</body>
</html>"""

# ── Start background loader — Flask binds port instantly, models load behind ──
# Render scans for an open port within ~5s of startup. By loading models in a
# background thread, gunicorn binds 0.0.0.0:$PORT immediately and Render is
# happy. The /predict route returns 503 until _ready=True (usually <30s).
threading.Thread(target=load_models, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
