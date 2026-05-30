"""
app.py - Heart Attack Prediction — Questionnaire UI
Features match train_model.py dataset: Age, Gender, BMI, Blood_Pressure,
Cholesterol_Level, Triglyceride_Level, Fasting_Blood_Sugar, CRP_Level,
Homocysteine_Level, Sleep_Hours, High_Blood_Pressure, Diabetes, Low_HDL,
High_LDL, Exercise_Habits, Smoking, Alcohol_Consumption, Sugar_Consumption,
Stress_Level, Family_Heart_Disease
"""

import os, pickle, threading
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

_lock          = threading.Lock()
_models_ready  = False
_nn_model      = None
_scaler        = None
_le_map        = None
_feature_names = None
_emb_model     = None
_sent_pipe     = None

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

CATEGORICAL_DEFAULTS = {
    "Gender": "Male", "Exercise_Habits": "Medium", "Smoking": "No",
    "Alcohol_Consumption": "Low", "Sugar_Consumption": "Medium",
    "Stress_Level": "Medium", "Family_Heart_Disease": "No",
    "High_Blood_Pressure": "No", "Diabetes": "No",
    "Low_HDL": "No", "High_LDL": "No",
}
NUMERIC_DEFAULTS = {
    "Age": 50, "BMI": 25.0, "Blood_Pressure": 120.0,
    "Cholesterol_Level": 200.0, "Triglyceride_Level": 150,
    "Fasting_Blood_Sugar": 95, "CRP_Level": 3.0,
    "Homocysteine_Level": 10.0, "Sleep_Hours": 7,
}

def _load_models():
    global _models_ready, _nn_model, _scaler, _le_map, _feature_names, _emb_model, _sent_pipe
    with _lock:
        if _models_ready:
            return
        print("[load] TensorFlow ...")
        from tensorflow.keras.models import load_model
        _nn_model      = load_model(os.path.join(MODEL_DIR, "transformer_nn.keras"))
        _scaler        = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"),         "rb"))
        _le_map        = pickle.load(open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "rb"))
        _feature_names = pickle.load(open(os.path.join(MODEL_DIR, "feature_names.pkl"),  "rb"))
        print("[load] SentenceTransformer ...")
        from sentence_transformers import SentenceTransformer
        _emb_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[load] Sentiment pipeline ...")
        from transformers import pipeline as hf_pipeline
        _sent_pipe = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True, max_length=512,
        )
        _models_ready = True
        print("[load] All models ready.")

def _encode_categoricals(values):
    import numpy as np
    encoded = {}
    for col, le in _le_map.items():
        if col == "Heart_Disease_Status":
            continue
        val = values.get(col, CATEGORICAL_DEFAULTS.get(col, le.classes_[0]))
        if val not in le.classes_:
            val = le.classes_[0]
        encoded[col] = int(le.transform([val])[0])
    return encoded

def _predict_risk(patient_info):
    import numpy as np
    free_text = (
        f"{patient_info.get('Age',50)} year old {patient_info.get('Gender','Male')}, "
        f"BMI {patient_info.get('BMI',25)}, BP {patient_info.get('Blood_Pressure',120)}, "
        f"Cholesterol {patient_info.get('Cholesterol_Level',200)}, "
        f"Smoking {patient_info.get('Smoking','No')}, "
        f"Diabetes {patient_info.get('Diabetes','No')}, "
        f"Stress {patient_info.get('Stress_Level','Medium')}, "
        f"Exercise {patient_info.get('Exercise_Habits','Medium')}."
    )
    embedding       = _emb_model.encode([free_text])[0]
    sentiment_score = _sent_pipe(free_text[:512])[0]["score"]
    cat_encoded     = _encode_categoricals(patient_info)
    num_values      = {k: patient_info.get(k, v) for k, v in NUMERIC_DEFAULTS.items()}
    row = {}
    row.update(num_values)
    row.update(cat_encoded)
    for i, v in enumerate(embedding):
        row[f"embedding_{i}"] = v
    row["Sentiment_Score"] = sentiment_score
    vec   = __import__('numpy').array([row.get(f, 0.0) for f in _feature_names], dtype=__import__('numpy').float32)
    vec_s = _scaler.transform(vec.reshape(1, -1))
    prob  = float(_nn_model.predict(vec_s, verbose=0)[0][0])
    return {"probability": round(prob, 4), "label": "High Risk" if prob > 0.5 else "Low Risk", "pct": round(prob*100,1)}

@app.route("/")
def index():
    return render_template_string(HTML_UI)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_ready": _models_ready})

@app.route("/predict", methods=["POST"])
def predict():
    _load_models()
    data = request.get_json(force=True)
    # cast numeric fields
    for k in NUMERIC_DEFAULTS:
        if k in data:
            try: data[k] = float(data[k])
            except: data[k] = NUMERIC_DEFAULTS[k]
    result = _predict_risk(data)
    return jsonify(result)

HTML_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>HeartGuard AI — Risk Assessment</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
:root{
  --bg:#f7f3ee;
  --card:#ffffff;
  --ink:#1a1208;
  --muted:#7a6e62;
  --accent:#c0392b;
  --accent2:#e8f5e9;
  --border:#e2dbd2;
  --gold:#b5831a;
  --radius:14px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font-family:'DM Sans',sans-serif;min-height:100vh}

/* ── Header ── */
.hero{
  background:var(--ink);
  color:#fff;
  text-align:center;
  padding:52px 24px 44px;
  position:relative;
  overflow:hidden;
}
.hero::before{
  content:'';
  position:absolute;inset:0;
  background:radial-gradient(ellipse 80% 60% at 50% 120%, #c0392b44 0%, transparent 70%);
}
.hero-emoji{font-size:3rem;display:block;margin-bottom:12px;position:relative}
.hero h1{
  font-family:'Playfair Display',serif;
  font-size:clamp(2rem,5vw,3.2rem);
  font-weight:900;
  letter-spacing:-.02em;
  position:relative;
}
.hero h1 span{color:#e74c3c}
.hero p{
  margin-top:10px;
  color:#b0a89e;
  font-size:.97rem;
  font-weight:300;
  position:relative;
}

/* ── Layout ── */
.container{max-width:780px;margin:0 auto;padding:36px 20px 80px}

/* ── Section card ── */
.card{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:32px 36px;
  margin-bottom:20px;
  box-shadow:0 2px 12px #1a120808;
}
.section-label{
  font-family:'Playfair Display',serif;
  font-size:.75rem;
  font-weight:700;
  letter-spacing:.14em;
  text-transform:uppercase;
  color:var(--gold);
  margin-bottom:20px;
  display:flex;
  align-items:center;
  gap:8px;
}
.section-label::after{
  content:'';flex:1;height:1px;background:var(--border);
}

/* ── Form grid ── */
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px 28px}
.form-grid.three{grid-template-columns:1fr 1fr 1fr}
.full{grid-column:1/-1}
@media(max-width:560px){.form-grid,.form-grid.three{grid-template-columns:1fr}}

/* ── Field ── */
.field label{
  display:block;
  font-size:.82rem;
  font-weight:600;
  color:var(--muted);
  margin-bottom:7px;
  letter-spacing:.02em;
}
.field input,.field select{
  width:100%;
  background:#faf8f5;
  border:1.5px solid var(--border);
  border-radius:9px;
  padding:11px 14px;
  font-family:'DM Sans',sans-serif;
  font-size:.95rem;
  color:var(--ink);
  outline:none;
  transition:border-color .18s,box-shadow .18s;
  appearance:none;
  -webkit-appearance:none;
}
.field input:focus,.field select:focus{
  border-color:var(--accent);
  box-shadow:0 0 0 3px #c0392b14;
}
.field input[readonly]{background:#f0ede8;color:var(--muted);cursor:default}
.select-wrap{position:relative}
.select-wrap::after{
  content:'▾';
  position:absolute;right:14px;top:50%;transform:translateY(-50%);
  color:var(--muted);pointer-events:none;font-size:.9rem;
}

/* ── Slider ── */
.slider-field label{
  font-size:.82rem;font-weight:600;color:var(--muted);
  display:flex;justify-content:space-between;margin-bottom:9px;
}
.slider-field label span{
  font-family:'Playfair Display',serif;
  font-size:1rem;color:var(--accent);font-weight:700;
}
input[type=range]{
  width:100%;
  -webkit-appearance:none;height:5px;
  background:linear-gradient(to right,var(--accent) var(--pct,50%),var(--border) var(--pct,50%));
  border-radius:4px;outline:none;cursor:pointer;
}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;width:18px;height:18px;
  background:var(--accent);border-radius:50%;
  border:2px solid #fff;box-shadow:0 1px 5px #0002;
}

/* ── Toggle pills ── */
.pill-group{display:flex;gap:8px;flex-wrap:wrap;margin-top:4px}
.pill{
  padding:8px 18px;border-radius:30px;font-size:.85rem;font-weight:600;
  border:1.5px solid var(--border);background:#faf8f5;color:var(--muted);
  cursor:pointer;transition:all .15s;user-select:none;
}
.pill.active{background:var(--accent);border-color:var(--accent);color:#fff}
.pill-label{font-size:.82rem;font-weight:600;color:var(--muted);margin-bottom:9px;letter-spacing:.02em}

/* ── Submit ── */
.predict-btn{
  display:block;width:100%;
  background:var(--accent);
  color:#fff;border:none;
  border-radius:12px;
  padding:17px;
  font-family:'Playfair Display',serif;
  font-size:1.15rem;font-weight:700;letter-spacing:.02em;
  cursor:pointer;
  transition:transform .13s,box-shadow .13s,background .13s;
  box-shadow:0 4px 20px #c0392b33;
  margin-top:8px;
}
.predict-btn:hover{background:#a93226;transform:translateY(-1px);box-shadow:0 6px 24px #c0392b44}
.predict-btn:active{transform:translateY(0)}
.predict-btn:disabled{background:#ccc;box-shadow:none;cursor:not-allowed}

/* ── Result panel ── */
#result{display:none;margin-top:24px}
.result-card{
  border-radius:var(--radius);
  padding:36px;
  text-align:center;
  border:1.5px solid;
  animation:pop .35s cubic-bezier(.34,1.56,.64,1);
}
@keyframes pop{from{opacity:0;transform:scale(.93)}to{opacity:1;transform:scale(1)}}
.result-card.high{background:#fff5f5;border-color:#e74c3c}
.result-card.low{background:#f0faf3;border-color:#27ae60}
.result-icon{font-size:2.8rem;margin-bottom:10px}
.result-label{
  font-family:'Playfair Display',serif;
  font-size:2rem;font-weight:900;
  margin-bottom:6px;
}
.result-card.high .result-label{color:#c0392b}
.result-card.low  .result-label{color:#27ae60}
.result-prob{
  font-size:1rem;color:var(--muted);margin-bottom:18px;
}
.result-bar-wrap{background:#e8e0d8;border-radius:30px;height:10px;overflow:hidden;margin:0 auto 20px;max-width:340px}
.result-bar{height:100%;border-radius:30px;transition:width .8s cubic-bezier(.34,1.2,.64,1)}
.result-card.high .result-bar{background:linear-gradient(90deg,#e74c3c,#c0392b)}
.result-card.low  .result-bar{background:linear-gradient(90deg,#27ae60,#1e8449)}
.result-advice{
  font-size:.9rem;line-height:1.65;color:var(--ink);
  background:#fff;border-radius:10px;padding:16px 20px;
  text-align:left;margin-top:4px;border:1px solid var(--border);
}
.result-disclaimer{font-size:.78rem;color:var(--muted);margin-top:14px;font-style:italic}

/* ── Loading overlay ── */
#loading{display:none;text-align:center;padding:20px}
.spinner{
  width:36px;height:36px;border:3px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;
  animation:spin .7s linear infinite;margin:0 auto 10px;
}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>

<div class="hero">
  <span class="hero-emoji">🫀</span>
  <h1>Heart<span>Guard</span> AI</h1>
  <p>Clinical-grade heart attack risk assessment powered by Transformer Neural Network</p>
</div>

<div class="container">

  <!-- Basic Info -->
  <div class="card">
    <div class="section-label">01 — Basic Information</div>
    <div class="form-grid">
      <div class="field">
        <label>Age (years)</label>
        <input type="number" id="Age" min="18" max="100" placeholder="e.g. 45" value=""/>
      </div>
      <div class="field">
        <label>Gender</label>
        <div class="select-wrap">
          <select id="Gender">
            <option value="">Select</option>
            <option>Male</option>
            <option>Female</option>
          </select>
        </div>
      </div>
      <div class="field">
        <label>Height (cm)</label>
        <input type="number" id="height_cm" min="100" max="220" placeholder="e.g. 170" oninput="calcBMI()"/>
      </div>
      <div class="field">
        <label>Weight (kg)</label>
        <input type="number" id="weight_kg" min="30" max="250" placeholder="e.g. 70" oninput="calcBMI()"/>
      </div>
      <div class="field">
        <label>BMI (auto-calculated)</label>
        <input type="text" id="BMI" readonly placeholder="Fill height & weight"/>
      </div>
    </div>
  </div>

  <!-- Vitals -->
  <div class="card">
    <div class="section-label">02 — Clinical Vitals</div>
    <div class="form-grid">
      <div class="field">
        <label>Resting Blood Pressure (mmHg)</label>
        <input type="number" id="Blood_Pressure" min="60" max="250" placeholder="e.g. 120"/>
      </div>
      <div class="field">
        <label>Cholesterol Level (mg/dL)</label>
        <input type="number" id="Cholesterol_Level" min="50" max="700" placeholder="e.g. 200"/>
      </div>
      <div class="field">
        <label>Triglyceride Level (mg/dL)</label>
        <input type="number" id="Triglyceride_Level" min="20" max="1000" placeholder="e.g. 150"/>
      </div>
      <div class="field">
        <label>Fasting Blood Sugar (mg/dL)</label>
        <input type="number" id="Fasting_Blood_Sugar" min="50" max="400" placeholder="e.g. 95"/>
      </div>
      <div class="field">
        <label>CRP Level (mg/L)</label>
        <input type="number" id="CRP_Level" min="0" max="50" step="0.1" placeholder="e.g. 3.0"/>
      </div>
      <div class="field">
        <label>Homocysteine Level (µmol/L)</label>
        <input type="number" id="Homocysteine_Level" min="0" max="50" step="0.1" placeholder="e.g. 10.0"/>
      </div>
    </div>
  </div>

  <!-- Conditions -->
  <div class="card">
    <div class="section-label">03 — Medical Conditions</div>
    <div class="form-grid">
      <div class="field full">
        <div class="pill-label">High Blood Pressure diagnosed?</div>
        <div class="pill-group" data-field="High_Blood_Pressure">
          <div class="pill active" data-val="No">No</div>
          <div class="pill" data-val="Yes">Yes</div>
        </div>
      </div>
      <div class="field full">
        <div class="pill-label">Diabetes diagnosed?</div>
        <div class="pill-group" data-field="Diabetes">
          <div class="pill active" data-val="No">No</div>
          <div class="pill" data-val="Yes">Yes</div>
        </div>
      </div>
      <div class="field full">
        <div class="pill-label">Low HDL Cholesterol?</div>
        <div class="pill-group" data-field="Low_HDL">
          <div class="pill active" data-val="No">No</div>
          <div class="pill" data-val="Yes">Yes</div>
        </div>
      </div>
      <div class="field full">
        <div class="pill-label">High LDL Cholesterol?</div>
        <div class="pill-group" data-field="High_LDL">
          <div class="pill active" data-val="No">No</div>
          <div class="pill" data-val="Yes">Yes</div>
        </div>
      </div>
      <div class="field full">
        <div class="pill-label">Family History of Heart Disease?</div>
        <div class="pill-group" data-field="Family_Heart_Disease">
          <div class="pill active" data-val="No">No</div>
          <div class="pill" data-val="Yes">Yes</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Lifestyle -->
  <div class="card">
    <div class="section-label">04 — Lifestyle Factors</div>

    <div class="slider-field" style="margin-bottom:22px">
      <label>Sleep Hours / Night <span id="sleep_val">7</span> hrs</label>
      <input type="range" id="Sleep_Hours" min="3" max="12" step="0.5" value="7"
        oninput="document.getElementById('sleep_val').textContent=this.value;updateSlider(this)"/>
    </div>

    <div class="form-grid">
      <div class="field">
        <label>Exercise Habits</label>
        <div class="select-wrap">
          <select id="Exercise_Habits">
            <option value="">Select</option>
            <option>Low</option>
            <option selected>Medium</option>
            <option>High</option>
          </select>
        </div>
      </div>
      <div class="field">
        <label>Stress Level</label>
        <div class="select-wrap">
          <select id="Stress_Level">
            <option value="">Select</option>
            <option>Low</option>
            <option selected>Medium</option>
            <option>High</option>
          </select>
        </div>
      </div>
      <div class="field">
        <label>Alcohol Consumption</label>
        <div class="select-wrap">
          <select id="Alcohol_Consumption">
            <option value="">Select</option>
            <option selected>Low</option>
            <option>Medium</option>
            <option>High</option>
          </select>
        </div>
      </div>
      <div class="field">
        <label>Sugar Consumption</label>
        <div class="select-wrap">
          <select id="Sugar_Consumption">
            <option value="">Select</option>
            <option>Low</option>
            <option selected>Medium</option>
            <option>High</option>
          </select>
        </div>
      </div>
    </div>

    <div style="margin-top:22px">
      <div class="pill-label">Smoking?</div>
      <div class="pill-group" data-field="Smoking">
        <div class="pill active" data-val="No">No</div>
        <div class="pill" data-val="Yes">Yes</div>
      </div>
    </div>
  </div>

  <!-- Predict -->
  <button class="predict-btn" id="predictBtn" onclick="submitForm()">Predict My Heart Attack Risk</button>

  <div id="loading">
    <div class="spinner"></div>
    <p style="color:var(--muted);font-size:.9rem">Analysing with Transformer Neural Network…</p>
  </div>

  <div id="result">
    <div id="resultCard" class="result-card">
      <div id="resultIcon" class="result-icon"></div>
      <div id="resultLabel" class="result-label"></div>
      <div id="resultProb"  class="result-prob"></div>
      <div class="result-bar-wrap"><div id="resultBar" class="result-bar" style="width:0%"></div></div>
      <div id="resultAdvice" class="result-advice"></div>
      <div class="result-disclaimer">⚕️ This AI assessment is for informational purposes only and does not replace professional medical advice.</div>
    </div>
  </div>

</div><!-- /container -->

<script>
// ── BMI calculator ──────────────────────────────────────────────────────────
function calcBMI(){
  const h = parseFloat(document.getElementById('height_cm').value);
  const w = parseFloat(document.getElementById('weight_kg').value);
  const el = document.getElementById('BMI');
  if(h>0 && w>0){
    el.value = (w / ((h/100)**2)).toFixed(1);
  } else {
    el.value = '';
  }
}

// ── Slider gradient fill ────────────────────────────────────────────────────
function updateSlider(el){
  const min=parseFloat(el.min),max=parseFloat(el.max),val=parseFloat(el.value);
  const pct=((val-min)/(max-min)*100).toFixed(1)+'%';
  el.style.setProperty('--pct', pct);
}
document.querySelectorAll('input[type=range]').forEach(updateSlider);

// ── Pill toggles ────────────────────────────────────────────────────────────
document.querySelectorAll('.pill-group').forEach(group=>{
  group.querySelectorAll('.pill').forEach(pill=>{
    pill.addEventListener('click',()=>{
      group.querySelectorAll('.pill').forEach(p=>p.classList.remove('active'));
      pill.classList.add('active');
    });
  });
});

function getPillVal(field){
  const group = document.querySelector(`.pill-group[data-field="${field}"]`);
  if(!group) return null;
  const active = group.querySelector('.pill.active');
  return active ? active.dataset.val : null;
}

// ── Form submission ─────────────────────────────────────────────────────────
async function submitForm(){
  const btn = document.getElementById('predictBtn');

  // Collect data
  const data = {
    Age:               parseFloat(document.getElementById('Age').value) || 50,
    Gender:            document.getElementById('Gender').value || 'Male',
    BMI:               parseFloat(document.getElementById('BMI').value) || 25,
    Blood_Pressure:    parseFloat(document.getElementById('Blood_Pressure').value) || 120,
    Cholesterol_Level: parseFloat(document.getElementById('Cholesterol_Level').value) || 200,
    Triglyceride_Level:parseFloat(document.getElementById('Triglyceride_Level').value) || 150,
    Fasting_Blood_Sugar:parseFloat(document.getElementById('Fasting_Blood_Sugar').value)||95,
    CRP_Level:         parseFloat(document.getElementById('CRP_Level').value) || 3,
    Homocysteine_Level:parseFloat(document.getElementById('Homocysteine_Level').value)||10,
    Sleep_Hours:       parseFloat(document.getElementById('Sleep_Hours').value) || 7,
    High_Blood_Pressure: getPillVal('High_Blood_Pressure') || 'No',
    Diabetes:            getPillVal('Diabetes') || 'No',
    Low_HDL:             getPillVal('Low_HDL') || 'No',
    High_LDL:            getPillVal('High_LDL') || 'No',
    Family_Heart_Disease:getPillVal('Family_Heart_Disease') || 'No',
    Exercise_Habits:   document.getElementById('Exercise_Habits').value || 'Medium',
    Stress_Level:      document.getElementById('Stress_Level').value || 'Medium',
    Alcohol_Consumption:document.getElementById('Alcohol_Consumption').value||'Low',
    Sugar_Consumption: document.getElementById('Sugar_Consumption').value || 'Medium',
    Smoking:           getPillVal('Smoking') || 'No',
  };

  btn.disabled = true;
  document.getElementById('loading').style.display='block';
  document.getElementById('result').style.display='none';

  try{
    const res  = await fetch('/predict',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(data)
    });
    const d = await res.json();

    document.getElementById('loading').style.display='none';
    btn.disabled = false;

    const isHigh = d.label === 'High Risk';
    const card   = document.getElementById('resultCard');
    card.className = 'result-card ' + (isHigh ? 'high' : 'low');
    document.getElementById('resultIcon').textContent  = isHigh ? '⚠️' : '✅';
    document.getElementById('resultLabel').textContent = d.label;
    document.getElementById('resultProb').textContent  = `Risk Probability: ${d.pct}%`;
    document.getElementById('resultBar').style.width   = d.pct + '%';

    const advice = isHigh
      ? `Your results indicate elevated cardiovascular risk. Key factors to address:\n\n` +
        `• Consult a cardiologist as soon as possible\n` +
        `• Monitor blood pressure and cholesterol regularly\n` +
        `• Adopt a heart-healthy diet (reduce saturated fats, increase fibre)\n` +
        `• Aim for 150 min moderate exercise per week\n` +
        `• Quit smoking if applicable — it dramatically lowers risk\n` +
        `• Manage stress through sleep, mindfulness, or therapy`
      : `Your results indicate lower cardiovascular risk — great news!\n\n` +
        `Maintain your healthy habits:\n\n` +
        `• Keep up regular physical activity\n` +
        `• Continue balanced diet and healthy sleep schedule\n` +
        `• Schedule annual health check-ups\n` +
        `• Monitor blood pressure and sugar levels periodically\n` +
        `• Avoid smoking and excessive alcohol`;

    document.getElementById('resultAdvice').textContent = advice;
    document.getElementById('result').style.display='block';
    document.getElementById('result').scrollIntoView({behavior:'smooth',block:'start'});

  } catch(e){
    document.getElementById('loading').style.display='none';
    btn.disabled = false;
    alert('Prediction failed. Please try again.\n' + e);
  }
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
