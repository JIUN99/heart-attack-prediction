"""
app.py - Heart Attack Prediction (Tabular Only, Render Free Tier)
No NLP, no embeddings. RAM ~120MB. Loads in <5s.
"""

import os, pickle, threading, time
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

MODEL_DIR     = os.path.join(os.path.dirname(__file__), "model")
_ready        = False
_load_error   = None
_nn_model     = None
_scaler       = None
_le_map       = None
_feature_names= None

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

# ── Model loader (runs in background thread) ──────────────────────────────────
def load_models():
    global _ready, _load_error, _nn_model, _scaler, _le_map, _feature_names
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        t0 = time.time()

        print("[startup] Importing TensorFlow ...", flush=True)
        from tensorflow.keras.models import load_model

        print("[startup] Loading model ...", flush=True)
        _nn_model = load_model(os.path.join(MODEL_DIR, "transformer_nn.keras"))

        print("[startup] Loading scaler ...", flush=True)
        with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
            _scaler = pickle.load(f)

        print("[startup] Loading encoders ...", flush=True)
        with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "rb") as f:
            _le_map = pickle.load(f)

        print("[startup] Loading feature names ...", flush=True)
        with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f:
            _feature_names = pickle.load(f)

        print(f"[startup] Warming up ({len(_feature_names)} features) ...", flush=True)
        dummy = np.zeros((1, len(_feature_names)), dtype=np.float32)
        _nn_model.predict(dummy, verbose=0)

        _ready = True
        print(f"[startup] ✅ Ready in {time.time()-t0:.1f}s", flush=True)

    except Exception:
        import traceback
        _load_error = traceback.format_exc()
        print(f"[startup] ❌ FAILED:\n{_load_error}", flush=True)

# Start background load immediately — Flask binds port first
threading.Thread(target=load_models, daemon=True).start()

# ── Prediction ────────────────────────────────────────────────────────────────
def _encode(values):
    encoded = {}
    for col, le in _le_map.items():
        val = values.get(col, CATEGORICAL_DEFAULTS.get(col, le.classes_[0]))
        if val not in le.classes_:
            val = le.classes_[0]
        encoded[col] = int(le.transform([val])[0])
    return encoded

def _predict(data):
    t0 = time.time()
    num = {k: float(data.get(k, v)) for k, v in NUMERIC_DEFAULTS.items()}
    cat = _encode(data)
    row = {**num, **cat}
    vec   = np.array([row.get(f, 0.0) for f in _feature_names], dtype=np.float32)
    vec_s = _scaler.transform(vec.reshape(1, -1))
    prob  = float(_nn_model.predict(vec_s, verbose=0)[0][0])
    ms    = round((time.time()-t0)*1000)
    print(f"[predict] {ms}ms  prob={prob:.4f}", flush=True)
    return {"probability": round(prob,4), "pct": round(prob*100,1),
            "label": "High Risk" if prob>0.5 else "Low Risk", "ms": ms}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/health")
def health():
    return jsonify({"ready": _ready, "error": _load_error})

@app.route("/debug")
def debug():
    return jsonify({
        "ready":       _ready,
        "load_error":  _load_error,
        "features":    _feature_names or [],
        "model_files": os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else [],
    })

@app.route("/reload")
def reload():
    global _ready, _load_error
    if _ready:
        return jsonify({"status": "already ready"})
    _ready = False; _load_error = None
    threading.Thread(target=load_models, daemon=True).start()
    return jsonify({"status": "reloading — check /debug in 30s"})

@app.route("/predict", methods=["POST"])
def predict():
    if not _ready:
        return jsonify({"error": "loading", "detail": _load_error}), 503
    try:
        data = request.get_json(force=True) or {}
        for k in NUMERIC_DEFAULTS:
            if k in data:
                try:    data[k] = float(data[k])
                except: data[k] = NUMERIC_DEFAULTS[k]
        return jsonify(_predict(data))
    except Exception:
        import traceback
        err = traceback.format_exc()
        print(f"[predict ERROR]\n{err}", flush=True)
        return jsonify({"error": err}), 500

# ── UI ────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
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
.predict-btn:disabled{background:#bbb;box-shadow:none;cursor:not-allowed;transform:none}
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
  <p>Clinical heart attack risk assessment · Transformer Neural Network</p>
</div>
<div id="warm-banner">⏳ <span id="warm-txt">AI model warming up...</span> &nbsp;Predict button unlocks automatically.</div>
<div id="err-banner">❌ <span id="err-txt"></span> &nbsp;<a href="/reload" style="color:inherit;font-weight:600">Retry reload →</a></div>
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
  <div id="loading"><div class="spinner"></div><p style="color:var(--muted);font-size:.88rem" id="load-msg">Analysing...</p></div>
  <div id="result">
    <div id="resultCard" class="result-card">
      <div id="rIcon" class="result-icon"></div>
      <div id="rLabel" class="result-label"></div>
      <div id="rProb"  class="result-prob"></div>
      <div class="result-bar-wrap"><div id="rBar" class="result-bar" style="width:0%"></div></div>
      <div id="rAdvice" class="result-advice"></div>
      <div id="rMs" class="timing"></div>
      <div class="disclaimer">⚕️ For informational purposes only. Not a substitute for medical advice.</div>
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
const gp=f=>{const g=document.querySelector(`.pill-group[data-f="${f}"]`);return g?.querySelector('.pill.active')?.dataset.v||null;};

async function checkReady(){
  try{ const r=await fetch('/health'); const d=await r.json(); return d.ready===true; }
  catch(e){ return false; }
}

// Poll on page load — disable button until ready
(async()=>{
  const btn=document.getElementById('predictBtn');
  const banner=document.getElementById('warm-banner');
  const errBanner=document.getElementById('err-banner');
  let ready=await checkReady();
  if(!ready){
    banner.style.display='block';
    btn.disabled=true;
    btn.textContent='Loading AI model...';
    let secs=0;
    const iv=setInterval(async()=>{
      secs+=2;
      document.getElementById('warm-txt').textContent='AI model warming up... '+secs+'s (usually 20–40s on first load)';
      const r=await fetch('/health').then(x=>x.json()).catch(()=>({}));
      if(r.ready){
        clearInterval(iv); banner.style.display='none';
        btn.disabled=false; btn.textContent='Predict My Heart Attack Risk';
      } else if(r.error){
        clearInterval(iv); banner.style.display='none';
        document.getElementById('err-txt').textContent='Load failed: '+r.error.split('\n').pop();
        errBanner.style.display='block';
        btn.disabled=false; btn.textContent='Predict My Heart Attack Risk';
      }
    },2000);
  }
})();

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
  document.getElementById('load-msg').textContent='Analysing with Neural Network...';
  try{
    const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const d=await r.json().catch(()=>null);
    document.getElementById('loading').style.display='none';
    btn.disabled=false;
    if(!d||r.status===503){
      alert('Model still loading. Please wait a moment and try again.'); return;
    }
    if(r.status!==200||d.error){
      alert('Prediction error:\n'+(d?.error||'Unknown').split('\n').slice(-3).join('\n')); return;
    }
    const hi=d.label==='High Risk';
    document.getElementById('resultCard').className='result-card '+(hi?'high':'low');
    document.getElementById('rIcon').textContent=hi?'⚠️':'✅';
    document.getElementById('rLabel').textContent=d.label;
    document.getElementById('rProb').textContent='Risk Probability: '+d.pct+'%';
    document.getElementById('rBar').style.width=d.pct+'%';
    document.getElementById('rMs').textContent='⚡ '+d.ms+'ms';
    document.getElementById('rAdvice').textContent=hi
      ?'Elevated cardiovascular risk detected.\n\n• Consult a cardiologist promptly\n• Monitor BP and cholesterol regularly\n• Reduce saturated fats, increase fibre\n• Aim for 150 min exercise per week\n• Quit smoking if applicable\n• Manage stress and improve sleep'
      :'Lower cardiovascular risk — keep it up!\n\n• Maintain regular physical activity\n• Continue balanced diet and good sleep\n• Schedule annual health check-ups\n• Monitor BP and blood sugar periodically\n• Avoid smoking and excess alcohol';
    document.getElementById('result').style.display='block';
    document.getElementById('result').scrollIntoView({behavior:'smooth',block:'start'});
  }catch(e){
    document.getElementById('loading').style.display='none';
    btn.disabled=false;
    alert('Request failed: '+e);
  }
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
