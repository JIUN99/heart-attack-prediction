"""
app.py - Heart Attack Prediction Chatbot
Conversation: HuggingFace Inference API (free tier, mistralai/Mistral-7B-Instruct-v0.3)
Heavy ML imports deferred — Flask binds port instantly on startup.
"""

import os, re, json, pickle, threading
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

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN   = os.environ.get("HF_TOKEN", "")   # set in Render dashboard

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
        print("[load] Scaler / encoders / feature names ...")
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

def _predict_risk(patient_info, free_text):
    import numpy as np
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
    vec   = np.array([row.get(f, 0.0) for f in _feature_names], dtype=np.float32)
    vec_s = _scaler.transform(vec.reshape(1, -1))
    prob  = float(_nn_model.predict(vec_s, verbose=0)[0][0])
    return {"probability": round(prob, 4), "label": "High Risk" if prob > 0.5 else "Low Risk"}

# ── HuggingFace Inference API ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a compassionate cardiac health assistant.
Your job is to:
1. Collect the patient health info conversationally (age, gender, BMI, blood pressure, cholesterol, smoking, diabetes, exercise, stress, sleep, family history, etc.).
2. Once you have enough info (at least age, gender, and 3+ other factors), summarise and ask the patient to confirm.
3. Return a JSON block EXACTLY like this when ready to predict:

<PREDICT>
{
  "patient_info": {
    "Age": 55, "Gender": "Male", "BMI": 28.5,
    "Blood_Pressure": 138, "Cholesterol_Level": 220,
    "Triglyceride_Level": 170, "Fasting_Blood_Sugar": 105,
    "CRP_Level": 4.0, "Homocysteine_Level": 12.0, "Sleep_Hours": 6,
    "High_Blood_Pressure": "Yes", "Diabetes": "No",
    "Low_HDL": "No", "High_LDL": "Yes",
    "Exercise_Habits": "Low", "Smoking": "Yes",
    "Alcohol_Consumption": "Medium", "Sugar_Consumption": "Medium",
    "Stress_Level": "High", "Family_Heart_Disease": "Yes"
  },
  "free_text": "55-year-old male smoker with high BP and cholesterol."
}
</PREDICT>

4. After the prediction result is returned, explain it clearly, mention key risk factors, and give lifestyle advice. Remind the user you are NOT a substitute for a real doctor.
Keep responses warm, clear, and concise."""

def _build_prompt(history, user_message, prediction_result=None):
    prompt = f"<s>[INST] {SYSTEM_PROMPT} [/INST] Understood! I am ready to help as a cardiac health assistant.</s>\n"
    for turn in history:
        role    = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            prompt += f"[INST] {content} [/INST] "
        elif role == "assistant":
            prompt += f"{content}</s>\n"
    msg = user_message
    if prediction_result:
        msg += (
            f"\n\n[ML model result: probability={prediction_result['probability']}, "
            f"label={prediction_result['label']}. Explain this to the patient with advice.]"
        )
    if msg.strip():
        prompt += f"[INST] {msg} [/INST] "
    return prompt

def _call_hf_api(prompt):
    import urllib.request, urllib.error
    payload = json.dumps({
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False,
        }
    }).encode("utf-8")
    req = urllib.request.Request(
        HF_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if isinstance(result, list) and result:
                text = result[0].get("generated_text", "").strip()
                if "[/INST]" in text:
                    text = text.split("[/INST]")[-1].strip()
                return text
            # Model loading response
            if isinstance(result, dict) and result.get("error", "").startswith("Loading"):
                return "The AI model is warming up (~20 seconds on first use). Please send your message again in a moment!"
            return str(result)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        if "loading" in body.lower() or e.code == 503:
            return "The AI model is warming up. Please resend your message in ~20 seconds!"
        return f"HuggingFace API error {e.code}: {body[:200]}"
    except Exception as e:
        return f"Could not reach AI service: {e}"

def _chat_with_hf(history, user_message, prediction_result=None):
    prompt = _build_prompt(history, user_message, prediction_result)
    return _call_hf_api(prompt)

def _extract_predict_block(text):
    match = re.search(r"<PREDICT>(.*?)</PREDICT>", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_UI)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_ready": _models_ready})

@app.route("/chat", methods=["POST"])
def chat():
    _load_models()
    data    = request.get_json(force=True)
    history = data.get("history", [])
    message = data.get("message", "")
    assistant_reply = _chat_with_hf(history, message)
    predict_data    = _extract_predict_block(assistant_reply)
    prediction      = None
    if predict_data:
        prediction  = _predict_risk(
            predict_data["patient_info"],
            predict_data.get("free_text", message),
        )
        clean = re.sub(r"<PREDICT>.*?</PREDICT>", "", assistant_reply, flags=re.DOTALL).strip()
        assistant_reply = _chat_with_hf(
            list(history) + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": clean},
            ],
            "",
            prediction_result=prediction,
        )
    return jsonify({"reply": assistant_reply, "prediction": prediction})

# ── UI ────────────────────────────────────────────────────────────────────────

HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Heart Attack Risk Chatbot</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',sans-serif;background:#f0f4f8;display:flex;flex-direction:column;height:100vh}
  header{background:#c0392b;color:#fff;padding:1rem 1.5rem;display:flex;align-items:center;gap:.75rem}
  header svg{width:30px;height:30px}
  header h1{font-size:1.1rem}
  #banner{background:#fff3cd;color:#856404;text-align:center;padding:.4rem;font-size:.83rem;display:none}
  #chat{flex:1;overflow-y:auto;padding:1.5rem;display:flex;flex-direction:column;gap:.75rem}
  .bubble{max-width:75%;padding:.75rem 1rem;border-radius:16px;line-height:1.5;font-size:.92rem;white-space:pre-wrap}
  .user{background:#c0392b;color:#fff;align-self:flex-end;border-bottom-right-radius:4px}
  .bot{background:#fff;color:#333;align-self:flex-start;border-bottom-left-radius:4px;box-shadow:0 1px 4px #0001}
  .risk-badge{display:inline-block;margin-top:.5rem;padding:.3rem .8rem;border-radius:20px;font-weight:700;font-size:.85rem}
  .high{background:#fde8e8;color:#c0392b}.low{background:#e8f8f0;color:#27ae60}
  #footer{padding:.75rem 1rem;background:#fff;border-top:1px solid #e0e0e0;display:flex;gap:.5rem}
  #input{flex:1;border:1px solid #ccc;border-radius:24px;padding:.6rem 1.1rem;font-size:.95rem;outline:none}
  #input:focus{border-color:#c0392b}
  #send{background:#c0392b;color:#fff;border:none;border-radius:24px;padding:.6rem 1.4rem;cursor:pointer;font-weight:600}
  #send:hover{background:#a93226}
  .typing{color:#999;font-style:italic;font-size:.85rem}
</style>
</head>
<body>
<header>
  <svg viewBox="0 0 24 24" fill="white">
    <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
  </svg>
  <h1>Heart Attack Risk Assessment — AI Chatbot</h1>
</header>
<div id="banner">Loading AI models on first use — please wait 30–60 seconds...</div>
<div id="chat">
  <div class="bubble bot">Hi! I am your cardiac health assistant powered by Mistral AI.
I will ask a few questions about your health to assess your heart attack risk.

Let us start — how old are you, and what is your gender?</div>
</div>
<div id="footer">
  <input id="input" type="text" placeholder="Type your message..." autocomplete="off"/>
  <button id="send">Send</button>
</div>
<script>
const chatEl=document.getElementById('chat'),inputEl=document.getElementById('input'),
      sendEl=document.getElementById('send'),bannerEl=document.getElementById('banner');
let history=[],firstMsg=true;
function addBubble(text,role,pred){
  const d=document.createElement('div');
  d.className='bubble '+(role==='user'?'user':'bot');
  d.textContent=text;
  if(pred){
    const b=document.createElement('div');
    b.className='risk-badge '+(pred.label==='High Risk'?'high':'low');
    b.textContent=pred.label+'  ('+(pred.probability*100).toFixed(1)+'%)';
    d.appendChild(document.createElement('br'));d.appendChild(b);
  }
  chatEl.appendChild(d);chatEl.scrollTop=chatEl.scrollHeight;
}
async function send(){
  const msg=inputEl.value.trim();if(!msg)return;
  inputEl.value='';addBubble(msg,'user');
  history.push({role:'user',content:msg});
  if(firstMsg){bannerEl.style.display='block';firstMsg=false;}
  const t=document.createElement('div');
  t.className='bubble bot typing';t.textContent='Thinking...';
  chatEl.appendChild(t);chatEl.scrollTop=chatEl.scrollHeight;
  try{
    const r=await fetch('/chat',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({history:history.slice(0,-1),message:msg})});
    const d=await r.json();
    chatEl.removeChild(t);bannerEl.style.display='none';
    addBubble(d.reply,'bot',d.prediction);
    history.push({role:'assistant',content:d.reply});
  }catch(e){
    chatEl.removeChild(t);bannerEl.style.display='none';
    addBubble('Sorry, something went wrong. Please try again.','bot');
  }
}
sendEl.addEventListener('click',send);
inputEl.addEventListener('keydown',e=>{if(e.key==='Enter')send();});
</script>
</body>
</html>"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
