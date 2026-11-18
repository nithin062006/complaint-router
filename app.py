"""Small API + demo page: paste a consumer complaint, get the likely product category."""
import json
import joblib, numpy as np, torch
from flask import Flask, jsonify, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TAG = "distilroberta-base"
labels = json.load(open("data/labels.json")); W = json.load(open("data/ensemble.json"))["w_transformer"]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(f"model_{TAG}")
model = AutoModelForSequenceClassification.from_pretrained(f"model_{TAG}").to(dev).eval()
vec, svm = joblib.load("data/tfidf_svm.joblib")
app = Flask(__name__)

def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def classify(text):
    x = tok([text], truncation=True, max_length=256, return_tensors="pt").to(dev)
    with torch.no_grad(): lt = model(**x).logits.float().cpu().numpy()
    p = W * softmax(lt) + (1 - W) * softmax(3 * svm.decision_function(vec.transform([text])))
    top = np.argsort(-p[0])[:3]
    return [{"label": labels[i], "probability": round(float(p[0][i]), 3)} for i in top]

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Complaint router</title><body style="font-family:system-ui;max-width:680px;margin:40px auto;padding:0 16px">
<h1>Complaint router</h1><p>Paste a consumer-finance complaint. Do not paste real personal details.</p>
<textarea id=t rows=8 style="width:100%"></textarea><p><button onclick="go()">Classify</button></p><pre id=o></pre>
<script>async function go(){const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:document.getElementById('t').value})});document.getElementById('o').textContent=JSON.stringify(await r.json(),null,2)}</script>"""

@app.get("/")
def home(): return PAGE

@app.post("/predict")
def predict():
    text = ((request.get_json(silent=True) or {}).get("text") or "").strip()
    if len(text) < 20: return jsonify({"error": "Send at least 20 characters of complaint text."}), 400
    return jsonify({"predictions": classify(text[:5000])})

if __name__ == "__main__":
    app.run(port=5060)
