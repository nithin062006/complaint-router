"""Combine transformer + TF-IDF scores. Weight chosen on 2022 validation, reported on untouched 2023+ test."""
import json, numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TAG = "distilroberta-base"
labels = json.load(open("data/labels.json"))
df = pd.read_parquet("data/clean.parquet")
va = df[df.year == 2022].sample(10_000, random_state=0)
te = df[df.year >= 2023].sample(20_000, random_state=1)
tok = AutoTokenizer.from_pretrained(f"model_{TAG}")
model = AutoModelForSequenceClassification.from_pretrained(f"model_{TAG}").to("mps").eval()

def logits(texts):
    order = np.argsort([len(t) for t in texts]); out = np.zeros((len(texts), len(labels)), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), 128):
            b = order[i:i + 128]
            x = tok([texts[j] for j in b], truncation=True, max_length=256, padding=True, return_tensors="pt").to("mps")
            out[b] = model(**x).logits.float().cpu().numpy()
    return out

def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
tv, tt = logits(va.text.tolist()), np.load(f"data/test_logits_{TAG}.npy")
sv, st = np.load("data/svm_val.npy"), np.load("data/svm_test.npy")
yv = va.label.map({l: i for i, l in enumerate(labels)}).values
yt = te.label.map({l: i for i, l in enumerate(labels)}).values
f1 = lambda y, p: f1_score(y, p, average="macro")
best = max(((w, f1(yv, (w * softmax(tv) + (1 - w) * softmax(3 * sv)).argmax(1))) for w in np.linspace(0, 1, 11)), key=lambda t: t[1])
w = best[0]; print(f"weight on transformer chosen on 2022 val: {w:.1f} (val macro-F1 {best[1]:.3f})")
for name, s in (("tfidf-svm", softmax(3 * st)), ("transformer", softmax(tt)), ("ensemble", w * softmax(tt) + (1 - w) * softmax(3 * st))):
    p = s.argmax(1); print(f"TEST 2023+  {name:12s} acc={accuracy_score(yt,p):.3f}  macro-F1={f1(yt,p):.3f}")
print(classification_report(yt, (w * softmax(tt) + (1 - w) * softmax(3 * st)).argmax(1), target_names=labels, digits=3))
json.dump({"w_transformer": float(w)}, open("data/ensemble.json", "w"))
