"""Selective routing: auto-route only confident complaints, send the rest to a human.
Threshold is chosen on 2022 validation, then reported on the untouched 2023+ test set."""
import json, os, numpy as np, pandas as pd
from sklearn.metrics import f1_score

TAG = "distilroberta-base"
labels = json.load(open("data/labels.json")); W = json.load(open("data/ensemble.json"))["w_transformer"]
df = pd.read_parquet("data/clean.parquet")
va = df[df.year == 2022].sample(10_000, random_state=0)
te = df[df.year >= 2023].sample(20_000, random_state=1)

def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
vp = "data/val_logits_distilroberta-base.npy"
if not os.path.exists(vp):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"model_{TAG}")
    m = AutoModelForSequenceClassification.from_pretrained(f"model_{TAG}").to("mps").eval()
    texts = va.text.tolist(); order = np.argsort([len(t) for t in texts]); out = np.zeros((len(texts), len(labels)), np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), 128):
            b = order[i:i+128]
            x = tok([texts[j] for j in b], truncation=True, max_length=256, padding=True, return_tensors="pt").to("mps")
            out[b] = m(**x).logits.float().cpu().numpy()
    np.save(vp, out)
tv, tt = np.load(vp), np.load(f"data/test_logits_{TAG}.npy")
sv, st = np.load("data/svm_val.npy"), np.load("data/svm_test.npy")
idx = {l: i for i, l in enumerate(labels)}
yv, yt = va.label.map(idx).values, te.label.map(idx).values
P = lambda t, s: W * softmax(t) + (1 - W) * softmax(3 * s)
pv, pt = P(tv, sv), P(tt, st)
cv, ct = pv.max(1), pt.max(1); av, at = pv.argmax(1) == yv, pt.argmax(1) == yt

print(f"{'target acc':>10} {'threshold':>9} | {'VAL coverage':>12} {'VAL acc':>8} | {'TEST coverage':>13} {'TEST acc':>8} {'TEST macroF1':>12}")
rows = []
for target in (0.90, 0.92, 0.95, 0.97, 0.98):
    ok = [t for t in np.unique(np.round(cv, 3)) if (cv >= t).sum() >= 300 and av[cv >= t].mean() >= target]
    if not ok: print(f"{target:>10.2f}  not reachable on validation"); continue
    t = min(ok); mv, mt = cv >= t, ct >= t
    f = f1_score(yt[mt], pt.argmax(1)[mt], average="macro")
    print(f"{target:>10.2f} {t:>9.3f} | {mv.mean():>12.1%} {av[mv].mean():>8.1%} | {mt.mean():>13.1%} {at[mt].mean():>8.1%} {f:>12.3f}")
    rows.append({"target": target, "threshold": float(t), "test_coverage": float(mt.mean()), "test_acc": float(at[mt].mean())})
json.dump(rows, open("data/threshold.json", "w"), indent=1)
print(f"\nNo threshold: coverage 100%, test acc {at.mean():.1%}")
