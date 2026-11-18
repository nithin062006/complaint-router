"""Fine-tune a small transformer on CFPB complaints (Apple GPU). Train <=2021, validate 2022, test 2023+."""
import sys, time, json
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL = sys.argv[1] if len(sys.argv) > 1 else "distilroberta-base"
NTRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 120_000
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
MAXLEN, BS, LR = 256, 32, 3e-5
dev = torch.device("mps"); torch.manual_seed(0)

df = pd.read_parquet("data/clean.parquet")
labels = sorted(df.label.unique()); l2i = {l: i for i, l in enumerate(labels)}
tr = df[df.year <= 2021].sample(NTRAIN, random_state=0)
va = df[df.year == 2022].sample(10_000, random_state=0)
te = df[df.year >= 2023].sample(20_000, random_state=1)  # identical test set to the earlier experiments

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(labels)).to(dev)

def enc(texts): return tok(texts, truncation=True, max_length=MAXLEN)["input_ids"]
def batches(ids, y, bs, shuffle):
    idx = np.random.permutation(len(ids)) if shuffle else np.argsort([len(x) for x in ids])
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]; L = max(len(ids[j]) for j in b)
        x = torch.full((len(b), L), tok.pad_token_id); m = torch.zeros((len(b), L), dtype=torch.long)
        for k, j in enumerate(b): x[k, :len(ids[j])] = torch.tensor(ids[j]); m[k, :len(ids[j])] = 1
        yield b, x.to(dev), m.to(dev), torch.tensor(y[b]).to(dev)

def predict(ids, y):
    model.eval(); out = np.zeros((len(ids), len(labels)), dtype=np.float32)
    with torch.no_grad():
        for b, x, m, _ in batches(ids, y, 128, False): out[b] = model(input_ids=x, attention_mask=m).logits.float().cpu().numpy()
    model.train(); return out

t0 = time.time()
Xtr, Xva, Xte = enc(tr.text.tolist()), enc(va.text.tolist()), enc(te.text.tolist())
ytr, yva, yte = (d.label.map(l2i).values for d in (tr, va, te))
print(f"tokenized in {time.time()-t0:.0f}s", flush=True)

freq = np.bincount(ytr, minlength=len(labels)) / len(ytr)
w = torch.tensor((1 / np.sqrt(freq)) / (1 / np.sqrt(freq)).mean(), dtype=torch.float32).to(dev)  # gentle class weighting
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * ((len(Xtr) + BS - 1) // BS)
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
lossf = torch.nn.CrossEntropyLoss(weight=w)
step = 0; model.train(); t0 = time.time()
for ep in range(EPOCHS):
    for _, x, m, y in batches(Xtr, ytr, BS, True):
        loss = lossf(model(input_ids=x, attention_mask=m).logits, y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad()
        step += 1
        if step % 200 == 0: print(f"step {step}/{steps} loss {loss.item():.3f}  {step/(time.time()-t0):.1f} it/s  eta {(steps-step)/(step/(time.time()-t0))/60:.0f} min", flush=True)
    pv = predict(Xva, yva).argmax(1)
    print(f"epoch {ep+1}: val(2022) acc={accuracy_score(yva,pv):.3f} macro-F1={f1_score(yva,pv,average='macro'):.3f}", flush=True)

lg = predict(Xte, yte); p = lg.argmax(1)
print(f"TEST(2023+) acc={accuracy_score(yte,p):.3f} macro-F1={f1_score(yte,p,average='macro'):.3f}", flush=True)
tag = MODEL.split("/")[-1]
np.save(f"data/test_logits_{tag}.npy", lg); json.dump(labels, open("data/labels.json", "w"))
model.save_pretrained(f"model_{tag}"); tok.save_pretrained(f"model_{tag}")
print("saved", flush=True)
