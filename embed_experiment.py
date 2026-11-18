"""Same time split, same samples: TF-IDF vs sentence embeddings vs both."""
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.svm import LinearSVC

NTR, NTE = 40_000, 20_000
df = pd.read_parquet("data/clean.parquet")
tr = df[df.year <= 2021].sample(NTR, random_state=1)
te = df[df.year >= 2023].sample(NTE, random_state=1)

def report(name, pred):
    print(f"{name:28s} acc={accuracy_score(te.label, pred):.3f}  macro-F1={f1_score(te.label, pred, average='macro'):.3f}", flush=True)

# --- TF-IDF baseline on this sample
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True)
Xtr_t = vec.fit_transform(tr.text); Xte_t = vec.transform(te.text)
report("tfidf + svm", LinearSVC(C=0.5).fit(Xtr_t, tr.label).predict(Xte_t))

# --- sentence embeddings (computed on the Apple GPU)
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="mps")
m.max_seq_length = 256
t = time.time()
Etr = m.encode(tr.text.tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
Ete = m.encode(te.text.tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
print(f"embedded {NTR+NTE:,} texts in {time.time()-t:.0f}s", flush=True)
np.save("data/Etr.npy", Etr); np.save("data/Ete.npy", Ete)
clf = LogisticRegression(C=10, max_iter=300).fit(Etr, tr.label)
report("minilm embeddings + LR", clf.predict(Ete))

# --- both together
Xtr = sp.hstack([Xtr_t, sp.csr_matrix(Etr)]).tocsr(); Xte = sp.hstack([Xte_t, sp.csr_matrix(Ete)]).tocsr()
p = LinearSVC(C=0.5).fit(Xtr, tr.label).predict(Xte)
report("tfidf + embeddings", p)
print(classification_report(te.label, p, digits=3))
