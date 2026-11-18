import json, time, joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC
df = pd.read_parquet("data/clean.parquet")
labels = sorted(df.label.unique())
tr = df[df.year <= 2021].sample(300_000, random_state=0)
va = df[df.year == 2022].sample(10_000, random_state=0)
te = df[df.year >= 2023].sample(20_000, random_state=1)
t = time.time()
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=400_000, sublinear_tf=True)
Xtr = vec.fit_transform(tr.text)
clf = LinearSVC(C=0.5, class_weight="balanced").fit(Xtr, tr.label)
assert list(clf.classes_) == labels
for name, d in (("val", va), ("test", te)):
    s = clf.decision_function(vec.transform(d.text)); np.save(f"data/svm_{name}.npy", s)
    p = np.array(labels)[s.argmax(1)]
    print(f"{name}: acc={accuracy_score(d.label,p):.3f} macro-F1={f1_score(d.label,p,average='macro'):.3f}")
print(f"{time.time()-t:.0f}s")
joblib.dump((vec, clf), "data/tfidf_svm.joblib")
