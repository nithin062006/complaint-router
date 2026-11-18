"""Baseline: TF-IDF + linear SVM. Compare a random split with a time-based split."""
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC

N = 150_000  # per-split cap so this runs in a couple of minutes on a laptop
df = pd.read_parquet("data/clean.parquet")


def fit_eval(train, test, name):
    t = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True)
    Xtr = vec.fit_transform(train["text"]); Xte = vec.transform(test["text"])
    clf = LinearSVC(C=0.5).fit(Xtr, train["label"])
    pred = clf.predict(Xte)
    acc = accuracy_score(test["label"], pred); f1 = f1_score(test["label"], pred, average="macro")
    print(f"{name:34s} acc={acc:.3f}  macro-F1={f1:.3f}  (train {len(train):,} / test {len(test):,}, {time.time()-t:.0f}s)")
    return test["label"], pred, acc, f1

# 1) random split (what most tutorials do)
s = df.sample(2 * N, random_state=0)
fit_eval(s.iloc[:N], s.iloc[N:], "random split")

# 2) time split: train on <=2021, test on 2023+ (the future)
tr = df[df.year <= 2021]; te = df[df.year >= 2023]
tr = tr.sample(min(N, len(tr)), random_state=0); te = te.sample(min(N, len(te)), random_state=0)
y, p, acc, f1 = fit_eval(tr, te, "time split (<=2021 -> 2023+)")

# 3) majority-class floor
maj = df["label"].value_counts().idxmax()
print(f"{'always predict ' + maj:34s} acc={(te['label']==maj).mean():.3f}")
print(classification_report(y, p, digits=3))
