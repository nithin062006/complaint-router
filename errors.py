import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
df = pd.read_parquet("data/clean.parquet")
tr = df[df.year <= 2021].sample(40_000, random_state=1); te = df[df.year >= 2023].sample(20_000, random_state=1)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True)
clf = LinearSVC(C=0.5).fit(vec.fit_transform(tr.text), tr.label)
te = te.assign(pred=clf.predict(vec.transform(te.text)))
for cls in ["Personal/payday loan", "Vehicle loan or lease"]:
    sub = te[te.label == cls]
    print(f"\n=== true class: {cls}  (n={len(sub)}) — where did the model send them?")
    print(sub.pred.value_counts(normalize=True).round(2).head(4).to_string())
    for _, r in sub[sub.pred != cls].sample(3, random_state=2).iterrows():
        print(f"\n[true={r.label} | predicted={r.pred}]\n  {r.text[:380].replace(chr(10),' ')}")
