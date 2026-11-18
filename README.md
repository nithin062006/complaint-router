# Complaint router

Classifies US consumer-finance complaints (CFPB database) into 9 product categories, and tests the model the way
it would be used: **trained on the past, evaluated on the future**.

| Model (test: 20,000 complaints from 2023+) | Accuracy | Macro-F1 |
|---|---|---|
| Always guess "Credit reporting" | 63.0% | n/a |
| TF-IDF + linear SVM (300k training complaints) | 87.0% | 0.757 |
| DistilRoBERTa, fine-tuned (120k complaints, 2 epochs, Apple GPU) | 87.5% | 0.767 |
| **Both combined (equal weight)** | **88.2%** | **0.781** |

Macro-F1 averages over the 9 classes equally, so small classes count as much as the big one. Accuracy alone flatters the
model because 63% of complaints are about credit reporting.

## Setup of the experiment
- **Data:** CFPB Consumer Complaint Database, the copy with narratives, 1.69M rows from 2015 to early 2024, licence CC0.
- **Split by time:** train on 2021 and earlier, choose settings on 2022, test on 2023 and later. Nothing in the test set is used for training or tuning.
- **Label cleaning:** CFPB renamed and merged product categories over the years (for example "Credit reporting" became
  "Credit reporting, credit repair services, or other personal consumer reports"). Left alone, a model would be scored wrong
  for using an old name. I merged 19 names into 9 classes (`prep.py`) and dropped two tiny ones.
- **Deduplication:** about 295,000 repeated texts (template complaints) were removed before splitting, so the same text can't be in both train and test.

## What I found
- A time split costs little here (macro-F1 0.769 on a random split vs 0.745 on a time split for the same baseline), unlike
  the job-scam datasets I worked on earlier, where scores collapsed across sources.
- The transformer alone is only about 1 point better than TF-IDF. Frozen sentence embeddings were *worse* than TF-IDF.
  The gain came from combining the two.
- **The weak classes are mostly a label problem.** "Personal/payday loan" (F1 0.54) and "Vehicle loan or lease" (F1 0.71)
  are hard because complaints overlap: a buy-now-pay-later complaint about Affirm is labelled personal loan, and a car-loan
  complaint about credit reporting is labelled credit reporting. A bigger model won't fix that. Cleaning or redefining the labels would.

## Limits
- Accuracy is 88%, not 99%. About 1 in 8 complaints is routed to the wrong category.
- The training data is US consumer finance. It will not transfer to other domains without new labelled data.
- The model reads one text and gives one product label. It does not understand the issue, urgency or sentiment.
- I did not compare against an LLM. That is an obvious next experiment.

## Run it
```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# data: BEE-spoke-data/consumer-finance-complaints (has-text) -> data/hf/has-text-*.parquet
.venv/bin/python prep.py && .venv/bin/python baseline.py     # baseline
.venv/bin/python tfidf_big.py && .venv/bin/python finetune.py distilroberta-base 120000 2
.venv/bin/python ensemble.py && .venv/bin/python app.py      # API on :5060
```
`POST /predict {"text": "..."}` returns the top 3 categories with probabilities.
