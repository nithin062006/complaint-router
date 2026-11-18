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

## Auto-route only what the model is sure of
Most support teams don't want 100% automation; they want to automate the easy majority and send the rest to a person.
I pick a confidence threshold on 2022 validation data, then check it on the untouched 2023+ test set (`threshold.py`):

| Target accuracy | Auto-routed (test) | Accuracy on those (test) | Macro-F1 on those |
|---|---|---|---|
| none (route everything) | 100% | 88.2% | 0.781 |
| 90% | 97.4% | 89.6% | 0.807 |
| 92% | 91.5% | 91.9% | 0.846 |
| 95% | 79.3% | 95.0% | 0.896 |
| 97% | 59.0% | 97.5% | 0.879 |
| 98% | 47.5% | 98.4% | 0.857 |

The validation thresholds held up on the later test years to within about a point, so the confidence scores are trustworthy.
In practice: auto-route about 79% of complaints at 95% accuracy and send the other 21% to a human.

## LLM comparison
Zero-shot Claude Sonnet 5.5 against the trained ensemble on the **same 180 held-out complaints from 2023+** (a balanced sample,
20 per category, so small classes are measured). The model saw only the first 600 characters of each complaint and the list of
9 category names, with no examples and no tuning. True labels were kept hidden until after it answered.

| | Accuracy | Macro-F1 |
|---|---|---|
| Claude Sonnet 5.5, zero-shot | 81.1% | 0.806 |
| Trained ensemble | 81.7% | 0.818 |

They agreed on 85% of complaints. The LLM was better on debt collection, student loans and vehicle loans; the trained model was
much better on money transfers (F1 0.76 vs 0.57), where the CFPB's labels follow conventions a zero-shot model can't guess.

Caveats: with 180 examples the margin of error is roughly 3 points either way, so the difference between the two is not
statistically meaningful. This was one pass by one model, with a single prompt and no cost or latency measurement.
The LLM may also have seen CFPB data during its training. `llm_eval.py` runs the same comparison through the Gemini API (not run yet).
The balanced sample scores lower than the 20,000-complaint test because small, confusable classes count equally.

## Limits
- Accuracy is 88%, not 99%. About 1 in 8 complaints is routed to the wrong category.
- The training data is US consumer finance. It will not transfer to other domains without new labelled data.
- The model reads one text and gives one product label. It does not understand the issue, urgency or sentiment.
- The LLM comparison above is small (180 complaints), so treat it as a sanity check, not a benchmark.

## Run it
```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# data: BEE-spoke-data/consumer-finance-complaints (has-text) -> data/hf/has-text-*.parquet
.venv/bin/python prep.py && .venv/bin/python baseline.py     # baseline
.venv/bin/python tfidf_big.py && .venv/bin/python finetune.py distilroberta-base 120000 2
.venv/bin/python ensemble.py && .venv/bin/python app.py      # API on :5060
```
`POST /predict {"text": "..."}` returns the top 3 categories with probabilities.
