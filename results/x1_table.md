# X1 -- fixed-yardstick evaluation of the urgency classifier

All three systems evaluated on identical inputs. Set A is the original 900-row
held-out test split (cleaned text, corpus labels); Set B is the gold slice of real
Hello Sarkar grievances (raw text, government `complain_type` labels).

| system | revision | Set A acc | Set A macro-F1 | B-test acc | B-test macro-F1 | B-all acc | B-train acc |
|---|---|---:|---:|---:|---:|---:|---:|
| XLM-R pre-loop | `e3a249c1ff8e` | 0.9333 | 0.9345 | 0.4444 | 0.3270 | 0.6019 | 0.6755 |
| XLM-R post-loop | `2e3ae2505f15` | 0.5489 | 0.5147 | 0.3056 | 0.2393 | 0.3426 | 0.3709 |
| TF-IDF char + LogReg | `-- (local fit)` | 0.9400 | 0.9414 | 0.5278 | 0.3876 | 0.8519 | 0.9868 |

Set A n = 900; B-test n = 36 (unbiased); B-train n = 151 (biased in favour of PRE); B-all n = 216.

> **Read the B-train and B-all columns with care.** Those partitions consist largely of rows that fall in the 4,200-row training split, which BOTH the pre-loop transformer AND the linear baseline were fitted on. The linear baseline's 0.9868 on B-train is memorisation, not generalisation. **B-test is the only partition of Set B held out from every system in this table**, and it is the one to quote.

## Paired bootstrap: POST − PRE on identical rows (2,000 resamples)

| set | n | PRE acc | POST acc | Δ accuracy | 95% CI | excludes 0 |
|---|---:|---:|---:|---:|---|---|
| SetA | 900 | 0.9333 | 0.5489 | -0.3844 | [-0.4189, -0.3489] | **yes** |
| SetB-test | 36 | 0.4444 | 0.3056 | -0.1389 | [-0.3611, +0.0833] | no |
| SetB-train | 151 | 0.6755 | 0.3709 | -0.3046 | [-0.4172, -0.1921] | **yes** |
| SetB-all | 216 | 0.6019 | 0.3426 | -0.2593 | [-0.3519, -0.1619] | **yes** |

Set A is the paper's headline: it is held out for both revisions, since the corpus the post-loop model was gated on shares 0.00% of its rows with the original 6,000-row corpus. On Set B-test the direction is the same but the interval includes zero at n=36 — that partition does not establish a regression on its own.


## Per-class recall, Set A

| system | NORMAL | URGENT | HIGHLY URGENT |
|---|---:|---:|---:|
| XLM-R pre-loop | 0.9022 | 0.9205 | 0.9846 |
| XLM-R post-loop | 0.2101 | 0.6082 | 0.8263 |
| TF-IDF char + LogReg | 0.8949 | 0.9534 | 0.9691 |

## Per-class recall, Set B-test (the unbiased real-data partition)

| system | NORMAL | URGENT | HIGHLY URGENT |
|---|---:|---:|---:|
| XLM-R pre-loop | 0.0000 | 0.3889 | 0.6923 |
| XLM-R post-loop | 0.0000 | 0.4444 | 0.2308 |
| TF-IDF char + LogReg | 0.0000 | 0.6111 | 0.6154 |

## Confusion matrices, Set A  (rows = true, cols = predicted)

**XLM-R pre-loop**

| true \ pred | NORMAL | URGENT | HIGHLY URGENT |
|---|---:|---:|---:|
| NORMAL | 249 | 24 | 3 |
| URGENT | 15 | 336 | 14 |
| HIGHLY URGENT | 2 | 2 | 255 |

**XLM-R post-loop**

| true \ pred | NORMAL | URGENT | HIGHLY URGENT |
|---|---:|---:|---:|
| NORMAL | 58 | 165 | 53 |
| URGENT | 25 | 222 | 118 |
| HIGHLY URGENT | 4 | 41 | 214 |

**TF-IDF char + LogReg**

| true \ pred | NORMAL | URGENT | HIGHLY URGENT |
|---|---:|---:|---:|
| NORMAL | 247 | 28 | 1 |
| URGENT | 10 | 348 | 7 |
| HIGHLY URGENT | 2 | 6 | 251 |

## Caveats

- **pre_revision_identity** — The PRE revision (e3a249c1ff8e) is the last commit before the first Auto-deploy. Its model card reports accuracy 0.9467 / F1-macro 0.9475 -- the VALIDATION figures from notebooks/Urgency_classifier.ipynb cell 6, not the 0.9344 test figure of cell 7. It is not proven byte-identical to the weights that produced either notebook number.
- **id2label_check** — PASS -- both revisions declare the same id2label mapping. Predictions were nevertheless mapped through label STRINGS, not integer indices.
- **truncation** — truncation=True, max_length=96, matching train_urgency.py; batch size 16, CPU, torch.no_grad().
- **text_preprocessing** — Set A is fed as `clean_text` to match the original evaluation. Set B is fed as RAW portal text, uncleaned, because that is what a deployed endpoint receives. The two sets are therefore not directly comparable to each other -- only across revisions.
- **setB_train_bias** — Partition B-train consists of gold rows that fall in the training split, which the PRE model was trained on. Those numbers are biased in FAVOUR of PRE -- i.e. against this paper's thesis. B-test is the unbiased partition and is the one to lead with.
- **partition_sizes** — {'B-train': 151, 'B-test': 36, 'B-val': 28, 'B-unseen': 1}
- **department_model** — No pre-loop department revision is recoverable: the repository's `initial commit` (5c5140c56f52, 2025-10-23T01:15:40Z) does predate the first department auto-deploy by ~4 minutes, but its tree contains only .gitattributes -- no model weights. X1 is therefore urgency-only.
- **linear_baseline_on_SetB** — The linear baseline was fitted on cleaned text but is scored on Set B's raw text, the same input the transformers receive. This is a mild disadvantage to the baseline.
