# Evidence ledger — Paper 1

One row per numerical claim the paper will make. Every value is read directly from
`results/*.json` by `scripts/build_evidence_ledger.py`; no number here is typed by hand.

**Tags** — `ALREADY DEMONSTRATED`: measured, reproducible from the scripts in this
directory. · `SUGGESTED BY EVIDENCE`: consistent with the artifacts but not identified.
· `NOT ESTABLISHED`: cannot be determined from the available evidence; the paper must
say so rather than infer.

X1 results present: **yes**

| # | Claim | Value | Source | Tag |
|---:|---|---|---|---|
| | **§ The corpus the loop started from** | | | |
| 1 | Training corpus size (sambodhan_balanced_dataset.csv) | 6000 | `data/processed/sambodhan_balanced_dataset.csv` | ALREADY DEMONSTRATED |
| 2 | Provenance stratum A — Hello Sarkar, gold labels | 211 | `scripts/repro_section24.py` | ALREADY DEMONSTRATED |
| 3 | Provenance stratum B — Indian tweets, Llama-3 pseudo-labels | 1386 | `scripts/repro_section24.py` | ALREADY DEMONSTRATED |
| 4 | Provenance stratum C — English, hand-labelled | 20 | `scripts/repro_section24.py` | ALREADY DEMONSTRATED |
| 5 | Provenance stratum D — untraceable to any repository file | 4383 | `scripts/repro_section24.py` | ALREADY DEMONSTRATED |
| 6 | Stratum D as % of corpus | 73.0500 | `scripts/repro_section24.py` | ALREADY DEMONSTRATED |
| 7 | Generating process of stratum D | not in the repository | `notebooks/preparing_datasets.ipynb terminates at the 1,640-row file` | NOT ESTABLISHED |
| 8 | Urgency splits at seed 42 (train/val/test) | [4200, 900, 900] | `data_prep.py:21-34 reimplemented` | ALREADY DEMONSTRATED |
| 9 | Department splits at seed 42 (train/eval/test) | [4800, 600, 600] | `preprocess_and_prepare_dataset.py:73-103 reimplemented` | ALREADY DEMONSTRATED |
| 10 | Urgency test leakage, seed 42 | 17.4444 | `scripts/repro_section24.py` | ALREADY DEMONSTRATED |
| 11 | Department test leakage, seed 42 | 19.1667 | `scripts/repro_section24.py` | ALREADY DEMONSTRATED |
| 12 | Leaked urgency test rows attributable to stratum D | 157 | `scripts/repro_section24.py` | ALREADY DEMONSTRATED |
| 13 | Urgency leakage across seeds {0,1,2,3,42}, mean | 19.1333 | `scripts/baselines_ci.py` | ALREADY DEMONSTRATED |
| 14 | Urgency leakage across seeds, range | [20.2222, 21.1111, 18.8889, 18.0000, 17.4444] | `scripts/baselines_ci.py` | ALREADY DEMONSTRATED |
| 15 | Department leakage across seeds, mean | 21.3667 | `scripts/baselines_ci.py` | ALREADY DEMONSTRATED |
| 16 | Leakage figures are lower bounds (exact match after normalisation only) | The seed-42 leakage figures the paper quotes (urgency 17.44%, department 19.17%) sit inside the across-seed range, so they are typical of the procedure rather than an unlucky draw. All figures are exact-match-after-normalisation and are therefore LOWER BOUNDS: near-duplicates are not counted. | `scripts/sambodhan_repro.py norm()` | ALREADY DEMONSTRATED |
| | **§ Linear baselines, with uncertainty** | | | |
| 17 | Urgency linear baseline, 5-seed mean accuracy | 0.9324 | `scripts/baselines_ci.py` | ALREADY DEMONSTRATED |
| 18 | Urgency linear baseline, 5-seed sd | 0.0066 | `scripts/baselines_ci.py` | ALREADY DEMONSTRATED |
| 19 | Urgency linear baseline, seed-42 accuracy 95% bootstrap CI | [0.9233, 0.9544] | `scripts/baselines_ci.py (2000 resamples)` | ALREADY DEMONSTRATED |
| 20 | Department linear baseline, 5-seed mean accuracy | 0.9660 | `scripts/baselines_ci.py` | ALREADY DEMONSTRATED |
| 21 | Department linear baseline, seed-42 accuracy 95% CI | [0.9500, 0.9783] | `scripts/baselines_ci.py` | ALREADY DEMONSTRATED |
| 22 | Urgency: reported XLM-R accuracy lies inside the linear baseline's 95% CI | True | `scripts/baselines_ci.py` | ALREADY DEMONSTRATED |
| 23 | Department: reported XLM-R accuracy lies inside the linear baseline's 95% CI | True | `scripts/baselines_ci.py` | ALREADY DEMONSTRATED |
| 24 | CLAIM WORDING: linear vs XLM-R on the original corpus is PARITY, not superiority | PARITY -- the reported XLM-R accuracy lies inside the linear baseline's 95% bootstrap CI, so the difference is not resolvable at this sample size. Do NOT write 'outperforms'. | `scripts/baselines_ci.py` | ALREADY DEMONSTRATED |
| 25 | Urgency leak-free linear accuracy (seed 42) | 0.9273 | `scripts/repro_section24.py` | ALREADY DEMONSTRATED |
| 26 | Department leak-free linear accuracy (seed 42) | 0.9567 | `scripts/repro_section24.py` | ALREADY DEMONSTRATED |
| | **§ The promotion gate** | | | |
| 27 | correct_ratio r (shipped default) | 0.5000 | `prepare_pd_df.py:9` | ALREADY DEMONSTRATED |
| 28 | decision_threshold τ | 0.0010 | `configs.py:42` | ALREADY DEMONSTRATED |
| 29 | deployed_sample_size | 300 | `configs.py:41` | ALREADY DEMONSTRATED |
| 30 | Incumbent's measured accuracy on the pool = r/(1+r); at r=0.5 | 0.3333 | `analysis/gate_model.md §2 (exact derivation)` | ALREADY DEMONSTRATED |
| 31 | Challenger accuracy required on flagged errors at r=0.5, a_C=1 | 0.0015 | `analysis/gate_model.md §3` | ALREADY DEMONSTRATED |
| 32 | P(promoted) at r=0.5, δ=−0.30 (challenger 30 pts worse) | 1.0000 | `scripts/gate_simulation.py` | ALREADY DEMONSTRATED |
| 33 | P(promoted) at r=0.5, δ=−0.10 | 1.0000 | `scripts/gate_simulation.py` | ALREADY DEMONSTRATED |
| 34 | Fail-open path yields unconditional acceptance | two routes: api_endpoint falsy (line 446) or all queries fail (line 443) | `model_pipeline.py:425-447` | ALREADY DEMONSTRATED |
| 35 | q — incumbent's true accuracy on deployment traffic | unobservable | `no ground-truth evaluation on deployment traffic exists` | NOT ESTABLISHED |
| 36 | d — fraction of incumbent errors admins report | unobservable | `would require re-adjudicating unflagged complaints; DB state gone` | NOT ESTABLISHED |
| 37 | a_M, a_C realised at each promotion | unrecorded | `model_metadata.json ships eval_metrics: {}` | NOT ESTABLISHED |
| | **§ The seven promotions** | | | |
| 38 | Total auto-deploy commits | 7 | `artifacts/MANIFEST.json + results/promotion_ledger.json` | ALREADY DEMONSTRATED |
| 39 | Urgency promotion 1 — v20251028_150816, reported eval accuracy | 0.7695 | `model card at commit 19466adb6226` | ALREADY DEMONSTRATED |
| 40 | Urgency promotion 2 — v20251030_114839, reported eval accuracy | 0.6867 | `model card at commit e04b4076b90c` | ALREADY DEMONSTRATED |
| 41 | Department promotions 1–5, reported eval accuracy sequence | 0.9417 → 0.9667 → 0.9918 → 0.9959 → 1.0000 | `results/promotion_ledger.json (cards at each auto-deploy commit)` | ALREADY DEMONSTRATED |
| 42 | Urgency promotions 1–2, reported eval accuracy sequence | 0.7695 → 0.6867 | `results/promotion_ledger.json — NOTE: different eval sets, not a regression series` | ALREADY DEMONSTRATED |
| 43 | The per-promotion metrics above are on DIFFERENT eval sets and are NOT a regression series | each promotion is scored on its own corpus's eval split | `results/promotion_ledger.json` | ALREADY DEMONSTRATED |
| 44 | Every deployed model_metadata.json ships eval_metrics: {} | confirmed on both deployed models | `model_pipeline.py:522 getattr bug; artifacts/model_cards/*` | ALREADY DEMONSTRATED |
| | **§ Dataset-revision attribution (Task B)** | | | |
| 45 | Urgency: promotions with an attributable dataset revision | 2/2 promotions have an attributable dataset revision. The timing caveat is RESOLVED for this task. | `scripts/task_b_dataset_revisions.py` | ALREADY DEMONSTRATED |
| 46 | Urgency promotion 2 ← v20251030_111553, pushed this many hours earlier | 0.5470 | `scripts/task_b_dataset_revisions.py` | ALREADY DEMONSTRATED |
| 47 | Attribution independently confirmed by model-card arithmetic | CONFIRMED | `scripts/promotion_ledger.py (steps/epoch and accuracy denominator)` | ALREADY DEMONSTRATED |
| 48 | Attributable urgency corpus v20251030_111553: rows | 1500 | `scripts/task_b_dataset_revisions.py` | ALREADY DEMONSTRATED |
| 49 | Attributable urgency corpus v20251030_111553: overlap with original 6,000-row corpus | 0.0000 | `scripts/task_b_dataset_revisions.py` | ALREADY DEMONSTRATED |
| 50 | Attributable urgency corpus v20251028_015640: overlap with original corpus | 0.0000 | `scripts/task_b_dataset_revisions.py` | ALREADY DEMONSTRATED |
| 51 | HEAD revision v20251030_115250 overlap with original corpus (the audit's 99.6% figure) | 99.5625 | `scripts/task_b_dataset_revisions.py — but this revision post-dates the last promotion` | ALREADY DEMONSTRATED |
| 52 | Department: promotions with an attributable dataset revision | 0/5 promotions have an attributable dataset revision. The dataset repository's first commit (2025-10-28T01:21:03+00:00) post-dates every promotion, so NO revision contemporaneous with any promotion exists. The timing caveat stands and is sharpened: the measured corpus provably post-dates all promotions. | `scripts/task_b_dataset_revisions.py` | ALREADY DEMONSTRATED |
| 53 | Department dataset repo first commit (post-dates all five promotions) | 2025-10-28T01:21:03Z | `scripts/task_b_dataset_revisions.py` | ALREADY DEMONSTRATED |
| 54 | Department promotions' training corpus identity | consistent with v20251028_015634 (n_train 1,940) by card arithmetic, but that revision post-dates every promotion | `results/promotion_ledger.json` | SUGGESTED BY EVIDENCE |
| 55 | Per-revision corpus measurements, urgency (rows; train/eval/test; overlap with original; train-test overlap; mean chars) | v20251028_013035: 8426 (6740/843/843) 71.09% 17.91% 178ch \| v20251028_015512: 6000 (4800/600/600) 99.83% 20.00% 128ch \| v20251028_015640: 2426 (1940/243/243) 0.00% 13.58% 303ch \| v20251030_111553: 1500 (1200/150/150) 0.00% 10.00% 302ch \| v20251030_115228: 6000 (4800/600/600) 99.83% 20.00% 128ch \| v20251030_115250: 1600 (1280/160/160) 99.56% 7.50% 128ch | `scripts/task_b_dataset_revisions.py` | ALREADY DEMONSTRATED |
| 56 | Per-revision corpus measurements, department | v20251028_012945: 8426 (6740/843/843) 71.09% 20.64% 178ch \| v20251028_015507: 6000 (4800/600/600) 99.83% 19.17% 128ch \| v20251028_015634: 2426 (1940/243/243) 0.00% 13.58% 303ch | `scripts/task_b_dataset_revisions.py` | ALREADY DEMONSTRATED |
| 57 | MIN_DATASET_LEN, the row count below which a corpus is not published | 1000 | `src/services/prepare_dataset/prepare_dataset_pipeline.py:30, :113` | ALREADY DEMONSTRATED |
| 58 | Augmented training rows used by the original urgency fine-tune | 5781 | `notebooks/Urgency_classifier.ipynb (against 4,200 raw rows for the baseline)` | ALREADY DEMONSTRATED |
| 59 | Set B partition sizes | {'B-train': 151, 'B-test': 36, 'B-val': 28, 'B-unseen': 1} | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 60 | Urgency and department retraining corpora contain identical text sets | 100.0000 | `scripts/task_b_dataset_revisions.py (100% at all three paired revisions)` | ALREADY DEMONSTRATED |
| 61 | The per-task SQL filters differ, so identical corpora are NOT forced by the code | prepare_pd_df.py:34 filters on correct_<task> IS DISTINCT FROM model_predicted_<task>, and prepare_dataset_pipeline.py:100-107 calls it once per task | `src/services/prepare_dataset/` | ALREADY DEMONSTRATED |
| 62 | Single-field flagging is permitted by the API, so identity is not an API constraint either | misclassification.py:63 rejects a report only if BOTH corrections are absent | `src/backend/app/routers/misclassification.py:63` | ALREADY DEMONSTRATED |
| 63 | WHY the two corpora are nevertheless identical (every flagged row corrected on both fields? one fetch reused? programmatic seeding?) | cannot be determined — the database state is gone | `no surviving DB or W&B record` | NOT ESTABLISHED |
| | **§ What the promoted models were trained on** | | | |
| 64 | Department corpus v20251028_015634 is perfectly separable by char n-grams (eval) | 1.0000 | `scripts/repro_section24.py` | ALREADY DEMONSTRATED |
| 65 | Department corpus v20251028_015634, TF-IDF test accuracy | 1.0000 | `scripts/repro_section24.py` | ALREADY DEMONSTRATED |
| 66 | Linear baseline vs promoted transformer, urgency Oct 30 (attributed) | 16.0000 | `scripts/baselines_ci.py — accuracy points, same published eval split` | ALREADY DEMONSTRATED |
| 67 | …its 95% CI (points) | [10.0000, 22.0000] | `scripts/baselines_ci.py` | ALREADY DEMONSTRATED |
| 68 | Linear baseline vs promoted transformer, urgency Oct 28 (attributed) | 14.8200 | `scripts/baselines_ci.py` | ALREADY DEMONSTRATED |
| 69 | …its 95% CI (points) | [11.1200, 18.1100] | `scripts/baselines_ci.py` | ALREADY DEMONSTRATED |
| 70 | Department 1.0 vs linear 1.0 is NOT a like-for-like comparison | SUGGESTED BY EVIDENCE -- the two numbers are NOT established to be on the same eval split; do not present this as a like-for-like gap | `scripts/baselines_ci.py` | SUGGESTED BY EVIDENCE |
| | **§ X1 — the fixed yardstick** | | | |
| 71 | PRE revision SHA | e3a249c1ff8e6d45eadbd9f303fa397030e8501f | `scripts/pin_revisions.py` | ALREADY DEMONSTRATED |
| 72 | POST revision SHA | 2e3ae2505f15784bd7866abcda1d6655a4f19575 | `scripts/pin_revisions.py` | ALREADY DEMONSTRATED |
| 73 | Department has no recoverable pre-loop revision | TIMESTAMP PRECEDES BY 3.8 MINUTES BUT THE TREE CONTAINS NO MODEL WEIGHTS (.gitattributes). No pre-loop revision is recoverable for this task. | `scripts/pin_revisions.py` | ALREADY DEMONSTRATED |
| 74 | id2label identical across PRE and POST | True | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 75 | PRE weights sha256 matches the Hub LFS oid | True | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 76 | POST weights sha256 matches the Hub LFS oid | True | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 77 | Set A (n=900): PRE accuracy | 0.9333 | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 78 | Set A (n=900): POST accuracy | 0.5489 | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 79 | Set A: POST − PRE accuracy delta | -0.3844 | `scripts/x1_eval.py (paired bootstrap)` | ALREADY DEMONSTRATED |
| 80 | Set A: delta 95% CI | [-0.4189, -0.3489] | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 81 | Set A: linear baseline accuracy | 0.9400 | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 82 | Set B-test (gold, unbiased): n | 36 | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 83 | Set B-test: PRE accuracy | 0.4444 | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 84 | Set B-test: POST accuracy | 0.3056 | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 85 | Set B-test: POST − PRE delta 95% CI | [-0.3611, 0.0833] | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 86 | Set B-train (biased in favour of PRE): n | 151 | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 87 | Set B-all: POST − PRE delta | -0.2593 | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 88 | Set B-all: delta 95% CI | [-0.3519, -0.1619] | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 89 | Set A: PRE macro-F1 | 0.9345 | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 90 | Set A: POST macro-F1 | 0.5147 | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 91 | Set A: linear baseline exceeds POST by (accuracy points) | 39.1100 | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 92 | CORROBORATION: PRE's Set A accuracy vs the notebook's reported 0.9344 | 0.9333 measured vs 0.9344 reported — a difference of one example out of 900 | `notebooks/Urgency_classifier.ipynb cell 7 vs scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 93 | MECHANISM: POST's NORMAL recall on Set A collapses (PRE → POST) | 0.902 → 0.210 | `scripts/x1_eval.py — the post-loop model over-escalates` | ALREADY DEMONSTRATED |
| 94 | POST's Set A predicted-class distribution vs true support | predicted {'NORMAL': 87, 'URGENT': 428, 'HIGHLY URGENT': 385} vs true {'NORMAL': 276, 'URGENT': 365, 'HIGHLY URGENT': 259} | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 95 | Class-prior shift does NOT explain it: attributable corpus is 32.6% NORMAL vs the original corpus's 30.6% | 489/1500 vs 1836/6000 | `results/dataset_revisions.json class_balance_all + the original corpus` | ALREADY DEMONSTRATED |
| 96 | On real gold data (Set B-test) NORMAL recall is 0.000 for ALL THREE systems | PRE=0.000, POST=0.000, LINEAR=0.000 | `scripts/x1_eval.py — a task/data property, not a model property (n=5 NORMAL)` | ALREADY DEMONSTRATED |
| 97 | Set B-test: linear baseline accuracy (best of the three on real data) | 0.5278 | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 98 | Set A is held out for BOTH revisions | PRE trained on the original 4,200-row train split; POST trained on a corpus with 0.00% overlap with the original 6,000-row corpus | `results/dataset_revisions.json + scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 99 | Set A is in-distribution for PRE and out-of-distribution for POST | an asymmetry that must be stated: POST's attributable corpus is disjoint from the original corpus, so Set A tests POST's transfer to the original task definition | `results/dataset_revisions.json` | ALREADY DEMONSTRATED |
| 100 | X1 per-partition accuracy, PRE / POST / LINEAR | SetA: n=900 PRE=0.9333 POST=0.5489 LIN=0.9400 \| SetB-test: n=36 PRE=0.4444 POST=0.3056 LIN=0.5278 \| SetB-train: n=151 PRE=0.6755 POST=0.3709 LIN=0.9868 \| SetB-all: n=216 PRE=0.6019 POST=0.3426 LIN=0.8519 | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 101 | X1 paired deltas with 95% CI, all partitions | SetA: -0.3844 [-0.4189, -0.3489] \| SetB-test: -0.1389 [-0.3611, +0.0833] \| SetB-train: -0.3046 [-0.4172, -0.1921] \| SetB-all: -0.2593 [-0.3519, -0.1619] | `scripts/x1_eval.py` | ALREADY DEMONSTRATED |
| 102 | Linear baseline on each retraining corpus: eval accuracy with 95% CI | v20251030_111553: 0.8467 [0.7867, 0.9067] \| v20251028_015640: 0.9177 [0.8807, 0.9506] \| v20251030_115250: 0.9187 [0.8750, 0.9625] \| v20251028_015634: 1.0000 [1.0000, 1.0000] | `scripts/baselines_ci.py` | ALREADY DEMONSTRATED |
| 103 | Gate: minimum a_M required at each r, for a_C in {1.00, 0.95, 0.90} | r=0.25: A_inc=0.2000 0.0013/0.0138/0.0262 \| r=0.5: A_inc=0.3333 0.0015/0.0265/0.0515 \| r=1.0: A_inc=0.5000 0.0020/0.0520/0.1020 \| r=2.0: A_inc=0.6667 0.0030/0.1030/0.2030 | `analysis/gate_model.md, scripts/gate_simulation.py` | ALREADY DEMONSTRATED |
| 104 | Gate simulation settings (q used in panel a, reporting rate, trials, pool size) | q=0.85, d=0.5, trials=300, pool n=1500, delta for panel b=-0.1 | `scripts/gate_simulation.py` | ALREADY DEMONSTRATED |
| 105 | Card arithmetic per promotion: steps/epoch and implied n_train window | v20251028_150816: 122 steps/epoch -> [1937, 1952] \| v20251030_114839: 75 steps/epoch -> [1185, 1200] \| v20251023_011818: 300 steps/epoch -> [4785, 4800] \| v20251023_012846: 300 steps/epoch -> [4785, 4800] \| v20251027_123835: 122 steps/epoch -> [1937, 1952] \| v20251027_133806: 122 steps/epoch -> [1937, 1952] \| v20251027_143006: 122 steps/epoch -> [1937, 1952] | `scripts/promotion_ledger.py` | ALREADY DEMONSTRATED |
| 106 | Epoch value printed at step 50 for promotion v20251030_114839 | 0.6667 | `model card at commit e04b4076b90c` | ALREADY DEMONSTRATED |
| 107 | PRE revision is byte-identical to the weights behind the notebook's 0.9344 | not provable | `the PRE card reports the VALIDATION figures (0.9467/0.9475), not the test figure; but the measured Set A accuracy matches the notebook to one example` | NOT ESTABLISHED |
| | **§ Artifact provenance** | | | |
| 108 | Local repository HEAD | 9b37728e2b4a68088300889090627030f8994af2 | `git rev-parse HEAD` | ALREADY DEMONSTRATED |
| 109 | Local repository clean at capture time | True | `git status --porcelain` | ALREADY DEMONSTRATED |
| 110 | External artifacts archived with SHA256 + UTC timestamp | 73 | `scripts/fetch_artifacts.py` | ALREADY DEMONSTRATED |
| 111 | kar137/sambodhan-urgency-classifier canonically resolves to sambodhan/… | sambodhan/sambodhan_urgency_classifier | `Hub redirect; resolves the urgency half of audit finding P15` | ALREADY DEMONSTRATED |
| 112 | Which endpoint the backend actually served at any given time | unresolved | `two artifact families; department repo does not redirect` | NOT ESTABLISHED |
| 113 | Real citizen traffic | no evidence anywhere in the repository | `analytics JSONs are internally inconsistent test fixtures (audit P9)` | NOT ESTABLISHED |
| 114 | Whether the gate ever REJECTED a candidate | W&B run logs not public | `referenced throughout the code; not in the repository` | NOT ESTABLISHED |
| | **§ Prevalence (Task F)** | | | |
| 115 | Projects composing error-conditioned corpus AND automated promotion | 1 | `scripts/write_prevalence.py` | SUGGESTED BY EVIDENCE |
| 116 | Projects with error-conditioned evaluation but advisory promotion | 2 | `scripts/write_prevalence.py` | SUGGESTED BY EVIDENCE |
| 117 | Correct implementations found (negative control) | 1 | `scripts/write_prevalence.py` | SUGGESTED BY EVIDENCE |
| 118 | A base rate for this pattern in the wild | not measured | `search was timeboxed and sampled; HF search does not index commit messages` | NOT ESTABLISHED |

## Tally

- ALREADY DEMONSTRATED: **103**
- SUGGESTED BY EVIDENCE: **5**
- NOT ESTABLISHED: **10**
- total rows: **118**

## Rules for Prompt 2

1. Do not promote a `SUGGESTED BY EVIDENCE` row to an assertion. In particular, the
   department model's 1.0 and the perfectly-separable department corpus are **not**
   established to be the same evaluation; say "consistent with", never "measured on".
2. On the original corpus, the linear-vs-XLM-R comparison is **parity**. Never write
   "outperforms".
3. The seven per-promotion metrics come from seven different evaluation sets. They are
   not a regression series. X1 is the only instrument that measures before/after on a
   fixed yardstick.
4. Every `NOT ESTABLISHED` row must appear in the paper's limitations, not be omitted.
