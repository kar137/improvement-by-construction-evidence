# FINDINGS — Paper 1 evidence closure

**Work performed:** 2026-08-11 / 2026-08-12. Every number below comes from a script in
`scripts/` or an artifact in `artifacts/` with a recorded retrieval timestamp. Exact
values, with their source paths, are tabulated in `EVIDENCE_LEDGER.md` (106 rows: 91
ALREADY DEMONSTRATED, 5 SUGGESTED BY EVIDENCE, 10 NOT ESTABLISHED).

**Repository integrity:** the Sambodhan repository is byte-identical to its state at the
start. `git status --porcelain` is empty; HEAD remains `9b37728e2b4a68088300889090627030f8994af2`.

**Harness validation:** every expected value in the task specification's Section 2.4 was
reproduced exactly before any new result was computed — provenance strata 211/1,386/20/4,383,
split sizes 4,200/900/900 and 4,800/600/600, leakage 17.44 % / 19.17 %, all 157 leaked
urgency test rows attributable to stratum D, baselines 0.9400/0.9414 and 0.9650/0.9648,
leak-free variants 0.9273 (n=743) and 0.9567 (n=485), both retraining corpora's row counts
and overlaps, and TF-IDF separability 1.0000 / 0.9187. `results/repro_section24.json`
records `"verdict": "ALL REPRODUCED"`, zero mismatches.

---

## 1. What X1 showed

X1 is the experiment the audit identified as the single most important missing piece. It
evaluates the pinned pre-loop revision (`e3a249c1ff8e`, 2025-10-10, the last commit before
the first `Auto-deploy`) and current HEAD (`2e3ae2505f15`, 2025-10-30, after two autonomous
promotions) on evaluation sets held fixed across revisions.

Both revisions' `model.safetensors` were hashed and **match the Hub's published LFS oids
exactly**, so the "you evaluated corrupted weights" objection is closed. Both revisions
declare the **identical** `id2label` mapping `{0: NORMAL, 1: URGENT, 2: HIGHLY URGENT}`;
predictions were nevertheless mapped through label *strings*, never integer indices.

### Set A — the original 900-row held-out test split

| system | accuracy | macro-F1 |
|---|---:|---:|
| XLM-R **pre-loop** (`e3a249c1ff8e`) | **0.9333** | 0.9345 |
| XLM-R **post-loop** (`2e3ae2505f15`) | **0.5489** | 0.5147 |
| TF-IDF char(2,5) + LogReg | 0.9400 | 0.9414 |

**Paired bootstrap on identical rows (2,000 resamples): POST − PRE = −0.3844, 95 % CI
[−0.4189, −0.3489].** The interval excludes zero by a wide margin.

Two facts make this the paper's central measurement rather than an artifact:

1. **The pre-loop revision authenticates itself.** Its measured Set A accuracy of 0.9333
   differs from the notebook's reported 0.9344 by **one example out of 900**. The pinned
   revision is, to within a single prediction, the model behind the repository's headline
   number. (The revision's own model card reports 0.9467/0.9475 — the *validation* figures
   — so byte-identity to the notebook's weights remains formally NOT ESTABLISHED.)
2. **Set A is held out for both revisions.** The pre-loop model was trained on the original
   4,200-row train split. The post-loop model's attributable retraining corpus has
   **0.00 %** overlap with the original 6,000-row corpus (§2), so it never saw Set A either.

**How the post-loop model fails.** NORMAL recall collapses from 0.902 to 0.210. It predicts
NORMAL 87 times against a true support of 276, and over-predicts the two escalated classes
(URGENT 428, HIGHLY URGENT 385). The post-loop model systematically **over-escalates**.
A class-prior shift does not explain this: the attributable retraining corpus is 32.6 %
NORMAL against the original corpus's 30.6 %. The mechanism consistent with the evidence is
the one the audit identified as P5 — every cycle cold-starts from `xlm-roberta-base` and
replays none of the original corpus, so a model fitted on a **disjoint** corpus transfers
poorly back to the original task definition.

### Set B — the gold slice (real Hello Sarkar grievances, government `complain_type` labels)

Constructed from the 218 gold-labelled rows of `sambodhan_complaint_details.csv` (excluding
29 BLOCKER and 30 missing), 216 of which join to recoverable text. Fed as **raw** portal
text. Gold labels agree with the corpus's own urgency labels on 214/215 matched rows.

| partition | n | PRE | POST | LINEAR | POST − PRE 95 % CI |
|---|---:|---:|---:|---:|---|
| **B-test** (unbiased — gold rows in Set A's held-out split) | 36 | 0.4444 | 0.3056 | **0.5278** | [−0.3611, **+0.0833**] |
| B-train (biased **in favour of** PRE, which trained on these) | 151 | 0.6755 | 0.3709 | — | [−0.4172, −0.1921] |
| B-all | 216 | 0.6019 | 0.3426 | — | [−0.3519, −0.1619] |

**The honest reading, stated plainly: on the unbiased real-data partition the direction is
the same but the confidence interval includes zero.** At n=36 the comparison is
underpowered and B-test does **not** on its own establish a regression. The paper must not
lead with B-test as evidence of degradation. B-all and B-train do exclude zero, but 151 of
those 216 rows were in the pre-loop model's training data, which biases them toward PRE.

Set B also produces a finding that survives all three systems: **NORMAL recall on real gold
data is 0.000 for the pre-loop model, the post-loop model, and the linear baseline alike**
(n=5 NORMAL in B-test). That is a property of the task and corpus, not of any model. And the
linear baseline is the **best** of the three on real data (0.5278).

### The asymmetry a reviewer will raise, and the answer

Set A is in-distribution for PRE and out-of-distribution for POST, because POST's corpus is
disjoint from the original. A reviewer will say POST does badly on Set A because it was
trained on something else. That is exactly right — **and it is the finding**, not a
confound: the deployment task never changed, the loop changed the training distribution
underneath it, and the promotion gate recorded that change as an improvement. The paper
should state the asymmetry explicitly (`EVIDENCE_LEDGER.md` rows 94–95) rather than let a
reviewer discover it.

---

## 2. Was the timing caveat resolved?

**Urgency: YES, resolved.** **Department: NO — and the caveat is now sharper, not softer.**

The audit measured each corpus at its current HEAD and noted that those revisions post-date
the promotions. Fetching the full dataset commit histories shows six published urgency
revisions and three department revisions, whose contents differ substantially from HEAD.

**Urgency — 2/2 promotions attributed, and confirmed by an independent signal.**

| promotion | attributable corpus revision | pushed before promotion | rows (train/eval/test) | overlap with original corpus |
|---|---|---:|---|---:|
| `v20251028_150816` | `v20251028_015640` | 13.19 h | 2,426 (1,940/243/243) | **0.00 %** |
| `v20251030_114839` | `v20251030_111553` | 0.55 h | 1,500 (1,200/150/150) | **0.00 %** |

The attribution does not rest on timestamps alone. Each promotion's model card, fetched **at
its own auto-deploy commit**, carries the Trainer's training-results table, which leaks the
corpus size two independent ways:

- *steps per epoch*: for `v20251030_114839`, step 50 is printed at epoch 0.6667, giving 75
  steps/epoch, hence n_train = 1,200 at batch size 16 — matching `v20251030_111553`
  exactly. For `v20251028_150816`, 122 steps/epoch gives n_train ∈ [1937, 1952], and the
  published split is 1,940.
- *the accuracy denominator*: the reported 0.6867 is 103/150 exactly. On the HEAD revision's
  160-row eval split, 0.6867 is **unreachable** at four decimal places.

Both promotions are marked `CONFIRMED` in `results/promotion_ledger.json`.

**Department — no attributable revision exists at all.** The department dataset repository's
*first* commit is 2025-10-28T01:21:03Z. The *last* of its five promotions is
2025-10-27T14:30:20Z. The repository did not exist when any department promotion was made.
The caveat therefore stands in a stronger form than the audit stated: the measured corpus
does not merely post-date the promotions by minutes, it **provably post-dates every one of
them**, by 11.4 hours to 5 days. The card arithmetic (122 steps/epoch → n_train ∈
[1937,1952]) is *consistent* with the published 1,940-row corpus, but consistency with a
16-row size window is not identification. Tagged SUGGESTED BY EVIDENCE.

---

## 3. The gate derivation's headline

Derivation in `analysis/gate_model.md`; simulation in `scripts/gate_simulation.py`; figure at
`figures/fig_gate_acceptance.pdf`.

**Exact result.** On group `M` the incumbent is wrong by the SQL predicate that defines
membership; on group `C` the label *is* the incumbent's own stored output, so it is right by
construction. Its measured accuracy on the pool is therefore

> **A_inc(r) = r / (1 + r)**, independent of its true accuracy `q` and of the admin
> reporting rate `d`. At the shipped `r = 0.5`, **A_inc = 1/3**.

**Acceptance condition.** Accept iff `a_M > r(1 − a_C) + τ(1 + r)`. At `r = 0.5`, `τ = 0.001`,
and `a_C → 1` (which training on `C`'s labels drives), this collapses to

> **a_M > 0.0015** — a challenger that gets **0.15 %** of the incumbent's known errors right
> is promoted.

**Acceptance probability at the observed operating point.** Simulating the comparison as
implemented (macro-F1, finite test split, 300 trials per point, challenger modelled as
*independent* of the incumbent, which is the conservative choice):

> **At `r = 0.5`, P(promoted) = 1.000 at every δ tested, down to δ = −0.30.** A challenger
> thirty accuracy points worse than the incumbent is promoted every time. `r = 0.25` behaves
> identically. Only at `r = 2.0` — four times the shipped value — does the gate begin to
> reject clearly worse challengers.

At the shipped operating point the criterion is **indistinguishable from the fail-open path**,
i.e. from not consulting the incumbent at all. Two routes reach that path: `api_endpoint`
falsy (line 446) and all 300 queries failing (line 443). The design pauses its own Spaces to
save compute, so endpoint unavailability is a normal state.

**Unobservable parameters, stated as such:** `q` and `d` are NOT ESTABLISHED, and the
pipeline contains no mechanism that could ever measure them — every label it sees is either
an admin correction of a flagged error or the incumbent's own output.

---

## 4. Which claims tightened, loosened, or must be dropped

### Tightened

- **The regression claim, from unsupported to measured.** −0.3844 accuracy [−0.4189, −0.3489]
  on a fixed yardstick that neither revision trained on. The audit correctly refused to
  assert this; it is now the paper's strongest number.
- **The gate's vacuity, from prose to derivation plus simulation.** `r/(1+r)` exactly, and
  P(promoted) = 1.000 at the shipped operating point across the whole δ < 0 band.
- **Dataset attribution for urgency, from caveat to measurement**, corroborated by card
  arithmetic that is independent of timestamps.
- **Baselines now carry intervals.** Leakage at seed 42 (17.44 %, 19.17 %) is the *lowest*
  of the five seeds tried (ranges 17.44–21.11 % and 19.17–22.67 %), so the quoted figures are
  conservative rather than cherry-picked.
- **The promotion ledger now carries per-promotion reported metrics** recovered from the card
  at each auto-deploy commit — urgency 0.7695 → 0.6867; department 0.9417 → 0.9667 → 0.9918
  → 0.9959 → 1.0000. These did not exist in the audit.
- **P15 (artifact-family ambiguity) is half resolved.** `kar137/sambodhan-urgency-classifier`
  canonically resolves to `sambodhan/sambodhan_urgency_classifier`, matching the
  `hub_model_id` in `model_metadata.json`. The department repository does not redirect.
- **New finding, not in the audit:** at every revision pushed in the same minute, the urgency
  and department retraining corpora contain **100.00 % identical text sets**, differing only
  in the `label` column. The two "task-specific" error-conditioned corpora are one pool of
  complaints labelled twice.

### Loosened

- **The linear-vs-XLM-R comparison on the original corpus is PARITY, not superiority.** The
  reported XLM-R accuracy falls *inside* the linear baseline's 95 % bootstrap CI for both
  tasks (urgency 0.9344 ∈ [0.9233, 0.9544]; department 0.97 ∈ [0.9500, 0.9783]). "Outperforms"
  must not be written for this comparison.
- **The department 1.0-versus-linear-1.0 comparison is no longer like-for-like.** Downgraded
  to SUGGESTED BY EVIDENCE.
- **Set B-test does not establish a regression on its own** (CI includes zero at n=36).

### Must be dropped or restated

- **"The urgency loop is recycling the same synthetic corpus (99.6 % overlap)."** Falsified —
  see §5.
- **"Beaten by a linear baseline by ~23 accuracy points."** Wrong magnitude — see §5.
- **Any framing of the seven per-promotion metrics as a regression series.** Each is scored on
  its own corpus's eval split; they are seven different yardsticks. X1 is the only instrument
  that measures before/after on a fixed one.

---

## 5. Claims this work FALSIFIED

Both falsified claims are the audit's own, and both were load-bearing.

### 5.1 The urgency "corpus recycling" characterisation is FALSE for the promotions

The audit (P12) measured the urgency corpus at HEAD — `v20251030_115250`, 1,600 rows,
**99.6 %** overlap with the original 6,000-row corpus — and concluded: *"The urgency loop is
recycling the same synthetic corpus at 27 % of its original size — not learning from new
data, just retraining from scratch on less of the same."*

That revision was pushed at 11:53:00 on 30 October, **four minutes after** the promotion at
11:48:53, and **no published model was trained on it** — the card arithmetic excludes it
(0.6867 is unreachable on a 160-row eval split). The corpora the two urgency promotions were
actually gated on are:

- `v20251028_015640` — 2,426 rows, **0.00 %** overlap with the original corpus
- `v20251030_111553` — 1,500 rows, **0.00 %** overlap with the original corpus

**Both urgency promotions were gated on corpora entirely disjoint from the original training
corpus.** The direction of the audit's claim is reversed: this is not recycling, it is total
distribution replacement — the same failure mode the audit attributed only to the department
loop. Both loops replaced the distribution; neither recycled it.

This correction *strengthens* the paper. It supplies the mechanism for X1's −38-point drop
(a disjoint corpus plus cold start with no replay) and it unifies the two loops under one
failure mode instead of contrasting "99.6 % vs 0.0 %".

The corpus contents also oscillate wildly across revisions rather than shrinking
monotonically — urgency train splits run 6,740 → 4,800 → 1,940 → 1,200 → 4,800 → 1,280, and
the 4,800-row revisions are 99.83 % the *original* corpus. Within 90 seconds on 28 October
the department corpus went from 6,000 rows to 2,426.

### 5.2 The "23-point" linear-baseline gap is the wrong number

The audit compared TF-IDF's 0.9187 on `v20251030_115250` against the card's 0.6867 — two
*different* eval splits (160 rows vs 150 rows). Recomputed like-for-like on each promotion's
attributable corpus and its own published eval split:

| promotion | linear | promoted transformer | gap (accuracy points) | 95 % CI |
|---|---:|---:|---:|---|
| `v20251030_114839` | 0.8467 | 0.6867 | **+16.00** | [+10.00, +22.00] |
| `v20251028_150816` | 0.9177 | 0.7695 | **+14.82** | [+11.12, +18.11] |

The claim **survives** — both intervals exclude zero — but the magnitude was overstated by
about seven points. Write **+16.0 [10.0, 22.0]**, not 23.

### 5.3 Not falsified, but corrected in scope

The audit's ">= 0.0000 threshold on two commits is unexplained" stands. Note additionally
that `deployed_sample_size = 300` **never binds** at the observed corpus sizes (test splits
of 150–243 rows), so challenger and incumbent were scored on the same rows, not on a
subsample — a small simplification in the paper's favour.

---

## 6. The sentence the paper should use for its central quantitative claim

Written to survive the Phase 14 hostile review point by point:

> **On the repository's own held-out test split of 900 examples — regenerated
> deterministically at the seed the repository specifies, and held out for both revisions,
> since the corpus the post-loop model was gated on shares 0.00 % of its rows with the
> original training corpus (exact match after normalisation) — accuracy fell from 0.9333 for
> the last pre-loop revision (`e3a249c1ff8e`, 2025-10-10) to 0.5489 for the revision produced
> by the loop's second autonomous promotion (`2e3ae2505f15`, 2025-10-30): a paired difference
> of −38.4 accuracy points, 95 % CI [−41.9, −34.9] over 2,000 bootstrap resamples. Both
> revisions' weights were verified by SHA-256 against the Hub's published LFS objects and
> declare identical label mappings; the pre-loop revision's measured accuracy differs from the
> figure reported in the repository's own notebook by one example out of 900. Over the same
> interval the pipeline emitted two commits recording the changes as improvements at
> `ΔF1 >= 0.0010`, and a character-n-gram logistic regression trained on the same 4,200-row
> training split scores 0.9400 on that same held-out test split — 39.1 points above the
> promoted model.**

**Why each hostile-review line fails against it.** *"Different evaluation sets"* — one set,
regenerated from the repository's own code at its own seed, held out for both. *"Static
analysis is not runtime evidence"* — these are the deployed weights, hash-verified against
the Hub. *"You may be measuring the wrong artifacts"* — the revisions are pinned by SHA and
the urgency corpus attribution is confirmed by card arithmetic independent of timestamps.
*"No error bars"* — a paired bootstrap CI on identical rows. *"Self-reported metrics from the
broken pipeline"* — not one number in the sentence comes from the pipeline's own reporting.
*"n=1"* — conceded in the limitations, and §7 records what the prevalence search did and did
not find.

**Companion sentence for the gate**, which needs no sample size at all:

> At the shipped `correct_ratio = 0.5`, the incumbent's measured accuracy on the retraining
> pool is exactly 1/3 whatever its true accuracy, and a challenger that reproduces the
> incumbent on the pseudo-labelled third needs to classify 0.15 % of the incumbent's known
> errors correctly to be promoted; simulating the comparison as implemented, the acceptance
> probability is 1.000 for challengers up to thirty accuracy points worse than the incumbent.

---

## 7. Prevalence (Task F) — what it does and does not license

Timeboxed and sampled, not exhaustive. Full records in `results/prevalence.json`.

The **ingredients are common** in public code. One project —
[`ismaeeeelshaikh/logguard`](https://github.com/ismaeeeelshaikh/logguard) — composes all
three and executes them autonomously on a weekly Airflow schedule: a corpus drawn entirely
from analyst verdicts on the incumbent's own alerts, promotion at `new_f1 > prod_f1 + 0.01`
through the MLflow registry, and a "no production model yet — promote directly" branch that
is an unconditional-acceptance analogue of `model_pipeline.py:443`. A second
([`Youssef-Guiga/spam-detection-api`](https://github.com/Youssef-Guiga/spam-detection-api))
reproduces the evaluation defect exactly — incumbent and challenger both scored on splits of
the same feedback pool — but leaves deployment as a code comment. A third
([`Kaif-Anwar/PakSentinel`](https://github.com/Kaif-Anwar/PakSentinel)) automates promotion
on metrics from separate runs. One project
([`aarushigupta22/drift-doctor`](https://github.com/aarushigupta22/drift-doctor)) does it
correctly, on a common held-out window, and is included as a negative control.

**What this licenses:** the composition is reachable by ordinary engineering practice and is
not unique to this system. **What it does not license:** any base rate. The search was
timeboxed and sampled, and the Hugging Face search API does not index commit messages — the
most diagnostic signal — so that probe cannot detect the pattern even where it exists. That
is a limitation of the method, not evidence of absence.

**The paper remains an n=1 artifact-complete post-mortem.** No other project was found in
which the corpus-construction code, the gate code, the *versioned* corpora the gate ran on,
the timestamped promotion decisions, and the resulting model artifacts are all publicly
inspectable. That combination is what makes this case measurable.

---

## 8. What remains NOT ESTABLISHED

Ten items, all in `EVIDENCE_LEDGER.md`. The ones that matter most:

- `q` and `d` — the incumbent's true accuracy on deployment traffic and the admin reporting
  rate. Unmeasurable in principle by this pipeline.
- The realised `a_M`, `a_C` at each promotion — `model_metadata.json` ships `eval_metrics: {}`
  on every deployed version, exactly as the `getattr` defect at `model_pipeline.py:522`
  predicts. Confirmed on both deployed models.
- Which corpus the five department promotions were trained on. No published revision predates
  any of them.
- *Why* the urgency and department corpora are byte-identical in text. The per-task SQL
  filters differ and the API permits single-field flagging, so neither the code nor the API
  forces it; the database state is gone.
- Whether the gate ever *rejected* a candidate. W&B logs are referenced throughout the code
  and are not public.
- Real citizen traffic. Say "deployed pipeline", never "production traffic".
- Which endpoint the backend actually served at any given time (P15, department half).
- Byte-identity of the PRE revision to the notebook's weights — though the measured accuracy
  matches to one example out of 900.

---

## 9. Deliverables

```
paper1-evidence/
├── EVIDENCE_LEDGER.md            106 claim rows, every value read from results/*.json
├── FINDINGS.md                   this file
├── analysis/gate_model.md         the derivation
├── data/gold_slice.csv            216 gold rows with partition labels
├── figures/fig_gate_acceptance.{pdf,png}
├── artifacts/                     73 archived items + MANIFEST.json (SHA256 + UTC each)
│   ├── commits_*.json             4 full commit histories
│   ├── model_cards/               PRE, POST, department HEAD
│   ├── promotion_cards/           the card at each of the 7 auto-deploy commits
│   ├── datasets/                  9 dataset revisions × (README, metadata, 3 parquets)
│   └── local_repo_provenance.json git HEAD 9b37728, clean
├── results/                       revisions · repro_section24 · dataset_revisions ·
│                                  promotion_ledger · gate_model · baselines_ci ·
│                                  x1_results · x1_table.md · prevalence
└── scripts/                       11 scripts + 1 shell fetcher, deterministic
```

Run order from a clean shell:

```
python fetch_artifacts.py          # archive + MANIFEST
python pin_revisions.py            # Task A.1
python repro_section24.py          # harness validation (must print ALL REPRODUCED)
python task_b_dataset_revisions.py # Task B
python promotion_ledger.py         # per-promotion metrics + corroboration
python gate_simulation.py          # Task C + figure
python baselines_ci.py             # Task D
python write_prevalence.py         # Task F
bash fetch_model_revisions.sh <dir>   # ~2.2 GB
python x1_eval.py <dir>            # Task A
python build_evidence_ledger.py    # regenerate the ledger last
```

**Determinism, verified rather than asserted.** All seven computation scripts
(`repro_section24`, `task_b_dataset_revisions`, `promotion_ledger`, `gate_simulation`,
`baselines_ci`, `write_prevalence`, `x1_eval`) were re-run from a clean shell and their
outputs diffed against the first run: **every one is byte-identical**, including
`x1_results.json` and `x1_table.md` — so the −38.4-point result reproduces exactly, weights
re-hashed, on an independent execution. The three network-facing scripts (`fetch_artifacts`,
`pin_revisions`, `fetch_model_revisions.sh`) are idempotent and re-hash rather than
re-download, but their records carry retrieval timestamps that necessarily change between
runs; that is by design, since the timestamps are part of the evidence.

Every 40-character commit SHA appearing anywhere in `scripts/`, `analysis/` or the two
top-level reports was checked against the archived commit histories and `MANIFEST.json`:
three distinct SHAs, all verified, none fabricated.

**Stopping here. The paper itself is Prompt 2.**
