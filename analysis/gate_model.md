# The promotion gate, derived

**What this establishes.** The audit argues in prose that the retraining loop's
promotion gate cannot measure improvement. This note replaces the prose with a
derivation. The central result is exact and requires no sample size, no
distributional assumption, and no knowledge of how good either model is:

> At `correct_ratio = r`, the incumbent's measured accuracy on the retraining
> pool is exactly **r / (1 + r)**, whatever its true accuracy. At the shipped
> `r = 0.5` that is **1/3**.

Everything else in this note follows from that.

Companion artifacts: `scripts/gate_simulation.py`, `results/gate_model.json`,
`figures/fig_gate_acceptance.pdf`.

---

## 1. The construction, read off the source

`src/services/prepare_dataset/prepare_pd_df.py:37-77` builds the retraining pool
`P = M ∪ C`:

**Group `M`** — every reviewed misclassification report. The SQL predicate is

```sql
mc.correct_<label> IS NOT NULL
AND mc.model_predicted_<label> IS DISTINCT FROM mc.correct_<label>
```

so membership in `M` *is the statement that the incumbent's prediction differs
from the stored label*. Let `n_mis = |M|`.

**Group `C`** — `n_correct = int(n_mis · r)` rows sampled from `complaints` with
no row in `misclassified_complaints`, labelled with `c.department` / `c.urgency`.
Those columns were written by `src/backend/app/routers/complaints.py:63-86` from
the incumbent's own output. Absence of a human complaint is treated as
confirmation.

With `r = 0.5` (`prepare_pd_df.py:9`), the pool is ⅔ `M` and ⅓ `C`:

| group | weight in `P` | incumbent's accuracy on it | why |
|---|---|---|---|
| `M` | `1/(1+r)` | **0** | by the SQL predicate |
| `C` | `r/(1+r)` | **1** | the label *is* the incumbent's prediction |

## 2. Proposition 1 — the incumbent's measured score is a constant

```
A_inc(r) = (1/(1+r))·0 + (r/(1+r))·1 = r / (1+r)
```

| `r` | `A_inc` |
|---|---|
| 0.25 | 0.2000 |
| **0.5 (shipped)** | **0.3333** |
| 1.0 | 0.5000 |
| 2.0 | 0.6667 |

Neither `q` (the incumbent's true accuracy) nor `d` (the fraction of its errors
that admins report) appears. **That constancy is the finding.** The number the
gate treats as the incumbent's quality is a property of the sampling ratio, and
a perfect model and a coin flip receive the same score.

Two precision notes, so the result is not overstated. First, `r/(1+r)` is exact
**on the pool**; the gate scores the incumbent on the pool's *test split*, which
the pipeline stratifies by label rather than by `M`/`C` membership, so the
identity holds there in expectation with the realised variance of a finite
sample — which is what the simulation in §7 measures. Second, the identity is
stated for **accuracy**; the gate compares **macro-F1**, which is not a linear
function of the per-group accuracies, so §7 simulates macro-F1 rather than
asserting a closed form for it.

## 3. Proposition 2 — what a challenger must clear

Write `a_M`, `a_C` for the challenger's measured accuracy on each group. Then
`A_chal = (a_M + r·a_C)/(1+r)`, and the gate
(`model_pipeline.py:447`, `decision_threshold = τ = 0.001` from `configs.py:42`)
accepts iff `A_chal > A_inc + τ`:

```
(a_M + r·a_C)/(1+r) > r/(1+r) + τ
        ⟺   a_M  >  r·(1 − a_C) + τ·(1 + r)
```

Required `a_M`, by `r` and `a_C`:

| `r` | `a_C = 1.00` | `a_C = 0.95` | `a_C = 0.90` |
|---|---|---|---|
| 0.25 | 0.0013 | 0.0138 | 0.0262 |
| **0.5** | **0.0015** | 0.0265 | 0.0515 |
| 1.0 | 0.0020 | 0.0520 | 0.1020 |
| 2.0 | 0.0030 | 0.1030 | 0.2030 |

**The operating point.** The challenger is trained on `P`'s own train split, so
it is fitted to exactly the label function that defines `C` — which drives
`a_C → 1`. At `r = 0.5, a_C = 1`, the acceptance condition collapses to

```
a_M > 0.0015
```

A challenger that reproduces the incumbent on the pseudo-labelled third and gets
**0.15 %** of the incumbent's known errors right is promoted.

Raising `τ` does not repair this. To demand a genuinely better model the
threshold would have to scale with the incumbent's true error rate `1 − q`,
and `q` is exactly what the pool cannot measure.

## 4. Where `q` and `d` do enter

They do not affect `A_inc`. They set the **label-noise rate of group `C`**.
Among complaints with no misclassification report, incumbent errors survive at
rate `(1−q)(1−d)` and correct predictions at rate `q`, so the fraction of `C`
whose stored label is actually wrong is

```
η(q, d) = (1−q)(1−d) / ( q + (1−q)(1−d) )
```

At `q = 0.85`: `η = 0.150` when `d = 0` (nothing reported), `0.081` at `d = 0.5`,
`0` at `d = 1` (every error reported). So `q` and `d` govern **what the challenger
is taught** — specifically, how much of the corpus instructs it to reproduce the
incumbent's unreported errors — not what the gate measures about the incumbent.

## 5. Observable vs. unobservable

| quantity | status | source |
|---|---|---|
| `r = 0.5` | **OBSERVABLE** | `prepare_pd_df.py:9` |
| `τ = 0.001` | **OBSERVABLE** | `configs.py:42`; echoed in all seven auto-deploy commit messages |
| `deployed_sample_size = 300` | **OBSERVABLE** | `configs.py:41` |
| `q` — incumbent's true accuracy on deployment traffic | **NOT ESTABLISHED** | no ground-truth evaluation on deployment traffic exists anywhere |
| `d` — fraction of incumbent errors admins report | **NOT ESTABLISHED** | would require re-adjudicating unflagged complaints; the database state is gone |
| `a_M`, `a_C` per promotion | **NOT ESTABLISHED** | not recorded; `model_metadata.json` ships `eval_metrics: {}` |

`q` and `d` are not merely unmeasured here — **the pipeline contains no mechanism
that could ever measure them**, because every label it can see is either an
admin correction of a flagged error or the incumbent's own output.

## 6. The gate as implemented, and the two fail-open routes

`model_pipeline.py:400-447` differs from the idealisation above in three ways,
all reflected in the simulation:

1. the comparison is **macro-F1**, not accuracy;
2. the challenger is scored over the pool's **whole test split**, the incumbent
   over `min(300, len(test))` rows of a seed-42 shuffle of that same split
   (at the observed corpus sizes — test splits of 150–243 rows — the 300 cap
   never binds, so both are scored on the same rows);
3. failed endpoint queries return `-1` and are dropped.

**Fail-open route 1 — no endpoint.** If `api_endpoint` is falsy the `if
api_endpoint:` block never runs, `deployed_f1_macro` stays `None`, and line 446
substitutes `0.0`.

**Fail-open route 2 — every query fails.** If all queries return `-1`,
`paired_true` is empty and line 443 sets `deployed_f1_macro = 0.0`.

In both routes the criterion becomes `f1_challenger > 0.001`: **unconditional
acceptance**. This is reachable in ordinary operation — the design pauses its own
Spaces to save compute (`train_model.py` calls `api.pause_space`), so endpoint
unavailability is a normal state, and it converts to auto-approval.

## 7. Simulation result

`scripts/gate_simulation.py` simulates the implemented comparison (macro-F1,
finite test split, 300 trials per point, pool `n = 1500`, 3 classes, `d = 0.5`),
with the challenger modelled as **independent** of the incumbent — the
conservative choice, since training on `C`'s labels would raise `a_C` and make
promotion easier. Reported acceptance probabilities are therefore lower bounds.

**At the shipped `r = 0.5`, the acceptance probability is 1.000 at every
`δ` tested, down to `δ = −0.30`** — a challenger thirty accuracy points worse
than the incumbent is promoted every time. `r = 0.25` behaves identically. Only
at `r = 2.0`, four times the shipped value, does the gate begin to reject clearly
worse challengers (`P = 0.00` at `δ = −0.30`).

Panel B holds `δ = −0.10` and sweeps `q`: at the shipped `r = 0.5` a challenger
ten points worse is promoted with probability ≥ 0.99 for every incumbent with
`q ≥ 0.6`.

**The headline, stated for the paper:** at the operating point the system
actually shipped, the promotion criterion's acceptance probability is
indistinguishable from the fail-open path — the case where the incumbent is not
consulted at all.

## 8. What this does *not* establish

- It does not show that the seven observed promotions degraded the deployed
  models. It shows the criterion under which they were made cannot distinguish
  improvement from degradation. The before/after question is experiment X1.
- `a_M` and `a_C` for the actual promotions are unrecorded; the simulation
  supplies a range, not the realised values.
- The simulation assumes admin corrections in `M` are correct. There is no
  adjudication step in `misclassification.py`, so this is an assumption
  favourable to the pipeline.
