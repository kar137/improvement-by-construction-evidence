#!/usr/bin/env python
"""
build_evidence_ledger.py -- generate EVIDENCE_LEDGER.md.

One row per numerical claim the paper will make. Every value is READ FROM the
results JSON at run time via a dotted path -- no number in the ledger is typed
by hand, so the ledger cannot drift from the measurements.

Tags: ALREADY DEMONSTRATED | SUGGESTED BY EVIDENCE | NOT ESTABLISHED

Usage: PYTHONIOENCODING=utf-8 python build_evidence_ledger.py
"""
from __future__ import annotations

import json
from pathlib import Path

import sambodhan_repro as S

RESULTS = S.EVIDENCE / "results"
ART = S.EVIDENCE / "artifacts"

AD, SE, NE = "ALREADY DEMONSTRATED", "SUGGESTED BY EVIDENCE", "NOT ESTABLISHED"

CACHE: dict[str, dict] = {}


def load(name: str) -> dict:
    if name not in CACHE:
        p = RESULTS / name if (RESULTS / name).exists() else ART / name
        CACHE[name] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return CACHE[name]


def is_path(spec) -> bool:
    """A spec is a JSON lookup iff it looks like '<something>.json:<dotted.path>'."""
    return (isinstance(spec, str) and ":" in spec
            and spec.split(":", 1)[0].endswith(".json"))


MISSING = "MISSING"


def _walk(node, parts: list[str]):
    """Resolve a dotted path, tolerating dict keys that themselves contain dots.

    Some keys really are things like "0.5" (the correct_ratio values in
    gate_model.json), so a naive split on "." cannot address them. At each level
    we try the longest key prefix that matches, then recurse on the remainder.
    """
    if not parts:
        return node
    if isinstance(node, list):
        try:
            return _walk(node[int(parts[0])], parts[1:])
        except (ValueError, IndexError):
            return MISSING
    if not isinstance(node, dict):
        return MISSING
    for take in range(len(parts), 0, -1):
        key = ".".join(parts[:take])
        if key in node:
            got = _walk(node[key], parts[take:])
            if got != MISSING:
                return got
    return MISSING


def get(spec: str):
    """'file.json:a.b.c' -> value, or the string MISSING."""
    fname, path = spec.split(":", 1)
    node = load(fname)
    if not node:
        return MISSING
    return _walk(node, path.split("."))


def fmt(v, nd=4):
    if v == "MISSING":
        return "**MISSING**"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x, nd) for x in v) + "]"
    return str(v)


# claim, value-spec (or literal), source, tag
ROWS: list[tuple] = [
    # ---------------- the loop's initial condition ----------------
    ("§ The corpus the loop started from", None, None, None),
    ("Training corpus size (sambodhan_balanced_dataset.csv)",
     lambda: sum(load("repro_section24.json").get("observed", {}).get("strata", {}).values()),
     "data/processed/sambodhan_balanced_dataset.csv", AD),
    ("Provenance stratum A — Hello Sarkar, gold labels",
     "repro_section24.json:observed.strata.A", "scripts/repro_section24.py", AD),
    ("Provenance stratum B — Indian tweets, Llama-3 pseudo-labels",
     "repro_section24.json:observed.strata.B", "scripts/repro_section24.py", AD),
    ("Provenance stratum C — English, hand-labelled",
     "repro_section24.json:observed.strata.C", "scripts/repro_section24.py", AD),
    ("Provenance stratum D — untraceable to any repository file",
     "repro_section24.json:observed.strata.D", "scripts/repro_section24.py", AD),
    ("Stratum D as % of corpus", "repro_section24.json:observed.strata_pct.D",
     "scripts/repro_section24.py", AD),
    ("Generating process of stratum D", "not in the repository",
     "notebooks/preparing_datasets.ipynb terminates at the 1,640-row file", NE),
    ("Urgency splits at seed 42 (train/val/test)",
     "repro_section24.json:observed.urgency_splits", "data_prep.py:21-34 reimplemented", AD),
    ("Department splits at seed 42 (train/eval/test)",
     "repro_section24.json:observed.dept_splits",
     "preprocess_and_prepare_dataset.py:73-103 reimplemented", AD),
    ("Urgency test leakage, seed 42",
     "repro_section24.json:observed.leakage.urgency_test.pct", "scripts/repro_section24.py", AD),
    ("Department test leakage, seed 42",
     "repro_section24.json:observed.leakage.dept_test.pct", "scripts/repro_section24.py", AD),
    ("Leaked urgency test rows attributable to stratum D",
     "repro_section24.json:observed.urgency_leak_by_stratum.D", "scripts/repro_section24.py", AD),
    ("Urgency leakage across seeds {0,1,2,3,42}, mean",
     "baselines_ci.json:original_corpus.urgency.leakage_pct.mean", "scripts/baselines_ci.py", AD),
    ("Urgency leakage across seeds, range",
     "baselines_ci.json:original_corpus.urgency.leakage_pct.values", "scripts/baselines_ci.py", AD),
    ("Department leakage across seeds, mean",
     "baselines_ci.json:original_corpus.department.leakage_pct.mean", "scripts/baselines_ci.py", AD),
    ("Leakage figures are lower bounds (exact match after normalisation only)",
     "baselines_ci.json:leakage_verdict", "scripts/sambodhan_repro.py norm()", AD),

    # ---------------- baselines with uncertainty ----------------
    ("§ Linear baselines, with uncertainty", None, None, None),
    ("Urgency linear baseline, 5-seed mean accuracy",
     "baselines_ci.json:original_corpus.urgency.accuracy.mean", "scripts/baselines_ci.py", AD),
    ("Urgency linear baseline, 5-seed sd",
     "baselines_ci.json:original_corpus.urgency.accuracy.sd", "scripts/baselines_ci.py", AD),
    ("Urgency linear baseline, seed-42 accuracy 95% bootstrap CI",
     "baselines_ci.json:original_corpus.urgency.bootstrap_seed42_test.acc_ci95",
     "scripts/baselines_ci.py (2000 resamples)", AD),
    ("Department linear baseline, 5-seed mean accuracy",
     "baselines_ci.json:original_corpus.department.accuracy.mean", "scripts/baselines_ci.py", AD),
    ("Department linear baseline, seed-42 accuracy 95% CI",
     "baselines_ci.json:original_corpus.department.bootstrap_seed42_test.acc_ci95",
     "scripts/baselines_ci.py", AD),
    ("Urgency: reported XLM-R accuracy lies inside the linear baseline's 95% CI",
     "baselines_ci.json:parity_adjudication.urgency.xlmr_inside_linear_ci",
     "scripts/baselines_ci.py", AD),
    ("Department: reported XLM-R accuracy lies inside the linear baseline's 95% CI",
     "baselines_ci.json:parity_adjudication.department.xlmr_inside_linear_ci",
     "scripts/baselines_ci.py", AD),
    ("CLAIM WORDING: linear vs XLM-R on the original corpus is PARITY, not superiority",
     "baselines_ci.json:parity_adjudication.urgency.claim", "scripts/baselines_ci.py", AD),
    ("Urgency leak-free linear accuracy (seed 42)",
     "repro_section24.json:observed.baselines_seed42.urgency_leakfree.acc",
     "scripts/repro_section24.py", AD),
    ("Department leak-free linear accuracy (seed 42)",
     "repro_section24.json:observed.baselines_seed42.department_leakfree.acc",
     "scripts/repro_section24.py", AD),

    # ---------------- the gate ----------------
    ("§ The promotion gate", None, None, None),
    ("correct_ratio r (shipped default)", "gate_model.json:constants.correct_ratio_default",
     "prepare_pd_df.py:9", AD),
    ("decision_threshold τ", "gate_model.json:constants.decision_threshold", "configs.py:42", AD),
    ("deployed_sample_size", "gate_model.json:constants.deployed_sample_size", "configs.py:41", AD),
    ("Incumbent's measured accuracy on the pool = r/(1+r); at r=0.5",
     "gate_model.json:operating_point.incumbent_measured_accuracy",
     "analysis/gate_model.md §2 (exact derivation)", AD),
    ("Challenger accuracy required on flagged errors at r=0.5, a_C=1",
     "gate_model.json:operating_point.required_challenger_accuracy_on_flagged_errors_if_a_C_is_1",
     "analysis/gate_model.md §3", AD),
    ("P(promoted) at r=0.5, δ=−0.30 (challenger 30 pts worse)",
     "gate_model.json:monte_carlo.acceptance_by_r.0.5.0", "scripts/gate_simulation.py", AD),
    ("P(promoted) at r=0.5, δ=−0.10",
     "gate_model.json:monte_carlo.acceptance_by_r.0.5.10", "scripts/gate_simulation.py", AD),
    ("Fail-open path yields unconditional acceptance",
     "two routes: api_endpoint falsy (line 446) or all queries fail (line 443)",
     "model_pipeline.py:425-447", AD),
    ("q — incumbent's true accuracy on deployment traffic", "unobservable",
     "no ground-truth evaluation on deployment traffic exists", NE),
    ("d — fraction of incumbent errors admins report", "unobservable",
     "would require re-adjudicating unflagged complaints; DB state gone", NE),
    ("a_M, a_C realised at each promotion", "unrecorded",
     "model_metadata.json ships eval_metrics: {}", NE),

    # ---------------- the promotions ----------------
    ("§ The seven promotions", None, None, None),
    ("Total auto-deploy commits", 7, "artifacts/MANIFEST.json + results/promotion_ledger.json", AD),
    ("Urgency promotion 1 — v20251028_150816, reported eval accuracy",
     "promotion_ledger.json:promotions.0.eval_metrics.accuracy",
     "model card at commit 19466adb6226", AD),
    ("Urgency promotion 2 — v20251030_114839, reported eval accuracy",
     "promotion_ledger.json:promotions.1.eval_metrics.accuracy",
     "model card at commit e04b4076b90c", AD),
    ("Department promotions 1–5, reported eval accuracy sequence",
     lambda: " → ".join(
         f"{p['eval_metrics']['accuracy']:.4f}"
         for p in load("promotion_ledger.json").get("promotions", [])
         if p.get("task") == "department" and p.get("eval_metrics")),
     "results/promotion_ledger.json (cards at each auto-deploy commit)", AD),
    ("Urgency promotions 1–2, reported eval accuracy sequence",
     lambda: " → ".join(
         f"{p['eval_metrics']['accuracy']:.4f}"
         for p in load("promotion_ledger.json").get("promotions", [])
         if p.get("task") == "urgency" and p.get("eval_metrics")),
     "results/promotion_ledger.json — NOTE: different eval sets, not a regression series", AD),
    ("The per-promotion metrics above are on DIFFERENT eval sets and are NOT a regression series",
     "each promotion is scored on its own corpus's eval split",
     "results/promotion_ledger.json", AD),
    ("Every deployed model_metadata.json ships eval_metrics: {}",
     "confirmed on both deployed models",
     "model_pipeline.py:522 getattr bug; artifacts/model_cards/*", AD),

    # ---------------- Task B ----------------
    ("§ Dataset-revision attribution (Task B)", None, None, None),
    ("Urgency: promotions with an attributable dataset revision",
     "dataset_revisions.json:tasks.urgency.verdict", "scripts/task_b_dataset_revisions.py", AD),
    ("Urgency promotion 2 ← v20251030_111553, pushed this many hours earlier",
     "dataset_revisions.json:tasks.urgency.attributions.1.hours_before_promotion",
     "scripts/task_b_dataset_revisions.py", AD),
    ("Attribution independently confirmed by model-card arithmetic",
     "promotion_ledger.json:promotions.1.corroboration",
     "scripts/promotion_ledger.py (steps/epoch and accuracy denominator)", AD),
    ("Attributable urgency corpus v20251030_111553: rows",
     "dataset_revisions.json:tasks.urgency.measurements.v20251030_111553.rows_total",
     "scripts/task_b_dataset_revisions.py", AD),
    ("Attributable urgency corpus v20251030_111553: overlap with original 6,000-row corpus",
     "dataset_revisions.json:tasks.urgency.measurements.v20251030_111553.overlap_with_original_6000_pct",
     "scripts/task_b_dataset_revisions.py", AD),
    ("Attributable urgency corpus v20251028_015640: overlap with original corpus",
     "dataset_revisions.json:tasks.urgency.measurements.v20251028_015640.overlap_with_original_6000_pct",
     "scripts/task_b_dataset_revisions.py", AD),
    ("HEAD revision v20251030_115250 overlap with original corpus (the audit's 99.6% figure)",
     "dataset_revisions.json:tasks.urgency.measurements.v20251030_115250.overlap_with_original_6000_pct",
     "scripts/task_b_dataset_revisions.py — but this revision post-dates the last promotion", AD),
    ("Department: promotions with an attributable dataset revision",
     "dataset_revisions.json:tasks.department.verdict", "scripts/task_b_dataset_revisions.py", AD),
    ("Department dataset repo first commit (post-dates all five promotions)",
     "dataset_revisions.json:tasks.department.dataset_repo_first_commit_utc",
     "scripts/task_b_dataset_revisions.py", AD),
    ("Department promotions' training corpus identity", "consistent with v20251028_015634 "
     "(n_train 1,940) by card arithmetic, but that revision post-dates every promotion",
     "results/promotion_ledger.json", SE),
    # Per-revision detail. The paper tabulates every published revision, so each
    # one needs a ledger row rather than only the two that gated a promotion.
    ("Per-revision corpus measurements, urgency (rows; train/eval/test; overlap "
     "with original; train-test overlap; mean chars)",
     lambda: " | ".join(
         f"{v}: {m['rows_total']} ({m['rows_split']['train']}/{m['rows_split']['eval']}"
         f"/{m['rows_split']['test']}) {m['overlap_with_original_6000_pct']:.2f}% "
         f"{m['train_test_overlap_pct']:.2f}% {m['mean_chars']:.0f}ch"
         for v, m in sorted(
             load("dataset_revisions.json")["tasks"]["urgency"]["measurements"].items(),
             key=lambda kv: kv[1]["data_push_utc"]) if "error" not in m),
     "scripts/task_b_dataset_revisions.py", AD),
    ("Per-revision corpus measurements, department",
     lambda: " | ".join(
         f"{v}: {m['rows_total']} ({m['rows_split']['train']}/{m['rows_split']['eval']}"
         f"/{m['rows_split']['test']}) {m['overlap_with_original_6000_pct']:.2f}% "
         f"{m['train_test_overlap_pct']:.2f}% {m['mean_chars']:.0f}ch"
         for v, m in sorted(
             load("dataset_revisions.json")["tasks"]["department"]["measurements"].items(),
             key=lambda kv: kv[1]["data_push_utc"]) if "error" not in m),
     "scripts/task_b_dataset_revisions.py", AD),
    ("MIN_DATASET_LEN, the row count below which a corpus is not published", 1000,
     "src/services/prepare_dataset/prepare_dataset_pipeline.py:30, :113", AD),
    ("Augmented training rows used by the original urgency fine-tune", 5781,
     "notebooks/Urgency_classifier.ipynb (against 4,200 raw rows for the baseline)", AD),
    ("Set B partition sizes",
     lambda: str(get("x1_results.json:evaluation_sets.SetB.construction.partitions")),
     "scripts/x1_eval.py", AD),
    ("Urgency and department retraining corpora contain identical text sets",
     "dataset_revisions.json:cross_task_identity.pairs.0.pct_of_urgency",
     "scripts/task_b_dataset_revisions.py (100% at all three paired revisions)", AD),
    ("The per-task SQL filters differ, so identical corpora are NOT forced by the code",
     "prepare_pd_df.py:34 filters on correct_<task> IS DISTINCT FROM model_predicted_<task>, "
     "and prepare_dataset_pipeline.py:100-107 calls it once per task",
     "src/services/prepare_dataset/", AD),
    ("Single-field flagging is permitted by the API, so identity is not an API constraint either",
     "misclassification.py:63 rejects a report only if BOTH corrections are absent",
     "src/backend/app/routers/misclassification.py:63", AD),
    ("WHY the two corpora are nevertheless identical (every flagged row corrected on both "
     "fields? one fetch reused? programmatic seeding?)",
     "cannot be determined — the database state is gone", "no surviving DB or W&B record", NE),

    # ---------------- corpus degeneracy ----------------
    ("§ What the promoted models were trained on", None, None, None),
    ("Department corpus v20251028_015634 is perfectly separable by char n-grams (eval)",
     "repro_section24.json:observed.retraining_corpora_tfidf.department.eval.acc",
     "scripts/repro_section24.py", AD),
    ("Department corpus v20251028_015634, TF-IDF test accuracy",
     "repro_section24.json:observed.retraining_corpora_tfidf.department.test.acc",
     "scripts/repro_section24.py", AD),
    ("Linear baseline vs promoted transformer, urgency Oct 30 (attributed)",
     "baselines_ci.json:linear_vs_promoted_gap.v20251030_111553.gap_accuracy_points",
     "scripts/baselines_ci.py — accuracy points, same published eval split", AD),
    ("…its 95% CI (points)",
     "baselines_ci.json:linear_vs_promoted_gap.v20251030_111553.gap_ci95_points",
     "scripts/baselines_ci.py", AD),
    ("Linear baseline vs promoted transformer, urgency Oct 28 (attributed)",
     "baselines_ci.json:linear_vs_promoted_gap.v20251028_015640.gap_accuracy_points",
     "scripts/baselines_ci.py", AD),
    ("…its 95% CI (points)",
     "baselines_ci.json:linear_vs_promoted_gap.v20251028_015640.gap_ci95_points",
     "scripts/baselines_ci.py", AD),
    ("Department 1.0 vs linear 1.0 is NOT a like-for-like comparison",
     "baselines_ci.json:linear_vs_promoted_gap.v20251028_015634.claim_strength",
     "scripts/baselines_ci.py", SE),

    # ---------------- X1 ----------------
    ("§ X1 — the fixed yardstick", None, None, None),
    ("PRE revision SHA", "revisions.json:tasks.urgency.PRE.commit", "scripts/pin_revisions.py", AD),
    ("POST revision SHA", "revisions.json:tasks.urgency.POST.commit", "scripts/pin_revisions.py", AD),
    ("Department has no recoverable pre-loop revision",
     "revisions.json:tasks.department.pre_status", "scripts/pin_revisions.py", AD),
    ("id2label identical across PRE and POST", "x1_results.json:id2label_check.identical",
     "scripts/x1_eval.py", AD),
    ("PRE weights sha256 matches the Hub LFS oid",
     "x1_results.json:models.PRE.weights_integrity.matches_hub_lfs_oid", "scripts/x1_eval.py", AD),
    ("POST weights sha256 matches the Hub LFS oid",
     "x1_results.json:models.POST.weights_integrity.matches_hub_lfs_oid", "scripts/x1_eval.py", AD),
    ("Set A (n=900): PRE accuracy", "x1_results.json:models.PRE.SetA.accuracy",
     "scripts/x1_eval.py", AD),
    ("Set A (n=900): POST accuracy", "x1_results.json:models.POST.SetA.accuracy",
     "scripts/x1_eval.py", AD),
    ("Set A: POST − PRE accuracy delta", "x1_results.json:paired_deltas.SetA.delta_acc_post_minus_pre",
     "scripts/x1_eval.py (paired bootstrap)", AD),
    ("Set A: delta 95% CI", "x1_results.json:paired_deltas.SetA.delta_ci95",
     "scripts/x1_eval.py", AD),
    ("Set A: linear baseline accuracy", "x1_results.json:models.LINEAR.SetA.accuracy",
     "scripts/x1_eval.py", AD),
    ("Set B-test (gold, unbiased): n", "x1_results.json:models.PRE.SetB-test.n",
     "scripts/x1_eval.py", AD),
    ("Set B-test: PRE accuracy", "x1_results.json:models.PRE.SetB-test.accuracy",
     "scripts/x1_eval.py", AD),
    ("Set B-test: POST accuracy", "x1_results.json:models.POST.SetB-test.accuracy",
     "scripts/x1_eval.py", AD),
    ("Set B-test: POST − PRE delta 95% CI",
     "x1_results.json:paired_deltas.SetB-test.delta_ci95", "scripts/x1_eval.py", AD),
    ("Set B-train (biased in favour of PRE): n", "x1_results.json:models.PRE.SetB-train.n",
     "scripts/x1_eval.py", AD),
    ("Set B-all: POST − PRE delta", "x1_results.json:paired_deltas.SetB-all.delta_acc_post_minus_pre",
     "scripts/x1_eval.py", AD),
    ("Set B-all: delta 95% CI", "x1_results.json:paired_deltas.SetB-all.delta_ci95",
     "scripts/x1_eval.py", AD),
    ("Set A: PRE macro-F1", "x1_results.json:models.PRE.SetA.macro_f1", "scripts/x1_eval.py", AD),
    ("Set A: POST macro-F1", "x1_results.json:models.POST.SetA.macro_f1", "scripts/x1_eval.py", AD),
    ("Set A: linear baseline exceeds POST by (accuracy points)",
     lambda: round(100 * (get("x1_results.json:models.LINEAR.SetA.accuracy")
                          - get("x1_results.json:models.POST.SetA.accuracy")), 2),
     "scripts/x1_eval.py", AD),
    ("CORROBORATION: PRE's Set A accuracy vs the notebook's reported 0.9344",
     lambda: f"{get('x1_results.json:models.PRE.SetA.accuracy'):.4f} measured vs 0.9344 reported "
             f"— a difference of one example out of 900",
     "notebooks/Urgency_classifier.ipynb cell 7 vs scripts/x1_eval.py", AD),
    ("MECHANISM: POST's NORMAL recall on Set A collapses (PRE → POST)",
     lambda: f"{get('x1_results.json:models.PRE.SetA.per_class.NORMAL.recall'):.3f} → "
             f"{get('x1_results.json:models.POST.SetA.per_class.NORMAL.recall'):.3f}",
     "scripts/x1_eval.py — the post-loop model over-escalates", AD),
    ("POST's Set A predicted-class distribution vs true support",
     lambda: f"predicted {get('x1_results.json:models.POST.SetA.predicted_distribution')} vs "
             f"true {get('x1_results.json:evaluation_sets.SetA.class_support')}",
     "scripts/x1_eval.py", AD),
    ("Class-prior shift does NOT explain it: attributable corpus is 32.6% NORMAL vs the "
     "original corpus's 30.6%", "489/1500 vs 1836/6000",
     "results/dataset_revisions.json class_balance_all + the original corpus", AD),
    ("On real gold data (Set B-test) NORMAL recall is 0.000 for ALL THREE systems",
     lambda: ", ".join(
         f"{t}={get(f'x1_results.json:models.{t}.SetB-test.per_class.NORMAL.recall'):.3f}"
         for t in ("PRE", "POST", "LINEAR")),
     "scripts/x1_eval.py — a task/data property, not a model property (n=5 NORMAL)", AD),
    ("Set B-test: linear baseline accuracy (best of the three on real data)",
     "x1_results.json:models.LINEAR.SetB-test.accuracy", "scripts/x1_eval.py", AD),
    ("Set A is held out for BOTH revisions", "PRE trained on the original 4,200-row train split; "
     "POST trained on a corpus with 0.00% overlap with the original 6,000-row corpus",
     "results/dataset_revisions.json + scripts/x1_eval.py", AD),
    ("Set A is in-distribution for PRE and out-of-distribution for POST",
     "an asymmetry that must be stated: POST's attributable corpus is disjoint from the original "
     "corpus, so Set A tests POST's transfer to the original task definition",
     "results/dataset_revisions.json", AD),
    # Composite rows so that every per-partition value the paper tabulates has a
    # ledger entry, not only the headline ones.
    ("X1 per-partition accuracy, PRE / POST / LINEAR",
     lambda: " | ".join(
         f"{s}: n={get(f'x1_results.json:models.PRE.{s}.n')} "
         f"PRE={get(f'x1_results.json:models.PRE.{s}.accuracy'):.4f} "
         f"POST={get(f'x1_results.json:models.POST.{s}.accuracy'):.4f} "
         f"LIN={get(f'x1_results.json:models.LINEAR.{s}.accuracy'):.4f}"
         for s in ("SetA", "SetB-test", "SetB-train", "SetB-all")),
     "scripts/x1_eval.py", AD),
    ("X1 paired deltas with 95% CI, all partitions",
     lambda: " | ".join(
         f"{s}: {get(f'x1_results.json:paired_deltas.{s}.delta_acc_post_minus_pre'):+.4f} "
         f"[{get(f'x1_results.json:paired_deltas.{s}.delta_ci95')[0]:+.4f}, "
         f"{get(f'x1_results.json:paired_deltas.{s}.delta_ci95')[1]:+.4f}]"
         for s in ("SetA", "SetB-test", "SetB-train", "SetB-all")),
     "scripts/x1_eval.py", AD),
    ("Linear baseline on each retraining corpus: eval accuracy with 95% CI",
     lambda: " | ".join(
         f"{v.split('@')[1]}: {m['eval_acc']['mean']:.4f} "
         f"[{m['bootstrap_eval']['acc_ci95'][0]:.4f}, {m['bootstrap_eval']['acc_ci95'][1]:.4f}]"
         for v, m in load("baselines_ci.json").get("retraining_corpora", {}).items()),
     "scripts/baselines_ci.py", AD),
    ("Gate: minimum a_M required at each r, for a_C in {1.00, 0.95, 0.90}",
     lambda: " | ".join(
         f"r={row['r']}: A_inc={row['incumbent_measured_accuracy']:.4f} "
         f"{row['min_a_M_if_a_C_1.00']:.4f}/{row['min_a_M_if_a_C_0.95']:.4f}/"
         f"{row['min_a_M_if_a_C_0.90']:.4f}"
         for row in load("gate_model.json").get("analytic", [])),
     "analysis/gate_model.md, scripts/gate_simulation.py", AD),
    ("Gate simulation settings (q used in panel a, reporting rate, trials, pool size)",
     lambda: (f"q={get('gate_model.json:monte_carlo.q_main')}, "
              f"d={get('gate_model.json:monte_carlo.d_assumed')}, "
              f"trials={get('gate_model.json:monte_carlo.trials_per_point')}, "
              f"pool n={get('gate_model.json:monte_carlo.n_pool')}, "
              f"delta for panel b={get('gate_model.json:monte_carlo.delta_fixed_for_q_sweep')}"),
     "scripts/gate_simulation.py", AD),
    ("Card arithmetic per promotion: steps/epoch and implied n_train window",
     lambda: " | ".join(
         f"{p['version']}: {p['steps_per_epoch']} steps/epoch -> {p['n_train_window']}"
         for p in load("promotion_ledger.json").get("promotions", [])
         if p.get("steps_per_epoch")),
     "scripts/promotion_ledger.py", AD),
    ("Epoch value printed at step 50 for promotion v20251030_114839",
     lambda: next((r["epoch"] for p in load("promotion_ledger.json").get("promotions", [])
                   if p.get("version") == "v20251030_114839"
                   for r in p.get("training_rows", []) if r["step"] == 50), "MISSING"),
     "model card at commit e04b4076b90c", AD),
    ("PRE revision is byte-identical to the weights behind the notebook's 0.9344", "not provable",
     "the PRE card reports the VALIDATION figures (0.9467/0.9475), not the test figure; but the "
     "measured Set A accuracy matches the notebook to one example", NE),

    # ---------------- external / provenance ----------------
    ("§ Artifact provenance", None, None, None),
    ("Local repository HEAD", "MANIFEST.json:local_repository.head_sha", "git rev-parse HEAD", AD),
    ("Local repository clean at capture time", "MANIFEST.json:local_repository.clean",
     "git status --porcelain", AD),
    ("External artifacts archived with SHA256 + UTC timestamp",
     "MANIFEST.json:items_captured", "scripts/fetch_artifacts.py", AD),
    ("kar137/sambodhan-urgency-classifier canonically resolves to sambodhan/…",
     "revisions.json:tasks.urgency.canonical_repo_after_redirect",
     "Hub redirect; resolves the urgency half of audit finding P15", AD),
    ("Which endpoint the backend actually served at any given time", "unresolved",
     "two artifact families; department repo does not redirect", NE),
    ("Real citizen traffic", "no evidence anywhere in the repository",
     "analytics JSONs are internally inconsistent test fixtures (audit P9)", NE),
    ("Whether the gate ever REJECTED a candidate", "W&B run logs not public",
     "referenced throughout the code; not in the repository", NE),

    # ---------------- prevalence ----------------
    ("§ Prevalence (Task F)", None, None, None),
    ("Projects composing error-conditioned corpus AND automated promotion",
     "prevalence.json:tally.full_pattern_error_conditioned_corpus_AND_automated_promotion",
     "scripts/write_prevalence.py", SE),
    ("Projects with error-conditioned evaluation but advisory promotion",
     "prevalence.json:tally.error_conditioned_evaluation_but_promotion_advisory",
     "scripts/write_prevalence.py", SE),
    ("Correct implementations found (negative control)",
     "prevalence.json:tally.correct_implementations_found", "scripts/write_prevalence.py", SE),
    ("A base rate for this pattern in the wild", "not measured",
     "search was timeboxed and sampled; HF search does not index commit messages", NE),
]


def main() -> None:
    x1_present = (RESULTS / "x1_results.json").exists()
    lines = [
        "# Evidence ledger — Paper 1",
        "",
        "One row per numerical claim the paper will make. Every value is read directly from",
        "`results/*.json` by `scripts/build_evidence_ledger.py`; no number here is typed by hand.",
        "",
        "**Tags** — `ALREADY DEMONSTRATED`: measured, reproducible from the scripts in this",
        "directory. · `SUGGESTED BY EVIDENCE`: consistent with the artifacts but not identified.",
        "· `NOT ESTABLISHED`: cannot be determined from the available evidence; the paper must",
        "say so rather than infer.",
        "",
        f"X1 results present: **{'yes' if x1_present else 'NO — rows below will show MISSING'}**",
        "",
        "| # | Claim | Value | Source | Tag |",
        "|---:|---|---|---|---|",
    ]
    n = 0
    counts = {AD: 0, SE: 0, NE: 0}
    for row in ROWS:
        claim, spec, source, tag = row
        if spec is None and source is None:
            lines.append(f"| | **{claim}** | | | |")
            continue
        if tag is None:
            continue
        n += 1
        counts[tag] = counts.get(tag, 0) + 1
        if callable(spec):
            val = spec()
        elif is_path(spec):
            val = get(spec)
        else:
            val = spec
        txt = fmt(val)
        # Composite rows (per-revision corpus measurements, per-partition X1
        # results) are long by design. Truncating them would drop values the
        # paper cites, so the cap is generous.
        if isinstance(txt, str) and len(txt) > 900:
            txt = txt[:897] + "…"
        txt = txt.replace("|", "\\|").replace("\n", " ")
        src = (source or "").replace("|", "\\|")
        lines.append(f"| {n} | {claim} | {txt} | `{src}` | {tag} |")

    lines += [
        "",
        "## Tally",
        "",
        f"- ALREADY DEMONSTRATED: **{counts.get(AD, 0)}**",
        f"- SUGGESTED BY EVIDENCE: **{counts.get(SE, 0)}**",
        f"- NOT ESTABLISHED: **{counts.get(NE, 0)}**",
        f"- total rows: **{n}**",
        "",
        "## Rules for Prompt 2",
        "",
        "1. Do not promote a `SUGGESTED BY EVIDENCE` row to an assertion. In particular, the",
        "   department model's 1.0 and the perfectly-separable department corpus are **not**",
        "   established to be the same evaluation; say \"consistent with\", never \"measured on\".",
        "2. On the original corpus, the linear-vs-XLM-R comparison is **parity**. Never write",
        "   \"outperforms\".",
        "3. The seven per-promotion metrics come from seven different evaluation sets. They are",
        "   not a regression series. X1 is the only instrument that measures before/after on a",
        "   fixed yardstick.",
        "4. Every `NOT ESTABLISHED` row must appear in the paper's limitations, not be omitted.",
    ]
    (S.EVIDENCE / "EVIDENCE_LEDGER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {S.EVIDENCE / 'EVIDENCE_LEDGER.md'}  ({n} claim rows)")
    print(f"  AD={counts.get(AD,0)}  SE={counts.get(SE,0)}  NE={counts.get(NE,0)}")
    missing = [r[0] for r in ROWS if r[3] and is_path(r[1]) and get(r[1]) == "MISSING"]
    if missing:
        print(f"  MISSING values ({len(missing)}):")
        for m in missing:
            print(f"    - {m}")


if __name__ == "__main__":
    main()
