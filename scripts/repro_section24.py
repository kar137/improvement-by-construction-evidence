#!/usr/bin/env python
"""
repro_section24.py -- reproduce every measurement the audit already made, from
a clean shell, and diff each one against its expected value.

This script exists to validate the measurement harness BEFORE any new result
(X1, Task B re-measurement, Task D CIs) is trusted. If anything here disagrees
with the audit by more than rounding, downstream results are suspect.

Reproduces:
  * provenance strata of the 6,000-row corpus            (211 / 1386 / 20 / 4383)
  * split sizes at seed 42                               (4200/900/900, 4800/600/600)
  * verbatim train->test leakage                         (17.44% urgency, 19.17% dept)
  * leakage attribution to stratum D                     (157/157)
  * linear baselines on the original corpus              (0.9400/0.9414, 0.9650/0.9648)
  * leak-free variants                                   (0.9273 n=743, 0.9567 n=485)
  * retraining-corpus measurements at the audit's revisions
  * TF-IDF separability of both retraining corpora       (dept 1.0000, urgency 0.9187)

Output: results/repro_section24.json
Usage:  PYTHONIOENCODING=utf-8 python repro_section24.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sambodhan_repro as S

RESULTS = S.EVIDENCE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
ART = S.EVIDENCE / "artifacts"

# The revisions the audit measured (Section 2.4 of the task spec).
AUDIT_URGENCY_REV = "v20251030_115250"
AUDIT_DEPT_REV = "v20251028_015634"

EXPECTED = {
    "strata": {"A": 211, "B": 1386, "C": 20, "D": 4383},
    "urgency_splits": [4200, 900, 900],
    "dept_splits": [4800, 600, 600],
    "urgency_test_leak_pct": 17.44,
    "urgency_test_leak_n": 157,
    "dept_test_leak_pct": 19.17,
    "dept_test_leak_n": 115,
    "urgency_leak_all_from_D": True,
    "urgency_base": {"acc": 0.9400, "f1": 0.9414},
    "dept_base": {"acc": 0.9650, "f1": 0.9648},
    "urgency_leakfree": {"acc": 0.9273, "n": 743},
    "dept_leakfree": {"acc": 0.9567, "n": 485},
    "urgency_corpus_rows": [1600, 1280, 160, 160],
    "dept_corpus_rows": [2426, 1940, 243, 243],
    "urgency_corpus_traintest_overlap_pct": 7.50,
    "dept_corpus_traintest_overlap_pct": 13.58,
    "urgency_corpus_overlap_with_6000_pct": 99.6,
    "dept_corpus_overlap_with_6000_pct": 0.0,
    "dept_corpus_tfidf": {"eval_acc": 1.0000, "test_acc": 1.0000},
    "urgency_corpus_tfidf": {"eval_acc": 0.9187, "test_acc": 0.9187},
}

out: dict = {"expected": EXPECTED, "observed": {}, "diffs": []}


def check(label, observed, expected, tol=0.01):
    """Record a comparison. tol is absolute, in the units of the quantity."""
    ok = None
    if expected is None:
        ok = None
    elif isinstance(expected, bool) or isinstance(observed, bool):
        ok = bool(observed) == bool(expected)
    elif isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        ok = abs(float(observed) - float(expected)) <= tol
    else:
        ok = observed == expected
    status = "OK " if ok else ("--" if ok is None else "MISMATCH")
    print(f"  [{status}] {label:52s} observed={observed!r:28s} expected={expected!r}")
    if ok is False:
        out["diffs"].append({"item": label, "observed": observed, "expected": expected})
    return ok


print("=" * 100)
print("1. PROVENANCE STRATA OF THE 6,000-ROW CORPUS")
print("=" * 100)
strat = S.provenance_strata()
counts = strat["stratum"].value_counts().to_dict()
obs_strata = {k: int(counts.get(k, 0)) for k in "ABCD"}
out["observed"]["strata"] = obs_strata
for k in "ABCD":
    check(f"stratum {k}", obs_strata[k], EXPECTED["strata"][k], tol=0)
out["observed"]["strata_pct"] = {k: round(100 * v / len(strat), 2) for k, v in obs_strata.items()}
print(f"  stratum D = {out['observed']['strata_pct']['D']}% of the corpus")

print()
print("=" * 100)
print("2. SPLIT SIZES AT SEED 42")
print("=" * 100)
u_tr, u_va, u_te = S.urgency_splits(42)
d_tr, d_ev, d_te = S.dept_splits(42)
check("urgency train/val/test", [len(u_tr), len(u_va), len(u_te)], EXPECTED["urgency_splits"], tol=0)
check("department train/eval/test", [len(d_tr), len(d_ev), len(d_te)], EXPECTED["dept_splits"], tol=0)
out["observed"]["urgency_splits"] = [len(u_tr), len(u_va), len(u_te)]
out["observed"]["dept_splits"] = [len(d_tr), len(d_ev), len(d_te)]
assert [len(u_tr), len(u_va), len(u_te)] == [4200, 900, 900], "urgency splits deviate -- STOP"
assert [len(d_tr), len(d_ev), len(d_te)] == [4800, 600, 600], "department splits deviate -- STOP"

print()
print("=" * 100)
print("3. VERBATIM TRAIN->TEST LEAKAGE")
print("=" * 100)
u_leak = S.leakage(u_tr["clean_text"], u_te["clean_text"])
u_leak_val = S.leakage(u_tr["clean_text"], u_va["clean_text"])
d_leak = S.leakage(d_tr["grievance"], d_te["grievance"])
d_leak_ev = S.leakage(d_tr["grievance"], d_ev["grievance"])
check("urgency test leakage n", u_leak["n_leaked"], EXPECTED["urgency_test_leak_n"], tol=0)
check("urgency test leakage %", u_leak["pct"], EXPECTED["urgency_test_leak_pct"], tol=0.01)
check("department test leakage n", d_leak["n_leaked"], EXPECTED["dept_test_leak_n"], tol=0)
check("department test leakage %", d_leak["pct"], EXPECTED["dept_test_leak_pct"], tol=0.01)
out["observed"]["leakage"] = {
    "urgency_test": {k: u_leak[k] for k in ("n_test", "n_leaked", "pct")},
    "urgency_val": {k: u_leak_val[k] for k in ("n_test", "n_leaked", "pct")},
    "dept_test": {k: d_leak[k] for k in ("n_test", "n_leaked", "pct")},
    "dept_eval": {k: d_leak_ev[k] for k in ("n_test", "n_leaked", "pct")},
}

# Attribution: which stratum do the leaked urgency test rows come from?
# Join on the corpus's own `id` column (unique across all 6,000 rows), NOT on text:
# the split frames carry `clean_text` while the stratification carries raw text, and
# for a handful of rows cleaning changes the string enough to break a text join.
id2stratum = dict(zip(strat["id"], strat["stratum"]))
leaked_ids = [i for i, m in zip(u_te["id"], u_leak["mask"]) if m]
leak_strata = pd.Series([id2stratum.get(i, "UNMATCHED") for i in leaked_ids]).value_counts().to_dict()
leak_strata = {k: int(v) for k, v in leak_strata.items()}
out["observed"]["urgency_leak_by_stratum"] = leak_strata
print(f"  leaked urgency test rows by stratum: {leak_strata}")
check("all urgency leakage from stratum D",
      leak_strata.get("D", 0) == u_leak["n_leaked"], EXPECTED["urgency_leak_all_from_D"])

print()
print("=" * 100)
print("4. LINEAR BASELINES ON THE ORIGINAL CORPUS, SEED 42")
print("=" * 100)
ur = S.fit_eval(u_tr["clean_text"], u_tr["label"],
                {"test": (list(u_te["clean_text"]), list(u_te["label"]))}, seed=42)["test"]
check("urgency TF-IDF char acc", round(ur["acc"], 4), EXPECTED["urgency_base"]["acc"], tol=0.0005)
check("urgency TF-IDF char macro-F1", round(ur["macro_f1"], 4), EXPECTED["urgency_base"]["f1"], tol=0.0005)

de = S.fit_eval(d_tr["grievance"], d_tr["label"],
                {"test": (list(d_te["grievance"]), list(d_te["label"]))}, seed=42)["test"]
check("department TF-IDF char acc", round(de["acc"], 4), EXPECTED["dept_base"]["acc"], tol=0.0005)
check("department TF-IDF char macro-F1", round(de["macro_f1"], 4), EXPECTED["dept_base"]["f1"], tol=0.0005)

# Leak-free variants: score the same fitted models on the non-leaked test rows only.
u_clean_idx = ~u_leak["mask"]
d_clean_idx = ~d_leak["mask"]
u_lf = S.fit_eval(u_tr["clean_text"], u_tr["label"],
                  {"lf": (list(np.array(u_te["clean_text"])[u_clean_idx]),
                          list(np.array(u_te["label"])[u_clean_idx]))}, seed=42)["lf"]
d_lf = S.fit_eval(d_tr["grievance"], d_tr["label"],
                  {"lf": (list(np.array(d_te["grievance"])[d_clean_idx]),
                          list(np.array(d_te["label"])[d_clean_idx]))}, seed=42)["lf"]
check("urgency leak-free n", u_lf["n"], EXPECTED["urgency_leakfree"]["n"], tol=0)
check("urgency leak-free acc", round(u_lf["acc"], 4), EXPECTED["urgency_leakfree"]["acc"], tol=0.0005)
check("department leak-free n", d_lf["n"], EXPECTED["dept_leakfree"]["n"], tol=0)
check("department leak-free acc", round(d_lf["acc"], 4), EXPECTED["dept_leakfree"]["acc"], tol=0.0005)

out["observed"]["baselines_seed42"] = {
    "urgency_test": {k: ur[k] for k in ("n", "acc", "macro_f1", "weighted_f1")},
    "department_test": {k: de[k] for k in ("n", "acc", "macro_f1", "weighted_f1")},
    "urgency_leakfree": {k: u_lf[k] for k in ("n", "acc", "macro_f1")},
    "department_leakfree": {k: d_lf[k] for k in ("n", "acc", "macro_f1")},
}

print()
print("=" * 100)
print("5. RETRAINING CORPORA AT THE AUDIT'S REVISIONS")
print("=" * 100)
big_norm = set(S.provenance_strata()["nt"])


def measure_corpus(slug: str, version: str, label_col: str = "label") -> dict:
    """Measure one published revision of a retraining corpus.

    The published parquets carry exactly two columns: `grievance` (text) and
    `label` (already integer-encoded). No label mapping is applied.
    """
    base = ART / "datasets" / slug / version
    parts = {s: pd.read_parquet(base / f"{s}.parquet") for s in ("train", "eval", "test")}
    allrows = pd.concat(parts.values(), ignore_index=True)
    text_col = "grievance" if "grievance" in allrows.columns else allrows.columns[0]
    for d in parts.values():
        d["nt"] = d[text_col].map(S.norm)
    tr_set = set(parts["train"]["nt"])
    ov = float(np.mean([t in tr_set for t in parts["test"]["nt"]])) * 100
    ov6000 = float(np.mean([t in big_norm for t in allrows[text_col].map(S.norm)])) * 100
    labels = allrows[label_col].value_counts().to_dict() if label_col in allrows.columns else {}
    return {
        "version": version,
        "columns": list(allrows.columns),
        "rows_total": int(len(allrows)),
        "rows_split": {s: int(len(d)) for s, d in parts.items()},
        "train_test_overlap_pct": round(ov, 4),
        "overlap_with_original_6000_pct": round(ov6000, 4),
        "class_balance": {str(k): int(v) for k, v in labels.items()},
        "mean_chars": round(float(allrows[text_col].astype(str).str.len().mean()), 2),
        "_parts": parts, "_text_col": text_col,
    }


u_corp = measure_corpus("misclassified_urgency_dataset", AUDIT_URGENCY_REV)
d_corp = measure_corpus("misclassified_department_dataset", AUDIT_DEPT_REV)

check("urgency corpus rows", [u_corp["rows_total"]] + [u_corp["rows_split"][s] for s in ("train", "eval", "test")],
      EXPECTED["urgency_corpus_rows"], tol=0)
check("dept corpus rows", [d_corp["rows_total"]] + [d_corp["rows_split"][s] for s in ("train", "eval", "test")],
      EXPECTED["dept_corpus_rows"], tol=0)
check("urgency corpus train-test overlap %", u_corp["train_test_overlap_pct"],
      EXPECTED["urgency_corpus_traintest_overlap_pct"], tol=0.02)
check("dept corpus train-test overlap %", d_corp["train_test_overlap_pct"],
      EXPECTED["dept_corpus_traintest_overlap_pct"], tol=0.02)
check("urgency corpus overlap with 6000 %", u_corp["overlap_with_original_6000_pct"],
      EXPECTED["urgency_corpus_overlap_with_6000_pct"], tol=0.15)
check("dept corpus overlap with 6000 %", d_corp["overlap_with_original_6000_pct"],
      EXPECTED["dept_corpus_overlap_with_6000_pct"], tol=0.05)

print()
print("=" * 100)
print("6. TF-IDF SEPARABILITY OF THE RETRAINING CORPORA")
print("=" * 100)


def tfidf_on_corpus(corp: dict, label_col: str = "label") -> dict:
    """Train the linear baseline on a retraining corpus's OWN published train split."""
    parts, tc = corp["_parts"], corp["_text_col"]
    y = {s: parts[s][label_col] for s in parts}
    return S.fit_eval(parts["train"][tc], y["train"],
                      {"eval": (list(parts["eval"][tc]), list(y["eval"])),
                       "test": (list(parts["test"][tc]), list(y["test"]))}, seed=42)


u_sep = tfidf_on_corpus(u_corp)
d_sep = tfidf_on_corpus(d_corp)
check("dept corpus TF-IDF eval acc", round(d_sep["eval"]["acc"], 4), EXPECTED["dept_corpus_tfidf"]["eval_acc"], tol=0.0005)
check("dept corpus TF-IDF test acc", round(d_sep["test"]["acc"], 4), EXPECTED["dept_corpus_tfidf"]["test_acc"], tol=0.0005)
check("urgency corpus TF-IDF eval acc", round(u_sep["eval"]["acc"], 4), EXPECTED["urgency_corpus_tfidf"]["eval_acc"], tol=0.0005)
check("urgency corpus TF-IDF test acc", round(u_sep["test"]["acc"], 4), EXPECTED["urgency_corpus_tfidf"]["test_acc"], tol=0.0005)

for c in (u_corp, d_corp):
    c.pop("_parts", None)
    c.pop("_text_col", None)
out["observed"]["retraining_corpora"] = {"urgency": u_corp, "department": d_corp}
out["observed"]["retraining_corpora_tfidf"] = {
    "urgency": {k: {kk: v[kk] for kk in ("n", "acc", "macro_f1")} for k, v in u_sep.items()},
    "department": {k: {kk: v[kk] for kk in ("n", "acc", "macro_f1")} for k, v in d_sep.items()},
}

print()
print("=" * 100)
n_bad = len(out["diffs"])
out["n_mismatches"] = n_bad
out["verdict"] = "ALL REPRODUCED" if n_bad == 0 else f"{n_bad} MISMATCH(ES) -- INVESTIGATE"
print(f"VERDICT: {out['verdict']}")
for d in out["diffs"]:
    print(f"   !! {d['item']}: observed {d['observed']} vs expected {d['expected']}")

(RESULTS / "repro_section24.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print(f"\nWrote {RESULTS / 'repro_section24.json'}")
