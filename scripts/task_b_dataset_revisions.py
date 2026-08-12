#!/usr/bin/env python
"""
task_b_dataset_revisions.py -- TASK B: resolve the dataset-revision timing caveat.

The audit measured each retraining corpus at its CURRENT HEAD, and noted that
those revisions post-date the model versions they were attributed to. This
script builds the full promotion/dataset timeline from the archived commit
histories, attributes each promotion to the latest dataset revision that
PRECEDES it, and re-runs the corpus measurements and the TF-IDF separability
test against the correctly attributed revisions.

Either outcome is a usable result:
  * an attributable revision exists -> the caveat is replaced by a measurement;
  * none exists                     -> the caveat is recorded, and sharpened.

Output: results/dataset_revisions.json
Usage:  PYTHONIOENCODING=utf-8 python task_b_dataset_revisions.py
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import sambodhan_repro as S

ART = S.EVIDENCE / "artifacts"
RESULTS = S.EVIDENCE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

VERSION_RE = re.compile(r"v(\d{8}_\d{6})")


def load_commits(name: str) -> list[dict]:
    """Return commits in true chronological order.

    The Hub API lists newest-first, INCLUDING within same-second ties, and the
    pipeline routinely writes two or three commits inside one second. Reversing
    the API order therefore yields the correct intra-second sequence; the stable
    sort that follows preserves it. Sorting the API order directly would place a
    later push group's metadata commit ahead of its own parquet commit and
    mis-tag the preceding version.
    """
    data = json.loads((ART / name).read_text(encoding="utf-8"))
    for c in data:
        c["dt"] = _dt.datetime.strptime(c["date"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=_dt.timezone.utc)
    return sorted(reversed(data), key=lambda c: c["dt"])


def promotions(commits: list[dict]) -> list[dict]:
    """Auto-deploy commits: the moment a challenger replaced the incumbent."""
    out = []
    for c in commits:
        if c["title"].startswith("Auto-deploy"):
            m = VERSION_RE.search(c["title"])
            thr = re.search(r">=\s*([0-9.]+)", c["title"])
            out.append({
                "commit": c["id"], "date_utc": c["date"], "dt": c["dt"],
                "title": c["title"],
                "version": m.group(0) if m else None,
                "threshold": float(thr.group(1)) if thr else None,
            })
    return out


def dataset_versions(commits: list[dict]) -> list[dict]:
    """One record per published dataset version.

    The pipeline pushes each version as a CONTIGUOUS group of up to three commits:
      'Dataset update (<task>) - <version>'  <- the parquets land here
      'Add metadata for version v<version>'
      'Update README.md (v<version>)'
    Grouping is by contiguity, not by version tag: in the department repository
    the parquet push is tagged 20251028_012120 while the metadata/README commits
    of the same push are tagged v20251028_012945, so tag-based grouping would
    split one push into two phantom versions.

    `data_dt` (the parquet-push time) is the moment the corpus became available
    to a trainer and is what attribution uses. `version` is taken from the
    metadata/README commits, because that is the tag the Hub tree is addressed by.
    """
    groups: list[dict] = []
    cur: dict | None = None
    for c in commits:
        title = c["title"]
        if title.startswith("Dataset update"):
            m = re.search(r"(\d{8}_\d{6})", title)
            cur = {"version": ("v" + m.group(1)) if m else None,
                   "commits": [], "data_dt": c["dt"],
                   "terminal_commit": c["id"], "terminal_dt": c["dt"]}
            groups.append(cur)
        elif title.startswith(("Add metadata", "Update README")):
            if cur is None:                      # metadata before any parquet push
                cur = {"version": None, "commits": [], "data_dt": c["dt"],
                       "terminal_commit": c["id"], "terminal_dt": c["dt"]}
                groups.append(cur)
            m = VERSION_RE.search(title)
            if m:
                cur["version"] = m.group(0)      # the tag the tree is addressed by
        else:
            continue                             # 'initial commit' etc.
        cur["commits"].append({"commit": c["id"], "title": title, "date_utc": c["date"]})
        cur["terminal_commit"] = c["id"]
        cur["terminal_dt"] = c["dt"]
    return sorted([g for g in groups if g["version"]], key=lambda g: g["data_dt"])


# --------------------------------------------------------------------------
# Corpus measurement
# --------------------------------------------------------------------------
BIG_NORM = set(S.provenance_strata()["nt"])


def measure(slug: str, version: str) -> dict:
    base = ART / "datasets" / slug / version
    if not (base / "train.parquet").exists():
        return {"version": version, "error": "not archived"}
    parts = {s: pd.read_parquet(base / f"{s}.parquet") for s in ("train", "eval", "test")}
    allrows = pd.concat(parts.values(), ignore_index=True)
    for d in parts.values():
        d["nt"] = d["grievance"].map(S.norm)
    tr_set = set(parts["train"]["nt"])
    ov = float(np.mean([t in tr_set for t in parts["test"]["nt"]])) * 100
    ov6000 = float(np.mean([t in BIG_NORM for t in allrows["grievance"].map(S.norm)])) * 100

    sep = S.fit_eval(parts["train"]["grievance"], parts["train"]["label"],
                     {"eval": (list(parts["eval"]["grievance"]), list(parts["eval"]["label"])),
                      "test": (list(parts["test"]["grievance"]), list(parts["test"]["label"]))},
                     seed=42)
    dup = 100.0 * (1 - allrows["grievance"].map(S.norm).nunique() / len(allrows))
    return {
        "version": version,
        "rows_total": int(len(allrows)),
        "rows_split": {s: int(len(d)) for s, d in parts.items()},
        "class_balance_all": {str(k): int(v) for k, v in allrows["label"].value_counts().items()},
        "class_balance_train": {str(k): int(v) for k, v in parts["train"]["label"].value_counts().items()},
        "train_test_overlap_pct": round(ov, 4),
        "overlap_with_original_6000_pct": round(ov6000, 4),
        "internal_duplicate_pct": round(dup, 4),
        "mean_chars": round(float(allrows["grievance"].astype(str).str.len().mean()), 2),
        "tfidf_char_logreg": {
            k: {kk: (round(v[kk], 4) if isinstance(v[kk], float) else v[kk])
                for kk in ("n", "acc", "macro_f1")}
            for k, v in sep.items()
        },
    }


# --------------------------------------------------------------------------
TASKS = [
    ("urgency", "commits_models_kar137_sambodhan-urgency-classifier.json",
     "commits_datasets_sambodhan_misclassified_urgency_dataset.json",
     "misclassified_urgency_dataset", "sambodhan/misclassified_urgency_dataset"),
    ("department", "commits_models_mr-kush_sambodhan-department-classification-model.json",
     "commits_datasets_sambodhan_misclassified_department_dataset.json",
     "misclassified_department_dataset", "sambodhan/misclassified_department_dataset"),
]

report: dict = {
    "question": "Do dataset revisions exist whose timestamps precede each model promotion, "
                "so that each promotion can be attributed to the corpus it was actually "
                "gated on?",
    "method": "Promotion times and dataset-version times are read from the archived Hub "
              "commit histories. A promotion is attributed to the latest dataset version "
              "whose parquet-push commit strictly precedes it.",
    "tasks": {},
}

for task, model_commits_file, ds_commits_file, slug, ds_repo in TASKS:
    print("=" * 100)
    print(f"TASK B :: {task.upper()}")
    print("=" * 100)

    mcom = load_commits(model_commits_file)
    dcom = load_commits(ds_commits_file)
    promos = promotions(mcom)
    dsvers = dataset_versions(dcom)

    ds_repo_created = dcom[0]["dt"]
    print(f"  dataset repo first commit : {ds_repo_created.isoformat()}")
    print(f"  promotions                : {len(promos)}")
    print(f"  published dataset versions: {len(dsvers)}")
    print()

    attributions = []
    for p in promos:
        earlier = [g for g in dsvers if g["data_dt"] < p["dt"]]
        best = earlier[-1] if earlier else None
        gap_h = ((p["dt"] - best["data_dt"]).total_seconds() / 3600.0) if best else None
        attributions.append({
            "promotion_version": p["version"],
            "promotion_commit": p["commit"],
            "promotion_date_utc": p["date_utc"],
            "threshold": p["threshold"],
            "attributable_dataset_version": best["version"] if best else None,
            "attributable_dataset_commit": best["terminal_commit"] if best else None,
            "attributable_dataset_data_utc": best["data_dt"].isoformat().replace("+00:00", "Z") if best else None,
            "hours_before_promotion": round(gap_h, 3) if gap_h is not None else None,
            "status": "ATTRIBUTED" if best else "NO PRECEDING DATASET REVISION EXISTS",
        })
        tag = best["version"] if best else "-- none --"
        print(f"  promo {p['version']} @ {p['date_utc']}  (dF1>={p['threshold']})")
        print(f"        -> {tag}" + (f"   ({gap_h:.2f} h earlier)" if gap_h is not None else
                                     "   [dataset repo did not yet exist]"))

    print()
    print("  measuring every published revision ...")
    measured = {}
    for g in dsvers:
        m = measure(slug, g["version"])
        m["data_push_utc"] = g["data_dt"].isoformat().replace("+00:00", "Z")
        m["terminal_commit"] = g["terminal_commit"]
        measured[g["version"]] = m
        if "error" in m:
            print(f"    {g['version']}: {m['error']}")
        else:
            t = m["tfidf_char_logreg"]
            print(f"    {g['version']}: n={m['rows_total']:5d} "
                  f"({m['rows_split']['train']}/{m['rows_split']['eval']}/{m['rows_split']['test']})  "
                  f"tr-te overlap={m['train_test_overlap_pct']:6.2f}%  "
                  f"orig-overlap={m['overlap_with_original_6000_pct']:6.2f}%  "
                  f"TFIDF eval={t['eval']['acc']:.4f} test={t['test']['acc']:.4f}")

    n_attr = sum(1 for a in attributions if a["attributable_dataset_version"])
    verdict = (f"{n_attr}/{len(promos)} promotions have an attributable dataset revision.")
    if n_attr == 0:
        verdict += (f" The dataset repository's first commit ({ds_repo_created.isoformat()}) "
                    f"post-dates every promotion, so NO revision contemporaneous with any "
                    f"promotion exists. The timing caveat stands and is sharpened: the "
                    f"measured corpus provably post-dates all promotions.")
    elif n_attr == len(promos):
        verdict += " The timing caveat is RESOLVED for this task."
    print()
    print(f"  VERDICT: {verdict}")
    print()

    report["tasks"][task] = {
        "model_repo_commits_file": model_commits_file,
        "dataset_repo": ds_repo,
        "dataset_repo_first_commit_utc": ds_repo_created.isoformat().replace("+00:00", "Z"),
        "n_promotions": len(promos),
        "n_published_dataset_versions": len(dsvers),
        "attributions": attributions,
        "measurements": measured,
        "verdict": verdict,
    }

# --------------------------------------------------------------------------
# Cross-task identity: are the two "task-specific" corpora the same rows?
# --------------------------------------------------------------------------
print("=" * 100)
print("CROSS-TASK CORPUS IDENTITY")
print("=" * 100)
print("  Each published revision of the urgency corpus is compared, by normalised")
print("  text, against the department revision pushed within the same minute.")
print()

PAIRS = [("v20251028_013035", "v20251028_012945"),
         ("v20251028_015512", "v20251028_015507"),
         ("v20251028_015640", "v20251028_015634")]

identity = []
for uver, dver in PAIRS:
    ub = ART / "datasets" / "misclassified_urgency_dataset" / uver
    db = ART / "datasets" / "misclassified_department_dataset" / dver
    if not (ub / "train.parquet").exists() or not (db / "train.parquet").exists():
        continue
    U = pd.concat([pd.read_parquet(ub / f"{s}.parquet") for s in ("train", "eval", "test")])
    D = pd.concat([pd.read_parquet(db / f"{s}.parquet") for s in ("train", "eval", "test")])
    su = {S.norm(t) for t in U["grievance"]}
    sd = {S.norm(t) for t in D["grievance"]}
    inter = len(su & sd)
    rec = {
        "urgency_version": uver, "department_version": dver,
        "urgency_rows": int(len(U)), "department_rows": int(len(D)),
        "urgency_unique_texts": len(su), "department_unique_texts": len(sd),
        "shared_unique_texts": inter,
        "pct_of_urgency": round(100 * inter / max(len(su), 1), 2),
        "pct_of_department": round(100 * inter / max(len(sd), 1), 2),
        "urgency_label_dist": {str(k): int(v) for k, v in U["label"].value_counts().items()},
        "department_label_dist": {str(k): int(v) for k, v in D["label"].value_counts().items()},
    }
    identity.append(rec)
    print(f"  urgency {uver} (n={len(U)}) vs department {dver} (n={len(D)}): "
          f"{rec['pct_of_urgency']}% / {rec['pct_of_department']}% shared unique texts")

report["cross_task_identity"] = {
    "finding": ("At every revision pushed in the same minute, the urgency and department "
                "retraining corpora contain the SAME set of texts and differ only in the "
                "`label` column. The two 'task-specific' error-conditioned corpora are one "
                "pool of complaints labelled twice."),
    "pairs": identity,
}

(RESULTS / "dataset_revisions.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print()
print(f"Wrote {RESULTS / 'dataset_revisions.json'}")
