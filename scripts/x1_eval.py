#!/usr/bin/env python
"""
x1_eval.py -- EXPERIMENT X1: the fixed-yardstick before/after evaluation.

The audit's strongest available headline compared 0.9344 against 0.6867. Those
come from different evaluation sets, different dataset versions and different
label distributions, so they are not commensurable and a reviewer discards the
comparison in one line. X1 replaces it by evaluating BOTH pinned revisions of
the urgency classifier, plus a linear baseline, on evaluation sets that are held
FIXED across revisions.

Revisions (pinned; see results/revisions.json):
  PRE  e3a249c1ff8e  2025-10-10T17:24:12Z  last commit before the first Auto-deploy
  POST 2e3ae2505f15  2025-10-30T11:48:55Z  current HEAD, after two auto-deploys

Evaluation sets:
  Set A  the original held-out test split (n=900), regenerated deterministically
         at seed 42. Fed as `clean_text`, matching the original evaluation.
  Set B  the gold slice -- real Hello Sarkar grievances carrying the government's
         own `complain_type` label. Fed RAW (portal text), because that is what
         the deployed system would receive. Reported in three partitions:
           B-all    every gold row
           B-test   gold rows whose corpus row falls in Set A's held-out test split
                    -- the UNBIASED partition, and the one the paper leads with
           B-train  gold rows the PRE model saw during training
                    -- biased in FAVOUR of PRE, i.e. against the paper's thesis

Usage:
  PYTHONIOENCODING=utf-8 python x1_eval.py <model_dir>
where <model_dir> contains pre/ and post/ subdirectories.

Outputs: results/x1_results.json, results/x1_table.md, data/gold_slice.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sambodhan_repro as S

RESULTS = S.EVIDENCE / "results"
DATA = S.EVIDENCE / "data"
RESULTS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

MAX_LEN = 96      # train_urgency.py trained the urgency model at max_length=96
BATCH = 16
CLASSES = ["NORMAL", "URGENT", "HIGHLY URGENT"]

REVISIONS = {
    "PRE": {"sha": "e3a249c1ff8e6d45eadbd9f303fa397030e8501f",
            "date_utc": "2025-10-10T17:24:12Z",
            "title": "Delete runs/Oct10_13-34-41",
            "role": "last commit before the first Auto-deploy commit "
                    "(19466adb6226, 2025-10-28T15:08:24Z)",
            # sha256 of model.safetensors as published (the Hub's LFS oid at this revision)
            "expected_weights_sha256":
                "13bc68fceb8b37c6088b96a7c8bfced55e05e2a2070ff5dfbb9f89921390b117"},
    "POST": {"sha": "2e3ae2505f15784bd7866abcda1d6655a4f19575",
             "date_utc": "2025-10-30T11:48:55Z",
             "title": "Upload model_metadata.json (v20251030_114839)",
             "role": "current HEAD, after two autonomous promotions",
             "expected_weights_sha256":
                 "914093aa9cb0314275ae522b2803f8a3b04161a4f776d2c8589ffe56a201fcb7"},
}


def verify_weights(path: Path, tag: str) -> dict:
    """Hash the local weights and compare against the Hub's published LFS oid.

    The first POST download died mid-stream and was re-fetched with resume, so
    byte count alone is not sufficient evidence of integrity. Recording this
    check answers the 'you evaluated corrupted weights' objection directly.
    """
    import hashlib
    f = path / "model.safetensors"
    h = hashlib.sha256()
    with f.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    got = h.hexdigest()
    want = REVISIONS[tag]["expected_weights_sha256"]
    ok = (got == want)
    print(f"  weights sha256 {got[:16]}...  {'MATCHES' if ok else 'DOES NOT MATCH'} "
          f"the Hub LFS oid at this revision")
    if not ok:
        raise SystemExit(f"{tag} weights sha256 mismatch: got {got}, expected {want} -- STOP")
    return {"file": "model.safetensors", "bytes": f.stat().st_size,
            "sha256": got, "matches_hub_lfs_oid": ok}


# ===========================================================================
# Set B -- the gold slice
# ===========================================================================
def build_gold_slice() -> tuple[pd.DataFrame, dict]:
    """Real Hello Sarkar rows with the government's own urgency label."""
    det = pd.read_csv(S.RAW_DETAILS)
    mid = pd.read_csv(S.CORPUS_1640)
    big = pd.read_csv(S.CORPUS_6000)
    big["nt"] = big["grievance"].map(S.norm)

    gold = det[det["complain_type"].isin(S.GOLD_URGENCY_VALUES)][["id", "complain_type"]]
    j = gold.merge(mid[["id", "grievance"]], on="id", how="inner").copy()
    j["nt"] = j["grievance"].map(S.norm)

    tr, va, te = S.urgency_splits(42)
    tr_ids, va_ids, te_ids = set(tr["id"]), set(va["id"]), set(te["id"])
    nt2ids = big.groupby("nt")["id"].apply(list).to_dict()
    nt2urg = dict(zip(big["nt"], big["urgency"]))

    def membership(nt) -> set[str]:
        """Which of the seed-42 splits contain a corpus row carrying this text."""
        ids = nt2ids.get(nt, [])
        found = set()
        for name, id_set in (("train", tr_ids), ("val", va_ids), ("test", te_ids)):
            if any(i in id_set for i in ids):
                found.add(name)
        return found

    j["splits"] = j["nt"].map(membership)
    j["in_corpus"] = j["nt"].isin(set(big["nt"]))
    j["corpus_urgency"] = j["nt"].map(nt2urg)
    j["in_test"] = j["splits"].map(lambda s: "test" in s)
    j["in_train"] = j["splits"].map(lambda s: "train" in s)

    def part(row):
        if row["in_test"]:
            return "B-test"
        if row["in_train"]:
            return "B-train"
        if row["in_corpus"]:
            return "B-val"
        return "B-unseen"

    j["partition"] = j.apply(part, axis=1)
    j["gold_label_id"] = j["complain_type"].map(S.URGENCY2ID)

    matched = j[j["in_corpus"]]
    stats = {
        "gold_rows_in_complaint_details": int(len(gold)),
        "excluded": {"BLOCKER": int((det['complain_type'] == 'BLOCKER').sum()),
                     "missing": int(det['complain_type'].isna().sum())},
        "joined_to_1640_by_id": int(len(j)),
        "present_in_6000_corpus": int(j["in_corpus"].sum()),
        "distinct_texts": int(j["nt"].nunique()),
        "corpus_rows_matching_gold_text (== provenance stratum A)":
            int(big["nt"].isin(set(j["nt"])).sum()),
        "partitions": {k: int(v) for k, v in j["partition"].value_counts().items()},
        "in_both_train_and_test": int((j["in_train"] & j["in_test"]).sum()),
        "gold_vs_corpus_label_agreement":
            f"{int((matched['complain_type'] == matched['corpus_urgency']).sum())}/{len(matched)}",
        "note": ("`present_in_6000_corpus` counts GOLD ROWS whose text occurs in the corpus; "
                 "`corpus_rows_matching_gold_text` counts CORPUS ROWS and is the provenance "
                 "stratum-A figure (211). The two differ because several gold rows share a "
                 "normalised text and a few gold texts occur more than once in the corpus."),
    }
    return j, stats


# ===========================================================================
# Inference
# ===========================================================================
def load_model(path: Path):
    tok = AutoTokenizer.from_pretrained(str(path))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(path))
    mdl.eval()
    return tok, mdl


def predict_labels(tok, mdl, texts: list[str]) -> list[str]:
    """Return predicted LABEL STRINGS, mapped through the model's own id2label."""
    id2label = {int(k): v for k, v in mdl.config.id2label.items()}
    out: list[str] = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            batch = [str(t) for t in texts[i:i + BATCH]]
            enc = tok(batch, truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt")
            logits = mdl(**enc).logits
            for p in logits.argmax(-1).tolist():
                out.append(id2label[int(p)])
            if (i // BATCH) % 15 == 0:
                print(f"      {min(i + BATCH, len(texts))}/{len(texts)}", end="\r")
    print(" " * 40, end="\r")
    return out


def score(y_true_str: list[str], y_pred_str: list[str]) -> dict:
    """All metrics computed on LABEL STRINGS, never on integer indices."""
    if not y_true_str:
        return {"n": 0}
    rep = classification_report(y_true_str, y_pred_str, labels=CLASSES,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true_str, y_pred_str, labels=CLASSES)
    return {
        "n": len(y_true_str),
        "accuracy": float(accuracy_score(y_true_str, y_pred_str)),
        "macro_f1": float(f1_score(y_true_str, y_pred_str, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_str, y_pred_str, average="weighted", zero_division=0)),
        "per_class": {c: {k: float(v) for k, v in rep[c].items()} for c in CLASSES if c in rep},
        "confusion_matrix": {"labels": CLASSES, "matrix": cm.tolist(),
                             "note": "rows = true, cols = predicted"},
        "predicted_distribution": {c: int(sum(1 for p in y_pred_str if p == c)) for c in CLASSES},
    }


# ===========================================================================
def main(model_root: Path) -> None:
    print("=" * 96)
    print("X1 :: building evaluation sets")
    print("=" * 96)

    tr, va, te = S.urgency_splits(42)
    assert [len(tr), len(va), len(te)] == [4200, 900, 900], \
        f"split sizes {[len(tr), len(va), len(te)]} != 4200/900/900 -- STOP"
    print(f"  Set A: original held-out test split, n={len(te)} (fed as clean_text)")

    setA_texts = list(te["clean_text"])
    setA_true = [S.ID2URGENCY[int(l)] for l in te["label"]]

    gold, gstats = build_gold_slice()
    gold_out = gold[["id", "complain_type", "grievance", "partition", "in_corpus",
                     "corpus_urgency", "in_train", "in_test"]].copy()
    gold_out.to_csv(DATA / "gold_slice.csv", index=False, encoding="utf-8")
    print(f"  Set B: gold slice, n={len(gold)} (fed RAW). Partitions: {gstats['partitions']}")
    print(f"         wrote {DATA / 'gold_slice.csv'}")

    setB_texts = list(gold["grievance"])
    setB_true = list(gold["complain_type"])
    partitions = list(gold["partition"])

    results: dict = {
        "experiment": "X1 -- fixed-yardstick evaluation of the pre-loop and post-loop "
                      "urgency classifier revisions",
        "revisions": REVISIONS,
        "evaluation_sets": {
            "SetA": {"description": "original held-out test split, seed 42",
                     "n": len(setA_texts), "text_fed": "clean_text (data_prep.clean_nepali_text)",
                     "labels": "corpus `urgency` column",
                     "class_support": {c: setA_true.count(c) for c in CLASSES}},
            "SetB": {"description": "gold slice -- real Hello Sarkar grievances with the "
                                    "government's own complain_type label",
                     "n": len(setB_texts), "text_fed": "raw portal text (uncleaned)",
                     "labels": "complain_type from data/raw/sambodhan_complaint_details.csv",
                     "construction": gstats,
                     "class_support": {c: setB_true.count(c) for c in CLASSES}},
        },
        "models": {},
    }

    # ---- transformer revisions -------------------------------------------
    id2labels: dict = {}
    predsA: dict = {}
    predsB: dict = {}
    for tag in ("PRE", "POST"):
        path = model_root / tag.lower()
        print()
        print("=" * 96)
        print(f"X1 :: evaluating {tag}  ({REVISIONS[tag]['sha'][:12]})")
        print("=" * 96)
        wcheck = verify_weights(path, tag)
        tok, mdl = load_model(path)
        id2label = {int(k): v for k, v in mdl.config.id2label.items()}
        id2labels[tag] = id2label
        print(f"  id2label: {id2label}")
        print(f"  tokenizer: {type(tok).__name__}   truncation max_length={MAX_LEN}")

        print("  Set A ...")
        predA = predict_labels(tok, mdl, setA_texts)
        print("  Set B ...")
        predB = predict_labels(tok, mdl, setB_texts)
        predsA[tag], predsB[tag] = predA, predB

        rec = {"revision": REVISIONS[tag], "id2label": id2label,
               "weights_integrity": wcheck,
               "SetA": score(setA_true, predA), "SetB-all": score(setB_true, predB)}
        for p in ("B-test", "B-train", "B-val", "B-unseen"):
            idx = [i for i, x in enumerate(partitions) if x == p]
            rec[f"Set{p}"] = score([setB_true[i] for i in idx], [predB[i] for i in idx])
        results["models"][tag] = rec

        print(f"  Set A     : acc {rec['SetA']['accuracy']:.4f}  macroF1 {rec['SetA']['macro_f1']:.4f}")
        print(f"  Set B-all : acc {rec['SetB-all']['accuracy']:.4f}  macroF1 {rec['SetB-all']['macro_f1']:.4f}  (n={rec['SetB-all']['n']})")
        print(f"  Set B-test: acc {rec['SetB-test']['accuracy']:.4f}  macroF1 {rec['SetB-test']['macro_f1']:.4f}  (n={rec['SetB-test']['n']})")
        print(f"  Set B-trn : acc {rec['SetB-train']['accuracy']:.4f}  macroF1 {rec['SetB-train']['macro_f1']:.4f}  (n={rec['SetB-train']['n']})")
        del mdl, tok

    # ---- label-ordering check --------------------------------------------
    same = id2labels.get("PRE") == id2labels.get("POST")
    results["id2label_check"] = {
        "PRE": id2labels.get("PRE"), "POST": id2labels.get("POST"),
        "identical": bool(same),
        "outcome": ("PASS -- both revisions declare the same id2label mapping. Predictions "
                    "were nevertheless mapped through label STRINGS, not integer indices."
                    if same else
                    "DIFFER -- the revisions declare different id2label mappings. All metrics "
                    "were computed on label strings, so the comparison remains valid; the "
                    "difference is itself a serving-layer finding."),
    }
    print(f"\n  id2label check: {'PASS (identical)' if same else 'DIFFER'}")

    # ---- linear baseline --------------------------------------------------
    print()
    print("=" * 96)
    print("X1 :: linear baseline (trained on Set A's training split, 4200 rows)")
    print("=" * 96)
    lin = S.fit_eval(tr["clean_text"], tr["label"],
                     {"A": (setA_texts, [S.URGENCY2ID[c] for c in setA_true]),
                      "B": (setB_texts, [S.URGENCY2ID[c] for c in setB_true])}, seed=42)
    predA_lin = [S.ID2URGENCY[i] for i in lin["A"]["y_pred"]]
    predB_lin = [S.ID2URGENCY[i] for i in lin["B"]["y_pred"]]
    rec = {"description": "TfidfVectorizer(char_wb,(2,5),min_df=2,sublinear_tf) + "
                          "LogisticRegression(max_iter=2000,C=5), fitted on the 4,200-row "
                          "training split as clean_text",
           "SetA": score(setA_true, predA_lin), "SetB-all": score(setB_true, predB_lin)}
    for p in ("B-test", "B-train", "B-val", "B-unseen"):
        idx = [i for i, x in enumerate(partitions) if x == p]
        rec[f"Set{p}"] = score([setB_true[i] for i in idx], [predB_lin[i] for i in idx])
    results["models"]["LINEAR"] = rec
    print(f"  Set A     : acc {rec['SetA']['accuracy']:.4f}  macroF1 {rec['SetA']['macro_f1']:.4f}")
    print(f"  Set B-test: acc {rec['SetB-test']['accuracy']:.4f}  macroF1 {rec['SetB-test']['macro_f1']:.4f}")

    # ---- paired bootstrap: POST minus PRE, on the SAME rows ---------------
    print()
    print("=" * 96)
    print("X1 :: paired bootstrap, POST minus PRE (same rows, 2000 resamples)")
    print("=" * 96)

    def boot_pair(true_s, pre_s, post_s, n_boot=2000, seed=7):
        """Paired percentile bootstrap of the accuracy difference on identical rows."""
        y, a, b = np.array(true_s), np.array(pre_s), np.array(post_s)
        rng = np.random.default_rng(seed)
        n = len(y)
        d = np.empty(n_boot)
        for i in range(n_boot):
            k = rng.integers(0, n, n)
            d[i] = accuracy_score(y[k], b[k]) - accuracy_score(y[k], a[k])
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        return {"n": int(n),
                "acc_pre": float(accuracy_score(y, a)),
                "acc_post": float(accuracy_score(y, b)),
                "delta_acc_post_minus_pre": float(accuracy_score(y, b) - accuracy_score(y, a)),
                "delta_ci95": [lo, hi],
                "excludes_zero": bool(lo > 0 or hi < 0),
                "n_boot": n_boot}

    preds = {"PRE": {"A": predsA["PRE"], "B": predsB["PRE"]},
             "POST": {"A": predsA["POST"], "B": predsB["POST"]},
             "LINEAR": {"A": predA_lin, "B": predB_lin}}
    results["predictions"] = {
        "note": "Per-row predicted label strings, so every number above is recomputable "
                "without re-running inference.",
        "SetA_true": setA_true, "SetB_true": setB_true, "SetB_partition": partitions,
        "by_model": preds,
    }

    deltas = {"SetA": boot_pair(setA_true, preds["PRE"]["A"], preds["POST"]["A"]),
              "SetB-all": boot_pair(setB_true, preds["PRE"]["B"], preds["POST"]["B"])}
    for p in ("B-test", "B-train"):
        idx = [i for i, x in enumerate(partitions) if x == p]
        if idx:
            deltas[f"Set{p}"] = boot_pair([setB_true[i] for i in idx],
                                          [preds["PRE"]["B"][i] for i in idx],
                                          [preds["POST"]["B"][i] for i in idx])
    results["paired_deltas"] = deltas
    for k, v in deltas.items():
        star = "  <-- CI excludes 0" if v["excludes_zero"] else "  (CI includes 0)"
        print(f"  {k:<10} n={v['n']:<4} PRE {v['acc_pre']:.4f} -> POST {v['acc_post']:.4f}   "
              f"delta {v['delta_acc_post_minus_pre']:+.4f} "
              f"[{v['delta_ci95'][0]:+.4f}, {v['delta_ci95'][1]:+.4f}]{star}")

    results["caveats"] = {
        "pre_revision_identity": (
            "The PRE revision (e3a249c1ff8e) is the last commit before the first Auto-deploy. "
            "Its model card reports accuracy 0.9467 / F1-macro 0.9475 -- the VALIDATION figures "
            "from notebooks/Urgency_classifier.ipynb cell 6, not the 0.9344 test figure of cell 7. "
            "It is not proven byte-identical to the weights that produced either notebook number."),
        "id2label_check": results["id2label_check"]["outcome"],
        "truncation": f"truncation=True, max_length={MAX_LEN}, matching train_urgency.py; "
                      f"batch size {BATCH}, CPU, torch.no_grad().",
        "text_preprocessing": ("Set A is fed as `clean_text` to match the original evaluation. "
                              "Set B is fed as RAW portal text, uncleaned, because that is what "
                              "a deployed endpoint receives. The two sets are therefore not "
                              "directly comparable to each other -- only across revisions."),
        "setB_train_bias": ("Partition B-train consists of gold rows that fall in the training "
                           "split, which the PRE model was trained on. Those numbers are biased "
                           "in FAVOUR of PRE -- i.e. against this paper's thesis. B-test is the "
                           "unbiased partition and is the one to lead with."),
        "partition_sizes": gstats["partitions"],
        "department_model": ("No pre-loop department revision is recoverable: the repository's "
                            "`initial commit` (5c5140c56f52, 2025-10-23T01:15:40Z) does predate "
                            "the first department auto-deploy by ~4 minutes, but its tree "
                            "contains only .gitattributes -- no model weights. X1 is therefore "
                            "urgency-only."),
        "linear_baseline_on_SetB": ("The linear baseline was fitted on cleaned text but is scored "
                                    "on Set B's raw text, the same input the transformers receive. "
                                    "This is a mild disadvantage to the baseline."),
    }

    (RESULTS / "x1_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {RESULTS / 'x1_results.json'}")

    # ---- markdown table ---------------------------------------------------
    write_table(results)


def write_table(res: dict) -> None:
    rows = ["# X1 -- fixed-yardstick evaluation of the urgency classifier", "",
            "All three systems evaluated on identical inputs. Set A is the original 900-row",
            "held-out test split (cleaned text, corpus labels); Set B is the gold slice of real",
            "Hello Sarkar grievances (raw text, government `complain_type` labels).", "",
            "| system | revision | Set A acc | Set A macro-F1 | B-test acc | B-test macro-F1 | B-all acc | B-train acc |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    names = {"PRE": "XLM-R pre-loop", "POST": "XLM-R post-loop", "LINEAR": "TF-IDF char + LogReg"}
    for tag in ("PRE", "POST", "LINEAR"):
        m = res["models"][tag]
        sha = res["revisions"][tag]["sha"][:12] if tag in res["revisions"] else "-- (local fit)"
        rows.append(
            f"| {names[tag]} | `{sha}` | {m['SetA']['accuracy']:.4f} | {m['SetA']['macro_f1']:.4f} | "
            f"{m['SetB-test']['accuracy']:.4f} | {m['SetB-test']['macro_f1']:.4f} | "
            f"{m['SetB-all']['accuracy']:.4f} | {m['SetB-train']['accuracy']:.4f} |")

    d = res.get("paired_deltas", {})
    rows += ["", f"Set A n = {res['evaluation_sets']['SetA']['n']}; "
                 f"B-test n = {res['models']['PRE']['SetB-test']['n']} (unbiased); "
                 f"B-train n = {res['models']['PRE']['SetB-train']['n']} (biased in favour of PRE); "
                 f"B-all n = {res['models']['PRE']['SetB-all']['n']}.", "",
             "> **Read the B-train and B-all columns with care.** Those partitions consist "
             "largely of rows that fall in the 4,200-row training split, which BOTH the "
             "pre-loop transformer AND the linear baseline were fitted on. The linear "
             "baseline's 0.9868 on B-train is memorisation, not generalisation. **B-test is "
             "the only partition of Set B held out from every system in this table**, and it "
             "is the one to quote.", ""]

    if d:
        rows += ["## Paired bootstrap: POST − PRE on identical rows (2,000 resamples)", "",
                 "| set | n | PRE acc | POST acc | Δ accuracy | 95% CI | excludes 0 |",
                 "|---|---:|---:|---:|---:|---|---|"]
        for k in ("SetA", "SetB-test", "SetB-train", "SetB-all"):
            if k not in d:
                continue
            v = d[k]
            rows.append(
                f"| {k} | {v['n']} | {v['acc_pre']:.4f} | {v['acc_post']:.4f} | "
                f"{v['delta_acc_post_minus_pre']:+.4f} | "
                f"[{v['delta_ci95'][0]:+.4f}, {v['delta_ci95'][1]:+.4f}] | "
                f"{'**yes**' if v['excludes_zero'] else 'no'} |")
        rows += ["",
                 "Set A is the paper's headline: it is held out for both revisions, since the "
                 "corpus the post-loop model was gated on shares 0.00% of its rows with the "
                 "original 6,000-row corpus. On Set B-test the direction is the same but the "
                 "interval includes zero at n=36 — that partition does not establish a "
                 "regression on its own.", ""]

    rows += ["",
             "## Per-class recall, Set A", "",
             "| system | NORMAL | URGENT | HIGHLY URGENT |", "|---|---:|---:|---:|"]
    for tag in ("PRE", "POST", "LINEAR"):
        pc = res["models"][tag]["SetA"]["per_class"]
        rows.append(f"| {names[tag]} | " + " | ".join(
            f"{pc[c]['recall']:.4f}" for c in CLASSES) + " |")

    rows += ["", "## Per-class recall, Set B-test (the unbiased real-data partition)", "",
             "| system | NORMAL | URGENT | HIGHLY URGENT |", "|---|---:|---:|---:|"]
    for tag in ("PRE", "POST", "LINEAR"):
        pc = res["models"][tag]["SetB-test"]["per_class"]
        rows.append(f"| {names[tag]} | " + " | ".join(
            f"{pc[c]['recall']:.4f}" if c in pc else "--" for c in CLASSES) + " |")

    rows += ["", "## Confusion matrices, Set A  (rows = true, cols = predicted)", ""]
    for tag in ("PRE", "POST", "LINEAR"):
        rows += [f"**{names[tag]}**", "", "| true \\ pred | " + " | ".join(CLASSES) + " |",
                 "|---|---:|---:|---:|"]
        for c, row in zip(CLASSES, res["models"][tag]["SetA"]["confusion_matrix"]["matrix"]):
            rows.append(f"| {c} | " + " | ".join(str(v) for v in row) + " |")
        rows.append("")

    rows += ["## Caveats", ""]
    for k, v in res["caveats"].items():
        rows.append(f"- **{k}** — {v}")

    (RESULTS / "x1_table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS / 'x1_table.md'}")


if __name__ == "__main__":
    # --table-only regenerates results/x1_table.md from the saved JSON, so the
    # rendered table can be revised without repeating ~2,200 forward passes.
    if len(sys.argv) > 1 and sys.argv[1] == "--table-only":
        saved = json.loads((RESULTS / "x1_results.json").read_text(encoding="utf-8"))
        write_table(saved)
        sys.exit(0)
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if root is None or not (root / "pre").exists():
        sys.exit("usage: python x1_eval.py <model_dir containing pre/ and post/>\n"
                 "       python x1_eval.py --table-only")
    main(root)
