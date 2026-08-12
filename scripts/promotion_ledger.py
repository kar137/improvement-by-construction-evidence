#!/usr/bin/env python
"""
promotion_ledger.py -- build the paper's promotion ledger, and corroborate the
Task B dataset attribution from an INDEPENDENT signal.

For every `Auto-deploy` commit in both model repositories this script fetches the
model card as it stood AT THAT COMMIT and parses the Trainer-generated
"Training results" table. That table leaks the size of the corpus the run was
trained on, in two independent ways:

  (1) steps-per-epoch.  The table prints (step, epoch) pairs. Since
      steps_per_epoch = ceil(n_train / train_batch_size) and epoch = step /
      steps_per_epoch, a printed epoch pins steps_per_epoch exactly, which
      brackets n_train to a 16-row window (batch size 16).

  (2) the accuracy denominator.  Reported accuracy is k / n_eval for integer k,
      so candidate n_eval values that cannot produce the printed accuracy at the
      printed precision are excluded.

Neither signal comes from a timestamp, so together they test the timestamp-based
attribution in results/dataset_revisions.json rather than assuming it.

Output: results/promotion_ledger.json
Usage:  PYTHONIOENCODING=utf-8 python promotion_ledger.py
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from fractions import Fraction
from pathlib import Path

import sambodhan_repro as S

ART = S.EVIDENCE / "artifacts"
RESULTS = S.EVIDENCE / "results"
CARDS = ART / "promotion_cards"
CARDS.mkdir(parents=True, exist_ok=True)

REPOS = {
    "urgency": ("kar137/sambodhan-urgency-classifier",
                "commits_models_kar137_sambodhan-urgency-classifier.json",
                "misclassified_urgency_dataset"),
    "department": ("mr-kush/sambodhan-department-classification-model",
                   "commits_models_mr-kush_sambodhan-department-classification-model.json",
                   "misclassified_department_dataset"),
}
BATCH = 16  # per_device_train_batch_size, from every model card and model_metadata.json

ROW_RE = re.compile(r"^\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
METRIC_RE = re.compile(r"^-\s*([A-Za-z0-9 _]+):\s*([0-9.]+)\s*$", re.M)


def curl(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    p = subprocess.run(["curl", "-sL", "--fail", "--max-time", "300", url, "-o", str(dest)],
                       capture_output=True, text=True)
    return p.returncode == 0


def parse_card(text: str) -> dict:
    """Extract the headline eval metrics and the training-results table."""
    head = text.split("## Model description")[0]
    metrics = {k.strip().lower().replace(" ", "_"): float(v) for k, v in METRIC_RE.findall(head)}
    rows = []
    for line in text.splitlines():
        m = ROW_RE.match(line.strip())
        if m:
            rows.append({"train_loss": float(m.group(1)), "epoch": float(m.group(2)),
                         "step": int(m.group(3)), "val_loss": float(m.group(4)),
                         "accuracy": float(m.group(5))})
    return {"eval_metrics": metrics, "training_rows": rows}


def infer_steps_per_epoch(rows: list[dict]) -> int | None:
    """steps_per_epoch from the printed (step, epoch) pairs; must agree across rows."""
    cands = set()
    for r in rows:
        if r["epoch"] <= 0:
            continue
        # epoch is printed to 4 dp; step/epoch recovers steps_per_epoch
        cands.add(round(r["step"] / r["epoch"]))
    if len(cands) == 1:
        return cands.pop()
    # Rows printed at whole epochs are exact; prefer the modal estimate.
    if cands:
        return max(set(cands), key=list(cands).count)
    return None


def n_train_window(spe: int, batch: int = BATCH) -> tuple[int, int]:
    """ceil(n/batch) == spe  =>  n in ((spe-1)*batch, spe*batch]."""
    return ((spe - 1) * batch + 1, spe * batch)


def eval_n_consistent(acc: float, n_eval: int, dp: int = 4) -> bool:
    """Could an eval set of size n_eval produce this accuracy at `dp` decimals?"""
    for k in range(n_eval + 1):
        if round(k / n_eval, dp) == round(acc, dp):
            return True
    return False


ledger = {
    "method": __doc__.strip().split("\n\n", 1)[1],
    "batch_size_assumed": BATCH,
    "promotions": [],
}

dsrev = json.loads((RESULTS / "dataset_revisions.json").read_text(encoding="utf-8"))

for task, (repo, commits_file, ds_slug) in REPOS.items():
    commits = json.loads((ART / commits_file).read_text(encoding="utf-8"))
    autos = [c for c in commits if c["title"].startswith("Auto-deploy")]
    autos = sorted(autos, key=lambda c: c["date"])

    # Sizes of every archived revision of this task's published corpus.
    meas = dsrev["tasks"][task]["measurements"]
    attr_by_version = {a["promotion_version"]: a
                       for a in dsrev["tasks"][task]["attributions"]}

    print("=" * 100)
    print(f"{task.upper()}  --  {len(autos)} auto-deploy commits")
    print("=" * 100)

    for c in autos:
        ver = re.search(r"v\d{8}_\d{6}", c["title"])
        ver = ver.group(0) if ver else None
        dest = CARDS / f"{task}__{ver}__{c['id'][:12]}__README.md"
        ok = curl(f"https://huggingface.co/{repo}/resolve/{c['id']}/README.md", dest)
        rec = {
            "task": task, "repo": repo, "version": ver,
            "commit": c["id"], "date_utc": c["date"], "title": c["title"],
            "threshold": float(re.search(r">=\s*([0-9.]+)", c["title"]).group(1)),
            "card_archived": ok,
        }
        if ok:
            parsed = parse_card(dest.read_text(encoding="utf-8", errors="replace"))
            rec.update(parsed)
            spe = infer_steps_per_epoch(parsed["training_rows"])
            rec["steps_per_epoch"] = spe
            if spe:
                lo, hi = n_train_window(spe)
                rec["n_train_window"] = [lo, hi]
                # Which archived revisions of this corpus are compatible?
                compat = []
                for v, m in meas.items():
                    if "error" in m:
                        continue
                    ntr, nev = m["rows_split"]["train"], m["rows_split"]["eval"]
                    ok_tr = lo <= ntr <= hi
                    acc = parsed["eval_metrics"].get("accuracy")
                    ok_ev = (acc is None) or eval_n_consistent(acc, nev)
                    if ok_tr and ok_ev:
                        compat.append({"version": v, "n_train": ntr, "n_eval": nev,
                                       "data_push_utc": m["data_push_utc"]})
                rec["revisions_compatible_with_card_arithmetic"] = compat
            a = attr_by_version.get(ver, {})
            rec["timestamp_attribution"] = a.get("attributable_dataset_version")
            rec["timestamp_attribution_status"] = a.get("status")
            comp_versions = [x["version"] for x in rec.get("revisions_compatible_with_card_arithmetic", [])]
            if rec["timestamp_attribution"] and comp_versions:
                rec["corroboration"] = ("CONFIRMED" if rec["timestamp_attribution"] in comp_versions
                                        else "CONFLICT")
            elif comp_versions:
                rec["corroboration"] = "CARD-ARITHMETIC ONLY (no revision predates this promotion)"
            else:
                rec["corroboration"] = "NOT ESTABLISHED"

            acc = parsed["eval_metrics"].get("accuracy")
            print(f"  {ver}  {c['date'][:19]}  dF1>={rec['threshold']}")
            print(f"      reported eval: acc={acc} f1_macro={parsed['eval_metrics'].get('f1_macro')} "
                  f"loss={parsed['eval_metrics'].get('loss')}")
            print(f"      steps/epoch={spe} -> n_train in {rec.get('n_train_window')}")
            print(f"      timestamp attribution : {rec['timestamp_attribution'] or '-- none --'}")
            print(f"      card-arithmetic match : {comp_versions or '-- none --'}")
            print(f"      => {rec['corroboration']}")
        else:
            print(f"  {ver}  card fetch FAILED")
        ledger["promotions"].append(rec)
    print()

RESULTS.mkdir(parents=True, exist_ok=True)
(RESULTS / "promotion_ledger.json").write_text(
    json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {RESULTS / 'promotion_ledger.json'}  ({len(ledger['promotions'])} promotions)")
