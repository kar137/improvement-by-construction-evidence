#!/usr/bin/env python
"""
pin_revisions.py -- TASK A.1: pin the PRE and POST revisions of both model repos.

PRE  = the last commit before the first `Auto-deploy` commit.
POST = current HEAD.

For the department model the question is whether the `initial commit` predates
the first auto-deploy. It does -- by about four minutes -- but its tree contains
only `.gitattributes`, so there are no pre-loop weights to evaluate. The
distinction matters and is recorded explicitly rather than collapsed into
"no earlier commit exists".

Output: results/revisions.json
Usage:  PYTHONIOENCODING=utf-8 python pin_revisions.py
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess

import sambodhan_repro as S

ART = S.EVIDENCE / "artifacts"
RESULTS = S.EVIDENCE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

REPOS = {
    "urgency": {
        "repo": "kar137/sambodhan-urgency-classifier",
        "commits_file": "commits_models_kar137_sambodhan-urgency-classifier.json",
    },
    "department": {
        "repo": "mr-kush/sambodhan-department-classification-model",
        "commits_file": "commits_models_mr-kush_sambodhan-department-classification-model.json",
    },
}


def tree(repo: str, sha: str) -> list[dict]:
    p = subprocess.run(
        ["curl", "-sL", "--max-time", "180",
         f"https://huggingface.co/api/models/{repo}/tree/{sha}?recursive=true"],
        capture_output=True, text=True)
    try:
        return json.loads(p.stdout)
    except Exception:
        return []


def canonical(repo: str) -> str:
    p = subprocess.run(
        ["curl", "-sL", "-o", "/dev/null", "-w", "%{url_effective}", "--max-time", "120",
         f"https://huggingface.co/api/models/{repo}"], capture_output=True, text=True)
    return p.stdout.strip().rsplit("/api/models/", 1)[-1]


out: dict = {
    "purpose": "Pinned revisions for experiment X1, with a record of why the department "
               "model contributes no PRE revision.",
    "tasks": {},
}

for task, cfg in REPOS.items():
    commits = json.loads((ART / cfg["commits_file"]).read_text(encoding="utf-8"))
    for c in commits:
        c["dt"] = _dt.datetime.strptime(c["date"], "%Y-%m-%dT%H:%M:%S.%fZ")
    chron = sorted(reversed(commits), key=lambda c: c["dt"])

    autos = [c for c in chron if c["title"].startswith("Auto-deploy")]
    first_auto = autos[0]
    before = [c for c in chron if c["dt"] < first_auto["dt"]]
    pre = before[-1] if before else None
    head = chron[-1]

    rec = {
        "repo": cfg["repo"],
        "canonical_repo_after_redirect": canonical(cfg["repo"]),
        "n_commits": len(chron),
        "n_auto_deploy_commits": len(autos),
        "first_auto_deploy": {"commit": first_auto["id"], "date_utc": first_auto["date"],
                              "title": first_auto["title"]},
        "POST": {"commit": head["id"], "date_utc": head["date"], "title": head["title"],
                 "role": "current HEAD"},
    }

    if pre is None:
        rec["PRE"] = None
        rec["pre_status"] = "NO COMMIT PRECEDES THE FIRST AUTO-DEPLOY"
    else:
        files = [f["path"] for f in tree(cfg["repo"], pre["id"]) if f.get("type") == "file"]
        has_weights = any(f.endswith((".safetensors", ".bin")) for f in files)
        rec["PRE"] = {"commit": pre["id"], "date_utc": pre["date"], "title": pre["title"],
                      "role": "last commit before the first Auto-deploy commit",
                      "files_at_revision": files,
                      "contains_model_weights": has_weights}
        gap_min = (first_auto["dt"] - pre["dt"]).total_seconds() / 60.0
        rec["pre_status"] = (
            f"USABLE -- weights present; {gap_min:.1f} minutes before the first auto-deploy"
            if has_weights else
            f"TIMESTAMP PRECEDES BY {gap_min:.1f} MINUTES BUT THE TREE CONTAINS NO MODEL "
            f"WEIGHTS ({', '.join(files)}). No pre-loop revision is recoverable for this task.")

    out["tasks"][task] = rec
    print(f"{task}: {cfg['repo']}")
    print(f"  canonical -> {rec['canonical_repo_after_redirect']}")
    print(f"  PRE  {rec['PRE']['commit'][:12] if rec['PRE'] else '--'}  "
          f"{rec['PRE']['date_utc'] if rec['PRE'] else ''}")
    print(f"       {rec['pre_status']}")
    print(f"  POST {rec['POST']['commit'][:12]}  {rec['POST']['date_utc']}")
    print()

out["x1_scope"] = ("X1 evaluates the urgency classifier only. The department model has no "
                   "recoverable pre-loop revision: its initial commit predates the first "
                   "auto-deploy by roughly four minutes but contains only .gitattributes.")
out["p15_artifact_family_note"] = (
    "The audit flagged (P15) that model_metadata.json targets "
    "hub_model_id `sambodhan/sambodhan_urgency_classifier` while the auto-deploy commits "
    "landed on `kar137/...`. Resolving the Hub redirect shows these are the SAME repository: "
    "`kar137/sambodhan-urgency-classifier` canonically resolves to "
    "`sambodhan/sambodhan_urgency_classifier`. The department repo does not redirect. "
    "This removes the urgency half of the artifact-family ambiguity.")

(RESULTS / "revisions.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
print(f"Wrote {RESULTS / 'revisions.json'}")
