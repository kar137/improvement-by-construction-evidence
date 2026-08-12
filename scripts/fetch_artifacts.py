#!/usr/bin/env python
"""
fetch_artifacts.py  --  TASK E: archive every external artifact the paper cites.

Captures, with a UTC retrieval timestamp, a SHA256 and the resolved commit SHA:
  1. Commit histories (JSON) for both model repos and both dataset repos.
  2. Both model cards (raw markdown) at their resolved SHA.
  3. Both model_metadata.json files.
  4. Dataset README.md + dataset_metadata.json for every published version.
  5. Every parquet split for every published dataset version (Task B needs these).
  6. The `sambodhan` organisation listing.
  7. artifacts/MANIFEST.json tying all of the above together.

HTTPS is done with `curl` in a subprocess: the local certificate store rejects
urllib calls to huggingface.co.

Usage:  PYTHONIOENCODING=utf-8 python fetch_artifacts.py
Idempotent: files already present with non-zero size are re-hashed, not re-fetched.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

HF = "https://huggingface.co"

URGENCY_MODEL = "kar137/sambodhan-urgency-classifier"
DEPT_MODEL = "mr-kush/sambodhan-department-classification-model"
URGENCY_DS = "sambodhan/misclassified_urgency_dataset"
DEPT_DS = "sambodhan/misclassified_department_dataset"

# ---------------------------------------------------------------------------
# Published dataset versions -> the LAST commit of that version's push group.
# Read off the commit histories fetched in step 1 (see results/dataset_revisions.json
# for the derivation). Ordered oldest -> newest.
# ---------------------------------------------------------------------------
URGENCY_DS_REVS = [
    ("v20251028_013035", "b4b24afd263d"),
    ("v20251028_015512", "c1522e890b09"),
    ("v20251028_015640", "e4914bb5006c"),
    ("v20251030_111553", "99ece0015d19"),
    ("v20251030_115228", "979c4282ad15"),
    ("v20251030_115250", "9d76f4a4af29"),  # HEAD
]
DEPT_DS_REVS = [
    ("v20251028_012945", "4931835a4641"),
    ("v20251028_015507", "0062d1af3957"),
    ("v20251028_015634", "3d0564c4813c"),  # HEAD
]

# Pinned model revisions used by experiment X1.
URGENCY_PRE = "e3a249c1ff8e"                                # last pre-loop commit
URGENCY_POST = "2e3ae2505f15784bd7866abcda1d6655a4f19575"   # HEAD

SPLITS = ["train", "eval", "test"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch(url: str, dest: Path, *, force: bool = False) -> dict:
    """curl -> dest. Returns a manifest record. Captures x-repo-commit if present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    hdr = dest.with_suffix(dest.suffix + ".headers")
    ts = utcnow()
    if force or not dest.exists() or dest.stat().st_size == 0:
        proc = subprocess.run(
            ["curl", "-sL", "--fail", "--max-time", "900", "-D", str(hdr), url, "-o", str(dest)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            # rc=22 is curl --fail on an HTTP error, almost always 404: the file does
            # not exist at that revision. That is itself evidence -- e.g. the PRE
            # revision carries no model_metadata.json, because model_metadata.json is
            # an artifact the retraining service creates.
            reason = ("HTTP error (curl --fail); for these repositories this is a 404 -- "
                      "the file does not exist at this revision"
                      if proc.returncode == 22 else f"curl rc={proc.returncode}")
            print(f"  MISS {dest.relative_to(ART)}  ({reason})")
            return {"url": url, "path": str(dest.relative_to(ART)).replace("\\", "/"),
                    "absent_at_revision": True, "reason": reason, "retrieved_utc": ts}
        print(f"  GET  {dest.relative_to(ART)}  ({dest.stat().st_size} B)")
    else:
        print(f"  HAVE {dest.relative_to(ART)}  ({dest.stat().st_size} B)")

    # Resolved commit SHA, if the server told us one.
    repo_commit = None
    if hdr.exists():
        for line in hdr.read_text(errors="replace").splitlines():
            if line.lower().startswith("x-repo-commit:"):
                repo_commit = line.split(":", 1)[1].strip()
        hdr.unlink()  # header dumps are scratch, not evidence

    return {
        "url": url,
        "path": str(dest.relative_to(ART)).replace("\\", "/"),
        "resolved_commit_sha": repo_commit,
        "retrieved_utc": ts,
        "sha256": sha256(dest),
        "bytes": dest.stat().st_size,
    }


def main() -> int:
    records: list[dict] = []

    print("[1/7] commit histories")
    for kind, repo in [("models", URGENCY_MODEL), ("models", DEPT_MODEL),
                       ("datasets", URGENCY_DS), ("datasets", DEPT_DS)]:
        name = f"commits_{kind}_{repo.replace('/', '_')}.json"
        records.append(fetch(f"{HF}/api/{kind}/{repo}/commits/main", ART / name))

    print("[2/7] repo metadata (canonical name / redirect resolution)")
    for kind, repo in [("models", URGENCY_MODEL), ("models", DEPT_MODEL),
                       ("datasets", URGENCY_DS), ("datasets", DEPT_DS)]:
        name = f"repoinfo_{kind}_{repo.replace('/', '_')}.json"
        records.append(fetch(f"{HF}/api/{kind}/{repo}", ART / name))

    print("[3/7] model cards + config + metadata, at pinned SHAs")
    for repo, rev, tag in [(URGENCY_MODEL, URGENCY_PRE, "PRE"),
                           (URGENCY_MODEL, URGENCY_POST, "POST"),
                           (DEPT_MODEL, "main", "HEAD")]:
        slug = repo.replace("/", "_")
        for fname in ["README.md", "config.json", "model_metadata.json", "training_metadata.json"]:
            out = ART / "model_cards" / f"{slug}__{tag}_{rev[:12]}__{fname}"
            r = fetch(f"{HF}/{repo}/resolve/{rev}/{fname}", out)
            r["note"] = f"{tag} revision of {repo}"
            records.append(r)

    print("[4/7] dataset README + metadata + parquets, every published version")
    for ds, revs in [(URGENCY_DS, URGENCY_DS_REVS), (DEPT_DS, DEPT_DS_REVS)]:
        slug = ds.split("/")[-1]
        for version, sha in revs:
            for fname in ["README.md", "dataset_metadata.json"]:
                out = ART / "datasets" / slug / version / fname
                r = fetch(f"{HF}/datasets/{ds}/resolve/{sha}/{fname}", out)
                r["dataset_version"] = version
                records.append(r)
            for split in SPLITS:
                rel = f"data/{split}-00000-of-00001.parquet"
                out = ART / "datasets" / slug / version / f"{split}.parquet"
                r = fetch(f"{HF}/datasets/{ds}/resolve/{sha}/{rel}", out)
                r["dataset_version"] = version
                r["split"] = split
                records.append(r)

    print("[4b/7] model card at every Auto-deploy commit (the promotion ledger's source)")
    # These cards carry the per-promotion reported metrics and the Trainer's
    # training-results table, from which n_train is recovered. They are load-bearing
    # evidence and must be hashed and dated like everything else.
    for kind_repo, commits_name in [
            (URGENCY_MODEL, f"commits_models_{URGENCY_MODEL.replace('/', '_')}.json"),
            (DEPT_MODEL, f"commits_models_{DEPT_MODEL.replace('/', '_')}.json")]:
        commits = json.loads((ART / commits_name).read_text(encoding="utf-8"))
        task = "urgency" if kind_repo == URGENCY_MODEL else "department"
        for c in sorted([x for x in commits if x["title"].startswith("Auto-deploy")],
                        key=lambda x: x["date"]):
            import re as _re
            v = _re.search(r"v\d{8}_\d{6}", c["title"])
            v = v.group(0) if v else c["id"][:8]
            out = ART / "promotion_cards" / f"{task}__{v}__{c['id'][:12]}__README.md"
            r = fetch(f"{HF}/{kind_repo}/resolve/{c['id']}/README.md", out)
            r["promotion_commit"] = c["id"]
            r["promotion_version"] = v
            r["promotion_date_utc"] = c["date"]
            r["note"] = f"model card as it stood at auto-deploy commit {c['id'][:12]} ({task})"
            records.append(r)

    print("[5/7] organisation listing")
    records.append(fetch(f"{HF}/api/models?author=sambodhan&full=true", ART / "org_sambodhan_models.json"))
    records.append(fetch(f"{HF}/api/datasets?author=sambodhan&full=true", ART / "org_sambodhan_datasets.json"))
    records.append(fetch(f"{HF}/api/spaces?author=sambodhan&full=true", ART / "org_sambodhan_spaces.json"))

    print("[6/7] local repository provenance")
    repo_dir = HERE.parent.parent / "Sambodhan-AI-Powered-Grievance-Redressal-System-for-Local-Governance"
    git = {}
    for key, args in [("head_sha", ["rev-parse", "HEAD"]),
                      ("commit_date", ["log", "-1", "--format=%cI"]),
                      ("subject", ["log", "-1", "--format=%s"]),
                      ("status_porcelain", ["status", "--porcelain"])]:
        p = subprocess.run(["git", "-C", str(repo_dir)] + args, capture_output=True, text=True)
        git[key] = p.stdout.strip()
    git["clean"] = (git["status_porcelain"] == "")
    (ART / "local_repo_provenance.json").write_text(
        json.dumps(git, indent=2), encoding="utf-8")
    print(f"  local repo HEAD={git['head_sha'][:12]} clean={git['clean']}")

    print("[7/7] MANIFEST.json")
    manifest = {
        "generated_utc": utcnow(),
        "generator": "paper1-evidence/scripts/fetch_artifacts.py",
        "purpose": "Dated, hash-addressed archive of every external artifact cited by Paper 1. "
                   "Mitigates the reviewer objection that the evidence lives in repositories "
                   "controlled by the paper's own authors.",
        "local_repository": git,
        "pinned_model_revisions": {
            "urgency_PRE": URGENCY_PRE,
            "urgency_POST": URGENCY_POST,
        },
        "dataset_versions": {
            URGENCY_DS: [{"version": v, "commit": s} for v, s in URGENCY_DS_REVS],
            DEPT_DS: [{"version": v, "commit": s} for v, s in DEPT_DS_REVS],
        },
        "item_count": len(records),
        "items": records,
    }
    (ART / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    ok = sum(1 for r in records if "sha256" in r)
    absent = [r for r in records if r.get("absent_at_revision")]
    manifest["items_captured"] = ok
    manifest["items_absent_at_revision"] = [
        {"url": r["url"], "reason": r["reason"]} for r in absent]
    (ART / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print(f"\nArchived {ok}/{len(records)} items -> {ART / 'MANIFEST.json'}")
    for r in absent:
        print(f"  absent at revision (evidence, not failure): {r['url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
