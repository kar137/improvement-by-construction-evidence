#!/usr/bin/env python
"""
check_release_safety.py -- find anything in the evidence pack that must not be
published.

The paper's Ethics Statement promises that the operational portal records are not
released. That promise is only true if the released directory actually excludes
them, so this checks rather than assumes.

Two categories are flagged:
  BLOCK  files carrying real citizen grievance text recovered from the portal.
  ALLOW-WITH-REASON  the archived retraining corpora, which were already public
         on the model hub before this work and are redistributed as a pinned,
         checksummed snapshot. The paper states this.

Usage: PYTHONIOENCODING=utf-8 python check_release_safety.py
Exit status is 1 if any BLOCK file would be included in a naive publish.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Already public on the model hub before this work; redistribution is disclosed.
ALLOWED_PREFIX = "artifacts/datasets/"

block: list[tuple[str, int]] = []
allowed: list[tuple[str, int]] = []

for p in sorted(ROOT.rglob("*")):
    if not p.is_file():
        continue
    rel = p.relative_to(ROOT).as_posix()
    if p.suffix.lower() in {".png", ".pdf", ".safetensors"}:
        continue
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        continue
    n = len(DEVANAGARI.findall(text))
    # gold_slice.csv is Latin-script in places, so also flag by column name.
    portal_text = n > 50 or ("complain_type" in text and "grievance" in text
                             and p.suffix == ".csv")
    if not portal_text:
        continue
    (allowed if rel.startswith(ALLOWED_PREFIX) else block).append((rel, n))

print("MUST NOT PUBLISH (real portal records recovered from the operational sample):")
if block:
    for rel, n in block:
        print(f"  BLOCK  {rel}   ({n} Devanagari chars)")
else:
    print("  none")

print()
print(f"Redistributable, already public on the model hub ({len(allowed)} files):")
for rel, n in allowed[:4]:
    print(f"  allow  {rel}")
if len(allowed) > 4:
    print(f"  allow  ... and {len(allowed) - 4} more under {ALLOWED_PREFIX}")

print()
print("Recommended .gitignore / exclude list for the public evidence release:")
for rel, _ in block:
    print(f"  {rel}")

sys.exit(1 if block else 0)
