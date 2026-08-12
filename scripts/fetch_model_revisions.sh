#!/usr/bin/env bash
# fetch_model_revisions.sh
# Downloads the two pinned revisions of the urgency classifier used by experiment X1.
# Weights are large (~1.1 GB each) and are staged OUTSIDE paper1-evidence/ so the
# evidence directory stays small enough to ship as supplementary material.
#
# Full SHAs resolved from the archived commit history
# (artifacts/commits_models_kar137_sambodhan-urgency-classifier.json, retrieved 2026-08-11):
#
#   PRE  e3a249c1ff8e6d45eadbd9f303fa397030e8501f  2025-10-10T17:24:12Z
#        last commit before the first Auto-deploy commit (19466adb6226, 2025-10-28T15:08:24Z)
#   POST 2e3ae2505f15784bd7866abcda1d6655a4f19575  2025-10-30T11:48:55Z
#        current HEAD, after two autonomous promotions
#
# x1_eval.py re-hashes both model.safetensors and compares against the Hub's
# published LFS oids, so a truncated or corrupted download fails loudly there.
#
# Usage: bash fetch_model_revisions.sh <dest_dir>
set -u

DEST="${1:?usage: fetch_model_revisions.sh <dest_dir>}"
REPO="kar137/sambodhan-urgency-classifier"
PRE="e3a249c1ff8e6d45eadbd9f303fa397030e8501f"
POST="2e3ae2505f15784bd7866abcda1d6655a4f19575"

fetch () {  # fetch <rev> <subdir> <file...>
  local rev="$1"; shift
  local sub="$1"; shift
  mkdir -p "$DEST/$sub"
  for f in "$@"; do
    local out="$DEST/$sub/$f"
    if [ -s "$out" ]; then echo "SKIP  $sub/$f"; continue; fi
    echo "GET   $sub/$f"
    # -C - resumes a partial file and --retry survives a mid-stream reset; the
    # 1.1 GB weight files did drop mid-download on the first attempt.
    curl -L --fail --retry 8 --retry-all-errors --retry-delay 5 -C - \
      --max-time 3600 --speed-time 120 --speed-limit 5000 \
      "https://huggingface.co/$REPO/resolve/$rev/$f" -o "$out" \
      || { echo "FAIL  $sub/$f (leaving partial file for -C - resume on re-run)"; }
  done
}

# PRE has no tokenizer.json (fast-tokenizer file); it ships sentencepiece.bpe.model only.
fetch "$PRE"  pre  config.json model.safetensors sentencepiece.bpe.model \
                   special_tokens_map.json tokenizer_config.json \
                   training_metadata.json README.md
fetch "$POST" post config.json model.safetensors sentencepiece.bpe.model \
                   special_tokens_map.json tokenizer_config.json tokenizer.json \
                   training_metadata.json model_metadata.json README.md

echo "=== sizes ==="
ls -l "$DEST/pre" "$DEST/post"
echo "=== DONE ==="
