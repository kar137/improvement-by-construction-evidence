# Evidence pack: *Improvement by Construction*

Analysis code, archived artifacts and per-claim evidence ledger for the preprint:

> **Improvement by Construction: An Artifact-Level Post-Mortem of an Autonomous
> Retraining Loop in a Deployed Low-Resource Public-Sector NLP System**
> Karan Bista, Kushal Regmi, Nayan Khusu.

The system under study is public and MIT licensed:
<https://github.com/fuseai-fellowship/Sambodhan-AI-Powered-Grievance-Redressal-System-for-Local-Governance>

Every claim in the paper that cites a file and line refers to commit
`9b37728e2b4a68088300889090627030f8994af2`, the tip of `main` in that repository.
Our analysis was frozen there and the repository was not modified at any point.

---

## Why this exists

The paper argues that a promotion gate whose evaluation pool is built from the
incumbent model's own errors cannot measure improvement. Two objections follow
naturally: that the mechanism is inferred from code that was never executed, and
that the supporting artifacts live in repositories the authors control and could
rewrite. This pack answers both. It archives the external artifacts with
checksums and retrieval timestamps, and it re-measures the published corpora using
a model the pipeline never touched, so that no conclusion depends on a number the
pipeline reported about itself.

## Layout

| path | contents |
|---|---|
| `EVIDENCE_LEDGER.md` | one row per numerical claim in the paper, each tagged `ALREADY DEMONSTRATED`, `SUGGESTED BY EVIDENCE` or `NOT ESTABLISHED`, with the script or artifact that produced it |
| `FINDINGS.md` | the analysis write-up, including the claims this work falsified |
| `analysis/gate_model.md` | the derivation of the gate's acceptance behaviour |
| `scripts/` | every analysis script; each reads from the frozen repository or from `artifacts/` |
| `results/*.json` | machine-readable outputs; the paper's figures read their values from these rather than hard-coding them |
| `artifacts/` | archived external artifacts with `MANIFEST.json` giving URL, resolved commit, SHA-256, byte size and UTC retrieval timestamp for each of 73 items |
| `figures/` | the gate acceptance figure |
| `data/` | derived data, see the exclusion note below |

## Deliberately excluded

`data/gold_slice.csv` is **not** in this repository.

It pairs 216 operational grievances with the government's own severity labels, and
it carries real complaint text. Publishing it would redistribute operational
records in a more usable form than they already exist in, which the paper's ethics
statement says we do not do. `scripts/x1_eval.py` reconstructs it deterministically
from the raw files in the system repository, so the Set B results remain
reproducible for anyone with lawful access to those records.

`scripts/check_release_safety.py` sweeps this directory for portal text and fails
if any such file is present. Run it before adding anything here.

## A disclosure about the archived corpora

`artifacts/datasets/` contains pinned snapshots of the two retraining corpora the
pipeline published to the model hub. They were public before this work, and they
are archived here so that the paper's measurements can be checked against the
exact bytes we measured rather than against whatever the upstream repositories
contain later. Their SHA-256 digests are in `artifacts/MANIFEST.json`.

Three of the archived revisions overlap heavily with the system's original
training corpus, which itself contains rows traceable to real portal grievances.
We archive them because they are the evidence, they were already public, and
omitting them would leave the paper's central corpus measurements uncheckable.
These files retain whatever terms their upstream publication carries; the MIT
licence in this repository covers our own code and analysis outputs.

## Reproducing the results

Requires Python 3.13 with `pandas`, `numpy`, `scikit-learn`, `pyarrow`,
`matplotlib`, and, for the fixed-yardstick experiment only, `torch` and
`transformers`. Clone the system repository next to this one, then:

```bash
cd scripts
python fetch_artifacts.py            # re-archive externals, rebuild MANIFEST.json
python pin_revisions.py              # resolve the two evaluated model revisions
python repro_section24.py            # harness validation: must print ALL REPRODUCED
python task_b_dataset_revisions.py   # attribute each promotion to a corpus revision
python promotion_ledger.py           # per-promotion metrics from the card at each commit
python gate_simulation.py            # gate derivation, simulation and figure
python baselines_ci.py               # multi-seed baselines with bootstrap intervals
python write_prevalence.py           # prevalence search record
bash   fetch_model_revisions.sh DIR  # ~2.2 GB, the only large download
python x1_eval.py DIR                # fixed-yardstick evaluation, verifies weight hashes
python build_evidence_ledger.py      # regenerate the ledger last
```

`repro_section24.py` asserts every previously established value and halts on any
mismatch, so run it before trusting anything downstream. Seven of these scripts,
including `x1_eval.py`, produce byte-identical output when re-run from a clean
shell; the three network-facing ones are idempotent but stamp fresh retrieval
timestamps by design.

## Reading the ledger

The ledger is the contract between the evidence and the paper. If a number
appears in the paper it appears there, with its source. Rows tagged
`NOT ESTABLISHED` are things we could not determine, including the incumbent's
true accuracy on deployment traffic, the fraction of its errors that reviewers
reported, and why the two task corpora contain identical texts. Those appear in
the paper's limitations rather than being quietly omitted.

## Licence

MIT, see `LICENSE`. The archived upstream corpora under `artifacts/datasets/`
retain their own terms as noted above.
