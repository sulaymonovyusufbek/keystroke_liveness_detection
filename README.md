# Keystroke liveness detection: human typing vs. synthetic and replayed typing

Can a detector tell **a real person typing** from **typing that a program generated or replayed**, using only key
press and release times? This repository contains the complete, leak-free evaluation pipeline:
- the data preparation;
- six attackers, from simple Gaussian bots to replays of real recordings;
- six detectors: a one-class SVM, an Isolation Forest, a Random Forest, an autoregressive GRU density model, a
  supervised BiGRU and a key-pair lookup model;
- the diagnostics and the bootstrap confidence intervals.

The human data is the Aalto **136M Keystrokes** corpus (Dhakal et al., CHI 2018). The corpus is **not** included here,
see [Data](#data).

Everything is deterministic (one global seed, 42). Every design decision is written down in a spec under `docs/`
**before** it was run. Phases 3b and 4 were pre-registered by SHA-256 hash. The single best overview of the project,
its decisions and all results is **[`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md)**.

---

## Contents

| phase | what it does | spec | scripts | report |
|---|---|---|---|---|
| 1 | Data sampling and cleaning, user-disjoint split, 23 session features, **D1** One-Class SVM trained on humans only, attack-free hyper-parameter selection, single test-split evaluation, sanity check vs. Gaussian bot A1, replica of the leaky "legacy" protocol | [`PHASE1_SPEC.md`](docs/PHASE1_SPEC.md) (incl. Revisions 1–2) | `scripts/01…07`, `99` | `outputs/full/METHODS.md` |
| 2 | Attackers **A1–A5b** fitted on a disjoint attacker pool; detectors **D2** (Isolation Forest) and **D3** (Random Forest); experiments E1 open set, E2 cross-attack, E3 leave-one-attacker-out | [`PHASE2_SPEC.md`](docs/PHASE2_SPEC.md) | `scripts/10…13` | `outputs/full/phase2/RESULTS_PHASE2.md`, `ATTACKS.md` |
| 3a | Diagnostics: centrality, feature-group ablation, permutation importance, per-user consistency (ICC) | [`PHASE3A_SPEC.md`](docs/PHASE3A_SPEC.md) | `scripts/20…21` | `outputs/full/phase3a/DIAGNOSTICS.md` |
| 3b | Sequence detectors **D4** (autoregressive GRU mixture density, humans only) and **D5** (supervised BiGRU), 3 seeds; E4–E6 | [`PHASE3B_PREREGISTRATION.md`](docs/PHASE3B_PREREGISTRATION.md) | `scripts/30…33`, `run_phase3b.py` | `outputs/full/phase3b/RESULTS_PHASE3B.md` |
| 3c | Post-hoc artefact check of D4/D5 (keycode realism, key/timing swap, identical-key pairs, letters-only score) | [`PHASE3C_SPEC.md`](docs/PHASE3C_SPEC.md) | `scripts/40…41` | `outputs/full/phase3c/DIAGNOSTICS_PHASE3C.md` |
| 4 | User-cluster bootstrap CIs for all headline numbers, identical-key subset with CIs, new detector **D6** (key-pair timing lookup) | [`PHASE4_SPEC.md`](docs/PHASE4_SPEC.md) | `scripts/phase4_run.py`, `phase4_report.py` | [`docs/PHASE4_RESULTS.md`](docs/PHASE4_RESULTS.md) |

Answers to specific implementation questions (attacker details, D4 masking, reference set, bootstrap rules,
pre-registration timeline) are in [`docs/PHASE4B_ANSWERS.md`](docs/PHASE4B_ANSWERS.md).

**Attackers:**
- A1: independent Gaussian timings.
- A2: per-digraph empirical timings.
- A3: A2 plus a per-session speed factor.
- A4: A3 plus pauses, typos with Backspace corrections and Shift overlap.
- A5a: verbatim replay of a real recording.
- A5b: a real session's timing transplanted onto the target sentence.

All of them are described in `outputs/full/phase2/ATTACKS.md`.

---

## What is in this repository and what is not

| included | not included (regenerate with the pipeline) |
|---|---|
| all source code (`src/kl/`), scripts, tests, configs | the Aalto corpus (`data/`, 18 GB; licence, see below) |
| every spec, pre-registration, freeze record and report (`docs/`) | keystroke / session / feature parquet files (derived from the corpus) |
| all result reports, tables, figures and small JSON files under `outputs/` | encoded keystroke sequences (`outputs/*/phase3b/interim/*.npz`, raw corpus timings) |
| | model weights (`*.pt`, `*.joblib`), per-session score files (`*.npz`), logs |

Because the bulky files are not published, the **freeze guards** (which hash every output file) and the few tests
that read saved scores **skip or fail on a fresh clone until you have rerun the pipeline**. The unit tests on
synthetic data run everywhere.

---

## Setup

Tested with Python 3.11.16, PyTorch 2.11.0 + CUDA 12.8, on Ubuntu 26.04 with an NVIDIA RTX 3050 Laptop GPU (6 GB),
12 CPU cores and 14 GB RAM. Exact versions: `docs/environment/requirements-lock.txt`, `docs/environment/environment.txt`.

```bash
git clone https://github.com/sulaymonovyusufbek/keystroke_liveness_detection.git
cd keystroke_liveness_detection
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r docs/environment/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
#   (or the minimum versions: python -m pip install -r requirements.txt)
pytest -q                     # unit tests on synthetic data; no dataset needed
```

Phases 1–3a and 4 run on the CPU. Phase 3b (training D4/D5) needs a CUDA GPU in practice; it also runs on the CPU, but
much more slowly.

## Data

1. Download the **136M Keystrokes** dataset from Aalto University (<https://userinterfaces.aalto.fi/136Mkeystrokes/>).
   It is free for **non-commercial** research; cite Dhakal et al. (2018), see below.
2. Unpack it so that the participant files are at `data/raw/files/`. You should have 168,593 `*_keystrokes.txt` files
   plus `metadata_participants.txt` and `readme.txt`.
3. Optional check: `sha256sum data/raw/files/metadata_participants.txt` should be
   `3ba22e2c23dbcbb5645197b9dd68c17e9ef68a1be7170e1bc1f0cd869a68ac6d` (`docs/environment/dataset.txt`).

Traps in the corpus, all of which the code handles:
- The files are latin-1, not UTF-8.
- Fields are unquoted, so quotes inside sentences are literal (`quoting=3`).
- `LETTER` is unreliable for special keys. Keys are identified by `KEYCODE`: 8/46 are corrections, 16 is Shift.
- The metadata layout column is `LAYOUT`, and the keyboard types are `full`, `laptop`, `small` and `on-screen`.

## Reproducing the results

Every script takes `--quick` for a smoke run on 2,000 users (outputs in `outputs/quick/`). Without it, the full run
on 20,000 users writes to `outputs/full/`. Run the quick mode first; it takes a few minutes per phase.

```bash
source .venv/bin/activate

# Phase 1 (CPU, ~10 min full)
for s in 01_sample_and_parse 02_clean_and_sessionize 03_split_users 04_extract_features \
         05_select_hyperparams 06_final_test 07_a1_sanity_check 99_legacy_protocol; do
  python scripts/$s.py            # add --quick for the smoke run
done

# Phase 2 (CPU, ~5 min full)
python scripts/10_fit_attackers.py && python scripts/11_generate_bots.py \
  && python scripts/12_run_experiments.py && python scripts/13_report_phase2.py

# Phase 3a (CPU)
python scripts/20_run_diagnostics.py && python scripts/21_report_diagnostics.py

# Phase 3b (GPU, ~3 h full on an RTX 3050; 39 models)
python scripts/30_build_sequences.py
nohup python scripts/run_phase3b.py > outputs/full/phase3b/launcher.out 2>&1 &
#   always use the launcher: it runs D4 (E4), D5 (E5, E6) and the report strictly in sequence,
#   holds locks, skips finished runs and refuses to touch the test split twice

# Phase 3c (GPU, ~4 min full)
python scripts/40_run_phase3c.py && python scripts/41_report_phase3c.py

# Phase 4 (CPU + a short GPU check, ~6 min full)
python scripts/phase4_run.py && python scripts/phase4_report.py
```

Notes:
- **One test-split use per model.** `06_final_test.py` and the Phase 3b training scripts refuse a second evaluation of
  the test split (`--allow-rerun` overrides; use it only for a fresh reproduction). `phase4_run.py` keeps its own
  ledger for D6.
- **Freeze guards.** Each phase checks, by SHA-256, that the sources, configs and outputs of the earlier phases are
  unchanged (`docs/phase*_frozen_hashes.json`, `docs/phase3b_prereg_hash.txt`, `docs/phase4_prereg_hash.txt`).
  - In your own reproduction the recorded **output** hashes describe the original run. Model training and attacker
    generation are seeded and deterministic, so a match is expected where the computation is deterministic, and the
    guards tell you where it is not (e.g. a different GPU or library version).
  - If you change code or configs on purpose, the guards will stop you. That is their job: record new hashes for your
    own variant.
- **Expected numbers** for comparison are in the reports listed in the table above, all of which are committed. For
  example, D4-primary accepts 0.6% [0.5, 0.6] of A2 bots at τ10, and D1 accepts 5.8% [5.6, 6.0] of A1 bots
  (`docs/PHASE4_RESULTS.md`).

## Repository layout

```
config.yaml, config_phase2.yaml, config_phase3a.yaml, config_phase3b.yaml   # all tunables (Phase 1 config is hashed)
src/kl/            io_aalto, sessions, split, features, model, selection, evaluate, report, config   (Phase 1)
src/kl/attacks/    A1–A5b generators + attacker-pool statistics                                    (Phase 2)
src/kl/phase2/ phase3a/ phase3b/ phase3c/   per-phase logic, evaluation, reports, freeze guards
src/kl/phase4_{guard,bootstrap,d6}.py       Phase 4
scripts/           thin command-line wrappers, numbered in run order
tests/             pytest suite (synthetic data; tests needing saved outputs skip when they are absent)
docs/              specs, pre-registrations, freeze records, PROJECT_STATE.md, results and answers
outputs/full/      result reports, tables and figures of the full run (bulky files not published)
outputs/quick/     the same for the 2,000-user smoke run
outputs/phase4/    Phase 4 tables and reports
```

`docs/README_PHASE1.md` is the original, more detailed Phase 1 guide: per-step outputs and protocol hygiene.

## Protocol in brief

- **User-disjoint split:** 60 / 15 / 15 / 10% of users go to train / validation / test / attacker pool. No person
  appears in two splits.
- **Attack-free model selection:**
  - Thresholds τ5 / τ10 are set on validation humans only, so that 5% / 10% of them are rejected.
  - No attack data is used to select or threshold the one-class detectors; the code raises an error if it is given
    any.
- **Attackers** are fitted only on attacker-pool users and fixed before any detector is run.
- **Metrics:**
  - APCER = attacks accepted.
  - BPCER = humans rejected.
  - ROC-AUC with humans as the positive class.
  - 95% CIs from a 1,000-replicate test-user cluster bootstrap (seed 42); each user carries their sessions and the
    bots matched to them.
- **Disclosures:**
  - Phase 1's selection rule was revised twice after quick-mode runs. Revision 2 was made with post-hoc A1 results
    known.
  - Phase 3c is post hoc.
  - D6 was pre-registered but designed after the main evaluation.

  See `docs/PHASE4B_ANSWERS.md` (B6) and `outputs/full/METHODS.md`.

## Citation and licence

If you use the data, cite:

> Vivek Dhakal, Anna Maria Feit, Per Ola Kristensson, and Antti Oulasvirta. 2018. Observations on Typing from
> 136 Million Keystrokes. In *Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems (CHI '18)*.
> ACM. https://doi.org/10.1145/3173574.3174220

The corpus is distributed by Aalto University for **non-commercial** use only; this repository does not redistribute
it or any keystroke sequences derived from it.
