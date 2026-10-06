# Keystroke liveness detection — Phase 1

Reproducible pipeline that separates human typing from machine-generated typing with a One-Class SVM
trained on **human data only** (Aalto 136M Keystrokes, Dhakal et al., CHI 2018). The full protocol is in
[`docs/PHASE1_SPEC.md`](docs/PHASE1_SPEC.md); everything is driven by [`config.yaml`](config.yaml) (one global seed, 42).

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate     # Python 3.11
pip install -r requirements.txt
pytest                                                     # 46 unit / integration tests, no dataset needed
```

The dataset must be at `data/raw/files/` (168,593 `*_keystrokes.txt` files + `metadata_participants.txt` + `readme.txt`).

## Run, step by step

Add `--quick` to any script for a small run (2,000 users, outputs in `outputs/quick/`); without it the run uses
`sampling.n_users` (20,000) and writes to `outputs/full/`.

| step | command | what it does | main outputs (`outputs/<mode>/`) |
|---|---|---|---|
| 1 | `python scripts/01_sample_and_parse.py` | keeps `KEYBOARD_TYPE ∈ {full, laptop}`, samples users with the seed, parses their files robustly | `interim/keystrokes.parquet`, `results/cleaning_report.json` |
| 2 | `python scripts/02_clean_and_sessionize.py` | session = (PARTICIPANT_ID, TEST_SECTION_ID); drop rules counted separately | `interim/sessions.parquet`, `interim/session_index.parquet` |
| 3 | `python scripts/03_split_users.py` | user-level 60/15/15/10 split (train/val/test/attacker_pool), train session cap | `results/split_manifest.json` |
| 4 | `python scripts/04_extract_features.py` | 23 features per session (same code for humans and bots); logs min/median/max of every `*_cv` on train | `interim/features_{split}.parquet`, `FEATURES.md` |
| 5 | `python scripts/05_select_hyperparams.py` | attack-free grid search: admissibility filter (|val FNR at native boundary − ν| ≤ max(0.02, 0.5ν)) then lowest product-of-marginals reference acceptance; degeneracy / no-admissible guards; freezes transform + model + τ5/τ10 | `results/grid_results.csv`, `results/selection.json`, `model/` |
| 6 | `python scripts/06_final_test.py` | touches the test split **once**: FNR at τ5/τ10 with user-level bootstrap CIs | `results/test_results.json`, `results/test_score_hist.png` |
| 7 | `python scripts/07_a1_sanity_check.py` | A1 Gaussian bots vs the frozen model: FPR, ROC-AUC, confusion matrices; plus a POST-HOC A1 table over all grid configs (not used for selection) | `results/a1_results.json`, `results/a1_roc.png`, `results/grid_attack_sensitivity.csv` |
| 99 | `python scripts/99_legacy_protocol.py` | legacy (in-sample, F1-selected) replica for the response letter | `results/legacy_vs_clean.md` |

`METHODS.md` (paper paragraph with the real numbers + assumptions list) is regenerated after every step.

Terminology: **FNR = humans rejected**, **FPR = bots accepted**; a session is accepted iff `score >= τ`.

## Protocol hygiene enforced in code

* users are split before anything is fitted; the four user sets are asserted disjoint;
* the input transform (signed log + RobustScaler) and the SVM see TRAIN only; τ comes from VAL only; `selection.py` raises `AttackDataError` on attack data;
* `attacker_pool` is never touched; the test split is evaluated once (`06` refuses to rerun without `--allow-rerun`);
* `05` stops (exit 2, nothing frozen, tolerance never relaxed) if reference acceptance spans < 1 pp over the grid or no config is admissible; otherwise it freezes the model with a config hash; `06`/`07` refuse to run if the protocol config changed afterwards.

## Data gotchas (verified on the real files)

* files are **latin-1**, not UTF-8 (decoded UTF-8-first with a latin-1 fallback);
* `LETTER` is unreliable for Backspace/Shift → keys are identified by `KEYCODE` (8/46 corrections, 16 Shift);
* the metadata column is `LAYOUT` and `KEYBOARD_TYPE` values are `full`, `laptop`, `small`, `on-screen`;
* `KEYSTROKE_ID` is not assumed monotonic: sort by `PRESS_TIME`, `KEYSTROKE_ID` as tie-break only.

## Revision 1 (2026-09-24)

The original Tax & Duin uniform-box selection rule was degenerate (all 35 configs accepted ~0% of uniform points). Revision 1 —
appended to `docs/PHASE1_SPEC.md`, original text kept — removes `flight_cv` (adds `flight_median`, `flight_iqr`), replaces the
StandardScaler by signed log + RobustScaler, and selects on the acceptance of a product-of-marginals reference set
(Hempstalk, Frank & Witten, 2008). The uniform-box acceptance is kept as the diagnostic column `uniform_accept_tau5`.

## Revision 2 (FINAL, 2026-09-24)

Adds the admissibility filter (ν-property of the OCSVM, Schölkopf et al. 2001) because the reference criterion alone picked the
overfitted γ = 0.3 (val FNR at the native boundary 11.5% vs ν = 1%; 1,952 support vectors). Features, transform, grid and selection rule
are now FROZEN for Phase 1 whatever the A1 results; A2–A5 (Phase 2) are designed only after this freeze.
