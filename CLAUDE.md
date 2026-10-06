# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## START HERE

- **Read `docs/PROJECT_STATE.md` first.** It is the single entry point: goal, phase timeline, decisions, all main results, incidents, limitations, next steps, resume commands and file map.
- **Phases 1, 2, 3a, 3b and 3c are COMPLETE and FROZEN.** Never modify a frozen file: sources, configs, specs/pre-registration, freeze records or anything under `outputs/*/`. The frozen source/config files are read-only (`docs/environment/readonly_files.txt`); never `chmod` them back to writable to edit them.
- **Run the guards and `pytest` before any work.** Commands are in `docs/PROJECT_STATE.md` §h. If a guard fails, stop and tell the user.
- **Ask before any protocol change.** New work = a new phase with its own spec (and pre-registration by hash for anything that touches the test split).
- **No git.** Pre-registration and freezes are SHA-256 records in `docs/`.
- **The user prefers plain-language explanations and complete files** (whole files, not fragments to paste).

## Project state

Phases 1, 2, 3a, 3b and 3c are implemented, run (quick + full) and frozen; see `docs/PROJECT_STATE.md`. The paragraph below is the original Phase 1–3a description.
Phases 1, 2 and 3a are implemented and run (quick + full). Phase 1 = `scripts/01…07,99` + `config.yaml` (`docs/PHASE1_SPEC.md`); Phase 2 = attackers/detectors, `scripts/10…13` + `config_phase2.yaml` (`docs/PHASE2_SPEC.md`); Phase 3a = diagnostics, `scripts/20…21` + `config_phase3a.yaml` (`docs/PHASE3A_SPEC.md`). The specs are authoritative; do not change protocol decisions without asking. `README.md` only covers Phase 1.

## Commands

```bash
source .venv/bin/activate          # Python 3.11 (uv-installed); deps in requirements.txt
pytest                             # all tests (synthetic data, no dataset needed)
pytest tests/test_features.py -k shift          # single test
python scripts/01_sample_and_parse.py --quick   # then 02, 03, 04, 05, 06, 07 (99 = legacy replica)
python scripts/10_fit_attackers.py --quick      # Phase 2: then 11 (bots + realism), 12 (E1–E3), 13 (report)
python scripts/20_run_diagnostics.py --quick --only centrality   # Phase 3a (centrality|ablation|importance|consistency); then 21
```

`--quick` = 2,000 users → `outputs/quick/`; without it → `outputs/full/`. Scripts are thin wrappers; logic lives in `src/kl/` (`io_aalto`, `sessions`, `split`, `features`, `selection`, `model`, `evaluate`, `report`, `attacks/`). Tunables live only in `config.yaml`; bulky intermediates are parquet under `outputs/<mode>/interim/` (one row group = whole participants, so sessions never span row groups).

## Architecture notes that need several files to see

- Data flow: raw files → `keystrokes.parquet` (01) → `sessions.parquet` + `session_index.parquet` (02) → `split_manifest.json` (03) → `features_{split}.parquet` (04) → frozen `model/` + `selection.json` (05) → test (06) → A1 (07).
- `features.py` is the single code path for humans and bots (`compute_features` for one session, `extract_features_frame` for many share `_features_sorted`). Bots are produced by `attacks/a1_gaussian.py` and must emit real JS keycodes.
- Hygiene is enforced in code: `selection.py` raises `AttackDataError` on attack data; 06 refuses to run twice (`--allow-rerun` overrides); 06/07 check the config hash stored with the frozen model (`config.HASH_EXCLUDED_KEYS` lists sections that do not affect it).
- Phase 2/3a outputs go to `outputs/<mode>/phase2/` (`pool.joblib`, `bots_<A>_<split>.parquet`, tables, `ATTACKS.md`, `RESULTS_PHASE2.md`) and `outputs/<mode>/phase3a/` (`DIAGNOSTICS.md`). Their configs live in separate YAML files precisely so `config.yaml` (and hence the Phase 1 config hash) never changes; seed, n_jobs, bootstrap settings are read from `config.yaml`, not duplicated.
- Freeze guards: every Phase 2 script calls `kl.phase2.guard.assert_frozen` (SHA-256 of `features.py`, `model.py`, `selection.py`, **the whole `config.yaml` file**, and the frozen model files, per `docs/phase1_frozen_hashes.json`); Phase 3a calls `kl.phase3a.guard.assert_frozen_p12`, which also hashes all Phase 2 sources/scripts/`config_phase2.yaml` and the Phase 2 output files (`docs/phase2_frozen_hashes.json`). So even a cosmetic edit to `config.yaml` (including `n_jobs`) or to any `attacks/`/`phase2/` file, or re-running 10–13, raises `FrozenViolation` downstream.
- Attackers (`src/kl/attacks/`, A1–A5b, common `generate(sentence, rng)` interface) are fitted only on `attacker_pool` users (`attacks/pool.py`); a held-out part of the pool is used only for the realism report, which must never tune an attacker. A5a (verbatim replay) is expected to be accepted and is never a D3 training attack. D1 = the frozen OCSVM (never refit); D2/D3 (Isolation Forest / Random Forest, `phase2/detectors.py`) use pre-declared hyperparameters and τ set on val humans only.
- Revisions 1 and 2 (FINAL; see the end of `docs/PHASE1_SPEC.md`). Revision 2 froze features, transform, grid and selection rule for Phase 1 — do not change them, whatever the A1 results. Revision 1: inputs go through `model.FeatureTransform` (signed log except the bounded rates in `features.BOUNDED_RATE_FEATURES`, then RobustScaler fit on train). Selection (`selection.py`): configs must be admissible (`|val_fnr_native − nu| ≤ max(0.02, 0.5·nu)`), then the lowest acceptance of a product-of-marginals reference set at τ5 wins; `DegenerateSelectionError` (spread over the grid < 1 pp) or `NoAdmissibleConfigError` make 05 exit 2 with nothing frozen (`selection_stopped.json`) — never relax the tolerance automatically. The old uniform-box acceptance is only a diagnostic column. `07` also writes `grid_attack_sensitivity.csv`, which is post-hoc and must never feed selection.
- `outputs/<mode>/results/*.json` feed `report.write_methods_md`, which every step regenerates; `docs/revision1_original_rule_diagnostics.json` archives the numbers of the abandoned rule.

## The dataset

`data/raw/files/` is the Aalto University *136M Keystrokes* corpus (Dhakal et al., CHI 2018; non-commercial use, cite the paper). ~18 GB, containing:

- **168,593 participant files**, `<PARTICIPANT_ID>_keystrokes.txt` (filename ID matches the `PARTICIPANT_ID` column). List with the glob `*_keystrokes.txt`.
- `metadata_participants.txt` and `readme.txt` in the same directory — **exclude them** when listing participants (a bare `ls | wc -l` gives 168,595).
- `data/raw/Keystrokes/` is an empty leftover from extraction.

### Keystroke files

Tab-separated, header on every file:

```
PARTICIPANT_ID  TEST_SECTION_ID  SENTENCE  USER_INPUT  KEYSTROKE_ID  PRESS_TIME  RELEASE_TIME  LETTER  KEYCODE
```

- Each participant typed 15 sentences; a **test section** (`TEST_SECTION_ID`) is one sentence. `SENTENCE` is the prompt, `USER_INPUT` the submitted string. Timing features (dwell = release − press, flight = next press − this release) are only meaningful *within* a test section.
- `PRESS_TIME` / `RELEASE_TIME` are epoch **milliseconds**.

### Use KEYCODE, not LETTER, to identify keys

`LETTER` is unreliable for special keys: in a 2,000-file sample, Backspace (keycode 8) shows up as `BKSP`, `\x08`, `=`, `\`, `e`, or empty, and Shift (16) as `SHIFT`, `?`, `I`, a space, etc. `KEYCODE` is the JavaScript keycode and is the source of truth:

| Key | KEYCODE |
|---|---|
| Backspace | 8 |
| Delete | 46 |
| Shift | 16 |
| Enter | 13 |

Corrections = `KEYCODE in {8, 46}`; Shift = `KEYCODE == 16`. Letters carry the uppercase ASCII code (`a` and `A` → 65), space is 32. Anything synthesised for the bot/attacker side must emit the same keycodes so it passes through the same feature code as human data.

### metadata_participants.txt

One row per participant (168,593 files vs 168,594 metadata rows — reconcile by joining on `PARTICIPANT_ID`, don't assume 1:1). Columns: `PARTICIPANT_ID AGE GENDER HAS_TAKEN_TYPING_COURSE COUNTRY LAYOUT NATIVE_LANGUAGE FINGERS TIME_SPENT_TYPING KEYBOARD_TYPE ERROR_RATE AVG_WPM_15 AVG_IKI ECPC KSPC ROR`. Note the actual layout column is `LAYOUT`, not `KEYBOARD_LAYOUT` as `readme.txt` says.

Observed values (counts over the whole file):

- `KEYBOARD_TYPE`: `laptop` 91,250 · `full` 73,759 · `small` 1,886 · `on-screen` 1,699. The readme calls these "full (desktop), laptop, small physical, touch"; the on-disk strings are `full`/`laptop`/`small`/`on-screen`. Physical-keyboard filter = `{full, laptop}`.
- `LAYOUT`: `qwerty` 165,324 · `qwertz` 1,979 · `azerty` 1,082 · `dvorak` 209. Keep as a metadata column for stratified analysis, not a model feature.

### Encoding — the other trap

Files are **ISO-8859-1 (latin-1), not UTF-8** (e.g. a `\xc9` byte in `100122_keystrokes.txt`; 2 of the first 300 files fail a UTF-8 decode). Always pass `encoding="latin-1"` — it never fails. Fields are unquoted, so `SENTENCE` / `USER_INPUT` can contain quote characters: use `csv.QUOTE_NONE` (`quoting=3` in pandas).

```python
pd.read_csv(path, sep="\t", encoding="latin-1", quoting=3)
```

### Working with 18 GB

- Never glob-read `data/raw/files/` into memory, and never `cat`/`grep` the whole directory in a shell tool — sample a handful of files instead.
- Preprocessing must stream per participant and write derived artifacts to a separate directory (e.g. `data/processed/`) in a columnar format (Parquet/feather); do not mutate `data/raw/`.
- Test new parsing code against a few hundred files before running the full corpus.

## Domain framing

The goal is keystroke **liveness / bot detection** — deciding whether a typing sequence came from a human or was synthesized/replayed. The corpus is human-only, so it supplies the positive class; negatives have to be generated or collected. Keep any synthetic-negative generator separate from the real-data loading path so the two never get silently mixed.

Phase 1 spec: docs/PHASE1_SPEC.md — follow it; ask before changing protocol decisions.

Phase 2 spec: docs/PHASE2_SPEC.md — Phase 1 is FROZEN (features.py, model.py, selection.py, config protocol keys, outputs/*/model/); verify with kl.phase2.guard; do not change any attacker after seeing detector results.

Phase 3a spec: docs/PHASE3A_SPEC.md — diagnostics only; Phases 1 and 2 FROZEN (Phase 1 record: docs/phase1_frozen_hashes.json, Phase 2 record: docs/phase2_frozen_hashes.json, both verified by kl.phase3a.guard); no new features/detectors, no attacker changes.

Phase 3b spec: docs/PHASE3B_SPEC.md (verbatim + the user's amendments at the end) — sequence-level detectors D4 (autoregressive GRU mixture density, humans only) and D5 (supervised BiGRU). Phases 1, 2 and 3a FROZEN (Phase 3a record: docs/phase3a_frozen_hashes.json; kl.phase3b.guard checks all three records plus the pre-registration hash in docs/phase3b_prereg_hash.txt before and after every step). Every design choice is fixed in docs/PHASE3B_PREREGISTRATION.md: never edit it; after results, fix only genuine bugs and report each one.
Phase 3b runs go through `scripts/run_phase3b.py` only (strictly sequential, launcher + training locks, skips COMPLETE runs; never a bash loop: bash treats a SIGSTOP-paused child as finished). Resume state: docs/PHASE3B_RESUME.md; bug/incident log: docs/phase3b_bugfixes.md.

Phase 3c spec: docs/PHASE3C_SPEC.md — POST-HOC artefact check of D4/D5 (keycode realism, key/timing swap test, letters-only D4 score); diagnostic only, Phases 1–3b FROZEN (no retraining, no model/attacker/threshold/split changes). Outputs: outputs/<mode>/phase3c/DIAGNOSTICS_PHASE3C.md.
