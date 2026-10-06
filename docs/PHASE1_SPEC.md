First: save this entire message verbatim as docs/PHASE1_SPEC.md and add a line to CLAUDE.md:
"Phase 1 spec: docs/PHASE1_SPEC.md — follow it; ask before changing protocol decisions."
Then implement it. The project root is THIS repo (replace the PyCharm template main.py).

# Phase 1 spec — keystroke liveness detection pipeline

Goal: a reproducible pipeline for a paper on keystroke-dynamics LIVENESS detection. It separates
human typing from machine-generated typing with a One-Class SVM trained on human data only. The
original code is lost, so rebuild it from scratch with strict protocol hygiene. Write complete,
working files (no stubs, no TODOs), with type hints and docstrings.

## Environment
- Python 3.11. Dependencies: numpy, pandas, pyarrow, scikit-learn, scipy, pyyaml, joblib,
  rapidfuzz, matplotlib, pytest, tqdm. Provide requirements.txt.
- Everything is driven by config.yaml. One global seed (default 42), passed explicitly to every
  random operation. No hidden randomness.

## Dataset facts (VERIFIED against the real files; these override the readme)
- Aalto 136M Keystrokes (Dhakal et al., CHI 2018, doi:10.1145/3173574.3174220).
- Location: data/raw/files/ (verify the path; if the files are at data/raw/Keystrokes/files/,
  use that and set it in config.yaml).
- 168,593 participant files matching "*_keystrokes.txt". Also present: readme.txt and
  metadata_participants.txt; exclude both from the listing.
- Keystroke columns: PARTICIPANT_ID, TEST_SECTION_ID, SENTENCE, USER_INPUT, KEYSTROKE_ID,
  PRESS_TIME, RELEASE_TIME, LETTER, KEYCODE. Times are in ms, epoch.
- Parse robustly: sep="\t", quoting=csv.QUOTE_NONE, on_bad_lines="skip", utf-8 with
  errors="replace" (latin-1 fallback). Never crash on a bad file; log it and count it.
- Metadata: metadata_participants.txt has 168,594 rows (not 1:1 with the files). Join on
  PARTICIPANT_ID. KEYBOARD_TYPE values are laptop, full, small, on-screen. The layout column
  is named LAYOUT (values: qwerty, qwertz, azerty, dvorak).
- LETTER is unreliable (Backspace and Shift often garbled). KEYCODE is the source of truth:
  Backspace=8, Delete=46, Shift=16, Enter=13 (Enter is rare/absent in the logs; do not rely on it).
- Do NOT assume KEYSTROKE_ID is globally increasing. Sort by PRESS_TIME, with KEYSTROKE_ID as
  the tie-break only.

## Layout
config.yaml, requirements.txt, README.md (how to run, step by step)
src/kl/
  io_aalto.py        # file discovery, robust parsing, metadata loading
  sessions.py        # session definition + cleaning filters
  split.py           # user-level split + manifest
  features.py        # session -> feature dict (one code path for humans AND bots)
  attacks/base.py    # Attacker interface
  attacks/a1_gaussian.py
  model.py           # scaler + OCSVM wrapper, save/load
  selection.py       # hyperparameter grid + attack-free selection rule
  evaluate.py        # metrics, bootstrap CIs, ROC, confusion matrices
  report.py          # writes METHODS.md, FEATURES.md, results JSON
scripts/ 01_sample_and_parse.py, 02_clean_and_sessionize.py, 03_split_users.py,
         04_extract_features.py, 05_select_hyperparams.py, 06_final_test.py,
         07_a1_sanity_check.py, 99_legacy_protocol.py
tests/ (pytest)
outputs/ (git-ignored except small JSON/MD/CSV)

## Step 1 — Eligibility + sampling (01)
- Load the metadata. Keep participants with KEYBOARD_TYPE in {full, laptop}; exclude small,
  on-screen, and missing. Keep only IDs that also have a keystroke file.
- Sort the eligible IDs, then draw N_USERS uniformly at random with the seed
  (default 20000; --quick mode = 2000).
- Parse only the sampled files -> outputs/interim/keystrokes.parquet. Attach LAYOUT as a
  metadata column (never a model feature).
- Log to cleaning_report.json: metadata rows, count per KEYBOARD_TYPE, eligible, eligible with a
  file, sampled, files failed, rows read.

## Step 2 — Sessions + cleaning (02)
Session = rows sharing (PARTICIPANT_ID, TEST_SECTION_ID) = one typed sentence. Drop a session
(counting each reason separately) if: fewer than 10 keystrokes; missing or non-numeric times;
any hold < 0 or > 2000 ms; any press-to-press gap > 5000 ms; empty SENTENCE or USER_INPUT.
All thresholds live in config.yaml. Outputs: sessions.parquet and the counts in
cleaning_report.json.

## Step 3 — User-level split (03)
Split PARTICIPANT_IDs into train 60% / val 15% / test 15% / attacker_pool 10%. attacker_pool is
never used for training or tuning (reserved for the Phase 2 replay attack). Optional cap
max_train_sessions (default 20000), applied by seeded uniform session sampling. Write
split_manifest.json with user and session IDs per split. Assert the four user sets are
pairwise disjoint.

## Step 4 — Features (04)
features.py takes ONE session's raw keystroke DataFrame (Aalto schema) and returns a dict. All
times in ms. For consecutive keys i, i+1:
  hold_i = RELEASE_i − PRESS_i;  pp_i = PRESS_{i+1} − PRESS_i;
  rr_i = RELEASE_{i+1} − RELEASE_i;  flight_i = PRESS_{i+1} − RELEASE_i (negative = rollover)
Groups (keep them; Phase 3 runs ablations by group):
  TIMING: for x in {hold, pp, rr, flight}: mean, std, cv (= std / max(|mean|, 1e-6))
  RHYTHM: pause_count (flight > 500 ms), pause_rate (= pause_count/(n_keys−1)), rollover_rate
  BEHAVIOUR: corrections_count (KEYCODE in {8, 46}), correction_rate,
             lev_norm (= Levenshtein(SENTENCE, USER_INPUT) / max of the two lengths),
             shift_overlap_count (non-Shift keys pressed while a KEYCODE-16 key is held,
             beyond the first per Shift hold)
  GLOBAL: n_keys, total_duration_ms (last PRESS − first PRESS),
          wpm (= (len(USER_INPUT)/5) / (total_duration_ms/60000))
Output features_{split}.parquet with session_id, user_id, split, LAYOUT, and the features.
Auto-generate FEATURES.md (name, group, formula, unit, value on the worked example).

## Step 5 — Attack-free hyperparameter selection (05)
- StandardScaler (fit on TRAIN only) -> OneClassSVM(kernel="rbf").
- Grid: nu ∈ {0.01, 0.02, 0.05, 0.10, 0.15}; gamma ∈ {"scale", 0.001, 0.003, 0.01, 0.03, 0.1, 0.3}.
- For each config: fit on train, score VAL humans, and set τ at the quantile giving val
  FNR = 5% (primary) and 10% (secondary).
- Selection rule (no attack data): Tax & Duin (2001) uniform artificial outliers. Draw 100,000
  seeded points uniformly in the per-feature [min, max] box of the scaled train data. Choose the
  config with the lowest accepted fraction at the 5% τ. Tie (within 0.1 percentage points) ->
  smaller nu, then smaller gamma.
- grid_results.csv columns: nu, gamma, fit_time, val FNR at the native boundary, τ5, τ10,
  uniform acceptance at τ5 and τ10, n_support_vectors.
- Save model, scaler, τ values, and a config hash to outputs/model/.
- selection.py must raise if it is ever given data labelled as an attack.

## Step 6 — Final human test (06)
Touch the test split once. Report test FNR at τ5 and τ10 with 95% bootstrap CIs (resample by
USER, 1000 reps, seeded), plus a histogram of the scores.

## Step 7 — A1 sanity check (07), only after selection is frozen
A1 reconstructs the original generator. Interface: generate(sentence, rng) -> DataFrame in
Aalto schema.
- Types SENTENCE exactly; USER_INPUT = SENTENCE.
- hold = max(30, N(116.24, 23.88)); flight = max(60, N(238.656, 111.6)), all draws independent;
  PRESS_{i+1} = RELEASE_i + flight. HOLD_MIN = 30 ms is an ASSUMPTION; state this in METHODS.md.
- Shifted chars: Shift(16) press -> key press/release -> Shift release, no overlap.
- Correct JS keycodes: letters use the uppercase ASCII code ('a'/'A' -> 65), space = 32; punctuation
  via a documented mapping table.
- One bot session per TEST human session, same sentence. Report FPR (bots accepted) at τ5 and
  τ10, ROC-AUC, and confusion matrices. Label clearly: FNR = humans rejected,
  FPR = bots accepted.

## Step 99 — Legacy protocol replica (response letter only)
Train and "test" on the SAME human sessions; select by best F1 on humans + A1, with
gamma ∈ {auto, 0.1, 0.01} and nu ∈ {0.01, 0.05, 0.1, 0.15}. Write legacy_vs_clean.md
showing the old protocol's numbers next to the clean ones.

## Tests
- Features: PRESS=[0,200,450], RELEASE=[100,280,560] -> hold=[100,80,110], pp=[200,250],
  rr=[180,280], flight=[100,170]; assert the means and stds.
- Shift held 0–300 ms, keys pressed at 50, 150, 250 -> shift_overlap_count = 2.
- Garbled LETTER with correct KEYCODE 8 -> counted as a correction.
- Split disjointness; determinism (same seed -> identical manifest and features); A1 output
  passes through features.py.

## Reporting
METHODS.md: a paper-ready paragraph with the real numbers (the keyboard-type filter and its
counts, users, sessions per split, drops per rule, grid, selection rule, chosen parameters, τ,
test FNR with CI, A1 results) plus an explicit list of assumptions.

Work order: implement -> run the tests -> run everything in --quick mode -> fix -> print a
summary of the outputs. Ask before changing any protocol decision above.


---

# Revision 1 — 2026-09-24 (selection-rule revision)

The original text above is kept unchanged for the record. Where it conflicts with this revision,
this revision is authoritative. Trigger: the Tax & Duin uniform-outlier rule was degenerate (all 35
grid configurations accepted ~0% of the uniform points, so the tie-break alone chose the model).

Decision on the degenerate selection rule. Update docs/PHASE1_SPEC.md with these changes (add a
dated "Revision 1" section; don't delete the original text), then implement them.

1. Feature fixes (features.py):
   - Remove flight_cv (undefined when mean flight is near 0 because of rollover). Add
     flight_median and flight_iqr (in ms).
   - Report the min/median/max of every remaining *_cv feature on the train split. If rr_cv (or
     any other cv) also shows values > 100, remove it the same way and tell me.
   - Update FEATURES.md and the worked-example tests.

2. Transform (model.py), replacing StandardScaler:
   - Bounded rates in [0,1] (pause_rate, rollover_rate, correction_rate, lev_norm): no log.
   - All other features: signed log, y = sign(x) * log1p(|x|).
   - Then RobustScaler (median/IQR), fit on TRAIN only. Save the transform with the model.

3. Selection rule (selection.py), replacing Tax & Duin uniform:
   - Reference set: M = 100,000 points drawn from the product of marginals of the TRANSFORMED
     train data. For each feature independently, sample values with replacement from that
     train column (seeded). Cite: Hempstalk, Frank & Witten (2008), One-class classification
     by combining density and class probability estimation, ECML PKDD.
   - Criterion: lowest reference-acceptance fraction at the 5% τ (val FNR = 5%). Same tie-break.
   - Keep computing the old uniform-box acceptance as a diagnostic column (uniform_accept_tau5)
     so the degeneracy is documented. It is NOT used for selection.
   - Degeneracy guard: if the spread (max − min) of reference acceptance across the grid is
     < 1 percentage point, stop and report instead of selecting.
   - Still no attack data anywhere in selection (keep the guard).

4. Post-hoc sensitivity (07_a1_sanity_check.py):
   - After the frozen selection, also evaluate A1 for ALL 35 grid configs (each with its own
     τ5/τ10 from val). Save grid_attack_sensitivity.csv with FPR@τ5, FPR@τ10, and AUC per config.
     Label it in the file and in METHODS.md: "post-hoc; not used for selection". Mark the
     selected config's rank.

5. METHODS.md: add a "Selection rule revision" paragraph. Say that the uniform rule was
   abandoned because all 35 configs accepted ~0% of uniform points (a degenerate tie in 22
   dimensions), with the diagnostic numbers, and that the revision was based on that
   degeneracy, not on attack performance.

6. Re-run the whole pipeline in --quick mode (use --allow-rerun for 06). Report: the cv
   diagnostics, the spread of reference acceptance across the grid, the selected config, test
   FNR (τ5, τ10), A1 FPR and AUC for the selected config, and where it ranks in the
   sensitivity table. Do NOT start full mode; wait for me.


---

# Revision 2 (FINAL) — 2026-09-24 (admissibility filter; protocol freeze)

Revision 2 is FINAL. Where it conflicts with the text above (original spec or Revision 1), this
revision is authoritative. Trigger: under Revision 1 the product-of-marginals criterion alone
favoured the largest γ (0.3), whose attack-free diagnostics indicated an overfitted boundary.

Decision on the γ/overfitting issue. Append a dated "Revision 2 (FINAL)" section to
docs/PHASE1_SPEC.md, implement it, then proceed to full mode without waiting for me.

1. Admissibility filter (selection.py), attack-free:
   - For each config compute val_fnr_native = the fraction of VAL humans with decision_function < 0.
   - admissible = |val_fnr_native − nu| ≤ max(0.02, 0.5·nu).
     Rationale: under the ν-property of One-Class SVM (Schölkopf et al., 2001, Neural
     Computation 13(7)), ν approximates the held-out outlier fraction; a large excess indicates
     overfitting of the boundary.
   - Among admissible configs, select the lowest product-of-marginals reference acceptance at
     τ5. Tie-break unchanged (smaller nu, then smaller gamma).
   - If no config is admissible, stop and report (do not relax the tolerance yourself).
   - grid_results.csv: add the columns val_fnr_native, admissible (bool), n_support_vectors.

2. Freeze: Revision 2 is FINAL. No further changes to features, transform, grid, or selection rule
   in Phase 1, whatever the A1 results. State this in METHODS.md.

3. METHODS.md "Selection rule revision" paragraph: add Revision 2. Say that the product-of-
   marginals criterion alone favoured the largest γ, and that this config showed attack-free
   overfitting diagnostics (val FNR at the native boundary 11.5% vs ν = 1%; 1,952 support vectors
   vs 173–236). Say that the admissibility filter was added based on those diagnostics.
   Disclose honestly that post-hoc A1 results had been seen when this revision was made, and
   that attacks A2–A5 (Phase 2) will be designed only after this freeze.

4. Run the whole pipeline in --quick mode once (--allow-rerun for 06) as a smoke test. Do not
   modify anything based on its numbers. If it completes without errors, run FULL mode
   (N_USERS = 20000, max_train_sessions = 20000). The full test split must be evaluated exactly
   once.

5. Final report to me (full mode): the counts from cleaning_report.json, split sizes, the
   admissible set (how many of 35), the selected config, test FNR at τ5/τ10 with CIs, A1 FPR at
   τ5/τ10 and AUC, the selected config's rank in grid_attack_sensitivity.csv, the
   legacy_vs_clean.md summary, and total runtime.
