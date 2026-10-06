# PROJECT STATE — keystroke liveness (single entry point)

_Written 2026-09-26 at the project handover. Every number below is copied from a result file named next to it;
nothing was recomputed for this document. Phases 1, 2, 3a, 3b and 3c are COMPLETE and FROZEN._

_Phase 4 (revision analyses, Part A) was added on 2026-10-06: see the timeline row in section b and "Phase 4" at the end of
section d. Phase 4 changed no Phase 1–3c file._

---

## a. Project goal

The project asks whether a computer can tell **a real person typing** from **typing that a program generated or
replayed** ("keystroke liveness" or bot detection), using only the timing of key presses and releases. The human
data is the Aalto *136M Keystrokes* corpus (Dhakal et al., CHI 2018), which contains only humans. The bot side
therefore comes from attackers we built ourselves, ranging from simple (random Gaussian timings) to realistic
(timings learnt from real people, typos, replays of real recordings). Detectors range from a classical one-class SVM
trained on humans only, to supervised classifiers and to sequence neural networks (GRUs). The aim is an honest,
leak-free measure of how well each detector works against each kind of attack.

---

## b. Timeline of phases

| phase | goal | spec | date | status | key scripts | outputs |
|---|---|---|---|---|---|---|
| **1** | Clean, leak-free baseline: 23 session features, One-Class SVM (D1) trained on humans only, hyper-parameters chosen **without any attack data**, test split touched once, sanity check against a simple Gaussian bot (A1); replica of the "legacy" leaky protocol for comparison | `docs/PHASE1_SPEC.md` (with Revisions 1 and 2 at the end) | 2026-09-24 | COMPLETE, FROZEN (`docs/phase1_frozen_hashes.json`) | `scripts/01…07`, `99_legacy_protocol.py`; `config.yaml` | `outputs/full/{interim,model,results}/`, `METHODS.md`, `FEATURES.md` |
| **2** | Five non-adaptive attackers A1–A5b fitted on a disjoint attacker pool; detectors D1 (frozen OCSVM), D2 (Isolation Forest), D3 (Random Forest, supervised); experiments E1 open set, E2 cross-attack, E3 leave-one-attack-out | `docs/PHASE2_SPEC.md` | 2026-09-24 | COMPLETE, FROZEN (`docs/phase2_frozen_hashes.json`) | `scripts/10…13`; `config_phase2.yaml` | `outputs/full/phase2/` (`RESULTS_PHASE2.md`, `ATTACKS.md`) |
| **3a** | Diagnostics only: why do A2–A5b fool D1? Centrality ("too typical"), feature-group ablation, permutation importance, per-user consistency | `docs/PHASE3A_SPEC.md` | 2026-09-24 | COMPLETE, FROZEN (`docs/phase3a_frozen_hashes.json`) | `scripts/20…21`; `config_phase3a.yaml` | `outputs/full/phase3a/DIAGNOSTICS.md` |
| **3b** | Sequence-level detectors: D4 (autoregressive GRU mixture density, humans only) and D5 (supervised BiGRU); E4 open set, E5 cross-attack, E6 leave-one-out; 3 seeds; **pre-registered by hash** | `docs/PHASE3B_SPEC.md`, `docs/PHASE3B_PREREGISTRATION.md` | 2026-09-25 → 2026-09-26 (finished 13:02) | COMPLETE, FROZEN (`docs/phase3b_prereg_hash.txt`, `docs/phase3b_frozen_hashes.json`) | `scripts/30…33`, launcher `scripts/run_phase3b.py`; `config_phase3b.yaml` | `outputs/full/phase3b/RESULTS_PHASE3B.md` |
| **3c** | POST-HOC artefact check: are D4/D5 detections timing given the keys (signal) or key-sequence differences (artefact)? Keycode realism, key/timing swap test, letters-only D4 score | `docs/PHASE3C_SPEC.md` | 2026-09-26 | COMPLETE, FROZEN (`docs/phase3c_frozen_hashes.json`, written at the handover) | `scripts/40_run_phase3c.py`, `41_report_phase3c.py` | `outputs/full/phase3c/DIAGNOSTICS_PHASE3C.md` |
| **4** | Revision analyses, Part A (supervisor): user-cluster bootstrap CIs for the headline numbers (A1), APCER @τ5 (A2), identical-key subset with CIs (A3), new pre-registered detector D6 = key-pair timing lookup model, humans only (A4) | `docs/PHASE4_SPEC.md` (hash in `docs/phase4_prereg_hash.txt`) | 2026-10-06 | COMPLETE (spec and results read-only) | `scripts/phase4_run.py`, `phase4_report.py`; `src/kl/phase4_{guard,bootstrap,d6}.py` | `docs/PHASE4_RESULTS.md`, `outputs/phase4/{full,quick}/` |

---

## c. Key decisions and why

- **User-disjoint split (60 / 15 / 15 / 10% of users → train / val / test / attacker pool).** No person appears in two
  splits, so a detector can never be scored on someone it has already seen. Train is capped at 20,000 sessions (D1's
  training set). The test split is used once (`06_final_test.py` refuses a second run; Phase 3b kept a per-model
  test ledger `test_access.json`). The legacy protocol (train = test, selection by F1 on humans + A1) looked much
  better (A1 accepted 2.94%) than the clean one (5.86% at τ10, 32.47% at τ5). That gap is the reason for the whole
  design (`outputs/full/results/legacy_vs_clean.md`).
- **Attacker pool.** 1,999 users (28,854 sessions) are reserved for building attackers. Every attacker parameter is
  fitted on pool users only. 400 of them (5,788 sessions) are held out and used only for the realism report, never
  to tune an attacker. Attackers were fixed before any detector was run and never changed afterwards
  (`outputs/full/phase2/ATTACKS.md`).
- **Attack-free hyper-parameter selection for D1 (Phase 1, Revisions 1–2).**
  1. *Original rule (abandoned):* uniform artificial outliers in the feature bounding box (Tax & Duin 2001). It was
     **degenerate**: in 22 dimensions with heavy-tailed features (`flight_cv` up to about 7×10⁷), 34 of 35
     configurations accepted none of the 100,000 uniform points and all 35 tied, so the tie-break alone chose the
     model.
  2. *Revision 1:* the reference set was changed to the **product of marginals** of the transformed training data
     (each feature resampled independently; Hempstalk, Frank & Witten 2008). Signed-log + RobustScaler transform;
     `flight_cv` replaced by `flight_median` + `flight_iqr`. A degeneracy guard stops selection if the spread over the
     grid is < 1 pp.
  3. *Revision 2 (FINAL):* the product-of-marginals rule alone picked the largest γ with **attack-free overfitting
     signs** (quick run: val FNR at the native boundary 11.5% vs ν = 1%; 1,952 support vectors vs 173–236). An
     **admissibility filter** was added: |val FNR_native − ν| ≤ max(0.02, 0.5·ν). Full run: 32 of 35 configurations
     admissible; chosen ν = 0.10, γ = 0.3.
  - *Disclosure (in METHODS.md):* post-hoc A1 results had been seen when Revision 2 was decided. Its trigger and
    formula are attack-free, and A2–A5 were designed only after the freeze.
- **Pre-registration by hash instead of git** (the user does not use git). `docs/PHASE3B_PREREGISTRATION.md` fixed
  every Phase 3b design choice before the full run. Its SHA-256 (and that of `config_phase3b.yaml`) is in
  `docs/phase3b_prereg_hash.txt` and is checked before and after every step. The user emailed the file to themselves
  as a timestamp. The same idea underlies all the freeze records (`docs/phase*_frozen_hashes.json`): each later
  phase verifies the earlier ones before it runs.
- **Thermal guard (Phase 3b, 2026-09-26).** An operational safety feature on a laptop GPU (RTX 3050), requested by
  the user. It changes no computation.
  - Between epochs it reads `nvidia-smi` temperature.
  - At ≥ 85 °C it pauses until ≤ 70 °C, and logs each pause in `runtime_log.json["thermal_pauses"]`.
  - If the GPU is still ≥ 90 °C after 30 min of pausing, it stops cleanly (exit 76) and writes the state to
    `PHASE3B_RESUME.md`.
  - The thresholds are in code (`src/kl/phase3b/thermal.py`), not in the pre-registered config.
  - A test checks that the weights are bit-identical with and without the guard.
- **Why 3c added an identical-key control (post hoc).** The specified swap test aligns a human's and a bot's
  sequences **by position**. Humans correct typos and bots mostly don't, so after the first correction the positions
  no longer hold the same character: only 51% of aligned positions have the same key for A1/A2/A3/A5b, 37% for A4
  and 9% for A5a. Swapped sequences therefore pair timings with the wrong keys. That breaks the key–timing link
  whoever produced the keys: the A5a control, which is two real humans, is flagged by D4 at AUC 0.967. The clean
  test is therefore the **identical-key pairs**, where the human typed exactly the bot's key sequence. There, any
  separation can only come from timing. It is labelled an addition to the spec.

---

## d. Main results

Conventions:
- **APCER = FPR = attacks accepted as human** (lower is better).
- **BPCER = FNR = humans rejected**.
- τ5 / τ10 are thresholds set on validation humans so that 5% / 10% of them are rejected.
- Test split: 43,248 human sessions from 2,999 users; one matched bot per human session per attacker.
- Brackets are 95% user-bootstrap CIs.
- Neural models: mean ± sd over seeds 42/43/44.

### Attackers (Phase 2; `ATTACKS.md`)

| attacker | what it is |
|---|---|
| A1 | Independent Gaussian holds / flights (Phase 1 sanity bot) |
| A2 | Digraph-conditioned timings from the pool |
| A3 | A2 + temporal (AR) structure and per-session speed |
| A4 | A3 + pauses, substitution typos with Backspace corrections, Shift overlap ("realistic imitator") |
| A5a | Verbatim replay of a real pool recording (a real human; expected to be accepted) |
| A5b | A real pool session's hold/flight sequence transplanted onto the target sentence |

### Phase 1 — D1 (One-Class SVM) and A1 (`outputs/full/METHODS.md`, `results/test_results.json`, `results/a1_results.json`, `results/legacy_vs_clean.md`)

| quantity | value |
|---|---|
| chosen configuration | ν = 0.10, γ = 0.3 (32 of 35 admissible; reference-acceptance spread 98.9 pp) |
| humans rejected (FNR) @τ5 / @τ10 | 4.78% [4.45, 5.12] / 9.90% [9.40, 10.40] |
| A1 accepted (FPR) @τ5 / @τ10 | 32.47% / 5.86% |
| ROC-AUC human vs A1 | 0.9540 |
| rank of the chosen config among 35 by A1 AUC (post hoc, not used) | 14 of 35 |

| | legacy (leaky) protocol | clean protocol |
|---|---|---|
| training humans | same sessions as the test | train split (disjoint users) |
| selection | best F1 on humans + A1 (uses attack data) | attack-free (Revision 2) |
| chosen (ν, γ) | (0.1, 0.1) | (0.1, 0.3) |
| humans rejected | 10.02% | 4.78% @τ5; 9.90% @τ10 |
| A1 accepted | 2.94% | 32.47% @τ5; 5.86% @τ10 |
| ROC-AUC vs A1 | 0.9578 | 0.9540 |

### Phase 2 — E1 open set (`outputs/full/phase2/RESULTS_PHASE2.md`): APCER @τ10 (AUC)

| detector | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|
| D1 OCSVM (humans only) | 5.8% (0.954) | 98.0% (0.462) | 96.8% (0.428) | 86.8% (0.499) | 89.9% (0.506) | 93.4% (0.528) |
| D2 Isolation Forest (humans only) | 99.9% (0.796) | 98.5% (0.318) | 98.0% (0.314) | 90.6% (0.483) | 89.5% (0.509) | 93.1% (0.426) |

BPCER @τ10: D1 9.9%, D2 9.6%.

### Phase 2 — E2 cross-attack, D3 Random Forest: APCER @τ10 (rows = trained on, columns = tested on)

| trained on | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|
| A1 | 0.0% | 97.9% | 83.3% | 90.0% | 94.7% | 93.1% |
| A2 | 99.9% | 0.2% | 40.1% | 76.0% | 90.1% | 48.5% |
| A3 | 0.0% | 68.7% | 1.0% | 77.9% | 90.5% | 68.1% |
| A4 | 11.8% | 27.0% | 18.1% | 23.7% | 90.8% | 82.6% |
| A5b | 91.2% | 15.0% | 57.4% | 81.7% | 89.0% | 32.8% |

D3 accepts on average 11.5% of the attacker it was trained on and 61.4% of the other attackers A1–A4/A5b.

### Phase 2 — E3 leave-one-attack-out, D3 (trained on all other attackers)

| held-out attacker | APCER @τ5 | APCER @τ10 | BPCER @τ10 | ROC-AUC |
|---|---|---|---|---|
| A1 | 11.5% | 3.7% | 10.0% | 0.978 |
| A2 | 30.3% | 11.1% | 10.1% | 0.959 |
| A3 | 36.6% | 19.0% | 9.8% | 0.941 |
| A4 | 74.9% | 70.0% | 10.0% | 0.583 |
| A5b | 76.3% | 65.7% | 9.5% | 0.694 |

Mean APCER @τ10 = 33.9% (range 3.7–70.0%). Realism (mean KS over 23 features vs held-out pool; lower = more
realistic): A1 0.527, A2 0.306, A3 0.304, A4 0.124, A5a 0.020, A5b 0.117.

### Phase 3a — ablation of D1 (`outputs/full/phase3a/DIAGNOSTICS.md`): ROC-AUC (APCER @τ10)

| feature subset | A1 | A2 | A3 | A4 | A5b |
|---|---|---|---|---|---|
| ALL 23 (frozen D1) | 0.954 (5.8%) | 0.462 (98.0%) | 0.428 (96.8%) | 0.499 (86.8%) | 0.528 (93.4%) |
| TIMING only (13) | 0.971 (0.8%) | 0.451 (97.7%) | 0.392 (92.7%) | 0.484 (90.2%) | 0.498 (90.5%) |
| TIMING + RHYTHM (16) | 0.974 (0.9%) | 0.406 (97.3%) | 0.392 (93.5%) | 0.484 (86.7%) | 0.487 (90.4%) |
| all minus BEHAVIOUR (19) | 0.971 (2.1%) | 0.385 (96.6%) | 0.436 (93.6%) | 0.479 (87.2%) | 0.481 (90.9%) |
| BEHAVIOUR only (4) | 0.774 (100.0%) | 0.774 (100.0%) | 0.774 (100.0%) | 0.543 (90.2%) | 0.768 (99.5%) |

No session-level feature subset makes D1 work against A2–A5b. Centrality (MCD Mahalanobis, median distance / human
median): A1 2.37×, A2 0.85× (62.1% of A2 sessions closer to the centre than the human median, i.e. "too typical"),
A3 1.19×, A4 1.05×, A5a 1.02×, A5b 1.00×.

### Phase 3a — per-user consistency (test users with ≥ 10 sessions, 2,969 users)

| group | feature | median per-user std (ms) | std / total std | ICC(1) |
|---|---|---|---|---|
| real users | pp_mean | 28.9 | 0.26 | 0.88 |
| shuffled humans (control) | pp_mean | 100.4 | 0.89 | 0.00 |
| A2 pseudo-users | pp_mean | 34.7 | 0.94 | −0.00 |
| A3 pseudo-users | pp_mean | 59.7 | 0.95 | −0.00 |
| real users | hold_mean | 9.3 | 0.37 | 0.80 |
| shuffled humans (control) | hold_mean | 24.0 | 0.95 | −0.00 |
| A2 pseudo-users | hold_mean | 8.8 | 0.90 | −0.00 |
| A3 pseudo-users | hold_mean | 10.3 | 0.91 | −0.00 |

Real users are self-consistent across sessions (ICC 0.80–0.88); the attackers have no persistence (ICC ≈ 0).

### Phase 3b — E4 open set, D4 (humans only) (`outputs/full/phase3b/RESULTS_PHASE3B.md`, `e4_open_set.csv`): APCER @τ10 (AUC)

| detector | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|
| D1 (Phase 1) | 5.8% (0.954) | 98.0% (0.462) | 96.8% (0.428) | 86.8% (0.499) | 89.9% (0.506) | 93.4% (0.528) |
| D4-primary | 64.0% (0.834) | **0.6%** (0.991) | **3.0%** (0.977) | **1.3%** (0.985) | 89.7% (0.494) | **7.5%** (0.968) |
| D4-typicality | 80.1% (0.690) | 1.3% (0.988) | 6.6% (0.967) | 3.2% (0.980) | 89.5% (0.498) | 11.8% (0.949) |
| D4-20k (data-size control) | 74.8% (0.796) | 1.2% (0.986) | 6.8% (0.964) | 3.2% (0.977) | 89.8% (0.494) | 11.1% (0.956) |
| D4-nokey (timing only) | 99.7% (0.247) | 25.0% (0.923) | 64.7% (0.798) | 39.5% (0.882) | 89.9% (0.495) | 85.2% (0.555) |

BPCER @τ10 ≈ 10.3–10.5% for every D4 variant. Seed sd is ≤ 1.9 pp.

### Phase 3b — E5 cross-attack, D5 (BiGRU) vs D3: APCER @τ10, cell = D5 / D3

| trained on | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|
| A1 | 0.0% / 0.0% | 95.2% / 97.9% | 78.9% / 83.3% | 87.6% / 90.0% | 89.4% / 94.7% | 67.1% / 93.1% |
| A2 | 91.6% / 99.9% | 0.0% / 0.2% | 12.8% / 40.1% | 54.7% / 76.0% | 90.1% / 90.1% | 32.4% / 48.5% |
| A3 | 0.9% / 0.0% | 4.8% / 68.7% | 0.1% / 1.0% | 53.1% / 77.9% | 90.2% / 90.5% | 28.7% / 68.1% |
| A4 | 7.4% / 11.8% | 1.0% / 27.0% | 0.7% / 18.1% | 0.4% / 23.7% | 90.3% / 90.8% | 29.8% / 82.6% |
| A5b | 16.2% / 91.2% | 13.0% / 15.0% | 16.4% / 57.4% | 60.9% / 81.7% | 89.6% / 89.0% | 0.1% / 32.8% |

**Off-diagonal averages (attacks accepted when the tested attacker differs from the training attacker):**
- **Excluding the replay A5a column (20 cells: A1–A4, A5b): D5 37.7% vs D3 61.4%.** 61.4% is the figure in
  `RESULTS_PHASE2.md`. 37.7% is the mean of the same 20 cells in `e5_cross_attack.csv`.
- **Including the replay A5a column (25 cells): D5 48.1% vs D3 67.3%.** This is the figure stated in
  `RESULTS_PHASE3B.md`.
- Use the first pair when comparing with Phase 2's text. The A5a column is about 90% for every model, by design.

### Phase 3b — E6 leave-one-attacker-out, D5 vs D3 (E3)

| held-out attacker | D5 APCER @τ10 | D3 APCER @τ10 | D5 AUC | D3 AUC |
|---|---|---|---|---|
| A1 | 3.3% ± 1.9 | 3.7% | 0.980 | 0.978 |
| A2 | 1.2% ± 0.4 | 11.1% | 0.993 | 0.959 |
| A3 | 0.8% ± 0.2 | 19.0% | 0.993 | 0.941 |
| A4 | 52.2% ± 2.0 | 70.0% | 0.719 | 0.583 |
| A5b | 25.0% ± 2.1 | 65.7% | 0.909 | 0.694 |

D5 BPCER @τ10 = 10.6–10.7%.

### Phase 3c — identical-key pairs (POST HOC; `outputs/full/phase3c/DIAGNOSTICS_PHASE3C.md`)

These are the pairs where the human typed exactly the bot's keycode sequence, so the keys are identical and only
timing differs. The detectors, weights and τ10 are frozen from Phase 3b.

| detector | attacker | pairs | humans rejected @τ10 | bots accepted @τ10 | AUC |
|---|---|---|---|---|---|
| D4-primary | A1 | 9,670 | 6.8% | 59.1% | 0.895 |
| D4-primary | A2 | 9,670 | 6.8% | 0.9% | 0.993 |
| D4-primary | A3 | 9,670 | 6.8% | 4.3% | 0.983 |
| D4-primary | A4 | 3,502 | 7.6% | 2.7% | 0.986 |
| D4-primary | A5b | 9,521 | 6.8% | 9.0% | 0.975 |
| D5 (E5, own attacker) | A1 | 9,670 | 24.4% | 0.0% | 1.000 |
| D5 (E5, own attacker) | A2 | 9,670 | 31.5% | 0.0% | 0.994 |
| D5 (E5, own attacker) | A3 | 9,670 | 33.0% | 0.1% | 0.989 |
| D5 (E5, own attacker) | A4 | 3,502 | 19.0% | 0.6% | 0.988 |
| D5 (E5, own attacker) | A5b | 9,521 | 32.9% | 0.2% | 0.992 |

Letters-only D4 (all non-letter/space steps masked; τ10 re-set on val humans) changes AUC by at most 0.008.

**Verdict (3c):**
- **D4 = SIGNAL.** A1, A2, A3 and A5b have byte-identical key sequences but are accepted at 64.0%, 0.6%, 3.0% and
  7.5%. With identical keys D4 still separates A2–A5b (AUC 0.975–0.993), and the letters-only score barely changes.
- **D5 = MIXED.**
  - Its bot detections are timing-driven: with identical keys, AUC ≥ 0.988.
  - But it rejects 19–33% of error-free humans (overall about 10.5%), while D4 rejects these humans at only about 7%.
  - The most plausible cause, supported only by indirect evidence, is a learned key-sequence cue: "no correction /
    CapsLock / arrow keys → bot".
- **Where that cue comes from:** A1, A2, A3 and A5b type the target sentence verbatim (0 Backspace, 0 CapsLock);
  A4 has typos but no CapsLock or arrow keys. No generator was changed.

### Phase 4 — revision analyses, Part A (`docs/PHASE4_RESULTS.md`, `outputs/phase4/full/`)

- **Spec:** `docs/PHASE4_SPEC.md`, SHA-256 `bebccb47bf173f6ec42a99883c81e2601741e85b0510e0cd699b7abc06ef190d`, recorded
  2026-10-06T14:02:58+02:00 in `docs/phase4_prereg_hash.txt`, before any Phase 4 code was written. `kl.phase4_guard` checks it,
  together with every Phase 1–3c freeze record, before and after each step. The spec and results files are read-only.
- **What was run (2026-10-06):**
  - `scripts/phase4_run.py --quick` (smoke run, 70 s; nothing was changed afterwards), then `scripts/phase4_run.py`
    (full, 346 s), then `scripts/phase4_report.py`.
  - It used the saved Phase 1–3b scores, weights and thresholds. Nothing earlier was retrained, re-selected or
    re-thresholded, except the approved D3 refit below.
  - 961 comparisons with saved Phase 1–3c values: 0 failures. `pytest`: 141 tests (19 new in `tests/test_phase4.py`).
- **A1 bootstrap:** the resampling unit is the test user; 1,000 replicates, seed 42, the same replicates as Phases 2
  and 3b. Every bot, A5a included, is matched to one test human session and moves with it. Thresholds are fixed at the
  frozen validation values. D4 / D5 are averaged over seeds inside each replicate, and the across-seed sd is reported
  separately. Sensitivity check for D4-primary: τ is re-set inside each replicate on resampled validation users.
- **Key numbers:**
  - D4-primary APCER @τ10: A2 0.6% [0.5, 0.6]; D1 on A1 5.8% [5.6, 6.0].
  - Cross-attack off-diagonal average excluding A5a: D3 61.4% [61.3, 61.5]; D5 37.7% [37.5, 37.8].
  - Identical-key pairs, BPCER @τ10: D4 6.8–7.6%; D5 (the E5 model trained on the same attacker, seeds 42–44)
    19.0–33.0%.
  - D6 (pre-registered, added after the main evaluation; all three variants) is far weaker than D4-primary on A2–A5b
    (best variant gauss-speed: AUC 0.75–0.89 vs 0.968–0.991).
- **Deviations** (`docs/PHASE4_RESULTS.md` section g):
  1. The spec hash was written to the new `docs/phase4_prereg_hash.txt`, because the earlier record
     `phase3b_prereg_hash.txt` is frozen and read-only.
  2. The five single-attacker D3 forests (E2) were refitted deterministically, with the user's approval. This was
     only for the CI of D3's cross-attack average, and it reproduced all 10 thresholds and 450 saved Phase 2 values
     exactly. The refitted forests were not saved.
  3. The D2 and D3 leave-one-out CIs were copied from Phase 2, which used the same replicates.
  4. The D5 validation logits were recomputed from the frozen weights, as a check only. They reproduce the saved
     τ5 / τ10 exactly.
  5. The identical-key pairs were regenerated with the frozen Phase 3c function. The counts and every per-seed value
     match Phase 3c.
  6. Additions beyond the spec: A1 in the identical-key tables, D4-nokey on that subset, averages including A5a.
  7. Outputs go to `outputs/phase4/{full,quick}/`.
- **D1 on A1, two saved values:**
  - 5.86% (Phase 1, `results/a1_results.json`) and 5.77% (Phase 2, `phase2/open_set_table.csv`).
  - Same model and τ10, but the A1 bots were drawn with different random streams.
  - Both values were kept unchanged.
- **Incident before Phase 4 (2026-10-06):** stray `cl` typed into `src/kl/sessions.py` line 40, in PyCharm.
  - It was removed, and the frozen hash `ff0d10b2…9d053e` is restored.
  - No result was affected (log in `docs/phase3b_bugfixes.md`).
  - All 45 `src/` files in the manifest are now read-only (`docs/environment/readonly_files.txt`).
- **Re-check (read-only):**
  `python -c "import sys; sys.path.insert(0,'src'); from kl.config import load_settings; from kl.phase4_guard import assert_frozen_p4; [assert_frozen_p4(load_settings(None,q)) for q in (False,True)]"`.
  `phase4_run.py` refuses a second test-split evaluation of D6 (`test_access_phase4.json`) without `--allow-rerun`.

---

## e. Incidents and fixes (details: `docs/phase3b_bugfixes.md`)

- **Stray bytes in the frozen `src/kl/features.py` (twice).**
  - 2026-09-25 (mtime 13:36): the file started with `cl`. 2026-09-26 (mtime 10:06): it started with `claude`.
  - Both times this happened outside any tool session, most likely from text typed into the open PyCharm editor.
  - Both times exactly those bytes were removed, which restored the frozen SHA-256 `5be4f320…a63ec`.
  - The guards and tests passed afterwards. No result was ever computed with the corrupted file (it could not even be
    imported).
  - **Since the handover, every frozen source/config file is read-only** (section i, `docs/environment/readonly_files.txt`).
- **Parallel-process launcher bug (2026-09-25 18:17–18:23).**
  - *What happened:* the first Phase 3b launcher was a bash `for` loop. When the user paused training with
    `kill -STOP`, non-interactive bash treated the paused child as finished (`exit=147`). It then started E5, and
    after a second pause started E6, so three processes held the GPU at once, though only one was computing at a
    time.
  - *Response:* everything was stopped. The 8 completed D4 models were checked and are valid.
  - *Fix:* `scripts/run_phase3b.py`.
    - It waits with `Popen.wait()`, which also waits while a child is paused, and runs one part at a time.
    - It holds a launcher lock and a training lock (`flock`).
    - It skips only runs that are COMPLETE and refuses to retrain a run that has already touched the test split.
- **Interrupted runs, retrained from scratch with the same seed on 2026-09-26:** D4-nokey seed 44, D5←A1 seed 42,
  D5←all-but-A1 seed 42. None of them had touched the test split. Training is deterministic.
- **Thermal pauses:** one pause during the whole run, at 10:35:07 on 2026-09-26 (D5←A1 seed 42 after epoch 7, GPU
  85 °C → 62 °C, 30 s). There was no thermal stop.
- **Leftover lock files** `outputs/full/phase3b/.launcher.lock` and `.training.lock` still exist but are not held
  (they are `flock` locks, which the OS releases when the process exits). They are harmless. Leave them or delete them.

---

## f. Known assumptions and limitations

- **HOLD_MIN = 30 ms (assumption)** and FLIGHT_MIN = 60 ms clamp A1's Gaussian draws. A1's Shift handling is a
  reconstruction: Shift is released HOLD_MIN after the key. This produces the unrealistic Shift pattern seen in 3c
  (Shift always released after the capital, median hold 384 ms vs human 248 ms).
- **A4 correction-delay model (assumption).** The delay d ∈ {0,1,2,3} comes from the pool's Backspace-run lengths.
  Runs longer than 4 presses (5.0% of runs) are treated as word-level deletions and left out of the distribution.
  A4 has no CapsLock, arrow keys or cursor editing, and its typo model makes some key pairs humans rarely produce
  (e.g. `Z → Backspace`).
- **γ at the grid edge.** D1's chosen γ = 0.3 is the largest value in the grid {scale, 0.001 … 0.3}. The frozen grid
  was never extended (Revision 2 froze it).
- **MCD rank deficiency (Phase 3a centrality).** The robust covariance is rank-deficient: `shift_overlap_count` has
  zero variance inside the MCD support (condition number 4.6×10²¹). scikit-learn uses a pseudo-inverse, so that
  feature does not count. A classical-covariance distance is reported as a supplement.
- **Phase 3c is post hoc** and **re-used the test split**. The frozen weights and τ10 were used and the Phase 3b test
  ledger was not touched. Its verdict text was written after the tables were read. Its swap test is confounded by
  positional misalignment (section c).
- **Truncation at max_len = 94 keystrokes** (99th percentile of train-human length) for D4/D5. About 1% of human
  sessions (train 0.98%, test 0.95%) and 0.6–0.9% of A4/A5a bots are cut; the other bots are not cut.
- **General scope:** one dataset (a web typing test with transcribed sentences, physical keyboards only); non-adaptive
  attackers (none queries the detector); replays (A5a) cannot be caught from timing, by design; D5 thresholds are
  set on val humans, and D5's human rejection is higher on error-free typists (3c).

---

## g. Open questions and next steps (in this order)

1. **Related-work comparison** (no GPU needed): position the results against BeCAPTCHA-Type and the literature on
   synthetic keystroke attacks (generators, detectors, evaluation protocols). Compare attacker realism and whether
   detectors were evaluated on held-out attackers.
2. **A possible pre-registered D1 + D4 combination.** D1 is strong on A1 (5.8%) where D4 is weak (64.0%), and the
   reverse holds for A2–A5b. A score fusion would have to be pre-registered with thresholds on val humans only
   before any test evaluation.
3. **Key-conditioned features for classical models.** D4-nokey vs D4 shows that conditioning timing on the key is
   what matters (e.g. digraph-normalised timing residuals as session features for an OCSVM).
4. **Enrolment-based liveness.** Real users are self-consistent across sessions (ICC 0.80–0.88) and attackers are
   not. A per-user model could exploit this.
5. **Adaptive / learned attackers** (e.g. a generator trained to fool D4/D5, and attackers that insert
   human-like corrections, CapsLock and cursor keys, which would remove the D5 cue found in 3c).
6. **Writing the manuscript and releasing the code** (cite Dhakal et al. 2018; respect the corpus's non-commercial
   licence; the backup in section j is the code snapshot).

Any of these is a **new phase**: write a spec first, get it approved, and pre-register it by hash as in 3b. Never
edit Phase 1–3c files.

---

## h. How to resume

Environment: `.venv` (Python 3.11.16; `docs/environment/`). Recreate if needed:
```bash
cd /home/nitro/PycharmProjects/keystroke_liveness
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r docs/environment/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

**First, every time: run the guards and the tests.**
```bash
source .venv/bin/activate
pytest -q                                   # 122 tests, synthetic data (2 use outputs/quick if present)
python - <<'EOF'
import sys, json; sys.path.insert(0, "src")
from kl.config import load_settings
from kl.phase3c.guard import assert_frozen_p1234      # Phases 1, 2, 3a, 3b pre-registration, 3b outputs
from kl.phase2.guard import sha256_file, verify_files
for quick in (False, True):
    st = load_settings(None, quick); assert_frozen_p1234(st)
    rec = json.load(open("docs/phase3c_frozen_hashes.json"))            # Phase 3c (no code guard; check by hand)
    verify_files(rec["source_files"])
    mode = "quick" if quick else "full"
    bad = [p for p, h in rec["modes"][mode]["output_files"].items() if sha256_file(__import__("pathlib").Path(p)) != h]
    assert not bad, bad
    print("all guards pass:", mode)
EOF
sha256sum -c <(grep -v '^#' docs/MANIFEST_SHA256.txt) | grep -v ': OK$'   # prints only files that changed
```

**Do not re-run phases in place.** Every result folder is frozen, re-running a step overwrites it and the guards
will then fail. `06_final_test.py` and the Phase 3b training scripts also refuse to touch the test split twice. To
reproduce from scratch, extract the backup (section j) into a **new folder**, link or copy `data/raw/files/`
there, and run:

| phase | quick (2,000 users → `outputs/quick/`) | full (20,000 users → `outputs/full/`) |
|---|---|---|
| 1 | `for s in 01_sample_and_parse 02_clean_and_sessionize 03_split_users 04_extract_features 05_select_hyperparams 06_final_test 07_a1_sanity_check 99_legacy_protocol; do python scripts/$s.py --quick; done` | the same without `--quick` |
| 2 | `python scripts/10_fit_attackers.py --quick`, then `11_generate_bots`, `12_run_experiments`, `13_report_phase2` | the same without `--quick` |
| 3a | `python scripts/20_run_diagnostics.py --quick` (optionally `--only centrality ablation importance consistency`), then `21_report_diagnostics.py --quick` | the same without `--quick` |
| 3b | `python scripts/30_build_sequences.py --quick`, then `nohup python scripts/run_phase3b.py --quick > launcher.out 2>&1 &` (runs 31, 32 e5, 32 e6 and 33, strictly in sequence) | `30_build_sequences.py`, then `nohup python scripts/run_phase3b.py > outputs/full/phase3b/launcher.out 2>&1 &` (about 3 h on the RTX 3050) |
| 3c | `python scripts/40_run_phase3c.py --quick && python scripts/41_report_phase3c.py --quick` | the same without `--quick` (about 4 min) |

A for-loop over the Phase 1 scripts is fine: they are CPU steps and nobody pauses them. Phase 3b must always go
through `run_phase3b.py`. Each step checks the freeze of the phases before it. In a fresh copy, the freeze records
(`docs/phase*_frozen_hashes.json`) describe the ORIGINAL outputs; a from-scratch re-run is expected to reproduce
them byte for byte only where the computation is deterministic, which the guards will tell you.

**Where every result lives:** see section i. The single most useful files are `outputs/full/METHODS.md`,
`phase2/RESULTS_PHASE2.md`, `phase3a/DIAGNOSTICS.md`, `phase3b/RESULTS_PHASE3B.md` and
`phase3c/DIAGNOSTICS_PHASE3C.md`.

---

## i. File map

| path | what it is |
|---|---|
| `CLAUDE.md` | Instructions for Claude Code sessions (START HERE block + per-phase pointers) |
| `README.md` | Phase 1 setup and run guide (covers Phase 1 only) |
| `config.yaml` | Phase 1 protocol config (seed 42, sampling, cleaning, grid, bootstrap, n_jobs); **frozen, read-only**, whole-file hashed |
| `config_phase2.yaml`, `config_phase3a.yaml`, `config_phase3b.yaml` | Per-phase configs (3b is part of the pre-registration hash); read-only |
| `pyproject.toml` | pytest settings (`pythonpath = src`) |
| `requirements.txt` | Minimum dependency versions; exact versions are in `docs/environment/requirements-lock.txt` |
| `docs/PROJECT_STATE.md` | **This file** |
| `docs/PHASE1_SPEC.md` … `docs/PHASE3C_SPEC.md` | Specs per phase (authoritative; Phase 1 includes Revisions 1–2) |
| `docs/PHASE3B_PREREGISTRATION.md`, `docs/phase3b_prereg_hash.txt` | Phase 3b pre-registration and its hash record |
| `docs/PHASE3B_RESUME.md` | Resume log of the interrupted 3b run (now COMPLETE) |
| `docs/phase3b_bugfixes.md` | Incident and bug-fix log (features.py, launcher, thermal guard) |
| `docs/PHASE4_SPEC.md`, `docs/phase4_prereg_hash.txt`, `docs/PHASE4_RESULTS.md` | Phase 4 spec (verbatim), its hash record, results report |
| `docs/phase{1,2,3a,3b,3c}_frozen_hashes.json` | Freeze records (SHA-256 of sources/configs and outputs per mode) |
| `docs/revision1_original_rule_diagnostics.json` | Numbers of the abandoned uniform-box selection rule |
| `docs/environment/` | `environment.txt` (Python, torch, CUDA, GPU, driver, OS), `requirements-lock.txt`, `dataset.txt` (corpus location, file count, SHA-256 of metadata), `readonly_files.txt` |
| `docs/MANIFEST_SHA256.txt` | SHA-256 of every code/doc/config file and every `outputs/full` file < 200 MB |
| `docs/reports/` | Does not exist (no separate reports folder; the reports are the `.md` files in `outputs/full/`) |
| `src/kl/` Phase 1 modules | `io_aalto` (reading the corpus), `sessions` (cleaning, sessions), `split` (user split), `features` (23 features; one code path for humans and bots), `model` (transform + OCSVM), `selection` (attack-free selection), `evaluate`, `report`, `config` |
| `src/kl/attacks/` | Attackers A1–A5b (`a1_gaussian`, `a2_digraph`, `a3_temporal`, `a4_imitator`, `a5_replay`), `pool` (attacker pool statistics), `common` |
| `src/kl/phase2/` | Bot generation, detectors D2/D3, evaluation (bootstrap), report, freeze guard |
| `src/kl/phase3a/` | Diagnostics analyses, plots, report, guard |
| `src/kl/phase3b/` | Sequences/encoder, nets (D4, D5, training), evaluation, runner (locks, run states), thermal guard, regen (bot keystroke regeneration), report, guard |
| `src/kl/phase3c/` | Swap / letters-only masking, keycode realism, guard (Phase 3b freeze record) |
| `scripts/01…07, 99` | Phase 1 pipeline steps; `99` = legacy (leaky) protocol replica |
| `scripts/10…13` | Phase 2: fit attackers, generate bots, experiments E1–E3, report |
| `scripts/20…21` | Phase 3a diagnostics and report |
| `scripts/30…33`, `run_phase3b.py` | Phase 3b: sequences, D4 training (E4), D5 training (E5/E6), report; launcher |
| `scripts/40…41` | Phase 3c artefact check and report |
| `src/kl/phase4_guard.py`, `phase4_bootstrap.py`, `phase4_d6.py`; `scripts/phase4_run.py`, `phase4_report.py` | Phase 4: freeze guard (Phases 1–3c + spec hash), user-cluster bootstrap with seed averaging, detector D6; run and report |
| `tests/` | 141 pytest tests (122 + 19 in `test_phase4.py`) (synthetic data; `synth.py`, `helpers.py` are fixtures) |
| `outputs/full/interim/` | Parquet intermediates: `keystrokes.parquet` (≈ 331 MB), `sessions.parquet` (≈ 319 MB), `session_index.parquet`, `features_{train,val,test,attacker_pool}.parquet`, `test_scores.parquet` |
| `outputs/full/model/` | Frozen D1: `model.joblib`, `thresholds.json` |
| `outputs/full/results/` | Phase 1 results: `selection.json`, `grid_results.csv`, `test_results.json`, `a1_results.json`, `grid_attack_sensitivity.csv` (post hoc), `legacy_vs_clean.md`, `split_manifest.json`, cleaning/feature reports, plots |
| `outputs/full/METHODS.md`, `FEATURES.md` | Paper-ready Phase 1 methods; feature definitions |
| `outputs/full/phase2/` | `RESULTS_PHASE2.md`, `ATTACKS.md`, `open_set_table.csv` (E1), `attack_matrix.csv` (E2), `loao_table.csv` (E3), `realism_report.csv`, `pool.joblib`, `bots_<A>_<split>.parquet`, heatmaps |
| `outputs/full/phase3a/` | `DIAGNOSTICS.md`, `ablation_table.csv`, `centrality_summary.csv`, `permutation_importance.csv`, `user_consistency_*.csv`, `variance_ratio.csv`, figures |
| `outputs/full/phase3b/` | `RESULTS_PHASE3B.md`, `e4_open_set.csv`, `e5_cross_attack.csv`, `e6_loao.csv`, `models/*.pt` (39 models), `scores/*.npz`, `interim/*.npz` (encoded sequences, `encoder.json`), `training_curves.csv`, `runtime_log.json`, `test_access.json` (test ledger), `truncation_report.csv`, figures, `log_archive/`, `launcher.out` |
| `outputs/full/phase3c/` | `DIAGNOSTICS_PHASE3C.md`, `verdict.md`, `realism_*.csv`, `swap_*.csv`, `identical_keys_per_seed.csv`, `letters_only_per_seed.csv`, `reproduction_check.csv` |
| `outputs/full/logs/` | Log of every step (01–41) |
| `outputs/phase4/full/` (and `quick/`) | `results_all.csv` (every cell: point, CI, seed sd, per-seed values), `results_cross_attack_averages.csv`, `reproduction_check.csv`, `thresholds_val_rejection.csv`, `d6_{model,scores}.npz`, `d6_{meta,thresholds}.json`, `d5_val_logits_recomputed.npz`, `test_access_phase4.json`, `run_meta.json`, `interpretation.md`, `logs/` |
| `outputs/quick/` | The same layout for the 2,000-user smoke runs |
| `data/raw/files/` | The corpus (18 GB, 168,593 participant files; not backed up, re-download from Aalto) |
| `backups/` | Backup archives (section j) |

---

## k. Public code release (GitHub, 2026-10-06)

- **Repository:** https://github.com/sulaymonovyusufbek/keystroke_liveness_detection (public, branch `main`).
  - `README.md` was rewritten as the replication guide for all phases.
  - The original Phase 1 guide is kept as `docs/README_PHASE1.md`.
- **Published:** all source code, scripts, tests, configs, `docs/`, and the result reports, tables and figures under
  `outputs/` (`.md` / `.csv` / `.json` / `.png`).
- **Not published** (`.gitignore`): `data/` (the Aalto corpus; non-commercial licence), `backups/`, all parquet /
  joblib / `.pt` / `.npz` files (corpus-derived keystroke sequences, model weights, per-session scores) and logs.
- **Fresh clone:** `pytest` gives 135 passed, 6 skipped. The skipped tests need the unpublished score files. The
  freeze guards fail until the pipeline has been rerun, because they hash the unpublished outputs.
- `tests/test_phase4.py` skip conditions were changed for this: they now check for the binary files the tests read.
  The test logic is unchanged.
- **Expected manifest mismatches:** `README.md`, `.gitignore`, `CLAUDE.md` and `tests/test_phase4.py` now differ
  from `docs/MANIFEST_SHA256.txt`. These are not frozen protocol files.
- **Pushing:** a GitHub personal access token with "Contents: read and write" on this repository is needed. It is
  used only per push and never stored.

---

## j. Backup

_Filled in after the archive was built (this section is the only part of the file written after
`docs/MANIFEST_SHA256.txt` was generated, so this file's manifest hash is expected to differ)._

| item | value |
|---|---|
| archive | `backups/keystroke_liveness_backup_20260926.tar.gz` (created 2026-09-26 13:59) |
| size | 898,029,181 bytes (857 MiB); 604 entries; 1.2 GB extracted |
| SHA-256 | `fb2f9cc70a1d39a8596148873a1a93f6512ab0b1dce6983ff4dccea71a83e88a` (also in `backups/keystroke_liveness_backup_20260926.tar.gz.sha256`) |
| contents | `src/`, `scripts/`, `tests/`, `docs/`, `config*.yaml`, `pyproject.toml`, `requirements.txt`, `CLAUDE.md`, `README.md`, `.gitignore`, `outputs/full/` (including model weights, scores and parquet files) and `outputs/quick/` (185 MB, included so that the two tests that read quick-run outputs also run from the backup) |
| excluded | `data/raw/` (18 GB corpus; re-download it from Aalto University, see `docs/environment/dataset.txt` for the file count and metadata hash to check against), `.venv/`, `__pycache__/`, `.pytest_cache/`. No file was over 2 GB, so none was excluded for size. |
| verification (2026-09-26) | extracted to a temp folder; all 340 `docs/MANIFEST_SHA256.txt` entries matched; the two >200 MB parquet files matched their listed hashes; `pytest`: 122 passed; all freeze guards (Phases 1, 2, 3a, 3b pre-registration, 3b outputs, 3c) passed for full and quick from the extracted copy; temp folder deleted |

Restore: `tar -xzf keystroke_liveness_backup_20260926.tar.gz -C <parent folder>` (it unpacks into `keystroke_liveness/`),
check `sha256sum -c` against the `.sha256` file first, recreate `.venv` (section h), put the corpus at
`data/raw/files/`, then run the guards and `pytest`.
