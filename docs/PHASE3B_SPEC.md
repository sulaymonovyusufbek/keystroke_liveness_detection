Phase 3b: sequence-level detectors. First save this message verbatim as docs/PHASE3B_SPEC.md and
add a pointer in CLAUDE.md. Phases 1, 2 and 3a are FROZEN: extend the hash guard to cover them
and check it before and after every step. Do not modify any existing detector, attacker, feature
or split.

## 0. Pre-registration (before any training)
- Write docs/PHASE3B_PREREGISTRATION.md listing every choice below (architecture, hyperparameters,
  scores, thresholds, seeds, experiments). Commit it to git and record the commit hash and the
  file's SHA-256 in docs/phase3b_prereg_hash.txt.
- After seeing results, only fix genuine bugs; do not tune. Report every bug fix, with the reason.

## 1. Data
- Splits: reuse the Phase 1 user split and manifest unchanged (train/val/test/attacker_pool).
- Human sequences: build them from the cleaned sessions (Phase 1 sessions.parquet).
- Bot sequences: Phase 2 may have saved only features. If raw keystroke events for the bots are
  not stored, regenerate them deterministically with the FROZEN Phase 2 attackers and the same
  seeds, then assert that features.py on the regenerated bots reproduces the stored Phase 2 bot
  features exactly (tolerance 1e-9). Stop and report if they don't match.
- Per keystroke i (sorted by PRESS_TIME, KEYSTROKE_ID as the tie-break):
    key_i  = KEYCODE (vocabulary built from TRAIN humans; keycodes with < 20 occurrences -> UNK)
    h_i    = log1p(hold_i)
    p_i    = log1p(pp_i), where pp_i = PRESS_{i+1} − PRESS_i (for the last key, p is undefined:
             mask it)
  Note: flight_i = pp_i − hold_i, so (hold, pp) fully determines the timing, including rollover.
  Standardise h and p with the mean/std of TRAIN humans.
- max_len: set it to the 99th percentile of train-human session length (report the value); truncate
  longer sessions and pad shorter ones with a mask. Report how many sessions were truncated per
  split and per attacker.

## 2. Detectors (PyTorch; use CUDA if available, else CPU; print which one)
Fixed hyperparameters for both: embedding 32, GRU hidden 128, 2 layers, dropout 0.1, Adam lr 1e-3,
batch 256, max 30 epochs, early-stopping patience 3, gradient clipping 1.0.
Three training seeds: 42, 43, 44. Deterministic torch settings.

D4, autoregressive human-timing model (one-class, humans only):
- A causal GRU over past steps: input at step i = [emb(key_i), h_{i−1}, p_{i−1}] (zeros at i=0).
- The output head also receives emb(key_{i+1}) (the next key is known from the text).
- It predicts the joint distribution of (h_i, p_i) as a mixture of K=5 diagonal bivariate
  Gaussians. The loss is NLL over unmasked targets (p_i masked on the last key).
- Trained on ALL train-user human sessions. Early stopping on the mean NLL of VAL humans.
- Pre-declared scores:
    D4-primary: session anomaly score = mean per-step NLL (higher = more anomalous).
    D4-typicality (secondary): |mean NLL − median mean NLL of val humans|, a two-sided
      typicality test (Nalisnick et al., 2019). It is included because Phase 3a showed that some
      attackers are more central than humans.
- Pre-declared controls:
    D4-20k: identical, but trained only on the 20,000 train sessions used by D1 (data-size
      control).
    D4-nokey: identical, but with key embeddings removed (timing only). This tests whether
      conditioning on the text is what matters.

D5, supervised sequence classifier (humans + bots):
- A bidirectional GRU (same sizes) over [emb(key_i), h_i, p_i], masked mean pooling, then a linear
  layer to a logit.
- Training sets exactly as for D3:
    E5: train humans + train bots from ONE attacker A_i.
    E6: leave-one-attacker-out.
  Class-balanced sampling. Early stopping on val humans + val bots of the TRAINING attacker(s)
  only (never the test attacker).

Thresholds for every detector and score: τ5/τ10 set on VAL humans only (5% and 10% of humans
rejected), exactly as in Phases 1–2.

## 3. Experiments (TEST split, evaluated once)
E4 open-set: D4-primary, D4-typicality, D4-20k and D4-nokey vs A1, A2, A3, A4, A5a, A5b. Put the
   frozen D1 and D2 rows in the same table for comparison.
E5 cross-attack matrix: D5 trained on A_i (rows A1–A4, A5b), tested on A_j (A1–A4, A5a, A5b).
   Add the D3 matrix from Phase 2 alongside for comparison.
E6 leave-one-attacker-out for D5, next to D3's E3 table.
Metrics per cell:
  - APCER and BPCER at τ5 and τ10, and ROC-AUC.
  - Neural models: mean ± sd across the 3 seeds, plus a 95% bootstrap CI (by user, 1000 reps,
    seeded) for seed 42.
  - Report AUC < 0.5 as it is (do not flip the scores).

## 4. Outputs (outputs/full/phase3b/)
e4_open_set.csv, e5_cross_attack.csv, e6_loao.csv, training curves (train/val loss per epoch per
seed), score histograms per attacker for D4-primary, a heatmap of E5 next to D3's, a
truncation report, a device/runtime log, and RESULTS_PHASE3B.md: a paper-style paragraph per
experiment with the numbers, plus a comparison table "session-level (D1, D2, D3) vs
sequence-level (D4, D5)".

## 5. Tests (pytest)
- Sequence builder: PRESS=[0,200,450], RELEASE=[100,280,560] -> hold=[100,80,110], pp=[200,250],
  with the last p masked.
- Padding: the NLL and D5 logits are identical with and without extra padding.
- Mixture NLL with K=1 equals the analytic bivariate diagonal Gaussian NLL (scipy) within 1e-6.
- Determinism: the same seed gives identical weights after 1 epoch on a tiny dataset.
- The regenerated bots match the Phase 2 features (the assertion from section 1).
- Guards: thresholds use val humans only; the test split is loaded once per detector.

Work order: pre-registration + commit -> implement -> run the tests -> --quick run (smoke test
only; change nothing based on its numbers) -> full run -> report the tables to me with runtime.
If the estimated full runtime exceeds 3 hours, tell me before starting and propose which part
to run first.

---

## Amendments (user's reply, 2026-09-25; the text above is kept verbatim)

1. No git. Remove the git commit from section 0. Instead: write
   docs/PHASE3B_PREREGISTRATION.md, then record its SHA-256 and the current date/time in
   docs/phase3b_prereg_hash.txt. At the start of every later step, verify that the hash is
   unchanged. Tell me when the pre-registration file is ready so I can email it to myself as a
   timestamp before the full run.

2. Yes: install the CUDA build of PyTorch in .venv and add it to requirements.txt. Log the torch
   version, the CUDA version and the GPU name in the runtime log.

3. Yes: regenerate the bot keystrokes with the frozen Phase 2 attackers and seeds, verify them
   against the stored features (1e-9), and stop and report if they don't match.

Runtime: after the --quick run, estimate the full runtime. Then run in this order, without
waiting for me between parts:
  (a) all D4 variants (E4), (b) D5 single-attacker (E5), (c) D5 leave-one-out (E6).
Write results after each part, so partial results survive if something stops. Only ask me
before starting if the total estimate is over 12 hours.
