# Phase 3b pre-registration — sequence-level detectors D4 and D5

Written on 2026-09-25, **before any sequence model has been implemented, trained or evaluated**. Its SHA-256 and that
of `config_phase3b.yaml` (the machine-readable copy of the numbers below) are recorded with the date and time in
`docs/phase3b_prereg_hash.txt`, and every Phase 3b step checks both at start and at the end. Protocol: `docs/PHASE3B_SPEC.md`
(the verbatim spec and the user's amendments). After results are seen, only genuine bugs are fixed. No tuning of any
kind is allowed. Every bug fix is listed in RESULTS_PHASE3B.md, with its reason, and reported to the user.

Only these facts were looked at before writing this file: the code and outputs of Phases 1–3a, and three descriptive
statistics of TRAIN-user human sessions (subsample of ¼ of the parquet row groups): the session-length percentiles
(1/50/90/99%: 18/47/73/94 keys), the number of keycodes (94; 77 with ≥ 20 occurrences) and the hold-time quantisation
(33% of holds are multiples of 8 ms, browser timer resolution; 0.006% are 0 ms). No bot, val or test sequence was examined.

## 1. What is frozen and how it is checked

* Phases 1, 2, 3a are frozen. A new record `docs/phase3a_frozen_hashes.json` stores the SHA-256 of the Phase 3a sources
  (`src/kl/phase3a/*.py`, `scripts/20_*.py`, `scripts/21_*.py`, `config_phase3a.yaml`) and of every file in
  `outputs/<mode>/phase3a/` (full and quick).
* `kl.phase3b.guard.assert_frozen_p123(st)` = the Phase 3a guard `kl.phase3a.guard.assert_frozen_p12` (Phase 1 sources,
  `config.yaml`, config hash, frozen model files; Phase 2 sources and outputs), plus the Phase 3a record, plus the
  pre-registration hashes. Every Phase 3b script calls it first and last. Any mismatch raises `FrozenViolation`.
* No existing detector, attacker, feature, split, config file or output is modified. Phase 3b code lives in
  `src/kl/phase3b/` and `scripts/30…33`. Its outputs go to `outputs/<mode>/phase3b/`.

## 2. Data

### 2.1 Human session sets (Phase 1 split and cleaned sessions, unchanged)

| set | definition | full-run size |
|---|---|---|
| `train_all` | every session in `interim/sessions.parquet` whose user is in the manifest's `train` users | 173,392 |
| `train_20k` | the manifest's `train` sessions (= `features_train.parquet`, the D1/D2/D3 training set) | 20,000 |
| `val` | the manifest's `val` sessions (= `features_val.parquet`) | 43,291 |
| `test` | the manifest's `test` sessions (= `features_test.parquet`) | 43,248 |

Sessions are ordered by `session_id`. Keystrokes in a session are ordered by `np.lexsort((KEYSTROKE_ID, PRESS_TIME))`,
which is the exact order used by `features.py`. Consistency check before anything else: for `train_20k`, `val` and `test`,
`n_keys`, `hold_mean` and `pp_mean` recomputed from the sequences must equal the Phase 1 feature files within 1e-9.
If they don't, the step stops.

### 2.2 Bot sessions (regenerated with the frozen Phase 2 attackers)

Phase 2 stored only features, so bot keystrokes are regenerated as follows:

* Attackers are built by `kl.phase2.generate.build_attackers` from the frozen `outputs/<mode>/phase2/pool.joblib`. A1 uses
  the `a1` section of `config.yaml`.
* Matching and seeds are exactly those of `scripts/11_generate_bots.py`: bot `i` of split `s` for attacker `a` types the
  `SENTENCE` of row `i` of `features_{s}.parquet` (rows in file order). Its RNG is
  `np.random.default_rng([42, ATTACKER_CODES[a], SPLIT_CODES[s], i])`, and it gets `PARTICIPANT_ID = -1`,
  `TEST_SECTION_ID = i`. A bot inherits its matched human's `session_id` and `user_id`. Bots with non-finite features
  are dropped, as in Phase 2 (Phase 2 dropped 0).
* **Verification (train and test, all six attackers):** the regenerated bots, featurised by the unchanged `features.py`,
  must have the stored Phase 2 row count, the same `session_id`/`user_id` in the same order, and
  `max |Δ| ≤ 1e-9` over all 23 features (NaN counts as a mismatch). **Any mismatch stops Phase 3b** (exit code 3,
  `bot_verification.json` written, nothing downstream runs), and the user is told.
* **Val bots (new).** Phase 2 made no val bots, but D5 early stopping needs them (spec §2). So attackers A1, A2, A3, A4
  and A5b generate one bot per `val` human session with the same scheme (`SPLIT_CODES["val"] = 2`, already defined in
  Phase 2). They can't be checked against stored features because none exist. They are used **only** for D5 early
  stopping, never for thresholds, never for evaluation. A5a val bots aren't generated, because A5a is never a training
  attacker.
* `assert_pool_disjoint` (Phase 2) is run on every bot-generation input.

### 2.3 Encoding

* **Vocabulary:** keycodes with ≥ 20 occurrences among all `train_all` keystrokes. Index 0 = PAD (zero embedding,
  `padding_idx`), 1 = UNK (every other keycode, including bot keycodes that are unseen or rare in training), 2… = kept
  keycodes in ascending order. All models share one vocabulary.
* `hold_i = RELEASE_i − PRESS_i`, `pp_i = PRESS_{i+1} − PRESS_i` (ms; `pp ≥ 0` by the sort order),
  `h_i = log1p(hold_i)`, `p_i = log1p(pp_i)`. A hold < 0 in any session is treated as a bug: the step stops. It is not
  clipped. (Cleaning already guarantees hold ≥ 0 for humans.)
* **Standardisation:** `h ← (h − μ_h)/σ_h`, `p ← (p − μ_p)/σ_p`. The means and population std come from all `train_all`
  keystrokes (for `p`, all defined `pp`). All models share these values. D4-20k uses them too, so the only thing it
  changes is the set of sessions the network is trained on.
* **max_len** = `ceil(np.percentile(n_keys of train_all sessions, 99))`, reported. A longer session keeps its first
  `max_len` keystrokes. **The last kept keystroke is treated as the session's last key**: its `p` is masked and its
  next key is PAD, whether or not the session was truncated. Truncations are counted per human set and per
  (attacker, split) in `truncation_report.csv`.
* **Padding:** a batch is padded to its longest sequence, and the GRUs run on `pack_padded_sequence`, so padding never
  enters the recurrence. Masked `p` values that appear as inputs (D5, last key) are set to 0, the standardised mean.
* No dequantisation: the spec defines `h = log1p(hold)`, and the timer quantisation is a property of the human data.

## 3. Detectors (PyTorch, float32, no mixed precision)

**Shared:** embedding 32 (`padding_idx = 0`), GRU hidden size 128, 2 layers, inter-layer dropout 0.1, Adam
(lr 1e-3, default betas/eps, no weight decay), batch size 256, at most 30 epochs, early-stopping patience 3,
`clip_grad_norm_` 1.0 over all parameters. Training seeds 42, 43, 44. Seed `s` sets `torch.manual_seed`,
`numpy` and `random`, and the `torch.Generator` that shuffles every epoch. Determinism: `torch.use_deterministic_algorithms(True)`,
`cudnn.deterministic = True`, `cudnn.benchmark = False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. An op with no
deterministic implementation raises an error instead of running non-deterministically. Device: CUDA if available,
otherwise CPU. The device is printed and logged. Inference batch size 1024.

**Early stopping (both):** after each epoch the validation loss is computed with dropout off. Training stops once
the val loss has gone 3 consecutive epochs without a strict improvement, or after 30 epochs. The weights of the
best-val-loss epoch are restored. Train and val loss are logged for every epoch.

### 3.1 D4 — autoregressive human-timing model (one-class, humans only)

* Input at step `i`: `x_i = [E(key_i), h_{i−1}, p_{i−1}]` (34-d), with `h_{−1} = p_{−1} = 0`. Unidirectional GRU(34→128,
  2 layers, dropout 0.1).
* Head: `z_i = [o_i, E(key_{i+1})]` (160-d; `key_L` = PAD = zero vector) → Linear(160,128) → ReLU → Dropout(0.1) →
  Linear(128, 5K). The outputs are the mixture logits `α`, means `μ_h, μ_p` and raw scales, with
  `σ = softplus(raw) + 0.01`. K = 5.
* Step NLL (standardised log1p units):
  `ℓ_i = −log Σ_k π_k N(h_i; μ_{h,k}, σ²_{h,k}) N(p_i; μ_{p,k}, σ²_{p,k})` for `i < L−1`. On the last step `p` is
  masked, and `ℓ_{L−1}` uses the exact marginal `−log Σ_k π_k N(h; μ_{h,k}, σ²_{h,k})`.
* Training loss: mean `ℓ` over every target step in the batch. Val loss: the same token-level mean over all `val` human
  steps.
* Training data: `train_all`, one pass per epoch in a seeded random order (the last partial batch is kept).
* **Scores** (anomaly: higher = more anomalous; the detector's human score is its negation):
  * **D4-primary:** `m(x)` = mean of `ℓ_i` over the session's `L` steps.
  * **D4-typicality** (secondary): `|m(x) − median_{val humans} m|`, using the same model and seed.
* **Controls** (both scored with the primary score only):
  * **D4-20k:** identical to D4, but trained on `train_20k`.
  * **D4-nokey:** identical to D4, but with no embedding anywhere: input `[h_{i−1}, p_{i−1}]` (2-d), head input `o_i` (128-d).

### 3.2 D5 — supervised bidirectional sequence classifier

* Input at step `i`: `[E(key_i), h_i, p_i]` (the last key's `p` = 0). BiGRU(34→128 per direction, 2 layers, dropout 0.1)
  → masked mean over the `L` real steps (256-d) → Linear(256, 1) = logit. Label 1 = human, 0 = bot. The score is the
  logit (higher = more human).
* Loss: `BCEWithLogits`, unweighted.
* Training sets exactly as for D3: humans = `train_20k`, bots = the regenerated Phase 2 train bots.
  * **E5:** one attacker `A_i ∈ {A1, A2, A3, A4, A5b}` (20,000 bots).
  * **E6:** for each held-out `A_j ∈ {A1, A2, A3, A4, A5b}`, the pooled train bots of the other four (80,000 bots).
* **Class-balanced sampling:** every epoch uses all `N_h` training humans plus `N_h` bots drawn uniformly without
  replacement from the pooled training bots (a new seeded draw each epoch), shuffled together. For E5 that is every bot.
* **Early-stopping val loss:** `0.5 · mean BCE(val humans) + 0.5 · mean BCE(val bots of the training attacker(s))`.
  For E6 the val bots of the four training attackers are pooled. The held-out/test attacker never enters training or
  early stopping, and the code asserts this.

### 3.3 Thresholds

For every detector, score and seed: `τ_f = np.quantile(human score on val humans, f)` for `f ∈ {0.05, 0.10}`
(`kl.model.tau_for_fnr`), exactly as in Phases 1–2. A session is accepted iff score ≥ τ. The threshold function refuses
any input that isn't the val-human set.

## 4. Experiments (TEST split, each (detector, seed) scored once)

* Test data: the `test` humans (43,248 sessions, 2,999 users) and the regenerated test bots of A1, A2, A3, A4, A5a
  and A5b, each matched to a test human session and carrying that human's user.
* Test access is logged in `outputs/<mode>/phase3b/test_access.json`. Loading the test set a second time for the same
  (detector, seed) raises an error unless `--allow-rerun` is given (the same rule as Phase 1 step 06).
* **E4 (open set):** D4-primary, D4-typicality, D4-20k and D4-nokey against A1, A2, A3, A4, A5a, A5b. The frozen D1 and D2
  rows are copied unchanged from Phase 2's `open_set_table.csv`.
* **E5 (cross-attack):** D5 trained on `A_i` (rows A1–A4, A5b), tested on `A_j` (A1–A4, A5a, A5b). D3's rows are copied from
  Phase 2's `attack_matrix.csv`.
* **E6 (leave-one-attacker-out):** D5 trained on all training attackers except `A_j`, tested on `A_j`. D3's (and D1/D2's) rows
  come from Phase 2's `loao_table.csv`.
* **Metrics per cell:** APCER (= FPR, attacks accepted) and BPCER (= FNR, humans rejected) at τ5 and τ10, and ROC-AUC
  (human = positive class, ties count ½). Both naming conventions appear in the headers. **AUC < 0.5 is reported as it
  is.** Scores are never flipped.
* **Aggregation:** for each metric, the mean and the sample sd (ddof = 1) over seeds 42, 43, 44, plus seed 42's value with
  a 95% percentile bootstrap CI. The bootstrap uses 1,000 replicates, resampling test users with replacement through
  `kl.phase2.evaluation.user_weights(n_test_users, 1000, seed=42)` and `cell_metrics`, which are the same resamples as
  the Phase 2 tables.
* **Execution order:** `30_build_sequences` → `31_train_d4` (E4) → `32_train_d5 --experiment e5` →
  `32_train_d5 --experiment e6` → `33_report_phase3b`. Each part writes its tables as soon as it finishes. First a
  `--quick` run (a smoke test only: nothing is changed because of its numbers), then the full run.

## 5. Outputs (`outputs/<mode>/phase3b/`)

`e4_open_set.csv`, `e5_cross_attack.csv`, `e6_loao.csv`, `training_curves.csv` + `training_curves_*.png`,
`d4_primary_score_hist.png` (seed 42, test humans vs each attacker), `e5_heatmaps.png` (D5 mean over seeds next to D3:
APCER@τ10 and AUC), `truncation_report.csv`, `bot_verification.json`, `runtime_log.json` (torch, CUDA and cuDNN
versions, GPU name, device, per-model epochs, best epoch and seconds), and `RESULTS_PHASE3B.md` (a paragraph per
experiment with the numbers, plus a "session-level (D1, D2, D3) vs sequence-level (D4, D5)" table, plus the bug-fix log).
