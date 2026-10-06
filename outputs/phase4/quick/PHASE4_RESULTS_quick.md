# Phase 4 results — revision analyses, Part A (quick run)

_Spec: `docs/PHASE4_SPEC.md` (SHA-256 `bebccb47bf173f6ec42a99883c81e2601741e85b0510e0cd699b7abc06ef190d`, recorded 2026-10-06T14:02:58+02:00 in `docs/phase4_prereg_hash.txt`). Every number below is read from `outputs/phase4/quick/` by `scripts/phase4_report.py`. Phases 1–3c are frozen; nothing earlier was retrained, re-selected or re-thresholded (the D3 exception is in section g)._

Conventions: APCER = attacks accepted (score ≥ τ); BPCER = humans rejected (score < τ); τ5 / τ10 = thresholds set on validation humans so that 5% / 10% of them are rejected. Brackets = 95% percentile CI over 1,000 test-user cluster-bootstrap replicates (seed 42). D4 / D5: point = mean over seeds 42, 43, 44; each replicate is averaged over the seeds before the CI is taken. Test split: 4,344 human sessions from 300 users; one matched bot per human session per attacker.

## (a) Human-only detectors × attackers (A1, A2)

| detector | attacker | APCER @τ10 | APCER @τ5 | ROC-AUC |
|---|---|---|---|---|
| D1 | A1 | 16.8% [15.7, 17.9] | 53.4% [51.9, 54.8] | 0.934 [0.923, 0.944] |
| D1 | A2 | 97.2% [96.6, 97.7] | 99.0% [98.7, 99.3] | 0.526 [0.507, 0.546] |
| D1 | A3 | 94.0% [93.3, 94.6] | 97.6% [97.1, 98.1] | 0.445 [0.429, 0.464] |
| D1 | A4 | 86.3% [85.2, 87.5] | 91.9% [91.1, 92.8] | 0.497 [0.480, 0.514] |
| D1 | A5a | 90.2% [89.5, 91.1] | 95.1% [94.5, 95.7] | 0.493 [0.475, 0.512] |
| D1 | A5b | 93.6% [92.7, 94.3] | 97.3% [96.8, 97.8] | 0.537 [0.519, 0.556] |
| D2 | A1 | 99.9% [99.8, 100.0] | 100.0% [100.0, 100.0] | 0.777 [0.749, 0.803] |
| D2 | A2 | 98.6% [98.2, 99.0] | 99.7% [99.5, 99.8] | 0.315 [0.297, 0.333] |
| D2 | A3 | 95.8% [95.2, 96.4] | 98.4% [98.1, 98.7] | 0.334 [0.316, 0.354] |
| D2 | A4 | 91.3% [90.4, 92.1] | 96.4% [95.9, 97.0] | 0.481 [0.460, 0.502] |
| D2 | A5a | 89.3% [88.3, 90.3] | 94.0% [93.3, 94.8] | 0.520 [0.497, 0.541] |
| D2 | A5b | 93.8% [93.0, 94.6] | 96.8% [96.3, 97.3] | 0.412 [0.391, 0.433] |
| D4-primary | A1 | 71.9% [70.5, 73.1] | 90.8% [90.0, 91.5] | 0.799 [0.775, 0.822] |
| D4-primary | A2 | 1.1% [0.8, 1.4] | 3.9% [3.3, 4.3] | 0.987 [0.984, 0.991] |
| D4-primary | A3 | 5.0% [4.4, 5.7] | 15.4% [14.5, 16.5] | 0.968 [0.961, 0.975] |
| D4-primary | A4 | 2.8% [2.4, 3.2] | 9.2% [8.4, 10.1] | 0.979 [0.973, 0.983] |
| D4-primary | A5a | 90.3% [89.5, 91.1] | 94.9% [94.2, 95.4] | 0.479 [0.454, 0.505] |
| D4-primary | A5b | 13.6% [12.6, 14.6] | 23.6% [22.4, 24.7] | 0.945 [0.937, 0.954] |
| D4-20k | A1 | 71.9% [70.5, 73.1] | 90.8% [90.0, 91.5] | 0.799 [0.775, 0.822] |
| D4-20k | A2 | 1.1% [0.8, 1.4] | 3.9% [3.3, 4.3] | 0.987 [0.984, 0.991] |
| D4-20k | A3 | 5.0% [4.4, 5.7] | 15.4% [14.5, 16.5] | 0.968 [0.961, 0.975] |
| D4-20k | A4 | 2.8% [2.4, 3.2] | 9.2% [8.4, 10.1] | 0.979 [0.973, 0.983] |
| D4-20k | A5a | 90.3% [89.5, 91.1] | 94.9% [94.2, 95.4] | 0.479 [0.454, 0.505] |
| D4-20k | A5b | 13.6% [12.6, 14.6] | 23.6% [22.4, 24.7] | 0.945 [0.937, 0.954] |
| D4-nokey | A1 | 99.4% [99.2, 99.6] | 99.8% [99.7, 99.9] | 0.252 [0.227, 0.276] |
| D4-nokey | A2 | 28.6% [27.3, 29.9] | 49.9% [48.3, 51.3] | 0.904 [0.888, 0.919] |
| D4-nokey | A3 | 65.3% [64.0, 66.7] | 78.8% [77.6, 80.0] | 0.783 [0.759, 0.805] |
| D4-nokey | A4 | 43.9% [42.7, 45.2] | 60.2% [58.9, 61.6] | 0.856 [0.837, 0.874] |
| D4-nokey | A5a | 89.2% [88.2, 90.0] | 94.3% [93.6, 94.9] | 0.462 [0.438, 0.487] |
| D4-nokey | A5b | 86.3% [85.3, 87.4] | 92.6% [91.8, 93.3] | 0.512 [0.487, 0.537] |

BPCER (humans rejected) of each detector. It is the same for every attacker, because the human sessions and replicates are the same:

| detector | BPCER @τ10 | BPCER @τ5 |
|---|---|---|
| D1 | 9.9% [8.4, 11.4] | 5.0% [4.0, 6.1] |
| D2 | 8.5% [6.7, 10.4] | 4.4% [3.2, 5.7] |
| D4-primary | 11.2% [9.2, 13.4] | 5.4% [4.1, 6.8] |
| D4-20k | 11.2% [9.2, 13.4] | 5.4% [4.1, 6.8] |
| D4-nokey | 11.7% [9.5, 14.0] | 6.0% [4.7, 7.7] |
| D6-gauss | 8.7% [6.5, 11.2] | 4.3% [2.8, 5.9] |
| D6-hist | 9.3% [6.7, 12.0] | 4.4% [2.7, 6.4] |
| D6-gauss-speed | 10.7% [8.9, 12.6] | 5.1% [4.0, 6.3] |

D1 is the frozen OCSVM; its bot sessions were re-scored with the frozen model, which reproduces the Phase 2 point estimates and CIs exactly. D2 (Isolation Forest) was not saved and is deterministic, so its rows are copied from `phase2/open_set_table.csv`, whose CIs were computed on the same test-user replicates.

**Sensitivity check (D4-primary only): τ5 / τ10 re-set inside every replicate on resampled validation users** (test users and validation users resampled independently):

| attacker | APCER @τ10, τ fixed | APCER @τ10, τ re-set | APCER @τ5, τ fixed | APCER @τ5, τ re-set |
|---|---|---|---|---|
| A1 | 71.9% [70.5, 73.1] | 71.9% [65.2, 79.6] | 90.8% [90.0, 91.5] | 90.8% [85.6, 94.5] |
| A2 | 1.1% [0.8, 1.4] | 1.1% [0.8, 1.8] | 3.9% [3.3, 4.3] | 3.9% [2.5, 6.1] |
| A3 | 5.0% [4.4, 5.7] | 5.0% [3.5, 7.4] | 15.4% [14.5, 16.5] | 15.4% [10.4, 22.4] |
| A4 | 2.8% [2.4, 3.2] | 2.8% [1.9, 4.3] | 9.2% [8.4, 10.1] | 9.2% [6.1, 14.4] |
| A5a | 90.3% [89.5, 91.1] | 90.3% [88.3, 92.2] | 94.9% [94.2, 95.4] | 94.9% [93.3, 96.4] |
| A5b | 13.6% [12.6, 14.6] | 13.6% [11.5, 16.5] | 23.6% [22.4, 24.7] | 23.6% [19.0, 29.4] |
| BPCER (humans) | 11.2% [9.2, 13.4] | 11.2% [8.2, 14.1] | 5.4% [4.1, 6.8] | 5.4% [3.6, 7.5] |

**A2 (τ5 for every detector and seed).** τ5 was saved for D1, for every D4 variant and seed (`tau5_primary`), and for every D5 model and seed. Nothing had to be computed from scratch. As a check, τ5 / τ10 were recomputed from validation humans only. For D4 the saved validation scores were used. For D5 the validation logits had not been saved, so they were recomputed from the frozen weights, on validation humans only. Validation rejection at the saved thresholds:
τ5 5.01–5.01%, τ10 10.00–10.00% over 43 detector/seed combinations (`thresholds_val_rejection.csv`).

## (b) D3 / D5 leave-one-attacker-out and cross-attack

| held-out attacker | detector | APCER @τ10 | APCER @τ5 | BPCER @τ10 (model) | BPCER @τ5 (model) | ROC-AUC |
|---|---|---|---|---|---|---|
| A1 | D3 | 6.7% [5.8, 7.4] | 39.2% [37.8, 40.8] | 8.6% [7.5, 9.7] | 3.4% [2.7, 4.1] | 0.962 [0.956, 0.968] |
| A1 | D5 | 4.2% [3.8, 4.7] | 11.8% [11.0, 12.5] | 8.2% [7.2, 9.3] | 3.8% [3.1, 4.4] | 0.984 [0.981, 0.986] |
| A2 | D3 | 10.6% [9.6, 11.5] | 28.7% [27.3, 30.0] | 8.4% [7.2, 9.5] | 3.9% [3.2, 4.6] | 0.967 [0.962, 0.972] |
| A2 | D5 | 0.8% [0.6, 1.0] | 2.7% [2.3, 3.0] | 7.7% [6.6, 8.8] | 3.5% [2.9, 4.2] | 0.996 [0.995, 0.997] |
| A3 | D3 | 19.0% [17.7, 20.2] | 39.7% [38.3, 41.2] | 8.7% [7.6, 9.8] | 4.2% [3.5, 5.0] | 0.947 [0.941, 0.954] |
| A3 | D5 | 1.1% [0.8, 1.3] | 3.6% [3.1, 4.0] | 8.4% [7.4, 9.5] | 3.9% [3.2, 4.6] | 0.994 [0.993, 0.995] |
| A4 | D3 | 69.2% [67.8, 70.4] | 75.4% [74.1, 76.8] | 8.4% [7.3, 9.4] | 3.8% [3.1, 4.5] | 0.609 [0.596, 0.623] |
| A4 | D5 | 50.8% [49.4, 52.2] | 53.3% [51.9, 54.6] | 8.6% [7.6, 9.8] | 4.1% [3.4, 4.7] | 0.735 [0.722, 0.749] |
| A5b | D3 | 69.5% [68.2, 70.8] | 81.4% [80.2, 82.5] | 7.7% [6.4, 8.9] | 3.6% [2.8, 4.3] | 0.703 [0.684, 0.722] |
| A5b | D5 | 27.7% [26.4, 28.8] | 38.5% [37.1, 40.0] | 7.8% [6.6, 9.2] | 3.8% [3.0, 4.6] | 0.913 [0.903, 0.922] |

D3 is a single deterministic model per row. Its LOAO rows are copied from `phase2/loao_table.csv`, which used the same replicates. D5 rows are new seed-averaged bootstrap CIs.

**Cross-attack point estimates, APCER @τ10** (rows = trained on, columns = tested on; cell = D5 mean over seeds / D3):

| trained on | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|
| A1 | 0.0% / 0.0% | 98.4% / 98.2% | 84.2% / 86.3% | 90.5% / 91.3% | 91.6% / 95.9% | 70.4% / 93.7% |
| A2 | 99.0% / 100.0% | 0.0% / 0.3% | 11.7% / 41.0% | 50.8% / 77.0% | 88.7% / 91.0% | 31.0% / 53.8% |
| A3 | 0.3% / 0.7% | 4.8% / 69.0% | 0.1% / 1.0% | 54.3% / 74.4% | 88.7% / 90.1% | 27.2% / 70.9% |
| A4 | 20.1% / 15.8% | 0.6% / 31.4% | 0.8% / 19.2% | 0.7% / 28.6% | 90.0% / 92.3% | 33.3% / 86.4% |
| A5b | 17.9% / 100.0% | 9.0% / 14.3% | 14.1% / 66.9% | 59.0% / 84.9% | 87.5% / 81.4% | 0.2% / 18.3% |

**Cross-attack point estimates, APCER @τ5** (rows = trained on, columns = tested on; cell = D5 mean over seeds / D3):

| trained on | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|
| A1 | 0.0% / 0.0% | 99.6% / 99.5% | 89.5% / 90.8% | 94.2% / 95.2% | 95.3% / 98.3% | 80.4% / 97.5% |
| A2 | 99.9% / 100.0% | 0.2% / 1.7% | 21.8% / 60.0% | 58.5% / 84.2% | 95.3% / 95.7% | 47.9% / 72.0% |
| A3 | 2.3% / 7.8% | 13.4% / 83.2% | 0.7% / 6.6% | 57.5% / 77.1% | 94.9% / 94.4% | 43.3% / 85.8% |
| A4 | 38.2% / 36.9% | 2.7% / 50.2% | 2.7% / 31.8% | 2.3% / 45.7% | 95.1% / 96.3% | 45.9% / 93.2% |
| A5b | 29.2% / 100.0% | 18.0% / 28.7% | 27.5% / 86.9% | 65.7% / 91.3% | 93.4% / 88.0% | 0.9% / 36.7% |

**Cross-attack averages with CIs** (the average is formed inside every replicate, so the CI accounts for the correlation between cells):

| detector | average | τ | cells | APCER [95% CI] | across-seed sd |
|---|---|---|---|---|---|
| D3 | off-diagonal, excluding A5a (20 cells) | τ10 | 20 | 63.8% [63.5, 64.0] | – |
| D3 | diagonal (own attacker, 5 cells) | τ10 | 5 | 9.7% [9.3, 10.0] | – |
| D3 | off-diagonal, excluding A5a (20 cells) | τ5 | 20 | 73.6% [73.3, 73.8] | – |
| D3 | diagonal (own attacker, 5 cells) | τ5 | 5 | 18.2% [17.7, 18.6] | – |
| D3 | off-diagonal, including A5a (25 cells) | τ10 | 25 | 69.0% [68.8, 69.2] | – |
| D3 | off-diagonal, including A5a (25 cells) | τ5 | 25 | 77.8% [77.5, 78.0] | – |
| D5 | off-diagonal, excluding A5a (20 cells) | τ10 | 20 | 38.9% [38.6, 39.2] | 0.97 pp |
| D5 | diagonal (own attacker, 5 cells) | τ10 | 5 | 0.2% [0.1, 0.2] | 0.02 pp |
| D5 | off-diagonal, excluding A5a (20 cells) | τ5 | 20 | 46.9% [46.6, 47.2] | 0.89 pp |
| D5 | diagonal (own attacker, 5 cells) | τ5 | 5 | 0.8% [0.7, 0.9] | 0.12 pp |
| D5 | off-diagonal, including A5a (25 cells) | τ10 | 25 | 49.0% [48.7, 49.2] | 0.80 pp |
| D5 | off-diagonal, including A5a (25 cells) | τ5 | 25 | 56.5% [56.2, 56.7] | 0.68 pp |

D3 refit reproduced Phase 2 exactly: **True** (section g).

## (c) Identical-key subset (A3; Table VIII of the manuscript)

These are the Phase 3c pairs. The pair lists themselves were not saved. They were regenerated by calling the frozen Phase 3c function `kl.phase3c.swap.identical_keys` on the frozen Phase 3b sequence files, the same call that `scripts/40_run_phase3c.py` makes, and the pair counts and every per-seed Phase 3c value match exactly. A pair is a test human session and the bot generated for it (same `session_id`, same target sentence) whose keycode sequences are identical: same length and same key at every position. Spec A3 asks for A2, A3, A4 and A5b; A1 is included for completeness.

| attacker | pairs | D4 BPCER @τ10 | D4 BPCER @τ5 | D4 APCER @τ10 | D4 AUC | D5 BPCER @τ10 | D5 BPCER @τ5 | D5 APCER @τ10 | D5 AUC |
|---|---|---|---|---|---|---|---|---|---|
| A1 | 864 | 8.0% [5.5, 10.6] | 3.9% [2.3, 5.5] | 66.3% [63.5, 69.0] | 0.863 [0.833, 0.891] | 21.1% [16.7, 25.5] | 11.2% [8.1, 14.4] | 0.0% [0.0, 0.0] | 1.000 [1.000, 1.000] |
| A2 | 864 | 8.0% [5.5, 10.6] | 3.9% [2.3, 5.5] | 1.8% [1.1, 2.6] | 0.990 [0.984, 0.994] | 27.2% [23.6, 31.3] | 12.8% [10.1, 15.8] | 0.0% [0.0, 0.1] | 0.993 [0.989, 0.996] |
| A3 | 864 | 8.0% [5.5, 10.6] | 3.9% [2.3, 5.5] | 6.9% [5.4, 8.5] | 0.973 [0.962, 0.982] | 26.7% [23.0, 30.7] | 12.1% [9.6, 14.9] | 0.2% [0.0, 0.4] | 0.992 [0.989, 0.994] |
| A4 | 315 | 8.1% [5.2, 11.3] | 4.2% [2.2, 6.6] | 4.6% [2.8, 6.6] | 0.978 [0.964, 0.988] | 16.6% [12.6, 20.8] | 8.3% [5.7, 11.1] | 1.6% [0.6, 2.8] | 0.987 [0.980, 0.992] |
| A5b | 853 | 7.9% [5.4, 10.4] | 3.8% [2.2, 5.4] | 16.8% [14.5, 19.0] | 0.952 [0.938, 0.965] | 31.4% [27.8, 35.1] | 17.2% [14.3, 20.0] | 0.1% [0.0, 0.2] | 0.988 [0.983, 0.991] |

D4-nokey (timing-only control) on the same pairs:

| attacker | BPCER @τ10 | BPCER @τ5 | APCER @τ10 | AUC |
|---|---|---|---|---|
| A1 | 5.7% [3.6, 8.0] | 3.0% [1.6, 4.7] | 99.7% [99.2, 100.0] | 0.404 [0.364, 0.446] |
| A2 | 5.7% [3.6, 8.0] | 3.0% [1.6, 4.7] | 31.5% [28.6, 34.4] | 0.941 [0.925, 0.957] |
| A3 | 5.7% [3.6, 8.0] | 3.0% [1.6, 4.7] | 67.7% [64.8, 70.9] | 0.863 [0.835, 0.889] |
| A4 | 6.3% [3.7, 9.3] | 3.7% [1.8, 6.0] | 57.8% [53.3, 63.0] | 0.882 [0.849, 0.909] |
| A5b | 5.5% [3.5, 7.8] | 3.1% [1.7, 4.7] | 84.8% [82.1, 87.3] | 0.623 [0.587, 0.662] |

**D5 variant used in Phase 3c (and here).** `D5 (E5, own attacker)` is the **E5 single-attacker model trained on the
same attacker it is tested on**: `D5<-A` (e.g. `D5<-A2`, trained on train humans + A2 train bots, early stopping on
validation humans + A2 validation bots). It is **not** the leave-one-out (E6) model. All three seeds (42, 43, 44) are
used, each with its own saved τ5 / τ10; the table shows the mean over seeds. Because every attacker has its own D5
model, D5's human scores, and therefore its BPCER on the same human sessions, differ between attackers. D4-primary is
one model per seed, and its BPCER differs between attackers only because the pair subsets differ.

**Pair counts and why A4 has fewer.** Pairs: A1 864, A2 864, A3 864, A4 315, A5b 853.
- A1, A2 and A3 always type the target sentence's canonical keycode sequence: no errors, and Shift before each
  capital. So they share one set of pairs: the human sessions in which the person typed exactly that sequence, without
  any correction.
- A4 inserts substitution typos with Backspace corrections. Only the A4 bots that happened to make no typo have the
  canonical sequence, so a pair needs both an error-free human and a typo-free A4 bot (or, rarely, the same typo
  corrected in the same way).
- A5b transplants a real pool session's timing onto the target keys. The keystrokes are then ordered by press time,
  as for every session. Where the transplanted timings overlap, two keys swap order (e.g. space pressed before the
  preceding letter), so some A5b bots leave the canonical sequence.

Counts from the frozen test sequence files (descriptive only):
- 4,344 test human sessions, of which 864 are canonical (= the A1/A2/A3 pairs).
- A4: 1,563 of 4,344 bots are typo-free (canonical). 315 of those coincide with a canonical human,
  and 0 pair(s) are humans who made exactly the same correction as their A4 bot.
- A5b: 101 bots differ from the canonical sequence (keys reordered by overlapping transplanted timings);
  11 of those fall on canonical humans, hence 864 − 11 = 853 pairs.

## (d) D6 (pre-registered, added after the main evaluation) vs D4-primary vs D4-nokey

D6 = key-pair timing lookup model, fitted on training humans only and thresholded on validation humans only. Each variant was evaluated once on the test split (ledger `outputs/phase4/full/test_access_phase4.json`). D6 has no seeds.

| attacker | detector | APCER @τ10 | APCER @τ5 | ROC-AUC |
|---|---|---|---|---|
| A1 | D6-gauss | 98.4% [98.0, 98.7] | 99.9% [99.8, 100.0] | 0.772 [0.740, 0.805] |
| A1 | D6-hist | 66.1% [64.5, 67.6] | 99.6% [99.4, 99.7] | 0.886 [0.857, 0.914] |
| A1 | D6-gauss-speed | 99.8% [99.7, 100.0] | 99.9% [99.8, 100.0] | 0.371 [0.344, 0.397] |
| A1 | D4-primary | 71.9% [70.5, 73.1] | 90.8% [90.0, 91.5] | 0.799 [0.775, 0.822] |
| A1 | D4-nokey | 99.4% [99.2, 99.6] | 99.8% [99.7, 99.9] | 0.252 [0.227, 0.276] |
| A2 | D6-gauss | 90.3% [89.4, 91.1] | 95.8% [95.3, 96.4] | 0.657 [0.626, 0.690] |
| A2 | D6-hist | 99.2% [99.0, 99.5] | 100.0% [99.9, 100.0] | 0.616 [0.579, 0.654] |
| A2 | D6-gauss-speed | 35.3% [33.8, 36.7] | 61.6% [60.2, 63.1] | 0.890 [0.876, 0.904] |
| A2 | D4-primary | 1.1% [0.8, 1.4] | 3.9% [3.3, 4.3] | 0.987 [0.984, 0.991] |
| A2 | D4-nokey | 28.6% [27.3, 29.9] | 49.9% [48.3, 51.3] | 0.904 [0.888, 0.919] |
| A3 | D6-gauss | 87.5% [86.6, 88.5] | 94.4% [93.8, 95.1] | 0.601 [0.570, 0.630] |
| A3 | D6-hist | 91.5% [90.6, 92.3] | 97.2% [96.7, 97.6] | 0.639 [0.605, 0.674] |
| A3 | D6-gauss-speed | 67.4% [66.2, 68.6] | 81.4% [80.3, 82.5] | 0.776 [0.755, 0.797] |
| A3 | D4-primary | 5.0% [4.4, 5.7] | 15.4% [14.5, 16.5] | 0.968 [0.961, 0.975] |
| A3 | D4-nokey | 65.3% [64.0, 66.7] | 78.8% [77.6, 80.0] | 0.783 [0.759, 0.805] |
| A4 | D6-gauss | 81.6% [80.5, 82.8] | 91.3% [90.5, 92.0] | 0.677 [0.649, 0.706] |
| A4 | D6-hist | 88.5% [87.6, 89.4] | 96.0% [95.4, 96.6] | 0.689 [0.656, 0.722] |
| A4 | D6-gauss-speed | 48.2% [46.8, 49.7] | 66.0% [64.7, 67.3] | 0.844 [0.826, 0.860] |
| A4 | D4-primary | 2.8% [2.4, 3.2] | 9.2% [8.4, 10.1] | 0.979 [0.973, 0.983] |
| A4 | D4-nokey | 43.9% [42.7, 45.2] | 60.2% [58.9, 61.6] | 0.856 [0.837, 0.874] |
| A5a | D6-gauss | 91.2% [90.3, 92.1] | 95.0% [94.3, 95.6] | 0.519 [0.488, 0.549] |
| A5a | D6-hist | 90.4% [89.5, 91.4] | 95.5% [94.9, 96.2] | 0.515 [0.486, 0.546] |
| A5a | D6-gauss-speed | 89.2% [88.3, 90.1] | 94.0% [93.3, 94.7] | 0.485 [0.462, 0.509] |
| A5a | D4-primary | 90.3% [89.5, 91.1] | 94.9% [94.2, 95.4] | 0.479 [0.454, 0.505] |
| A5a | D4-nokey | 89.2% [88.2, 90.0] | 94.3% [93.6, 94.9] | 0.462 [0.438, 0.487] |
| A5b | D6-gauss | 74.5% [73.1, 75.8] | 86.2% [85.1, 87.2] | 0.740 [0.713, 0.767] |
| A5b | D6-hist | 73.3% [71.9, 74.6] | 88.2% [87.2, 89.1] | 0.758 [0.730, 0.789] |
| A5b | D6-gauss-speed | 46.2% [44.7, 47.8] | 64.0% [62.5, 65.5] | 0.816 [0.799, 0.831] |
| A5b | D4-primary | 13.6% [12.6, 14.6] | 23.6% [22.4, 24.7] | 0.945 [0.937, 0.954] |
| A5b | D4-nokey | 86.3% [85.3, 87.4] | 92.6% [91.8, 93.3] | 0.512 [0.487, 0.537] |

BPCER: see the second table in (a).

**Identical-key subset** (same pairs as in (c)):

| attacker | detector | pairs | AUC | APCER @τ10 | BPCER @τ10 |
|---|---|---|---|---|---|
| A1 | D6-gauss | 864 | 0.818 [0.773, 0.858] | 97.8% [96.9, 98.7] | 5.6% [3.4, 7.9] |
| A1 | D6-hist | 864 | 0.907 [0.875, 0.936] | 63.7% [60.3, 66.7] | 7.2% [4.4, 10.2] |
| A1 | D6-gauss-speed | 864 | 0.516 [0.476, 0.556] | 99.9% [99.6, 100.0] | 5.1% [3.1, 7.5] |
| A1 | D4-primary | 864 | 0.863 [0.833, 0.891] | 66.3% [63.5, 69.0] | 8.0% [5.5, 10.6] |
| A1 | D4-nokey | 864 | 0.404 [0.364, 0.446] | 99.7% [99.2, 100.0] | 5.7% [3.6, 8.0] |
| A2 | D6-gauss | 864 | 0.681 [0.637, 0.723] | 92.4% [90.4, 94.3] | 5.6% [3.4, 7.9] |
| A2 | D6-hist | 864 | 0.643 [0.593, 0.689] | 99.4% [98.9, 99.9] | 7.2% [4.4, 10.2] |
| A2 | D6-gauss-speed | 864 | 0.928 [0.909, 0.945] | 41.2% [38.1, 44.5] | 5.1% [3.1, 7.5] |
| A2 | D4-primary | 864 | 0.990 [0.984, 0.994] | 1.8% [1.1, 2.6] | 8.0% [5.5, 10.6] |
| A2 | D4-nokey | 864 | 0.941 [0.925, 0.957] | 31.5% [28.6, 34.4] | 5.7% [3.6, 8.0] |
| A3 | D6-gauss | 864 | 0.623 [0.577, 0.665] | 89.6% [87.9, 91.5] | 5.6% [3.4, 7.9] |
| A3 | D6-hist | 864 | 0.661 [0.612, 0.705] | 91.9% [90.0, 93.6] | 7.2% [4.4, 10.2] |
| A3 | D6-gauss-speed | 864 | 0.845 [0.814, 0.871] | 74.1% [71.5, 77.0] | 5.1% [3.1, 7.5] |
| A3 | D4-primary | 864 | 0.973 [0.962, 0.982] | 6.9% [5.4, 8.5] | 8.0% [5.5, 10.6] |
| A3 | D4-nokey | 864 | 0.863 [0.835, 0.889] | 67.7% [64.8, 70.9] | 5.7% [3.6, 8.0] |
| A4 | D6-gauss | 315 | 0.661 [0.602, 0.711] | 81.6% [77.4, 86.1] | 5.4% [2.9, 8.3] |
| A4 | D6-hist | 315 | 0.689 [0.630, 0.742] | 87.3% [83.4, 91.4] | 7.6% [4.1, 11.7] |
| A4 | D6-gauss-speed | 315 | 0.860 [0.824, 0.892] | 62.5% [57.7, 67.7] | 5.1% [2.7, 7.9] |
| A4 | D4-primary | 315 | 0.978 [0.964, 0.988] | 4.6% [2.8, 6.6] | 8.1% [5.2, 11.3] |
| A4 | D4-nokey | 315 | 0.882 [0.849, 0.909] | 57.8% [53.3, 63.0] | 6.3% [3.7, 9.3] |
| A5b | D6-gauss | 853 | 0.763 [0.727, 0.799] | 76.8% [74.1, 79.6] | 5.6% [3.4, 8.0] |
| A5b | D6-hist | 853 | 0.763 [0.724, 0.802] | 76.3% [73.6, 79.0] | 7.3% [4.5, 10.3] |
| A5b | D6-gauss-speed | 853 | 0.859 [0.834, 0.885] | 50.8% [47.3, 53.8] | 4.9% [3.0, 7.2] |
| A5b | D4-primary | 853 | 0.952 [0.938, 0.965] | 16.8% [14.5, 19.0] | 7.9% [5.4, 10.4] |
| A5b | D4-nokey | 853 | 0.623 [0.587, 0.662] | 84.8% [82.1, 87.3] | 5.5% [3.5, 7.8] |

D6 fitting set: 17,143 training-human sessions (1,199 users), 827,053
hold steps, 809,910 press-to-press steps; vocabulary 71 (incl. PAD, UNK), max_len
94. Contexts with ≥ 50 observations: 56 keys (hold), 600 key
pairs and 56 next keys (press-to-press). On validation humans, holds were scored at the key level for
99.95% of steps (global 0.05%), and press-to-press times at the pair / next-key /
global level for 98.64% / 1.31% / 0.05%. Histogram
range (log1p ms): hold [2.833, 6.878], press-to-press [3.178, 7.832].
Thresholds: gauss: τ5 -2.1533, τ10 -1.8299, hist: τ5 -2.0247, τ10 -1.7121, gauss-speed: τ5 -2.8995, τ10 -2.7261
(on −score, i.e. higher = more human).

## (e) Across-seed standard deviation (D4, D5; point estimates, no bootstrap, ddof = 1, 3 seeds)

| model | trained on | tested on | APCER @τ10 | APCER @τ5 | BPCER @τ10 | BPCER @τ5 | AUC |
|---|---|---|---|---|---|---|---|
| D4-primary | attack-free | A1 | 1.84 pp | 0.92 pp | 0.13 pp | 0.15 pp | 0.0059 |
| D4-primary | attack-free | A2 | 0.10 pp | 0.39 pp | 0.13 pp | 0.15 pp | 0.0007 |
| D4-primary | attack-free | A3 | 0.56 pp | 1.28 pp | 0.13 pp | 0.15 pp | 0.0018 |
| D4-primary | attack-free | A4 | 0.34 pp | 0.98 pp | 0.13 pp | 0.15 pp | 0.0014 |
| D4-primary | attack-free | A5a | 0.21 pp | 0.31 pp | 0.13 pp | 0.15 pp | 0.0017 |
| D4-primary | attack-free | A5b | 0.40 pp | 0.85 pp | 0.13 pp | 0.15 pp | 0.0019 |
| D4-20k | attack-free | A1 | 1.84 pp | 0.92 pp | 0.13 pp | 0.15 pp | 0.0059 |
| D4-20k | attack-free | A2 | 0.10 pp | 0.39 pp | 0.13 pp | 0.15 pp | 0.0007 |
| D4-20k | attack-free | A3 | 0.56 pp | 1.28 pp | 0.13 pp | 0.15 pp | 0.0018 |
| D4-20k | attack-free | A4 | 0.34 pp | 0.98 pp | 0.13 pp | 0.15 pp | 0.0014 |
| D4-20k | attack-free | A5a | 0.21 pp | 0.31 pp | 0.13 pp | 0.15 pp | 0.0017 |
| D4-20k | attack-free | A5b | 0.40 pp | 0.85 pp | 0.13 pp | 0.15 pp | 0.0019 |
| D4-nokey | attack-free | A1 | 0.05 pp | 0.02 pp | 0.05 pp | 0.21 pp | 0.0057 |
| D4-nokey | attack-free | A2 | 0.64 pp | 0.81 pp | 0.05 pp | 0.21 pp | 0.0027 |
| D4-nokey | attack-free | A3 | 0.85 pp | 0.80 pp | 0.05 pp | 0.21 pp | 0.0056 |
| D4-nokey | attack-free | A4 | 1.02 pp | 0.58 pp | 0.05 pp | 0.21 pp | 0.0037 |
| D4-nokey | attack-free | A5a | 0.02 pp | 0.16 pp | 0.05 pp | 0.21 pp | 0.0003 |
| D4-nokey | attack-free | A5b | 0.07 pp | 0.12 pp | 0.05 pp | 0.21 pp | 0.0006 |
| D5 (LOAO) | all-but-A1 | A1 | 1.94 pp | 3.54 pp | 0.20 pp | 0.12 pp | 0.0041 |
| D5 (LOAO) | all-but-A2 | A2 | 0.42 pp | 0.89 pp | 0.09 pp | 0.29 pp | 0.0013 |
| D5 (LOAO) | all-but-A3 | A3 | 0.21 pp | 0.52 pp | 0.39 pp | 0.28 pp | 0.0011 |
| D5 (LOAO) | all-but-A4 | A4 | 0.48 pp | 0.39 pp | 0.11 pp | 0.10 pp | 0.0280 |
| D5 (LOAO) | all-but-A5b | A5b | 3.24 pp | 3.64 pp | 0.02 pp | 0.17 pp | 0.0124 |
| D4-primary (identical keys) | attack-free | A1 | 1.61 pp | 0.95 pp | 0.29 pp | 0.35 pp | 0.0024 |
| D5 E5 own attacker (identical keys) | A1 | A1 | 0.00 pp | 0.00 pp | 0.81 pp | 0.81 pp | 0.0001 |
| D4-primary (identical keys) | attack-free | A2 | 0.07 pp | 0.29 pp | 0.29 pp | 0.35 pp | 0.0005 |
| D5 E5 own attacker (identical keys) | A2 | A2 | 0.07 pp | 0.07 pp | 2.22 pp | 0.94 pp | 0.0010 |
| D4-primary (identical keys) | attack-free | A3 | 0.64 pp | 1.77 pp | 0.29 pp | 0.35 pp | 0.0016 |
| D5 E5 own attacker (identical keys) | A3 | A3 | 0.07 pp | 0.31 pp | 0.72 pp | 0.57 pp | 0.0005 |
| D4-primary (identical keys) | attack-free | A4 | 0.80 pp | 1.50 pp | 0.48 pp | 0.48 pp | 0.0017 |
| D5 E5 own attacker (identical keys) | A4 | A4 | 0.32 pp | 0.32 pp | 0.48 pp | 1.45 pp | 0.0015 |
| D4-primary (identical keys) | attack-free | A5b | 0.73 pp | 0.82 pp | 0.30 pp | 0.36 pp | 0.0015 |
| D5 E5 own attacker (identical keys) | A5b | A5b | 0.07 pp | 0.18 pp | 1.69 pp | 0.96 pp | 0.0008 |

D5 cross-attack (E5), across-seed sd of APCER @τ10 (rows = trained on):

| trained on | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|
| A1 | 0.00 pp | 0.99 pp | 3.64 pp | 2.33 pp | 0.06 pp | 2.89 pp |
| A2 | 0.35 pp | 0.02 pp | 1.30 pp | 1.60 pp | 0.49 pp | 2.03 pp |
| A3 | 0.31 pp | 1.73 pp | 0.04 pp | 0.79 pp | 0.28 pp | 2.46 pp |
| A4 | 16.97 pp | 0.09 pp | 0.16 pp | 0.12 pp | 0.32 pp | 3.26 pp |
| A5b | 5.08 pp | 0.22 pp | 0.89 pp | 1.20 pp | 0.28 pp | 0.06 pp |

The across-seed sd of the cross-attack averages is the last column of the averages table in (b).

## (f) Bootstrap rules actually used

- **Unit:** the test user (cluster bootstrap). Each replicate draws the 300 test users with
  replacement. This is stored as a draw count per user and applied as a weight to every session of that user.
- **Replicates:** 1,000, seed 42. They are generated by `kl.phase2.evaluation.user_weights(n_test_users, reps, seed) over np.sort(unique test user ids) (= Phase 2 / 3b replicates)`, i.e. **the same
  replicates as the Phase 2 and Phase 3b CIs**. The D1 CIs recomputed here equal the saved Phase 2 CIs to ≤ 1e-15.
- **Matched bots:** for every attacker, including **A5a**, the test bot file holds exactly one bot per test human
  session. The bot carries that session's `session_id` and `user_id`. For A1–A4 and A5b it was generated for the
  session's target sentence. A5a (verbatim replay) ignores the sentence and replays a random attacker-pool recording,
  but it is still generated for, and recorded against, one test human session. **So A5a is resampled with its test
  session**, like every other attacker, and the "resample A5a independently" fallback was not needed. A drawn user
  therefore brings all their human test sessions and every bot matched to them, for every attacker.
  Rule per attacker: A1: one bot per test human session (4344 of 4344); bot.session_id = human session_id, bot.user_id = human user_id; A2: one bot per test human session (4344 of 4344); bot.session_id = human session_id, bot.user_id = human user_id; A3: one bot per test human session (4344 of 4344); bot.session_id = human session_id, bot.user_id = human user_id; A4: one bot per test human session (4344 of 4344); bot.session_id = human session_id, bot.user_id = human user_id; A5a: one bot per test human session (4344 of 4344); bot.session_id = human session_id, bot.user_id = human user_id; A5b: one bot per test human session (4344 of 4344); bot.session_id = human session_id, bot.user_id = human user_id.
- **Thresholds:** fixed at the frozen validation values (primary analysis).
- **Seeds:** for D4 / D5, each replicate computes the metric for seeds 42, 43 and 44 and averages them; the CI is the
  2.5% / 97.5% percentile of these averages. The point estimate is the mean over seeds of the full-sample metric. The
  across-seed sd (section e) is separate.
- **Sensitivity check (D4-primary):** in each replicate the validation users (300; 4,330
  sessions) are also resampled, independently of the test users (multinomial, np.random.default_rng(SeedSequence(seed, spawn_key=(4,))) over np.unique(val user ids)). τ5 / τ10 are then recomputed with
  the frozen rule (`np.quantile` of the validation-human scores, as if the drawn sessions had been repeated), for each
  seed in that replicate.
- **Averages over cells** (cross-attack): the mean over cells is taken inside each replicate, then the percentile CI.
- **Identical-key subset:** the same replicates; only the paired human and bot sessions enter. A human session and its
  paired bot belong to the same user and move together.
- **AUC:** humans positive, ties counted ½, computed with each session weighted by its user's draw count.

## (g) Deviations from the spec, with reasons

1. **Hash record file.** The spec says to append the spec's hash to "the same hash record file used in earlier
   phases". That file, `docs/phase3b_prereg_hash.txt`, is frozen and read-only, and editing it would break the Phase 3b
   freeze guard. The hash was instead written, with a timestamp, to a new file in the same format,
   `docs/phase4_prereg_hash.txt`. It is checked by `kl.phase4_guard` before and after every Phase 4 step.
2. **D3 refit (approved by the user before implementation).** Phase 2 saved neither D3's per-session scores nor the
   fitted forests, and the CI of D3's off-diagonal cross-attack average needs per-session scores. The five
   single-attacker D3 forests (E2) were therefore refitted with the frozen code (`kl.phase2.detectors.fit_d3`), frozen
   data, pre-declared hyper-parameters and seed 42. They were to be used only if they reproduced Phase 2 exactly. Result:
   all 10 thresholds identical, and all 30 cells × 5 metrics × (point, CI lo, CI hi) = 450 saved values reproduced, with
   max |diff| 1.1e-16. Reproduced exactly: **True**. The refitted forests were not saved and
   were used for nothing else. The D3 LOAO models were not refitted: their CIs were already saved.
3. **D2 and D3 LOAO CIs copied, not recomputed.** Their per-session scores were not saved. The saved Phase 2 CIs used
   the identical test-user replicates (shown by the exact D1 reproduction), and D2 / D3 are single deterministic
   models, so no seed averaging applies.
4. **D5 validation logits recomputed (check only).** They were not saved. They were recomputed from the frozen D5
   weights, on validation humans only, to confirm the saved τ5 / τ10; the saved thresholds are the ones used. They are
   saved in `d5_val_logits_recomputed.npz`.
5. **Identical-key pairs regenerated with the frozen Phase 3c function** (they were not saved as a list). The same
   frozen function was called on the same frozen inputs, so the pairs are the same. The counts and every per-seed Phase 3c
   value (BPCER, APCER, AUC at τ10) are reproduced exactly.
6. **Additions beyond the spec:**
   - A1 in the identical-key tables.
   - D4-nokey on the identical-key subset.
   - Cross-attack averages including A5a (25 cells) and the diagonal averages.
   - The AUC CI for the LOAO rows.
7. **Implementation choices the spec left open (fixed before any D6 test evaluation):**
   - D6 Gaussian SD = population SD (ddof = 0), then floored at 0.05.
   - UNK (vocabulary id 1, the keycodes D4 also maps to UNK) is a key context like any other.
   - The histogram range is computed separately for hold and press-to-press values.
   - D6-gauss-speed uses the D6-gauss contexts, subtracts the session mean of the residuals over that session's scored
     h steps and p steps, and scores the centred residual with the N(0, 1) density in z units.
   - Validation-user replicates for the sensitivity check come from an independent stream derived from seed 42.
   - Outputs go to `outputs/phase4/full/` and `outputs/phase4/quick/` (the spec says `outputs/phase4/`).

**D1 on A1: two saved values (both reported, neither changed).** `outputs/full/results/a1_results.json` (Phase 1,
step 07): APCER @τ10 = **5.86%** (2,536 / 43,248). `outputs/full/phase2/open_set_table.csv` (Phase 2, step 12):
**5.77%** (2,496 / 43,248), which rounds to 5.8%. Same frozen D1, same τ10 and same test humans, but **different A1 bot
sessions**. Phase 1 drew all A1 bots from one sequential random stream, `default_rng([42, 3])`
(`kl.evaluate.build_bot_features`). Phase 2 regenerated A1 with one independent stream per bot,
`default_rng([42, attacker, split, i])` (`kl.phase2.generate`), so its results do not depend on chunking or
parallelism. Same generator and parameters, different random draws. Only the Phase 2 bots were saved, so the CIs
here refer to 5.77%.

**Reproduction checks:** 961 comparisons with saved Phase 1–3c values (`reproduction_check.csv`); failed: **0**.

## (h) Runtime

| step | wall time |
|---|---|
| D1_D2 | 6 s |
| D4 | 8 s |
| D5 | 2 s |
| D3_refit | 40 s |
| identical_keys | 1 s |
| D6 | 3 s |
| D5_val_check | 10 s |
| total | 70 s |

CPU for everything except the D5 validation check (GPU, NVIDIA RTX 3050 Laptop). The quick smoke run took about 70 s.

## Short interpretation

_(not written yet)_
