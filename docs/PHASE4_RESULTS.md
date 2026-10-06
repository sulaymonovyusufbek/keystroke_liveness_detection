# Phase 4 results — revision analyses, Part A (full run)

_Spec: `docs/PHASE4_SPEC.md` (SHA-256 `bebccb47bf173f6ec42a99883c81e2601741e85b0510e0cd699b7abc06ef190d`, recorded 2026-10-06T14:02:58+02:00 in `docs/phase4_prereg_hash.txt`). Every number below is read from `outputs/phase4/full/` by `scripts/phase4_report.py`. Phases 1–3c are frozen; nothing earlier was retrained, re-selected or re-thresholded (the D3 exception is in section g)._

Conventions: APCER = attacks accepted (score ≥ τ); BPCER = humans rejected (score < τ); τ5 / τ10 = thresholds set on validation humans so that 5% / 10% of them are rejected. Brackets = 95% percentile CI over 1,000 test-user cluster-bootstrap replicates (seed 42). D4 / D5: point = mean over seeds 42, 43, 44; each replicate is averaged over the seeds before the CI is taken. Test split: 43,248 human sessions from 2,999 users; one matched bot per human session per attacker.

## (a) Human-only detectors × attackers (A1, A2)

| detector | attacker | APCER @τ10 | APCER @τ5 | ROC-AUC |
|---|---|---|---|---|
| D1 | A1 | 5.8% [5.6, 6.0] | 32.4% [31.9, 32.8] | 0.954 [0.951, 0.957] |
| D1 | A2 | 98.0% [97.8, 98.1] | 99.4% [99.3, 99.4] | 0.462 [0.455, 0.468] |
| D1 | A3 | 96.8% [96.6, 96.9] | 98.8% [98.7, 98.9] | 0.428 [0.422, 0.434] |
| D1 | A4 | 86.8% [86.5, 87.2] | 92.6% [92.4, 92.9] | 0.499 [0.493, 0.505] |
| D1 | A5a | 89.9% [89.6, 90.2] | 95.0% [94.8, 95.2] | 0.506 [0.500, 0.512] |
| D1 | A5b | 93.4% [93.1, 93.6] | 97.2% [97.1, 97.4] | 0.528 [0.522, 0.534] |
| D2 | A1 | 99.9% [99.9, 99.9] | 100.0% [100.0, 100.0] | 0.796 [0.788, 0.805] |
| D2 | A2 | 98.5% [98.3, 98.6] | 99.6% [99.6, 99.7] | 0.318 [0.311, 0.325] |
| D2 | A3 | 98.0% [97.9, 98.1] | 99.5% [99.5, 99.6] | 0.314 [0.307, 0.321] |
| D2 | A4 | 90.6% [90.3, 90.9] | 96.1% [95.9, 96.3] | 0.483 [0.476, 0.491] |
| D2 | A5a | 89.5% [89.2, 89.8] | 94.8% [94.6, 95.0] | 0.509 [0.501, 0.516] |
| D2 | A5b | 93.1% [92.9, 93.4] | 96.8% [96.6, 97.0] | 0.426 [0.419, 0.433] |
| D4-primary | A1 | 64.0% [63.6, 64.4] | 85.2% [84.9, 85.5] | 0.834 [0.827, 0.841] |
| D4-primary | A2 | 0.6% [0.5, 0.6] | 1.9% [1.8, 2.0] | 0.991 [0.989, 0.992] |
| D4-primary | A3 | 3.0% [2.8, 3.1] | 9.1% [8.9, 9.4] | 0.977 [0.974, 0.979] |
| D4-primary | A4 | 1.3% [1.2, 1.4] | 4.5% [4.3, 4.7] | 0.985 [0.983, 0.987] |
| D4-primary | A5a | 89.7% [89.4, 90.0] | 94.5% [94.3, 94.7] | 0.494 [0.487, 0.502] |
| D4-primary | A5b | 7.5% [7.3, 7.7] | 14.2% [13.9, 14.5] | 0.968 [0.966, 0.970] |
| D4-20k | A1 | 74.8% [74.4, 75.2] | 90.5% [90.2, 90.7] | 0.796 [0.788, 0.804] |
| D4-20k | A2 | 1.2% [1.1, 1.3] | 3.8% [3.7, 4.0] | 0.986 [0.984, 0.988] |
| D4-20k | A3 | 6.8% [6.5, 7.0] | 17.8% [17.5, 18.2] | 0.964 [0.961, 0.967] |
| D4-20k | A4 | 3.2% [3.0, 3.3] | 9.3% [9.1, 9.6] | 0.977 [0.975, 0.980] |
| D4-20k | A5a | 89.8% [89.6, 90.1] | 94.5% [94.3, 94.7] | 0.494 [0.486, 0.502] |
| D4-20k | A5b | 11.1% [10.8, 11.4] | 19.5% [19.2, 19.9] | 0.956 [0.953, 0.959] |
| D4-nokey | A1 | 99.7% [99.6, 99.7] | 99.9% [99.9, 99.9] | 0.247 [0.240, 0.255] |
| D4-nokey | A2 | 25.0% [24.6, 25.4] | 45.1% [44.6, 45.5] | 0.923 [0.918, 0.927] |
| D4-nokey | A3 | 64.7% [64.2, 65.1] | 77.8% [77.4, 78.2] | 0.798 [0.791, 0.805] |
| D4-nokey | A4 | 39.5% [39.0, 39.9] | 56.2% [55.8, 56.7] | 0.882 [0.877, 0.887] |
| D4-nokey | A5a | 89.9% [89.6, 90.2] | 94.7% [94.5, 94.9] | 0.495 [0.487, 0.503] |
| D4-nokey | A5b | 85.2% [84.9, 85.5] | 92.1% [91.8, 92.3] | 0.555 [0.547, 0.563] |

BPCER (humans rejected) of each detector. It is the same for every attacker, because the human sessions and replicates are the same:

| detector | BPCER @τ10 | BPCER @τ5 |
|---|---|---|
| D1 | 9.9% [9.4, 10.4] | 4.8% [4.5, 5.1] |
| D2 | 9.6% [9.0, 10.2] | 4.6% [4.1, 5.0] |
| D4-primary | 10.5% [9.9, 11.1] | 5.5% [5.1, 6.0] |
| D4-20k | 10.4% [9.8, 11.1] | 5.5% [5.1, 6.0] |
| D4-nokey | 10.3% [9.7, 11.0] | 5.4% [4.9, 5.8] |
| D6-gauss | 9.9% [9.1, 10.7] | 4.8% [4.2, 5.3] |
| D6-hist | 9.8% [9.0, 10.6] | 4.8% [4.3, 5.4] |
| D6-gauss-speed | 10.3% [9.6, 11.0] | 5.3% [4.9, 5.8] |

D1 is the frozen OCSVM; its bot sessions were re-scored with the frozen model, which reproduces the Phase 2 point estimates and CIs exactly. D2 (Isolation Forest) was not saved and is deterministic, so its rows are copied from `phase2/open_set_table.csv`, whose CIs were computed on the same test-user replicates.

**Sensitivity check (D4-primary only): τ5 / τ10 re-set inside every replicate on resampled validation users** (test users and validation users resampled independently):

| attacker | APCER @τ10, τ fixed | APCER @τ10, τ re-set | APCER @τ5, τ fixed | APCER @τ5, τ re-set |
|---|---|---|---|---|
| A1 | 64.0% [63.6, 64.4] | 64.0% [61.6, 66.3] | 85.2% [84.9, 85.5] | 85.2% [83.3, 86.8] |
| A2 | 0.6% [0.5, 0.6] | 0.6% [0.5, 0.7] | 1.9% [1.8, 2.0] | 1.9% [1.6, 2.2] |
| A3 | 3.0% [2.8, 3.1] | 3.0% [2.6, 3.4] | 9.1% [8.9, 9.4] | 9.1% [8.0, 10.3] |
| A4 | 1.3% [1.2, 1.4] | 1.3% [1.2, 1.5] | 4.5% [4.3, 4.7] | 4.5% [3.9, 5.1] |
| A5a | 89.7% [89.4, 90.0] | 89.7% [89.1, 90.3] | 94.5% [94.3, 94.7] | 94.5% [94.0, 95.0] |
| A5b | 7.5% [7.3, 7.7] | 7.5% [7.0, 8.0] | 14.2% [13.9, 14.5] | 14.2% [13.2, 15.2] |
| BPCER (humans) | 10.5% [9.9, 11.1] | 10.5% [9.6, 11.4] | 5.5% [5.1, 6.0] | 5.5% [4.9, 6.2] |

**A2 (τ5 for every detector and seed).** τ5 was saved for D1, for every D4 variant and seed (`tau5_primary`), and for every D5 model and seed. Nothing had to be computed from scratch. As a check, τ5 / τ10 were recomputed from validation humans only. For D4 the saved validation scores were used. For D5 the validation logits had not been saved, so they were recomputed from the frozen weights, on validation humans only. Validation rejection at the saved thresholds:
τ5 5.00–5.00%, τ10 10.00–10.00% over 43 detector/seed combinations (`thresholds_val_rejection.csv`).

## (b) D3 / D5 leave-one-attacker-out and cross-attack

| held-out attacker | detector | APCER @τ10 | APCER @τ5 | BPCER @τ10 (model) | BPCER @τ5 (model) | ROC-AUC |
|---|---|---|---|---|---|---|
| A1 | D3 | 3.7% [3.5, 3.9] | 11.5% [11.2, 11.8] | 10.0% [9.6, 10.5] | 4.9% [4.6, 5.2] | 0.978 [0.977, 0.980] |
| A1 | D5 | 3.3% [3.1, 3.4] | 9.8% [9.5, 10.0] | 10.7% [10.2, 11.1] | 5.5% [5.2, 5.8] | 0.980 [0.979, 0.981] |
| A2 | D3 | 11.1% [10.8, 11.4] | 30.3% [29.9, 30.7] | 10.1% [9.6, 10.5] | 4.9% [4.6, 5.2] | 0.959 [0.957, 0.961] |
| A2 | D5 | 1.2% [1.1, 1.3] | 3.1% [3.0, 3.3] | 10.7% [10.2, 11.1] | 5.5% [5.1, 5.7] | 0.993 [0.993, 0.994] |
| A3 | D3 | 19.0% [18.6, 19.4] | 36.6% [36.1, 37.1] | 9.8% [9.4, 10.2] | 4.9% [4.6, 5.1] | 0.941 [0.939, 0.944] |
| A3 | D5 | 0.8% [0.7, 0.9] | 2.7% [2.5, 2.8] | 10.7% [10.3, 11.2] | 5.4% [5.1, 5.7] | 0.993 [0.993, 0.994] |
| A4 | D3 | 70.0% [69.6, 70.4] | 74.9% [74.5, 75.3] | 10.0% [9.6, 10.4] | 4.8% [4.6, 5.1] | 0.583 [0.578, 0.587] |
| A4 | D5 | 52.2% [51.8, 52.7] | 54.1% [53.6, 54.5] | 10.6% [10.1, 11.0] | 5.4% [5.1, 5.7] | 0.719 [0.715, 0.724] |
| A5b | D3 | 65.7% [65.2, 66.1] | 76.3% [75.9, 76.7] | 9.5% [9.0, 9.9] | 4.9% [4.5, 5.2] | 0.694 [0.689, 0.701] |
| A5b | D5 | 25.0% [24.6, 25.3] | 36.1% [35.7, 36.5] | 10.6% [10.1, 11.1] | 5.4% [5.0, 5.7] | 0.909 [0.906, 0.912] |

D3 is a single deterministic model per row. Its LOAO rows are copied from `phase2/loao_table.csv`, which used the same replicates. D5 rows are new seed-averaged bootstrap CIs.

**Cross-attack point estimates, APCER @τ10** (rows = trained on, columns = tested on; cell = D5 mean over seeds / D3):

| trained on | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|
| A1 | 0.0% / 0.0% | 95.2% / 97.9% | 78.9% / 83.3% | 87.6% / 90.0% | 89.4% / 94.7% | 67.1% / 93.1% |
| A2 | 91.6% / 99.9% | 0.0% / 0.2% | 12.8% / 40.1% | 54.7% / 76.0% | 90.1% / 90.1% | 32.4% / 48.5% |
| A3 | 0.9% / 0.0% | 4.8% / 68.7% | 0.1% / 1.0% | 53.1% / 77.9% | 90.2% / 90.5% | 28.7% / 68.1% |
| A4 | 7.4% / 11.8% | 1.0% / 27.0% | 0.7% / 18.1% | 0.4% / 23.7% | 90.3% / 90.8% | 29.8% / 82.6% |
| A5b | 16.2% / 91.2% | 13.0% / 15.0% | 16.4% / 57.4% | 60.9% / 81.7% | 89.6% / 89.0% | 0.1% / 32.8% |

**Cross-attack point estimates, APCER @τ5** (rows = trained on, columns = tested on; cell = D5 mean over seeds / D3):

| trained on | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|
| A1 | 0.0% / 0.0% | 98.7% / 97.9% | 85.7% / 83.3% | 92.2% / 90.0% | 95.0% / 94.7% | 78.8% / 93.1% |
| A2 | 98.3% / 100.0% | 0.1% / 1.1% | 22.0% / 59.1% | 60.7% / 82.8% | 94.8% / 95.0% | 48.3% / 67.2% |
| A3 | 3.3% / 3.6% | 11.6% / 84.0% | 0.5% / 5.3% | 57.3% / 80.7% | 95.0% / 95.5% | 44.1% / 84.3% |
| A4 | 18.2% / 21.0% | 2.8% / 50.0% | 2.4% / 31.5% | 1.4% / 40.7% | 95.3% / 95.7% | 41.8% / 90.6% |
| A5b | 27.1% / 99.4% | 23.0% / 31.3% | 30.6% / 81.9% | 67.3% / 89.4% | 94.7% / 94.6% | 0.6% / 54.3% |

**Cross-attack averages with CIs** (the average is formed inside every replicate, so the CI accounts for the correlation between cells):

| detector | average | τ | cells | APCER [95% CI] | across-seed sd |
|---|---|---|---|---|---|
| D3 | off-diagonal, excluding A5a (20 cells) | τ10 | 20 | 61.4% [61.3, 61.5] | – |
| D3 | diagonal (own attacker, 5 cells) | τ10 | 5 | 11.5% [11.4, 11.7] | – |
| D3 | off-diagonal, excluding A5a (20 cells) | τ5 | 20 | 71.1% [71.0, 71.1] | – |
| D3 | diagonal (own attacker, 5 cells) | τ5 | 5 | 20.3% [20.1, 20.4] | – |
| D3 | off-diagonal, including A5a (25 cells) | τ10 | 25 | 67.3% [67.2, 67.4] | – |
| D3 | off-diagonal, including A5a (25 cells) | τ5 | 25 | 75.9% [75.8, 75.9] | – |
| D5 | off-diagonal, excluding A5a (20 cells) | τ10 | 20 | 37.7% [37.5, 37.8] | 0.34 pp |
| D5 | diagonal (own attacker, 5 cells) | τ10 | 5 | 0.1% [0.1, 0.1] | 0.01 pp |
| D5 | off-diagonal, excluding A5a (20 cells) | τ5 | 20 | 45.7% [45.6, 45.8] | 0.40 pp |
| D5 | diagonal (own attacker, 5 cells) | τ5 | 5 | 0.5% [0.5, 0.6] | 0.06 pp |
| D5 | off-diagonal, including A5a (25 cells) | τ10 | 25 | 48.1% [48.0, 48.2] | 0.29 pp |
| D5 | off-diagonal, including A5a (25 cells) | τ5 | 25 | 55.6% [55.5, 55.6] | 0.32 pp |

D3 refit reproduced Phase 2 exactly: **True** (section g).

## (c) Identical-key subset (A3; Table VIII of the manuscript)

These are the Phase 3c pairs. The pair lists themselves were not saved. They were regenerated by calling the frozen Phase 3c function `kl.phase3c.swap.identical_keys` on the frozen Phase 3b sequence files, the same call that `scripts/40_run_phase3c.py` makes, and the pair counts and every per-seed Phase 3c value match exactly. A pair is a test human session and the bot generated for it (same `session_id`, same target sentence) whose keycode sequences are identical: same length and same key at every position. Spec A3 asks for A2, A3, A4 and A5b; A1 is included for completeness.

| attacker | pairs | D4 BPCER @τ10 | D4 BPCER @τ5 | D4 APCER @τ10 | D4 AUC | D5 BPCER @τ10 | D5 BPCER @τ5 | D5 APCER @τ10 | D5 AUC |
|---|---|---|---|---|---|---|---|---|---|
| A1 | 9670 | 6.8% [6.1, 7.5] | 3.6% [3.1, 4.1] | 59.1% [58.2, 60.0] | 0.895 [0.887, 0.902] | 24.4% [22.7, 26.0] | 13.5% [12.2, 14.7] | 0.0% [0.0, 0.0] | 1.000 [1.000, 1.000] |
| A2 | 9670 | 6.8% [6.1, 7.5] | 3.6% [3.1, 4.1] | 0.9% [0.7, 1.1] | 0.993 [0.991, 0.995] | 31.5% [30.2, 32.9] | 15.9% [14.9, 16.9] | 0.0% [0.0, 0.0] | 0.994 [0.993, 0.995] |
| A3 | 9670 | 6.8% [6.1, 7.5] | 3.6% [3.1, 4.1] | 4.3% [3.9, 4.6] | 0.983 [0.980, 0.985] | 33.0% [31.6, 34.3] | 17.0% [16.0, 18.0] | 0.1% [0.1, 0.2] | 0.989 [0.987, 0.990] |
| A4 | 3502 | 7.6% [6.6, 8.6] | 4.0% [3.3, 4.7] | 2.7% [2.2, 3.3] | 0.986 [0.982, 0.989] | 19.0% [17.5, 20.3] | 9.1% [8.0, 10.0] | 0.6% [0.4, 0.9] | 0.988 [0.986, 0.990] |
| A5b | 9521 | 6.8% [6.1, 7.6] | 3.6% [3.1, 4.1] | 9.0% [8.5, 9.6] | 0.975 [0.973, 0.978] | 32.9% [31.8, 34.0] | 16.8% [16.0, 17.7] | 0.2% [0.1, 0.2] | 0.992 [0.991, 0.993] |

D4-nokey (timing-only control) on the same pairs:

| attacker | BPCER @τ10 | BPCER @τ5 | APCER @τ10 | AUC |
|---|---|---|---|---|
| A1 | 4.5% [3.9, 5.0] | 2.3% [2.0, 2.7] | 99.8% [99.8, 99.9] | 0.393 [0.380, 0.406] |
| A2 | 4.5% [3.9, 5.0] | 2.3% [2.0, 2.7] | 27.5% [26.7, 28.4] | 0.958 [0.954, 0.961] |
| A3 | 4.5% [3.9, 5.0] | 2.3% [2.0, 2.7] | 68.3% [67.4, 69.2] | 0.876 [0.869, 0.883] |
| A4 | 4.8% [4.0, 5.6] | 2.4% [1.9, 3.0] | 52.6% [51.0, 54.4] | 0.905 [0.897, 0.913] |
| A5b | 4.5% [4.0, 5.0] | 2.3% [2.0, 2.8] | 85.9% [85.2, 86.6] | 0.661 [0.650, 0.672] |

**D5 variant used in Phase 3c (and here).** `D5 (E5, own attacker)` is the **E5 single-attacker model trained on the
same attacker it is tested on**: `D5<-A` (e.g. `D5<-A2`, trained on train humans + A2 train bots, early stopping on
validation humans + A2 validation bots). It is **not** the leave-one-out (E6) model. All three seeds (42, 43, 44) are
used, each with its own saved τ5 / τ10; the table shows the mean over seeds. Because every attacker has its own D5
model, D5's human scores, and therefore its BPCER on the same human sessions, differ between attackers. D4-primary is
one model per seed, and its BPCER differs between attackers only because the pair subsets differ.

**Pair counts and why A4 has fewer.** Pairs: A1 9,670, A2 9,670, A3 9,670, A4 3,502, A5b 9,521.
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
- 43,248 test human sessions, of which 9,670 are canonical (= the A1/A2/A3 pairs).
- A4: 14,707 of 43,248 bots are typo-free (canonical). 3,501 of those coincide with a canonical human,
  and 1 pair(s) are humans who made exactly the same correction as their A4 bot.
- A5b: 875 bots differ from the canonical sequence (keys reordered by overlapping transplanted timings);
  149 of those fall on canonical humans, hence 9,670 − 149 = 9,521 pairs.

## (d) D6 (pre-registered, added after the main evaluation) vs D4-primary vs D4-nokey

D6 = key-pair timing lookup model, fitted on training humans only and thresholded on validation humans only. Each variant was evaluated once on the test split (ledger `outputs/phase4/full/test_access_phase4.json`). D6 has no seeds.

| attacker | detector | APCER @τ10 | APCER @τ5 | ROC-AUC |
|---|---|---|---|---|
| A1 | D6-gauss | 96.3% [96.1, 96.4] | 99.9% [99.9, 99.9] | 0.785 [0.775, 0.795] |
| A1 | D6-hist | 54.8% [54.3, 55.3] | 97.9% [97.8, 98.1] | 0.891 [0.883, 0.899] |
| A1 | D6-gauss-speed | 99.7% [99.7, 99.8] | 100.0% [99.9, 100.0] | 0.387 [0.379, 0.396] |
| A1 | D4-primary | 64.0% [63.6, 64.4] | 85.2% [84.9, 85.5] | 0.834 [0.827, 0.841] |
| A1 | D4-nokey | 99.7% [99.6, 99.7] | 99.9% [99.9, 99.9] | 0.247 [0.240, 0.255] |
| A2 | D6-gauss | 89.9% [89.6, 90.2] | 96.0% [95.9, 96.2] | 0.633 [0.623, 0.643] |
| A2 | D6-hist | 99.3% [99.2, 99.4] | 99.9% [99.9, 100.0] | 0.596 [0.585, 0.607] |
| A2 | D6-gauss-speed | 37.7% [37.3, 38.2] | 60.3% [59.8, 60.7] | 0.889 [0.883, 0.894] |
| A2 | D4-primary | 0.6% [0.5, 0.6] | 1.9% [1.8, 2.0] | 0.991 [0.989, 0.992] |
| A2 | D4-nokey | 25.0% [24.6, 25.4] | 45.1% [44.6, 45.5] | 0.923 [0.918, 0.927] |
| A3 | D6-gauss | 88.9% [88.6, 89.2] | 95.1% [94.9, 95.3] | 0.557 [0.547, 0.566] |
| A3 | D6-hist | 93.0% [92.7, 93.2] | 97.7% [97.6, 97.8] | 0.610 [0.599, 0.621] |
| A3 | D6-gauss-speed | 71.2% [70.8, 71.7] | 81.8% [81.4, 82.1] | 0.750 [0.743, 0.757] |
| A3 | D4-primary | 3.0% [2.8, 3.1] | 9.1% [8.9, 9.4] | 0.977 [0.974, 0.979] |
| A3 | D4-nokey | 64.7% [64.2, 65.1] | 77.8% [77.4, 78.2] | 0.798 [0.791, 0.805] |
| A4 | D6-gauss | 80.7% [80.4, 81.1] | 91.0% [90.8, 91.3] | 0.660 [0.651, 0.669] |
| A4 | D6-hist | 88.5% [88.2, 88.8] | 96.1% [95.9, 96.3] | 0.676 [0.666, 0.686] |
| A4 | D6-gauss-speed | 49.2% [48.7, 49.7] | 64.1% [63.7, 64.6] | 0.840 [0.835, 0.846] |
| A4 | D4-primary | 1.3% [1.2, 1.4] | 4.5% [4.3, 4.7] | 0.985 [0.983, 0.987] |
| A4 | D4-nokey | 39.5% [39.0, 39.9] | 56.2% [55.8, 56.7] | 0.882 [0.877, 0.887] |
| A5a | D6-gauss | 89.3% [89.0, 89.6] | 94.6% [94.4, 94.9] | 0.502 [0.493, 0.510] |
| A5a | D6-hist | 89.3% [89.1, 89.6] | 94.4% [94.2, 94.7] | 0.503 [0.494, 0.512] |
| A5a | D6-gauss-speed | 89.6% [89.3, 89.9] | 94.3% [94.1, 94.6] | 0.499 [0.492, 0.507] |
| A5a | D4-primary | 89.7% [89.4, 90.0] | 94.5% [94.3, 94.7] | 0.494 [0.487, 0.502] |
| A5a | D4-nokey | 89.9% [89.6, 90.2] | 94.7% [94.5, 94.9] | 0.495 [0.487, 0.503] |
| A5b | D6-gauss | 70.0% [69.6, 70.4] | 83.4% [83.1, 83.8] | 0.746 [0.738, 0.754] |
| A5b | D6-hist | 69.4% [69.0, 69.9] | 84.1% [83.7, 84.4] | 0.771 [0.762, 0.780] |
| A5b | D6-gauss-speed | 42.7% [42.2, 43.2] | 58.5% [58.1, 59.0] | 0.840 [0.835, 0.846] |
| A5b | D4-primary | 7.5% [7.3, 7.7] | 14.2% [13.9, 14.5] | 0.968 [0.966, 0.970] |
| A5b | D4-nokey | 85.2% [84.9, 85.5] | 92.1% [91.8, 92.3] | 0.555 [0.547, 0.563] |

BPCER: see the second table in (a).

**Identical-key subset** (same pairs as in (c)):

| attacker | detector | pairs | AUC | APCER @τ10 | BPCER @τ10 |
|---|---|---|---|---|---|
| A1 | D6-gauss | 9670 | 0.836 [0.823, 0.848] | 95.3% [94.9, 95.7] | 6.9% [6.0, 7.9] |
| A1 | D6-hist | 9670 | 0.911 [0.901, 0.922] | 53.9% [52.9, 55.0] | 7.8% [6.8, 8.9] |
| A1 | D6-gauss-speed | 9670 | 0.537 [0.524, 0.549] | 99.6% [99.5, 99.7] | 4.5% [4.0, 5.1] |
| A1 | D4-primary | 9670 | 0.895 [0.887, 0.902] | 59.1% [58.2, 60.0] | 6.8% [6.1, 7.5] |
| A1 | D4-nokey | 9670 | 0.393 [0.380, 0.406] | 99.8% [99.8, 99.9] | 4.5% [3.9, 5.0] |
| A2 | D6-gauss | 9670 | 0.680 [0.666, 0.694] | 90.9% [90.4, 91.5] | 6.9% [6.0, 7.9] |
| A2 | D6-hist | 9670 | 0.645 [0.630, 0.661] | 99.1% [98.9, 99.3] | 7.8% [6.8, 8.9] |
| A2 | D6-gauss-speed | 9670 | 0.933 [0.928, 0.938] | 42.7% [41.8, 43.6] | 4.5% [4.0, 5.1] |
| A2 | D4-primary | 9670 | 0.993 [0.991, 0.995] | 0.9% [0.7, 1.1] | 6.8% [6.1, 7.5] |
| A2 | D4-nokey | 9670 | 0.958 [0.954, 0.961] | 27.5% [26.7, 28.4] | 4.5% [3.9, 5.0] |
| A3 | D6-gauss | 9670 | 0.600 [0.587, 0.613] | 90.6% [90.0, 91.2] | 6.9% [6.0, 7.9] |
| A3 | D6-hist | 9670 | 0.653 [0.638, 0.667] | 93.1% [92.7, 93.6] | 7.8% [6.8, 8.9] |
| A3 | D6-gauss-speed | 9670 | 0.828 [0.820, 0.836] | 75.8% [75.0, 76.6] | 4.5% [4.0, 5.1] |
| A3 | D4-primary | 9670 | 0.983 [0.980, 0.985] | 4.3% [3.9, 4.6] | 6.8% [6.1, 7.5] |
| A3 | D4-nokey | 9670 | 0.876 [0.869, 0.883] | 68.3% [67.4, 69.2] | 4.5% [3.9, 5.0] |
| A4 | D6-gauss | 3502 | 0.681 [0.665, 0.698] | 82.6% [81.3, 83.9] | 7.5% [6.3, 8.7] |
| A4 | D6-hist | 3502 | 0.705 [0.687, 0.722] | 87.7% [86.7, 88.8] | 8.0% [6.8, 9.3] |
| A4 | D6-gauss-speed | 3502 | 0.873 [0.864, 0.882] | 60.3% [58.6, 61.9] | 4.4% [3.6, 5.2] |
| A4 | D4-primary | 3502 | 0.986 [0.982, 0.989] | 2.7% [2.2, 3.3] | 7.6% [6.6, 8.6] |
| A4 | D4-nokey | 3502 | 0.905 [0.897, 0.913] | 52.6% [51.0, 54.4] | 4.8% [4.0, 5.6] |
| A5b | D6-gauss | 9521 | 0.782 [0.771, 0.793] | 70.5% [69.6, 71.4] | 7.0% [6.0, 8.0] |
| A5b | D6-hist | 9521 | 0.793 [0.781, 0.805] | 69.5% [68.7, 70.5] | 7.8% [6.8, 8.9] |
| A5b | D6-gauss-speed | 9521 | 0.886 [0.880, 0.892] | 47.7% [46.7, 48.7] | 4.5% [4.0, 5.1] |
| A5b | D4-primary | 9521 | 0.975 [0.973, 0.978] | 9.0% [8.5, 9.6] | 6.8% [6.1, 7.6] |
| A5b | D4-nokey | 9521 | 0.661 [0.650, 0.672] | 85.9% [85.2, 86.6] | 4.5% [4.0, 5.0] |

D6 fitting set: 173,392 training-human sessions (11,996 users), 8,380,407
hold steps, 8,207,015 press-to-press steps; vocabulary 86 (incl. PAD, UNK), max_len
94. Contexts with ≥ 50 observations: 80 keys (hold), 1,141 key
pairs and 80 next keys (press-to-press). On validation humans, holds were scored at the key level for
100.00% of steps (global 0.00%), and press-to-press times at the pair / next-key /
global level for 99.78% / 0.22% / 0.00%. Histogram
range (log1p ms): hold [2.773, 6.853], press-to-press [3.178, 7.777].
Thresholds: gauss: τ5 -2.1331, τ10 -1.7913, hist: τ5 -1.9680, τ10 -1.6915, gauss-speed: τ5 -2.8753, τ10 -2.7255
(on −score, i.e. higher = more human).

## (e) Across-seed standard deviation (D4, D5; point estimates, no bootstrap, ddof = 1, 3 seeds)

| model | trained on | tested on | APCER @τ10 | APCER @τ5 | BPCER @τ10 | BPCER @τ5 | AUC |
|---|---|---|---|---|---|---|---|
| D4-primary | attack-free | A1 | 1.35 pp | 1.01 pp | 0.06 pp | 0.02 pp | 0.0028 |
| D4-primary | attack-free | A2 | 0.02 pp | 0.00 pp | 0.06 pp | 0.02 pp | 0.0002 |
| D4-primary | attack-free | A3 | 0.04 pp | 0.10 pp | 0.06 pp | 0.02 pp | 0.0003 |
| D4-primary | attack-free | A4 | 0.04 pp | 0.10 pp | 0.06 pp | 0.02 pp | 0.0003 |
| D4-primary | attack-free | A5a | 0.08 pp | 0.06 pp | 0.06 pp | 0.02 pp | 0.0003 |
| D4-primary | attack-free | A5b | 0.22 pp | 0.38 pp | 0.06 pp | 0.02 pp | 0.0009 |
| D4-20k | attack-free | A1 | 1.92 pp | 1.01 pp | 0.09 pp | 0.03 pp | 0.0092 |
| D4-20k | attack-free | A2 | 0.09 pp | 0.17 pp | 0.09 pp | 0.03 pp | 0.0004 |
| D4-20k | attack-free | A3 | 0.36 pp | 0.46 pp | 0.09 pp | 0.03 pp | 0.0007 |
| D4-20k | attack-free | A4 | 0.22 pp | 0.29 pp | 0.09 pp | 0.03 pp | 0.0005 |
| D4-20k | attack-free | A5a | 0.12 pp | 0.03 pp | 0.09 pp | 0.03 pp | 0.0003 |
| D4-20k | attack-free | A5b | 0.14 pp | 0.13 pp | 0.09 pp | 0.03 pp | 0.0004 |
| D4-nokey | attack-free | A1 | 0.03 pp | 0.00 pp | 0.02 pp | 0.04 pp | 0.0025 |
| D4-nokey | attack-free | A2 | 0.26 pp | 0.22 pp | 0.02 pp | 0.04 pp | 0.0003 |
| D4-nokey | attack-free | A3 | 0.14 pp | 0.33 pp | 0.02 pp | 0.04 pp | 0.0006 |
| D4-nokey | attack-free | A4 | 0.24 pp | 0.04 pp | 0.02 pp | 0.04 pp | 0.0003 |
| D4-nokey | attack-free | A5a | 0.09 pp | 0.09 pp | 0.02 pp | 0.04 pp | 0.0002 |
| D4-nokey | attack-free | A5b | 0.09 pp | 0.08 pp | 0.02 pp | 0.04 pp | 0.0005 |
| D5 (LOAO) | all-but-A1 | A1 | 1.92 pp | 3.64 pp | 0.14 pp | 0.04 pp | 0.0048 |
| D5 (LOAO) | all-but-A2 | A2 | 0.37 pp | 0.63 pp | 0.08 pp | 0.07 pp | 0.0012 |
| D5 (LOAO) | all-but-A3 | A3 | 0.16 pp | 0.30 pp | 0.06 pp | 0.12 pp | 0.0005 |
| D5 (LOAO) | all-but-A4 | A4 | 1.97 pp | 1.33 pp | 0.03 pp | 0.10 pp | 0.0264 |
| D5 (LOAO) | all-but-A5b | A5b | 2.10 pp | 3.16 pp | 0.13 pp | 0.13 pp | 0.0070 |
| D4-primary (identical keys) | attack-free | A1 | 1.38 pp | 1.38 pp | 0.11 pp | 0.04 pp | 0.0013 |
| D5 E5 own attacker (identical keys) | A1 | A1 | 0.00 pp | 0.00 pp | 0.70 pp | 0.41 pp | 0.0000 |
| D4-primary (identical keys) | attack-free | A2 | 0.06 pp | 0.02 pp | 0.11 pp | 0.04 pp | 0.0002 |
| D5 E5 own attacker (identical keys) | A2 | A2 | 0.01 pp | 0.04 pp | 0.28 pp | 0.30 pp | 0.0005 |
| D4-primary (identical keys) | attack-free | A3 | 0.06 pp | 0.09 pp | 0.11 pp | 0.04 pp | 0.0002 |
| D5 E5 own attacker (identical keys) | A3 | A3 | 0.03 pp | 0.16 pp | 0.80 pp | 0.14 pp | 0.0015 |
| D4-primary (identical keys) | attack-free | A4 | 0.23 pp | 0.48 pp | 0.19 pp | 0.23 pp | 0.0002 |
| D5 E5 own attacker (identical keys) | A4 | A4 | 0.02 pp | 0.13 pp | 0.39 pp | 0.53 pp | 0.0008 |
| D4-primary (identical keys) | attack-free | A5b | 0.17 pp | 0.31 pp | 0.12 pp | 0.04 pp | 0.0007 |
| D5 E5 own attacker (identical keys) | A5b | A5b | 0.02 pp | 0.10 pp | 1.40 pp | 0.66 pp | 0.0009 |

D5 cross-attack (E5), across-seed sd of APCER @τ10 (rows = trained on):

| trained on | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|
| A1 | 0.00 pp | 1.33 pp | 3.71 pp | 2.15 pp | 0.30 pp | 2.05 pp |
| A2 | 1.72 pp | 0.00 pp | 0.94 pp | 0.58 pp | 0.10 pp | 1.93 pp |
| A3 | 1.30 pp | 2.98 pp | 0.02 pp | 0.95 pp | 0.23 pp | 4.34 pp |
| A4 | 1.32 pp | 0.10 pp | 0.14 pp | 0.03 pp | 0.17 pp | 2.15 pp |
| A5b | 0.73 pp | 1.63 pp | 1.50 pp | 0.98 pp | 0.23 pp | 0.01 pp |

The across-seed sd of the cross-attack averages is the last column of the averages table in (b).

## (f) Bootstrap rules actually used

- **Unit:** the test user (cluster bootstrap). Each replicate draws the 2,999 test users with
  replacement. This is stored as a draw count per user and applied as a weight to every session of that user.
- **Replicates:** 1,000, seed 42. They are generated by `kl.phase2.evaluation.user_weights(n_test_users, reps, seed) over np.sort(unique test user ids) (= Phase 2 / 3b replicates)`, i.e. **the same
  replicates as the Phase 2 and Phase 3b CIs**. The D1 CIs recomputed here equal the saved Phase 2 CIs to ≤ 1e-15.
- **Matched bots:** for every attacker, including **A5a**, the test bot file holds exactly one bot per test human
  session. The bot carries that session's `session_id` and `user_id`. For A1–A4 and A5b it was generated for the
  session's target sentence. A5a (verbatim replay) ignores the sentence and replays a random attacker-pool recording,
  but it is still generated for, and recorded against, one test human session. **So A5a is resampled with its test
  session**, like every other attacker, and the "resample A5a independently" fallback was not needed. A drawn user
  therefore brings all their human test sessions and every bot matched to them, for every attacker.
  Rule per attacker: A1: one bot per test human session (43248 of 43248); bot.session_id = human session_id, bot.user_id = human user_id; A2: one bot per test human session (43248 of 43248); bot.session_id = human session_id, bot.user_id = human user_id; A3: one bot per test human session (43248 of 43248); bot.session_id = human session_id, bot.user_id = human user_id; A4: one bot per test human session (43248 of 43248); bot.session_id = human session_id, bot.user_id = human user_id; A5a: one bot per test human session (43248 of 43248); bot.session_id = human session_id, bot.user_id = human user_id; A5b: one bot per test human session (43248 of 43248); bot.session_id = human session_id, bot.user_id = human user_id.
- **Thresholds:** fixed at the frozen validation values (primary analysis).
- **Seeds:** for D4 / D5, each replicate computes the metric for seeds 42, 43 and 44 and averages them; the CI is the
  2.5% / 97.5% percentile of these averages. The point estimate is the mean over seeds of the full-sample metric. The
  across-seed sd (section e) is separate.
- **Sensitivity check (D4-primary):** in each replicate the validation users (3,000; 43,291
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
| D1_D2 | 66 s |
| D4 | 109 s |
| D5 | 28 s |
| D3_refit | 55 s |
| identical_keys | 13 s |
| D6 | 27 s |
| D5_val_check | 47 s |
| total | 346 s |

CPU for everything except the D5 validation check (GPU, NVIDIA RTX 3050 Laptop). The quick smoke run took about 70 s.

## Short interpretation

Uncertainty from the test population is small: 98% of the open-set APCER @τ10 CIs span ≤ 1 pp (widest 1.0 pp), and
AUC CIs span ≤ 0.023. Seed variation is of a similar or larger size. The across-seed sd of APCER is at most 1.9 pp at τ10 / 1.0 pp at τ5
for the D4 variants, 2.1 / 3.6 pp for D5 leave-one-out, and 4.3 pp at τ10 (trained on A3, tested on A5b) / 6.4 pp at τ5
(trained on A3, tested on A2) for single D5 cross-attack cells. The across-seed sd of BPCER is ≤ 0.2 pp. Section (e)
lists APCER @τ10 for the cross-attack cells. Re-setting τ on resampled validation
users widens the D4-primary CIs about 2–5×, but no headline value moves outside them. The two cross-attack averages
have narrow, non-overlapping CIs: D3 61.4% [61.3, 61.5] vs D5 37.7% [37.5, 37.8]. On the identical-key pairs,
D5's BPCER CIs (19–33%) do not overlap D4-primary's (6.8–7.6%). D6, a key-conditioned lookup model without a
recurrent network, is far weaker than D4-primary on A2, A3, A4 and A5b in every variant (best: D6-gauss-speed, AUC
0.75–0.89 vs 0.968–0.991), and is also weaker than the timing-only D4-nokey on A2–A4. Key conditioning in a context
table, even with a per-session speed correction, therefore does not reproduce D4's detection rates. (Note: A2's
timings are themselves drawn from attacker-pool digraph distributions, the same kind of statistic that D6 tabulates.)
