# Phase 3b — sequence-level detectors (results)

_Run mode **quick**; Phase 1 config hash `107e859345c78e77` (frozen); design pre-registered in `docs/PHASE3B_PREREGISTRATION.md` (hash verified at the start and end of every step). Terminology: **APCER = FPR = attacks accepted**, **BPCER = FNR = humans rejected**; neural cells are *mean ± sd over training seeds 42/43/44 [95% user-bootstrap CI of seed 42]*; Phase 2 rows (D1, D2, D3) are a single deterministic model: *value [95% CI]*. AUC < 0.5 is reported as it is._

## Data and set-up

Human sessions: train_all 17,143 (1,199 users), val 4,330 (300 users), test 4,344 (300 users), train_20k 17,143 (1,199 users).
Sequences are truncated at **max_len = 94** keystrokes (99th percentile of train-user human session length); vocabulary 71 ids (69 keycodes with ≥ 20 train occurrences + PAD + UNK); standardisation log1p(hold) 4.660 ± 0.403, log1p(pp) 5.207 ± 0.687.
Bot keystrokes were regenerated with the frozen Phase 2 attackers and seeds: **12 of 12 train/test sets reproduce the stored Phase 2 features exactly** (tolerance 1e-9, observed max |diff| 0); val bots of A1–A4 and A5b (new, used only for D5 early stopping) were generated with the same scheme.

Truncated sessions (first max_len keystrokes kept):

| set | sessions | truncated | fraction |
|---|---|---|---|
| train_all | 17,143 | 163 | 0.95% |
| val | 4,330 | 30 | 0.69% |
| test | 4,344 | 43 | 0.99% |
| train_20k | 17,143 | 163 | 0.95% |
| bot_A1_train | 17,143 | 0 | 0.00% |
| bot_A2_train | 17,143 | 0 | 0.00% |
| bot_A3_train | 17,143 | 0 | 0.00% |
| bot_A4_train | 17,143 | 130 | 0.76% |
| bot_A5a_train | 17,143 | 170 | 0.99% |
| bot_A5b_train | 17,143 | 0 | 0.00% |
| bot_A1_test | 4,344 | 0 | 0.00% |
| bot_A2_test | 4,344 | 0 | 0.00% |
| bot_A3_test | 4,344 | 0 | 0.00% |
| bot_A4_test | 4,344 | 46 | 1.06% |
| bot_A5a_test | 4,344 | 58 | 1.34% |
| bot_A5b_test | 4,344 | 0 | 0.00% |
| bot_A1_val | 4,330 | 0 | 0.00% |
| bot_A2_val | 4,330 | 0 | 0.00% |
| bot_A3_val | 4,330 | 0 | 0.00% |
| bot_A4_val | 4,330 | 25 | 0.58% |
| bot_A5b_val | 4,330 | 0 | 0.00% |

Device: **cuda** (NVIDIA GeForce RTX 3050 6GB Laptop GPU), torch 2.11.0+cu128, CUDA 12.8, cuDNN 91900; deterministic algorithms on.

## E4 — open set: D4 (one-class, humans only) vs the frozen D1 / D2

| detector | tested_on | APCER/FPR @τ5 | APCER/FPR @τ10 | BPCER/FNR @τ5 | BPCER/FNR @τ10 | ROC-AUC |
|---|---|---|---|---|---|---|
| D1 | A1 | 53.4% [51.9, 54.8] | 16.8% [15.7, 17.9] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.934 [0.923, 0.944] |
| D2 | A1 | 100.0% [100.0, 100.0] | 99.9% [99.8, 100.0] | 4.4% [3.2, 5.7] | 8.5% [6.7, 10.4] | 0.777 [0.749, 0.803] |
| D4-primary | A1 | 90.8% ± 0.9 [90.1, 91.7] | 71.9% ± 1.8 [70.8, 73.6] | 5.4% ± 0.2 [4.2, 6.8] | 11.2% ± 0.1 [9.1, 13.4] | 0.799 ± 0.006 [0.771, 0.820] |
| D4-typicality | A1 | 94.4% ± 0.1 [93.8, 95.1] | 85.8% ± 0.8 [84.9, 86.9] | 5.7% ± 0.1 [4.6, 7.1] | 10.7% ± 0.3 [9.2, 12.7] | 0.646 ± 0.006 [0.624, 0.664] |
| D4-20k | A1 | 90.8% ± 0.9 [90.1, 91.7] | 71.9% ± 1.8 [70.8, 73.6] | 5.4% ± 0.2 [4.2, 6.8] | 11.2% ± 0.1 [9.1, 13.4] | 0.799 ± 0.006 [0.771, 0.820] |
| D4-nokey | A1 | 99.8% ± 0.0 [99.7, 99.9] | 99.4% ± 0.0 [99.2, 99.7] | 6.0% ± 0.2 [4.6, 7.6] | 11.7% ± 0.0 [9.5, 14.0] | 0.252 ± 0.006 [0.223, 0.273] |
| D1 | A2 | 99.0% [98.7, 99.3] | 97.2% [96.6, 97.7] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.526 [0.507, 0.546] |
| D2 | A2 | 99.7% [99.5, 99.8] | 98.6% [98.2, 99.0] | 4.4% [3.2, 5.7] | 8.5% [6.7, 10.4] | 0.315 [0.297, 0.333] |
| D4-primary | A2 | 3.9% ± 0.4 [2.9, 3.9] | 1.1% ± 0.1 [0.7, 1.3] | 5.4% ± 0.2 [4.2, 6.8] | 11.2% ± 0.1 [9.1, 13.4] | 0.987 ± 0.001 [0.985, 0.991] |
| D4-typicality | A2 | 5.8% ± 0.5 [4.6, 5.8] | 2.5% ± 0.2 [1.8, 2.6] | 5.7% ± 0.1 [4.6, 7.1] | 10.7% ± 0.3 [9.2, 12.7] | 0.983 ± 0.001 [0.981, 0.988] |
| D4-20k | A2 | 3.9% ± 0.4 [2.9, 3.9] | 1.1% ± 0.1 [0.7, 1.3] | 5.4% ± 0.2 [4.2, 6.8] | 11.2% ± 0.1 [9.1, 13.4] | 0.987 ± 0.001 [0.985, 0.991] |
| D4-nokey | A2 | 49.9% ± 0.8 [48.1, 51.1] | 28.6% ± 0.6 [27.2, 29.7] | 6.0% ± 0.2 [4.6, 7.6] | 11.7% ± 0.0 [9.5, 14.0] | 0.904 ± 0.003 [0.890, 0.920] |
| D1 | A3 | 97.6% [97.1, 98.1] | 94.0% [93.3, 94.6] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.445 [0.429, 0.464] |
| D2 | A3 | 98.4% [98.1, 98.7] | 95.8% [95.2, 96.4] | 4.4% [3.2, 5.7] | 8.5% [6.7, 10.4] | 0.334 [0.316, 0.354] |
| D4-primary | A3 | 15.4% ± 1.3 [13.2, 15.2] | 5.0% ± 0.6 [3.8, 5.0] | 5.4% ± 0.2 [4.2, 6.8] | 11.2% ± 0.1 [9.1, 13.4] | 0.968 ± 0.002 [0.963, 0.977] |
| D4-typicality | A3 | 21.9% ± 1.6 [19.0, 21.3] | 10.5% ± 0.6 [8.9, 10.7] | 5.7% ± 0.1 [4.6, 7.1] | 10.7% ± 0.3 [9.2, 12.7] | 0.953 ± 0.003 [0.949, 0.963] |
| D4-20k | A3 | 15.4% ± 1.3 [13.2, 15.2] | 5.0% ± 0.6 [3.8, 5.0] | 5.4% ± 0.2 [4.2, 6.8] | 11.2% ± 0.1 [9.1, 13.4] | 0.968 ± 0.002 [0.963, 0.977] |
| D4-nokey | A3 | 78.8% ± 0.8 [77.5, 79.9] | 65.3% ± 0.9 [63.5, 66.2] | 6.0% ± 0.2 [4.6, 7.6] | 11.7% ± 0.0 [9.5, 14.0] | 0.783 ± 0.006 [0.763, 0.808] |
| D1 | A4 | 91.9% [91.1, 92.8] | 86.3% [85.2, 87.5] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.497 [0.480, 0.514] |
| D2 | A4 | 96.4% [95.9, 97.0] | 91.3% [90.4, 92.1] | 4.4% [3.2, 5.7] | 8.5% [6.7, 10.4] | 0.481 [0.460, 0.502] |
| D4-primary | A4 | 9.2% ± 1.0 [7.6, 9.3] | 2.8% ± 0.3 [2.1, 3.0] | 5.4% ± 0.2 [4.2, 6.8] | 11.2% ± 0.1 [9.1, 13.4] | 0.979 ± 0.001 [0.975, 0.985] |
| D4-typicality | A4 | 13.9% ± 1.2 [11.6, 13.5] | 6.1% ± 0.4 [5.0, 6.4] | 5.7% ± 0.1 [4.6, 7.1] | 10.7% ± 0.3 [9.2, 12.7] | 0.969 ± 0.002 [0.966, 0.977] |
| D4-20k | A4 | 9.2% ± 1.0 [7.6, 9.3] | 2.8% ± 0.3 [2.1, 3.0] | 5.4% ± 0.2 [4.2, 6.8] | 11.2% ± 0.1 [9.1, 13.4] | 0.979 ± 0.001 [0.975, 0.985] |
| D4-nokey | A4 | 60.2% ± 0.6 [59.1, 61.8] | 43.9% ± 1.0 [42.1, 44.9] | 6.0% ± 0.2 [4.6, 7.6] | 11.7% ± 0.0 [9.5, 14.0] | 0.856 ± 0.004 [0.839, 0.876] |
| D1 | A5a | 95.1% [94.5, 95.7] | 90.2% [89.5, 91.1] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.493 [0.475, 0.512] |
| D2 | A5a | 94.0% [93.3, 94.8] | 89.3% [88.3, 90.3] | 4.4% [3.2, 5.7] | 8.5% [6.7, 10.4] | 0.520 [0.497, 0.541] |
| D4-primary | A5a | 94.9% ± 0.3 [94.1, 95.4] | 90.3% ± 0.2 [89.6, 91.3] | 5.4% ± 0.2 [4.2, 6.8] | 11.2% ± 0.1 [9.1, 13.4] | 0.479 ± 0.002 [0.452, 0.503] |
| D4-typicality | A5a | 94.5% ± 0.3 [93.6, 95.0] | 89.8% ± 0.2 [89.0, 90.8] | 5.7% ± 0.1 [4.6, 7.1] | 10.7% ± 0.3 [9.2, 12.7] | 0.498 ± 0.003 [0.483, 0.521] |
| D4-20k | A5a | 94.9% ± 0.3 [94.1, 95.4] | 90.3% ± 0.2 [89.6, 91.3] | 5.4% ± 0.2 [4.2, 6.8] | 11.2% ± 0.1 [9.1, 13.4] | 0.479 ± 0.002 [0.452, 0.503] |
| D4-nokey | A5a | 94.3% ± 0.2 [93.7, 95.1] | 89.2% ± 0.0 [88.2, 90.0] | 6.0% ± 0.2 [4.6, 7.6] | 11.7% ± 0.0 [9.5, 14.0] | 0.462 ± 0.000 [0.438, 0.488] |
| D1 | A5b | 97.3% [96.8, 97.8] | 93.6% [92.7, 94.3] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.537 [0.519, 0.556] |
| D2 | A5b | 96.8% [96.3, 97.3] | 93.8% [93.0, 94.6] | 4.4% [3.2, 5.7] | 8.5% [6.7, 10.4] | 0.412 [0.391, 0.433] |
| D4-primary | A5b | 23.6% ± 0.9 [21.9, 24.2] | 13.6% ± 0.4 [12.2, 14.3] | 5.4% ± 0.2 [4.2, 6.8] | 11.2% ± 0.1 [9.1, 13.4] | 0.945 ± 0.002 [0.939, 0.956] |
| D4-typicality | A5b | 28.9% ± 0.6 [27.0, 29.6] | 19.2% ± 0.3 [17.7, 20.1] | 5.7% ± 0.1 [4.6, 7.1] | 10.7% ± 0.3 [9.2, 12.7] | 0.915 ± 0.003 [0.910, 0.927] |
| D4-20k | A5b | 23.6% ± 0.9 [21.9, 24.2] | 13.6% ± 0.4 [12.2, 14.3] | 5.4% ± 0.2 [4.2, 6.8] | 11.2% ± 0.1 [9.1, 13.4] | 0.945 ± 0.002 [0.939, 0.956] |
| D4-nokey | A5b | 92.6% ± 0.1 [91.9, 93.5] | 86.3% ± 0.1 [85.1, 87.4] | 6.0% ± 0.2 [4.6, 7.6] | 11.7% ± 0.0 [9.5, 14.0] | 0.512 ± 0.001 [0.488, 0.538] |

At the τ10 operating point (10% of validation humans rejected) — A1: D4-primary accepts 71.9% of attacks at τ10 (AUC 0.799) vs D1 16.8% (AUC 0.934); D4-typicality 85.8%; A2: D4-primary accepts 1.1% of attacks at τ10 (AUC 0.987) vs D1 97.2% (AUC 0.526); D4-typicality 2.5%; A3: D4-primary accepts 5.0% of attacks at τ10 (AUC 0.968) vs D1 94.0% (AUC 0.445); D4-typicality 10.5%; A4: D4-primary accepts 2.8% of attacks at τ10 (AUC 0.979) vs D1 86.3% (AUC 0.497); D4-typicality 6.1%; A5a: D4-primary accepts 90.3% of attacks at τ10 (AUC 0.479) vs D1 90.2% (AUC 0.493); D4-typicality 89.8%; A5b: D4-primary accepts 13.6% of attacks at τ10 (AUC 0.945) vs D1 93.6% (AUC 0.537); D4-typicality 19.2%. D4-20k (data-size control) and D4-nokey (timing only, no key conditioning) are in the table; A1: D4 71.9%, D4-20k 71.9%, D4-nokey 99.4%; A2: D4 1.1%, D4-20k 1.1%, D4-nokey 28.6%; A3: D4 5.0%, D4-20k 5.0%, D4-nokey 65.3%; A4: D4 2.8%, D4-20k 2.8%, D4-nokey 43.9%; A5a: D4 90.3%, D4-20k 90.3%, D4-nokey 89.2%; A5b: D4 13.6%, D4-20k 13.6%, D4-nokey 86.3%.

## E5 — cross-attack matrix: D5 (supervised BiGRU) next to D3 (Random Forest, Phase 2)

**APCER / FPR @τ10 (attacks accepted)** — rows: trained on, columns: tested on; each cell `D5 mean / D3`.

| trained on | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|
| A1 | 0.0% / 0.0% | 98.4% / 98.2% | 84.2% / 86.3% | 90.5% / 91.3% | 91.6% / 95.9% | 70.4% / 93.7% |
| A2 | 99.0% / 100.0% | 0.0% / 0.3% | 11.7% / 41.0% | 50.8% / 77.0% | 88.7% / 91.0% | 31.0% / 53.8% |
| A3 | 0.3% / 0.7% | 4.8% / 69.0% | 0.1% / 1.0% | 54.3% / 74.4% | 88.7% / 90.1% | 27.2% / 70.9% |
| A4 | 20.1% / 15.8% | 0.6% / 31.4% | 0.8% / 19.2% | 0.7% / 28.6% | 90.0% / 92.3% | 33.3% / 86.4% |
| A5b | 17.9% / 100.0% | 9.0% / 14.3% | 14.1% / 66.9% | 59.0% / 84.9% | 87.5% / 81.4% | 0.2% / 18.3% |

**ROC-AUC** — rows: trained on, columns: tested on; each cell `D5 mean / D3`.

| trained on | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|
| A1 | 1.000 / 1.000 | 0.571 / 0.483 | 0.605 / 0.544 | 0.480 / 0.518 | 0.485 / 0.495 | 0.729 / 0.506 |
| A2 | 0.603 / 0.507 | 0.999 / 0.993 | 0.966 / 0.861 | 0.743 / 0.602 | 0.530 / 0.509 | 0.919 / 0.837 |
| A3 | 0.995 / 0.982 | 0.985 / 0.847 | 0.998 / 0.990 | 0.733 / 0.581 | 0.532 / 0.522 | 0.941 / 0.820 |
| A4 | 0.954 / 0.950 | 0.996 / 0.929 | 0.995 / 0.949 | 0.996 / 0.932 | 0.545 / 0.506 | 0.892 / 0.581 |
| A5b | 0.955 / 0.755 | 0.973 / 0.960 | 0.961 / 0.867 | 0.723 / 0.566 | 0.538 / 0.541 | 0.997 / 0.949 |

Full cells with sd and the seed-42 CI are in `e5_cross_attack.csv`. On the diagonal (trained and tested on the same attacker) D5 accepts A1 0.0% (D3 0.0%), A2 0.0% (D3 0.3%), A3 0.1% (D3 1.0%), A4 0.7% (D3 28.6%), A5b 0.2% (D3 18.3%) at τ10. Averaged over the off-diagonal cells, D5 accepts 49.0% of attacks (D3 69.0%); against the verbatim replay A5a D5 accepts 91.6% (←A1), 88.7% (←A2), 88.7% (←A3), 90.0% (←A4), 87.5% (←A5b).

## E6 — leave one attacker out: D5 next to D3 (E3)

| detector | trained_on | tested_on | APCER/FPR @τ5 | APCER/FPR @τ10 | BPCER/FNR @τ10 | ROC-AUC |
|---|---|---|---|---|---|---|
| D5 | all-but-A1 | A1 | 11.8% ± 3.5 [13.0, 15.1] | 4.2% ± 1.9 [4.9, 6.2] | 8.2% ± 0.2 [7.2, 9.5] | 0.984 ± 0.004 [0.978, 0.984] |
| D3 | all-but-A1 | A1 | 39.2% [37.8, 40.8] | 6.7% [5.8, 7.4] | 8.6% [7.5, 9.7] | 0.962 [0.956, 0.968] |
| D1 | attack-free | A1 | 53.4% [51.9, 54.8] | 16.8% [15.7, 17.9] | 9.9% [8.4, 11.4] | 0.934 [0.923, 0.944] |
| D2 | attack-free | A1 | 100.0% [100.0, 100.0] | 99.9% [99.8, 100.0] | 8.5% [6.7, 10.4] | 0.777 [0.749, 0.803] |
| D5 | all-but-A2 | A2 | 2.7% ± 0.9 [1.8, 2.7] | 0.8% ± 0.4 [0.4, 0.9] | 7.7% ± 0.1 [6.6, 9.1] | 0.996 ± 0.001 [0.995, 0.997] |
| D3 | all-but-A2 | A2 | 28.7% [27.3, 30.0] | 10.6% [9.6, 11.5] | 8.4% [7.2, 9.5] | 0.967 [0.962, 0.972] |
| D1 | attack-free | A2 | 99.0% [98.7, 99.3] | 97.2% [96.6, 97.7] | 9.9% [8.4, 11.4] | 0.526 [0.507, 0.546] |
| D2 | attack-free | A2 | 99.7% [99.5, 99.8] | 98.6% [98.2, 99.0] | 8.5% [6.7, 10.4] | 0.315 [0.297, 0.333] |
| D5 | all-but-A3 | A3 | 3.6% ± 0.5 [3.4, 4.5] | 1.1% ± 0.2 [1.0, 1.6] | 8.4% ± 0.4 [7.2, 9.5] | 0.994 ± 0.001 [0.991, 0.994] |
| D3 | all-but-A3 | A3 | 39.7% [38.3, 41.2] | 19.0% [17.7, 20.2] | 8.7% [7.6, 9.8] | 0.947 [0.941, 0.954] |
| D1 | attack-free | A3 | 97.6% [97.1, 98.1] | 94.0% [93.3, 94.6] | 9.9% [8.4, 11.4] | 0.445 [0.429, 0.464] |
| D2 | attack-free | A3 | 98.4% [98.1, 98.7] | 95.8% [95.2, 96.4] | 8.5% [6.7, 10.4] | 0.334 [0.316, 0.354] |
| D5 | all-but-A4 | A4 | 53.3% ± 0.4 [51.7, 54.5] | 50.8% ± 0.5 [48.8, 51.6] | 8.6% ± 0.1 [7.4, 9.7] | 0.735 ± 0.028 [0.753, 0.780] |
| D3 | all-but-A4 | A4 | 75.4% [74.1, 76.8] | 69.2% [67.8, 70.4] | 8.4% [7.3, 9.4] | 0.609 [0.596, 0.623] |
| D1 | attack-free | A4 | 91.9% [91.1, 92.8] | 86.3% [85.2, 87.5] | 9.9% [8.4, 11.4] | 0.497 [0.480, 0.514] |
| D2 | attack-free | A4 | 96.4% [95.9, 97.0] | 91.3% [90.4, 92.1] | 8.5% [6.7, 10.4] | 0.481 [0.460, 0.502] |
| D5 | all-but-A5b | A5b | 38.5% ± 3.6 [34.6, 37.6] | 27.7% ± 3.2 [23.9, 26.4] | 7.8% ± 0.0 [6.4, 9.2] | 0.913 ± 0.012 [0.916, 0.934] |
| D3 | all-but-A5b | A5b | 81.4% [80.2, 82.5] | 69.5% [68.2, 70.8] | 7.7% [6.4, 8.9] | 0.703 [0.684, 0.722] |
| D1 | attack-free | A5b | 97.3% [96.8, 97.8] | 93.6% [92.7, 94.3] | 9.9% [8.4, 11.4] | 0.537 [0.519, 0.556] |
| D2 | attack-free | A5b | 96.8% [96.3, 97.3] | 93.8% [93.0, 94.6] | 8.5% [6.7, 10.4] | 0.412 [0.391, 0.433] |

For each held-out attacker, APCER at τ10 — A1: D5 4.2% vs D3 6.7% (AUC 0.984 vs 0.962); A2: D5 0.8% vs D3 10.6% (AUC 0.996 vs 0.967); A3: D5 1.1% vs D3 19.0% (AUC 0.994 vs 0.947); A4: D5 50.8% vs D3 69.2% (AUC 0.735 vs 0.609); A5b: D5 27.7% vs D3 69.5% (AUC 0.913 vs 0.703).

## Session-level (D1, D2, D3) vs sequence-level (D4, D5)

**APCER / FPR at τ10** (attacks accepted; lower is better; BPCER ≈ 10% by construction for every row):

| level | detector / setting | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|---|
| session | D1 (OCSVM), attack-free | 16.8% | 97.2% | 94.0% | 86.3% | 90.2% | 93.6% |
| session | D2 (Isolation Forest), attack-free | 99.9% | 98.6% | 95.8% | 91.3% | 89.3% | 93.8% |
| sequence | D4-primary, attack-free | 71.9% ± 1.8 | 1.1% ± 0.1 | 5.0% ± 0.6 | 2.8% ± 0.3 | 90.3% ± 0.2 | 13.6% ± 0.4 |
| sequence | D4-typicality, attack-free | 85.8% ± 0.8 | 2.5% ± 0.2 | 10.5% ± 0.6 | 6.1% ± 0.4 | 89.8% ± 0.2 | 19.2% ± 0.3 |
| session | D3 trained on the same attacker (E2 diagonal) | 0.0% | 0.3% | 1.0% | 28.6% | — | 18.3% |
| sequence | D5 trained on the same attacker (E5 diagonal) | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.1% ± 0.0 | 0.7% ± 0.1 | — | 0.2% ± 0.1 |
| session | D3, attacker unseen in training (E3) | 6.7% | 10.6% | 19.0% | 69.2% | — | 69.5% |
| sequence | D5, attacker unseen in training (E6) | 4.2% ± 1.9 | 0.8% ± 0.4 | 1.1% ± 0.2 | 50.8% ± 0.5 | — | 27.7% ± 3.2 |

**ROC-AUC** (human vs attack; 0.5 = chance; < 0.5 = attacks look *more* human than humans):

| level | detector / setting | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|---|
| session | D1 (OCSVM), attack-free | 0.934 | 0.526 | 0.445 | 0.497 | 0.493 | 0.537 |
| session | D2 (Isolation Forest), attack-free | 0.777 | 0.315 | 0.334 | 0.481 | 0.520 | 0.412 |
| sequence | D4-primary, attack-free | 0.799 ± 0.006 | 0.987 ± 0.001 | 0.968 ± 0.002 | 0.979 ± 0.001 | 0.479 ± 0.002 | 0.945 ± 0.002 |
| sequence | D4-typicality, attack-free | 0.646 ± 0.006 | 0.983 ± 0.001 | 0.953 ± 0.003 | 0.969 ± 0.002 | 0.498 ± 0.003 | 0.915 ± 0.003 |
| session | D3 trained on the same attacker (E2 diagonal) | 1.000 | 0.993 | 0.990 | 0.932 | — | 0.949 |
| sequence | D5 trained on the same attacker (E5 diagonal) | 1.000 ± 0.000 | 0.999 ± 0.000 | 0.998 ± 0.000 | 0.996 ± 0.001 | — | 0.997 ± 0.000 |
| session | D3, attacker unseen in training (E3) | 0.962 | 0.967 | 0.947 | 0.609 | — | 0.703 |
| sequence | D5, attacker unseen in training (E6) | 0.984 ± 0.004 | 0.996 ± 0.001 | 0.994 ± 0.001 | 0.735 ± 0.028 | — | 0.913 ± 0.012 |

## Training and runtime

| run | train set | epochs run | best epoch | best val loss | stopped early | train time |
|---|---|---|---|---|---|---|
| D4/seed42 | train_all | 30 | 30 | 1.74440 | False | 2.4 min |
| D4/seed43 | train_all | 30 | 29 | 1.74877 | False | 2.4 min |
| D4/seed44 | train_all | 30 | 30 | 1.74784 | False | 2.3 min |
| D4-20k/seed42 | train_20k | 30 | 30 | 1.74440 | False | 2.1 min |
| D4-20k/seed43 | train_20k | 30 | 29 | 1.74877 | False | 2.1 min |
| D4-20k/seed44 | train_20k | 30 | 30 | 1.74784 | False | 2.1 min |
| D4-nokey/seed42 | train_all | 30 | 30 | 2.19310 | False | 2.1 min |
| D4-nokey/seed43 | train_all | 26 | 23 | 2.19560 | True | 1.8 min |
| D4-nokey/seed44 | train_all | 30 | 28 | 2.19541 | False | 2.1 min |
| D5<-A1/seed42 | 17,143 humans + 17,143 bots | 7 | 4 | 0.00480 | True | 1.7 min |
| D5<-A1/seed43 | 17,143 humans + 17,143 bots | 11 | 8 | 0.00206 | True | 2.7 min |
| D5<-A1/seed44 | 17,143 humans + 17,143 bots | 14 | 11 | 0.00172 | True | 3.4 min |
| D5<-A2/seed42 | 17,143 humans + 17,143 bots | 13 | 10 | 0.04726 | True | 3.2 min |
| D5<-A2/seed43 | 17,143 humans + 17,143 bots | 11 | 8 | 0.04937 | True | 2.7 min |
| D5<-A2/seed44 | 17,143 humans + 17,143 bots | 12 | 9 | 0.04627 | True | 3.0 min |
| D5<-A3/seed42 | 17,143 humans + 17,143 bots | 11 | 8 | 0.06711 | True | 2.7 min |
| D5<-A3/seed43 | 17,143 humans + 17,143 bots | 13 | 10 | 0.06597 | True | 3.3 min |
| D5<-A3/seed44 | 17,143 humans + 17,143 bots | 14 | 11 | 0.07028 | True | 3.5 min |
| D5<-A4/seed42 | 17,143 humans + 17,143 bots | 11 | 8 | 0.09413 | True | 2.9 min |
| D5<-A4/seed43 | 17,143 humans + 17,143 bots | 12 | 9 | 0.08979 | True | 3.2 min |
| D5<-A4/seed44 | 17,143 humans + 17,143 bots | 10 | 7 | 0.09655 | True | 2.6 min |
| D5<-A5b/seed42 | 17,143 humans + 17,143 bots | 14 | 11 | 0.07071 | True | 3.4 min |
| D5<-A5b/seed43 | 17,143 humans + 17,143 bots | 16 | 13 | 0.07215 | True | 3.9 min |
| D5<-A5b/seed44 | 17,143 humans + 17,143 bots | 13 | 10 | 0.06738 | True | 3.2 min |
| D5<-all-but-A1/seed42 | 17,143 humans + 68,572 bots | 17 | 14 | 0.09468 | True | 4.4 min |
| D5<-all-but-A1/seed43 | 17,143 humans + 68,572 bots | 15 | 12 | 0.10309 | True | 3.8 min |
| D5<-all-but-A1/seed44 | 17,143 humans + 68,572 bots | 14 | 11 | 0.10311 | True | 3.6 min |
| D5<-all-but-A2/seed42 | 17,143 humans + 68,572 bots | 15 | 12 | 0.08945 | True | 3.9 min |
| D5<-all-but-A2/seed43 | 17,143 humans + 68,572 bots | 15 | 12 | 0.09218 | True | 3.8 min |
| D5<-all-but-A2/seed44 | 17,143 humans + 68,572 bots | 11 | 8 | 0.10132 | True | 2.8 min |
| D5<-all-but-A3/seed42 | 17,143 humans + 68,572 bots | 13 | 10 | 0.09029 | True | 3.3 min |
| D5<-all-but-A3/seed43 | 17,143 humans + 68,572 bots | 15 | 12 | 0.09030 | True | 3.8 min |
| D5<-all-but-A3/seed44 | 17,143 humans + 68,572 bots | 20 | 17 | 0.08683 | True | 5.1 min |
| D5<-all-but-A4/seed42 | 17,143 humans + 68,572 bots | 15 | 12 | 0.07710 | True | 3.8 min |
| D5<-all-but-A4/seed43 | 17,143 humans + 68,572 bots | 14 | 11 | 0.07487 | True | 3.5 min |
| D5<-all-but-A4/seed44 | 17,143 humans + 68,572 bots | 14 | 11 | 0.07749 | True | 3.5 min |
| D5<-all-but-A5b/seed42 | 17,143 humans + 68,572 bots | 17 | 14 | 0.07490 | True | 4.4 min |
| D5<-all-but-A5b/seed43 | 17,143 humans + 68,572 bots | 13 | 10 | 0.07681 | True | 3.3 min |
| D5<-all-but-A5b/seed44 | 17,143 humans + 68,572 bots | 17 | 14 | 0.06759 | True | 4.4 min |

Wall-clock per part: E4 (31_train_d4) 19.6 min, E5 (32_train_d5) 45.9 min, E6 (32_train_d5) 57.7 min.

## Bug fixes and incidents


Rules (docs/PHASE3B_PREREGISTRATION.md): after results are seen, only genuine bugs are fixed, never tuning. Each
entry gives the date, what changed, why it is a bug and not tuning, and whether any result had been seen.
RESULTS_PHASE3B.md reproduces this file.

### Incidents (not Phase 3b bugs)

- **2026-09-25, before any Phase 3b model was trained: frozen `src/kl/features.py` was found modified.** Its first
  line started with two stray characters (`cl"""Session -> …`), which made the file unparseable, and it failed the
  Phase 1 hash. The file's mtime was 2026-09-25 13:36:26, before this Phase 3b session started; no tool in this
  session wrote to it. Removing exactly those two bytes gives back the frozen SHA-256 `5be4f320…a63ec` from
  `docs/phase1_frozen_hashes.json`, so the file was restored to its frozen bytes. The modified copy was kept
  (`features.py.with_stray_cl` in the session scratchpad). All Phase 1/2/3a guards passed afterwards. No result
  depended on the modified file: it could not even be imported.

### Bug fixes

None so far.
