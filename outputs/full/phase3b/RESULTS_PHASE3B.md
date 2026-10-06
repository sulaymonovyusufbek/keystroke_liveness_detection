# Phase 3b — sequence-level detectors (results)

_Run mode **full**; Phase 1 config hash `8b2fc2193f74312e` (frozen); design pre-registered in `docs/PHASE3B_PREREGISTRATION.md` (hash verified at the start and end of every step). Terminology: **APCER = FPR = attacks accepted**, **BPCER = FNR = humans rejected**; neural cells are *mean ± sd over training seeds 42/43/44 [95% user-bootstrap CI of seed 42]*; Phase 2 rows (D1, D2, D3) are a single deterministic model: *value [95% CI]*. AUC < 0.5 is reported as it is._

## Data and set-up

Human sessions: train_all 173,392 (11,996 users), val 43,291 (3,000 users), test 43,248 (2,999 users), train_20k 20,000 (9,923 users).
Sequences are truncated at **max_len = 94** keystrokes (99th percentile of train-user human session length); vocabulary 86 ids (84 keycodes with ≥ 20 train occurrences + PAD + UNK); standardisation log1p(hold) 4.658 ± 0.406, log1p(pp) 5.194 ± 0.680.
Bot keystrokes were regenerated with the frozen Phase 2 attackers and seeds: **12 of 12 train/test sets reproduce the stored Phase 2 features exactly** (tolerance 1e-9, observed max |diff| 0); val bots of A1–A4 and A5b (new, used only for D5 early stopping) were generated with the same scheme.

Truncated sessions (first max_len keystrokes kept):

| set | sessions | truncated | fraction |
|---|---|---|---|
| train_all | 173,392 | 1,695 | 0.98% |
| val | 43,291 | 384 | 0.89% |
| test | 43,248 | 410 | 0.95% |
| train_20k | 20,000 | 195 | 0.97% |
| bot_A1_train | 20,000 | 0 | 0.00% |
| bot_A2_train | 20,000 | 0 | 0.00% |
| bot_A3_train | 20,000 | 0 | 0.00% |
| bot_A4_train | 20,000 | 126 | 0.63% |
| bot_A5a_train | 20,000 | 163 | 0.81% |
| bot_A5b_train | 20,000 | 0 | 0.00% |
| bot_A1_test | 43,248 | 0 | 0.00% |
| bot_A2_test | 43,248 | 0 | 0.00% |
| bot_A3_test | 43,248 | 0 | 0.00% |
| bot_A4_test | 43,248 | 294 | 0.68% |
| bot_A5a_test | 43,248 | 373 | 0.86% |
| bot_A5b_test | 43,248 | 0 | 0.00% |
| bot_A1_val | 43,291 | 0 | 0.00% |
| bot_A2_val | 43,291 | 0 | 0.00% |
| bot_A3_val | 43,291 | 0 | 0.00% |
| bot_A4_val | 43,291 | 335 | 0.77% |
| bot_A5b_val | 43,291 | 0 | 0.00% |

Device: **cuda** (NVIDIA GeForce RTX 3050 6GB Laptop GPU), torch 2.11.0+cu128, CUDA 12.8, cuDNN 91900; deterministic algorithms on.

## E4 — open set: D4 (one-class, humans only) vs the frozen D1 / D2

| detector | tested_on | APCER/FPR @τ5 | APCER/FPR @τ10 | BPCER/FNR @τ5 | BPCER/FNR @τ10 | ROC-AUC |
|---|---|---|---|---|---|---|
| D1 | A1 | 32.4% [31.9, 32.8] | 5.8% [5.6, 6.0] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.954 [0.951, 0.957] |
| D2 | A1 | 100.0% [100.0, 100.0] | 99.9% [99.9, 99.9] | 4.6% [4.1, 5.0] | 9.6% [9.0, 10.2] | 0.796 [0.788, 0.805] |
| D4-primary | A1 | 85.2% ± 1.0 [85.7, 86.3] | 64.0% ± 1.3 [64.9, 65.8] | 5.5% ± 0.0 [5.0, 6.0] | 10.5% ± 0.1 [9.8, 11.1] | 0.834 ± 0.003 [0.824, 0.838] |
| D4-typicality | A1 | 91.1% ± 0.5 [91.1, 91.6] | 80.1% ± 1.1 [80.4, 81.2] | 5.5% ± 0.0 [5.0, 5.9] | 10.5% ± 0.1 [9.9, 11.0] | 0.690 ± 0.005 [0.679, 0.691] |
| D4-20k | A1 | 90.5% ± 1.0 [89.1, 89.6] | 74.8% ± 1.9 [72.4, 73.3] | 5.5% ± 0.0 [5.1, 6.0] | 10.4% ± 0.1 [9.7, 11.0] | 0.796 ± 0.009 [0.796, 0.812] |
| D4-nokey | A1 | 99.9% ± 0.0 [99.9, 99.9] | 99.7% ± 0.0 [99.6, 99.7] | 5.4% ± 0.0 [4.9, 5.8] | 10.3% ± 0.0 [9.7, 11.0] | 0.247 ± 0.003 [0.242, 0.257] |
| D1 | A2 | 99.4% [99.3, 99.4] | 98.0% [97.8, 98.1] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.462 [0.455, 0.468] |
| D2 | A2 | 99.6% [99.6, 99.7] | 98.5% [98.3, 98.6] | 4.6% [4.1, 5.0] | 9.6% [9.0, 10.2] | 0.318 [0.311, 0.325] |
| D4-primary | A2 | 1.9% ± 0.0 [1.7, 2.0] | 0.6% ± 0.0 [0.5, 0.6] | 5.5% ± 0.0 [5.0, 6.0] | 10.5% ± 0.1 [9.8, 11.1] | 0.991 ± 0.000 [0.989, 0.992] |
| D4-typicality | A2 | 3.2% ± 0.1 [3.0, 3.3] | 1.3% ± 0.0 [1.2, 1.4] | 5.5% ± 0.0 [5.0, 5.9] | 10.5% ± 0.1 [9.9, 11.0] | 0.988 ± 0.000 [0.987, 0.990] |
| D4-20k | A2 | 3.8% ± 0.2 [3.8, 4.2] | 1.2% ± 0.1 [1.2, 1.4] | 5.5% ± 0.0 [5.1, 6.0] | 10.4% ± 0.1 [9.7, 11.0] | 0.986 ± 0.000 [0.984, 0.988] |
| D4-nokey | A2 | 45.1% ± 0.2 [44.4, 45.3] | 25.0% ± 0.3 [24.3, 25.1] | 5.4% ± 0.0 [4.9, 5.8] | 10.3% ± 0.0 [9.7, 11.0] | 0.923 ± 0.000 [0.919, 0.927] |
| D1 | A3 | 98.8% [98.7, 98.9] | 96.8% [96.6, 96.9] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.428 [0.422, 0.434] |
| D2 | A3 | 99.5% [99.5, 99.6] | 98.0% [97.9, 98.1] | 4.6% [4.1, 5.0] | 9.6% [9.0, 10.2] | 0.314 [0.307, 0.321] |
| D4-primary | A3 | 9.1% ± 0.1 [8.9, 9.4] | 3.0% ± 0.0 [2.8, 3.2] | 5.5% ± 0.0 [5.0, 6.0] | 10.5% ± 0.1 [9.8, 11.1] | 0.977 ± 0.000 [0.975, 0.979] |
| D4-typicality | A3 | 14.5% ± 0.3 [13.9, 14.5] | 6.6% ± 0.1 [6.4, 6.8] | 5.5% ± 0.0 [5.0, 5.9] | 10.5% ± 0.1 [9.9, 11.0] | 0.967 ± 0.001 [0.965, 0.970] |
| D4-20k | A3 | 17.8% ± 0.5 [18.0, 18.7] | 6.8% ± 0.4 [6.9, 7.3] | 5.5% ± 0.0 [5.1, 6.0] | 10.4% ± 0.1 [9.7, 11.0] | 0.964 ± 0.001 [0.961, 0.966] |
| D4-nokey | A3 | 77.8% ± 0.3 [77.6, 78.4] | 64.7% ± 0.1 [64.2, 65.1] | 5.4% ± 0.0 [4.9, 5.8] | 10.3% ± 0.0 [9.7, 11.0] | 0.798 ± 0.001 [0.790, 0.804] |
| D1 | A4 | 92.6% [92.4, 92.9] | 86.8% [86.5, 87.2] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.499 [0.493, 0.505] |
| D2 | A4 | 96.1% [95.9, 96.3] | 90.6% [90.3, 90.9] | 4.6% [4.1, 5.0] | 9.6% [9.0, 10.2] | 0.483 [0.476, 0.491] |
| D4-primary | A4 | 4.5% ± 0.1 [4.4, 4.8] | 1.3% ± 0.0 [1.2, 1.5] | 5.5% ± 0.0 [5.0, 6.0] | 10.5% ± 0.1 [9.8, 11.1] | 0.985 ± 0.000 [0.984, 0.987] |
| D4-typicality | A4 | 7.5% ± 0.2 [7.1, 7.6] | 3.2% ± 0.1 [3.0, 3.3] | 5.5% ± 0.0 [5.0, 5.9] | 10.5% ± 0.1 [9.9, 11.0] | 0.980 ± 0.000 [0.979, 0.982] |
| D4-20k | A4 | 9.3% ± 0.3 [9.4, 9.9] | 3.2% ± 0.2 [3.3, 3.6] | 5.5% ± 0.0 [5.1, 6.0] | 10.4% ± 0.1 [9.7, 11.0] | 0.977 ± 0.000 [0.974, 0.979] |
| D4-nokey | A4 | 56.2% ± 0.0 [55.7, 56.6] | 39.5% ± 0.2 [38.8, 39.6] | 5.4% ± 0.0 [4.9, 5.8] | 10.3% ± 0.0 [9.7, 11.0] | 0.882 ± 0.000 [0.877, 0.887] |
| D1 | A5a | 95.0% [94.8, 95.2] | 89.9% [89.6, 90.2] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.506 [0.500, 0.512] |
| D2 | A5a | 94.8% [94.6, 95.0] | 89.5% [89.2, 89.8] | 4.6% [4.1, 5.0] | 9.6% [9.0, 10.2] | 0.509 [0.501, 0.516] |
| D4-primary | A5a | 94.5% ± 0.1 [94.3, 94.7] | 89.7% ± 0.1 [89.4, 90.0] | 5.5% ± 0.0 [5.0, 6.0] | 10.5% ± 0.1 [9.8, 11.1] | 0.494 ± 0.000 [0.486, 0.502] |
| D4-typicality | A5a | 94.5% ± 0.1 [94.3, 94.7] | 89.5% ± 0.1 [89.2, 89.7] | 5.5% ± 0.0 [5.0, 5.9] | 10.5% ± 0.1 [9.9, 11.0] | 0.498 ± 0.001 [0.494, 0.505] |
| D4-20k | A5a | 94.5% ± 0.0 [94.2, 94.6] | 89.8% ± 0.1 [89.5, 90.0] | 5.5% ± 0.0 [5.1, 6.0] | 10.4% ± 0.1 [9.7, 11.0] | 0.494 ± 0.000 [0.487, 0.502] |
| D4-nokey | A5a | 94.7% ± 0.1 [94.6, 95.0] | 89.9% ± 0.1 [89.6, 90.1] | 5.4% ± 0.0 [4.9, 5.8] | 10.3% ± 0.0 [9.7, 11.0] | 0.495 ± 0.000 [0.487, 0.503] |
| D1 | A5b | 97.2% [97.1, 97.4] | 93.4% [93.1, 93.6] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.528 [0.522, 0.534] |
| D2 | A5b | 96.8% [96.6, 97.0] | 93.1% [92.9, 93.4] | 4.6% [4.1, 5.0] | 9.6% [9.0, 10.2] | 0.426 [0.419, 0.433] |
| D4-primary | A5b | 14.2% ± 0.4 [13.5, 14.2] | 7.5% ± 0.2 [7.0, 7.5] | 5.5% ± 0.0 [5.0, 6.0] | 10.5% ± 0.1 [9.8, 11.1] | 0.968 ± 0.001 [0.967, 0.971] |
| D4-typicality | A5b | 18.6% ± 0.7 [17.5, 18.3] | 11.8% ± 0.4 [11.1, 11.7] | 5.5% ± 0.0 [5.0, 5.9] | 10.5% ± 0.1 [9.9, 11.0] | 0.949 ± 0.002 [0.949, 0.954] |
| D4-20k | A5b | 19.5% ± 0.1 [19.2, 20.0] | 11.1% ± 0.1 [11.0, 11.6] | 5.5% ± 0.0 [5.1, 6.0] | 10.4% ± 0.1 [9.7, 11.0] | 0.956 ± 0.000 [0.953, 0.958] |
| D4-nokey | A5b | 92.1% ± 0.1 [91.9, 92.4] | 85.2% ± 0.1 [84.8, 85.4] | 5.4% ± 0.0 [4.9, 5.8] | 10.3% ± 0.0 [9.7, 11.0] | 0.555 ± 0.000 [0.547, 0.563] |

At the τ10 operating point (10% of validation humans rejected) — A1: D4-primary accepts 64.0% of attacks at τ10 (AUC 0.834) vs D1 5.8% (AUC 0.954); D4-typicality 80.1%; A2: D4-primary accepts 0.6% of attacks at τ10 (AUC 0.991) vs D1 98.0% (AUC 0.462); D4-typicality 1.3%; A3: D4-primary accepts 3.0% of attacks at τ10 (AUC 0.977) vs D1 96.8% (AUC 0.428); D4-typicality 6.6%; A4: D4-primary accepts 1.3% of attacks at τ10 (AUC 0.985) vs D1 86.8% (AUC 0.499); D4-typicality 3.2%; A5a: D4-primary accepts 89.7% of attacks at τ10 (AUC 0.494) vs D1 89.9% (AUC 0.506); D4-typicality 89.5%; A5b: D4-primary accepts 7.5% of attacks at τ10 (AUC 0.968) vs D1 93.4% (AUC 0.528); D4-typicality 11.8%. D4-20k (data-size control) and D4-nokey (timing only, no key conditioning) are in the table; A1: D4 64.0%, D4-20k 74.8%, D4-nokey 99.7%; A2: D4 0.6%, D4-20k 1.2%, D4-nokey 25.0%; A3: D4 3.0%, D4-20k 6.8%, D4-nokey 64.7%; A4: D4 1.3%, D4-20k 3.2%, D4-nokey 39.5%; A5a: D4 89.7%, D4-20k 89.8%, D4-nokey 89.9%; A5b: D4 7.5%, D4-20k 11.1%, D4-nokey 85.2%.

## E5 — cross-attack matrix: D5 (supervised BiGRU) next to D3 (Random Forest, Phase 2)

**APCER / FPR @τ10 (attacks accepted)** — rows: trained on, columns: tested on; each cell `D5 mean / D3`.

| trained on | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|
| A1 | 0.0% / 0.0% | 95.2% / 97.9% | 78.9% / 83.3% | 87.6% / 90.0% | 89.4% / 94.7% | 67.1% / 93.1% |
| A2 | 91.6% / 99.9% | 0.0% / 0.2% | 12.8% / 40.1% | 54.7% / 76.0% | 90.1% / 90.1% | 32.4% / 48.5% |
| A3 | 0.9% / 0.0% | 4.8% / 68.7% | 0.1% / 1.0% | 53.1% / 77.9% | 90.2% / 90.5% | 28.7% / 68.1% |
| A4 | 7.4% / 11.8% | 1.0% / 27.0% | 0.7% / 18.1% | 0.4% / 23.7% | 90.3% / 90.8% | 29.8% / 82.6% |
| A5b | 16.2% / 91.2% | 13.0% / 15.0% | 16.4% / 57.4% | 60.9% / 81.7% | 89.6% / 89.0% | 0.1% / 32.8% |

**ROC-AUC** — rows: trained on, columns: tested on; each cell `D5 mean / D3`.

| trained on | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|
| A1 | 1.000 / 1.000 | 0.607 / 0.486 | 0.621 / 0.560 | 0.507 / 0.526 | 0.492 / 0.502 | 0.730 / 0.510 |
| A2 | 0.750 / 0.757 | 0.998 / 0.994 | 0.958 / 0.851 | 0.710 / 0.597 | 0.493 / 0.494 | 0.910 / 0.836 |
| A3 | 0.990 / 0.986 | 0.981 / 0.793 | 0.997 / 0.988 | 0.728 / 0.549 | 0.494 / 0.501 | 0.922 / 0.810 |
| A4 | 0.967 / 0.959 | 0.994 / 0.925 | 0.994 / 0.945 | 0.996 / 0.933 | 0.491 / 0.497 | 0.891 / 0.587 |
| A5b | 0.953 / 0.808 | 0.961 / 0.956 | 0.951 / 0.874 | 0.686 / 0.557 | 0.497 / 0.501 | 0.998 / 0.920 |

Full cells with sd and the seed-42 CI are in `e5_cross_attack.csv`. On the diagonal (trained and tested on the same attacker) D5 accepts A1 0.0% (D3 0.0%), A2 0.0% (D3 0.2%), A3 0.1% (D3 1.0%), A4 0.4% (D3 23.7%), A5b 0.1% (D3 32.8%) at τ10. Averaged over the off-diagonal cells, D5 accepts 48.1% of attacks (D3 67.3%); against the verbatim replay A5a D5 accepts 89.4% (←A1), 90.1% (←A2), 90.2% (←A3), 90.3% (←A4), 89.6% (←A5b).

## E6 — leave one attacker out: D5 next to D3 (E3)

| detector | trained_on | tested_on | APCER/FPR @τ5 | APCER/FPR @τ10 | BPCER/FNR @τ10 | ROC-AUC |
|---|---|---|---|---|---|---|
| D5 | all-but-A1 | A1 | 9.8% ± 3.6 [8.5, 9.0] | 3.3% ± 1.9 [2.7, 3.0] | 10.7% ± 0.1 [10.1, 11.0] | 0.980 ± 0.005 [0.981, 0.983] |
| D3 | all-but-A1 | A1 | 11.5% [11.2, 11.8] | 3.7% [3.5, 3.9] | 10.0% [9.6, 10.5] | 0.978 [0.977, 0.980] |
| D1 | attack-free | A1 | 32.4% [31.9, 32.8] | 5.8% [5.6, 6.0] | 9.9% [9.4, 10.4] | 0.954 [0.951, 0.957] |
| D2 | attack-free | A1 | 100.0% [100.0, 100.0] | 99.9% [99.9, 99.9] | 9.6% [9.0, 10.2] | 0.796 [0.788, 0.805] |
| D5 | all-but-A2 | A2 | 3.1% ± 0.6 [2.9, 3.2] | 1.2% ± 0.4 [0.9, 1.1] | 10.7% ± 0.1 [10.1, 11.0] | 0.993 ± 0.001 [0.993, 0.994] |
| D3 | all-but-A2 | A2 | 30.3% [29.9, 30.7] | 11.1% [10.8, 11.4] | 10.1% [9.6, 10.5] | 0.959 [0.957, 0.961] |
| D1 | attack-free | A2 | 99.4% [99.3, 99.4] | 98.0% [97.8, 98.1] | 9.9% [9.4, 10.4] | 0.462 [0.455, 0.468] |
| D2 | attack-free | A2 | 99.6% [99.6, 99.7] | 98.5% [98.3, 98.6] | 9.6% [9.0, 10.2] | 0.318 [0.311, 0.325] |
| D5 | all-but-A3 | A3 | 2.7% ± 0.3 [2.4, 2.7] | 0.8% ± 0.2 [0.6, 0.8] | 10.7% ± 0.1 [10.2, 11.2] | 0.993 ± 0.001 [0.993, 0.994] |
| D3 | all-but-A3 | A3 | 36.6% [36.1, 37.1] | 19.0% [18.6, 19.4] | 9.8% [9.4, 10.2] | 0.941 [0.939, 0.944] |
| D1 | attack-free | A3 | 98.8% [98.7, 98.9] | 96.8% [96.6, 96.9] | 9.9% [9.4, 10.4] | 0.428 [0.422, 0.434] |
| D2 | attack-free | A3 | 99.5% [99.5, 99.6] | 98.0% [97.9, 98.1] | 9.6% [9.0, 10.2] | 0.314 [0.307, 0.321] |
| D5 | all-but-A4 | A4 | 54.1% ± 1.3 [53.2, 54.1] | 52.2% ± 2.0 [50.5, 51.4] | 10.6% ± 0.0 [10.2, 11.0] | 0.719 ± 0.026 [0.742, 0.751] |
| D3 | all-but-A4 | A4 | 74.9% [74.5, 75.3] | 70.0% [69.6, 70.4] | 10.0% [9.6, 10.4] | 0.583 [0.578, 0.587] |
| D1 | attack-free | A4 | 92.6% [92.4, 92.9] | 86.8% [86.5, 87.2] | 9.9% [9.4, 10.4] | 0.499 [0.493, 0.505] |
| D2 | attack-free | A4 | 96.1% [95.9, 96.3] | 90.6% [90.3, 90.9] | 9.6% [9.0, 10.2] | 0.483 [0.476, 0.491] |
| D5 | all-but-A5b | A5b | 36.1% ± 3.2 [34.2, 35.1] | 25.0% ± 2.1 [23.4, 24.2] | 10.6% ± 0.1 [10.2, 11.3] | 0.909 ± 0.007 [0.910, 0.916] |
| D3 | all-but-A5b | A5b | 76.3% [75.9, 76.7] | 65.7% [65.2, 66.1] | 9.5% [9.0, 9.9] | 0.694 [0.689, 0.701] |
| D1 | attack-free | A5b | 97.2% [97.1, 97.4] | 93.4% [93.1, 93.6] | 9.9% [9.4, 10.4] | 0.528 [0.522, 0.534] |
| D2 | attack-free | A5b | 96.8% [96.6, 97.0] | 93.1% [92.9, 93.4] | 9.6% [9.0, 10.2] | 0.426 [0.419, 0.433] |

For each held-out attacker, APCER at τ10 — A1: D5 3.3% vs D3 3.7% (AUC 0.980 vs 0.978); A2: D5 1.2% vs D3 11.1% (AUC 0.993 vs 0.959); A3: D5 0.8% vs D3 19.0% (AUC 0.993 vs 0.941); A4: D5 52.2% vs D3 70.0% (AUC 0.719 vs 0.583); A5b: D5 25.0% vs D3 65.7% (AUC 0.909 vs 0.694).

## Session-level (D1, D2, D3) vs sequence-level (D4, D5)

**APCER / FPR at τ10** (attacks accepted; lower is better; BPCER ≈ 10% by construction for every row):

| level | detector / setting | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|---|
| session | D1 (OCSVM), attack-free | 5.8% | 98.0% | 96.8% | 86.8% | 89.9% | 93.4% |
| session | D2 (Isolation Forest), attack-free | 99.9% | 98.5% | 98.0% | 90.6% | 89.5% | 93.1% |
| sequence | D4-primary, attack-free | 64.0% ± 1.3 | 0.6% ± 0.0 | 3.0% ± 0.0 | 1.3% ± 0.0 | 89.7% ± 0.1 | 7.5% ± 0.2 |
| sequence | D4-typicality, attack-free | 80.1% ± 1.1 | 1.3% ± 0.0 | 6.6% ± 0.1 | 3.2% ± 0.1 | 89.5% ± 0.1 | 11.8% ± 0.4 |
| session | D3 trained on the same attacker (E2 diagonal) | 0.0% | 0.2% | 1.0% | 23.7% | — | 32.8% |
| sequence | D5 trained on the same attacker (E5 diagonal) | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.1% ± 0.0 | 0.4% ± 0.0 | — | 0.1% ± 0.0 |
| session | D3, attacker unseen in training (E3) | 3.7% | 11.1% | 19.0% | 70.0% | — | 65.7% |
| sequence | D5, attacker unseen in training (E6) | 3.3% ± 1.9 | 1.2% ± 0.4 | 0.8% ± 0.2 | 52.2% ± 2.0 | — | 25.0% ± 2.1 |

**ROC-AUC** (human vs attack; 0.5 = chance; < 0.5 = attacks look *more* human than humans):

| level | detector / setting | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|---|
| session | D1 (OCSVM), attack-free | 0.954 | 0.462 | 0.428 | 0.499 | 0.506 | 0.528 |
| session | D2 (Isolation Forest), attack-free | 0.796 | 0.318 | 0.314 | 0.483 | 0.509 | 0.426 |
| sequence | D4-primary, attack-free | 0.834 ± 0.003 | 0.991 ± 0.000 | 0.977 ± 0.000 | 0.985 ± 0.000 | 0.494 ± 0.000 | 0.968 ± 0.001 |
| sequence | D4-typicality, attack-free | 0.690 ± 0.005 | 0.988 ± 0.000 | 0.967 ± 0.001 | 0.980 ± 0.000 | 0.498 ± 0.001 | 0.949 ± 0.002 |
| session | D3 trained on the same attacker (E2 diagonal) | 1.000 | 0.994 | 0.988 | 0.933 | — | 0.920 |
| sequence | D5 trained on the same attacker (E5 diagonal) | 1.000 ± 0.000 | 0.998 ± 0.000 | 0.997 ± 0.001 | 0.996 ± 0.000 | — | 0.998 ± 0.000 |
| session | D3, attacker unseen in training (E3) | 0.978 | 0.959 | 0.941 | 0.583 | — | 0.694 |
| sequence | D5, attacker unseen in training (E6) | 0.980 ± 0.005 | 0.993 ± 0.001 | 0.993 ± 0.001 | 0.719 ± 0.026 | — | 0.909 ± 0.007 |

## Training and runtime

| run | train set | epochs run | best epoch | best val loss | stopped early | train time |
|---|---|---|---|---|---|---|
| D4/seed42 | train_all | 28 | 25 | 1.66611 | True | 20.1 min |
| D4/seed43 | train_all | 30 | 28 | 1.66473 | False | 21.5 min |
| D4/seed44 | train_all | 30 | 29 | 1.66563 | False | 21.5 min |
| D4-20k/seed42 | train_20k | 30 | 30 | 1.74478 | False | 2.8 min |
| D4-20k/seed43 | train_20k | 30 | 30 | 1.74162 | False | 2.9 min |
| D4-20k/seed44 | train_20k | 30 | 30 | 1.74215 | False | 2.8 min |
| D4-nokey/seed42 | train_all | 30 | 27 | 2.18033 | True | 20.7 min |
| D4-nokey/seed43 | train_all | 29 | 26 | 2.18144 | True | 20.0 min |
| D4-nokey/seed44 | train_all | 30 | 27 | 2.18050 | True | 21.0 min |
| D5<-A1/seed42 | 20,000 humans + 20,000 bots | 11 | 8 | 0.00327 | True | 3.6 min |
| D5<-A1/seed43 | 20,000 humans + 20,000 bots | 10 | 7 | 0.00418 | True | 3.3 min |
| D5<-A1/seed44 | 20,000 humans + 20,000 bots | 11 | 8 | 0.00305 | True | 3.6 min |
| D5<-A2/seed42 | 20,000 humans + 20,000 bots | 11 | 8 | 0.04315 | True | 3.6 min |
| D5<-A2/seed43 | 20,000 humans + 20,000 bots | 9 | 6 | 0.04463 | True | 2.9 min |
| D5<-A2/seed44 | 20,000 humans + 20,000 bots | 10 | 7 | 0.04410 | True | 3.3 min |
| D5<-A3/seed42 | 20,000 humans + 20,000 bots | 13 | 10 | 0.06026 | True | 4.2 min |
| D5<-A3/seed43 | 20,000 humans + 20,000 bots | 11 | 8 | 0.06472 | True | 3.6 min |
| D5<-A3/seed44 | 20,000 humans + 20,000 bots | 13 | 10 | 0.06193 | True | 4.2 min |
| D5<-A4/seed42 | 20,000 humans + 20,000 bots | 13 | 10 | 0.07587 | True | 4.5 min |
| D5<-A4/seed43 | 20,000 humans + 20,000 bots | 11 | 8 | 0.08392 | True | 3.8 min |
| D5<-A4/seed44 | 20,000 humans + 20,000 bots | 12 | 9 | 0.08091 | True | 4.2 min |
| D5<-A5b/seed42 | 20,000 humans + 20,000 bots | 14 | 11 | 0.05433 | True | 4.6 min |
| D5<-A5b/seed43 | 20,000 humans + 20,000 bots | 11 | 8 | 0.06094 | True | 3.6 min |
| D5<-A5b/seed44 | 20,000 humans + 20,000 bots | 13 | 10 | 0.05527 | True | 4.2 min |
| D5<-all-but-A1/seed42 | 20,000 humans + 80,000 bots | 14 | 11 | 0.08856 | True | 5.5 min |
| D5<-all-but-A1/seed43 | 20,000 humans + 80,000 bots | 15 | 12 | 0.08410 | True | 5.9 min |
| D5<-all-but-A1/seed44 | 20,000 humans + 80,000 bots | 14 | 11 | 0.08566 | True | 5.5 min |
| D5<-all-but-A2/seed42 | 20,000 humans + 80,000 bots | 17 | 14 | 0.08165 | True | 6.7 min |
| D5<-all-but-A2/seed43 | 20,000 humans + 80,000 bots | 13 | 10 | 0.09108 | True | 5.1 min |
| D5<-all-but-A2/seed44 | 20,000 humans + 80,000 bots | 16 | 13 | 0.08496 | True | 6.3 min |
| D5<-all-but-A3/seed42 | 20,000 humans + 80,000 bots | 15 | 12 | 0.07968 | True | 5.9 min |
| D5<-all-but-A3/seed43 | 20,000 humans + 80,000 bots | 16 | 13 | 0.08425 | True | 6.3 min |
| D5<-all-but-A3/seed44 | 20,000 humans + 80,000 bots | 15 | 12 | 0.07853 | True | 5.9 min |
| D5<-all-but-A4/seed42 | 20,000 humans + 80,000 bots | 17 | 14 | 0.07338 | True | 6.5 min |
| D5<-all-but-A4/seed43 | 20,000 humans + 80,000 bots | 15 | 12 | 0.07503 | True | 5.8 min |
| D5<-all-but-A4/seed44 | 20,000 humans + 80,000 bots | 15 | 12 | 0.07451 | True | 5.8 min |
| D5<-all-but-A5b/seed42 | 20,000 humans + 80,000 bots | 15 | 12 | 0.07139 | True | 5.9 min |
| D5<-all-but-A5b/seed43 | 20,000 humans + 80,000 bots | 11 | 8 | 0.07807 | True | 4.3 min |
| D5<-all-but-A5b/seed44 | 20,000 humans + 80,000 bots | 15 | 12 | 0.06575 | True | 5.9 min |

Wall-clock per part: E4 (31_train_d4) 21.4 min, E5 (32_train_d5) 60.6 min, E6 (32_train_d5) 88.4 min.

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

- **2026-09-25 18:17–18:23, full run: three Phase 3b processes were alive at once, and the run was stopped.** The
  user paused the D4 training (`kill -STOP`, PID 79790) to free the GPU. The launcher was a plain bash loop, and a
  non-interactive bash treats a SIGSTOP-paused foreground child as *finished*: it logged `exit=147` (= 128 + SIGSTOP),
  ran the report step and started part (b) E5 (PID 97986). When that was paused too (exit 147 at 18:20:02) it started
  part (c) E6 (PID 103032). Only one process was ever *computing* at a time, because each earlier one was paused, but
  all three held GPU memory. At the user's request everything was stopped at 18:23 (loop frozen first, then CONT+TERM
  for each process). Damage check: no score file, model file, test-ledger entry or training curve was written after
  18:06:09. The only files written during the overlap were derived report files (regenerated on every report) and
  the `device` entries of `runtime_log.json`. All 8 complete D4 models were checked (score file loads, expected
  arrays, finite, sizes, ledger entry; reloaded weights reproduce the stored val scores, CPU vs GPU max |diff|
  ≤ 2.1e-4), so none is invalid. The three interrupted models (D4-nokey seed 44, D5←A1 seed 42, D5←all-but-A1 seed 42)
  had saved nothing and not touched the test split. They are retrained from scratch with the same seed. See
  `docs/PHASE3B_RESUME.md`.

- **2026-09-26 10:06, frozen `src/kl/features.py` corrupted outside any Phase 3b process.** Before the resume, the
  file started with the 6 stray bytes `claude` (mtime 10:06:03, a minute before the resume session's first command;
  most likely typed into the open editor, which autosaves). Every freeze guard would have raised. With the user's
  approval, exactly those 6 bytes were removed. The file's SHA-256 is again `5be4f320…a63ec`, identical to
  `docs/phase1_frozen_hashes.json`. `assert_frozen_p123` passes for full and quick, and all tests pass. No Phase 3b
  step ran while the file was corrupted.

### Bug fixes

- **2026-09-25, launcher (after the incident above; D4 results for 8 models had been logged, no E4 table existed yet).**
  *Bug:* the bash-loop launcher started the next part when the current one was paused with SIGSTOP, so parts could
  overlap on the GPU. *Fix:* new `scripts/run_phase3b.py`. It runs parts strictly one after the other and waits with
  `Popen.wait()`, which does not return while a child is paused. A launcher lock (`.launcher.lock`) stops a second
  launcher. A training lock (`.training.lock`, taken by `31_train_d4.py` and `32_train_d5.py` for their whole
  lifetime and kept while paused) stops a second training process, however it is started. If a part fails, later
  parts are not started. SIGTERM is forwarded to the running part. Old logs are archived before being overwritten.
  The training scripts now skip only runs that are **complete** (weights + loadable score file with every expected
  finite array + test-ledger entry + runtime entry, `runner.run_state`), rather than any run with a score file. A run
  that used the test split but has no complete result refuses to retrain without `--allow-rerun`. *Why this is not
  tuning:* nothing about models, data, seeds, scores, thresholds or evaluation changed. Only the order of execution
  and the bookkeeping changed, and every model is still trained exactly as pre-registered. Tests: lock exclusivity,
  lock held while paused, `Popen.wait` across SIGSTOP, completion states.

- **2026-09-26, thermal guard (operational safety, requested by the user; not a protocol change).** New
  `src/kl/phase3b/thermal.py`. After every epoch, `nets.fit` calls an optional `between_epochs` hook. The training
  scripts pass `ThermalGuard.check`, which reads `nvidia-smi --query-gpu=temperature.gpu`. At ≥ 85 °C it sleeps until
  ≤ 70 °C and logs every pause (timestamp, temperatures, peak, duration) in the step log and in
  `runtime_log.json["thermal_pauses"]`. If the GPU is still ≥ 90 °C after 30 min of pausing, it writes
  `thermal_stop.json` and exits with code 76. The launcher then prepends the stop state to `PHASE3B_RESUME.md` and
  stops; the interrupted run is retrained from scratch with the same seed on relaunch. The thresholds are constants
  in code, not in `config_phase3b.yaml`, so the pre-registration hash is unchanged. The hook only sleeps and never
  touches the model, optimiser, data or RNGs. A test checks that the weights are bit-identical with and without it.
  `train_seconds` / `seconds_per_epoch` now exclude pause time, and the pause time is logged separately
  (`thermal_paused_seconds`).
