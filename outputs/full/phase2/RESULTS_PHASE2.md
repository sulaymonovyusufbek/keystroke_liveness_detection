# Phase 2 results

_Run mode: **full**, seed 42, Phase 1 config hash `8b2fc2193f74312e` (frozen). Attackers were fixed before any detector was run. APCER = FPR = attacks accepted; BPCER = FNR = humans rejected; a session is accepted iff score ≥ τ. 95% CIs: bootstrap over test users (1,000 resamples, the same resamples in every cell)._

## Paper-style summary

We evaluated three detectors on the held-out test split (43,248 human sessions from 2,999 users) against six non-adaptive attackers whose parameters were fitted only on disjoint attacker-pool users. The frozen one-class SVM (D1) rejected 4.8% of humans at τ5 and 9.9% at τ10 (BPCER/FNR); the Isolation Forest (D2), thresholded on validation humans in the same way, 4.6% and 9.6%. At τ10 the attack accepted most often by D1 among A1–A5b was A2 (APCER/FPR 98.0%, AUC 0.462) and the least often A1 (5.8%, AUC 0.954); for D2 the most accepted was A1 (99.9%, AUC 0.796). The verbatim replay A5a, a genuine human recording, was accepted at τ10 by D1 in 89.9% and by D2 in 89.5% of cases (AUC 0.506 and 0.509), as expected: session-level behavioural liveness cannot separate a fresh human session from a recorded one, which motivates challenge sentences and freshness checks rather than a better classifier.

A supervised Random Forest (D3) trained on one attacker accepted, at τ10, on average 11.5% of attacks from the attacker it was trained on but 61.4% of the other attackers A1–A4/A5b (cross-attack matrix); when the tested attacker was left out of training entirely (leave-one-attack-out), the mean APCER/FPR at τ10 was 33.9% (range 3.7–70.0%).

Attacker realism against held-out attacker-pool sessions (mean Kolmogorov–Smirnov statistic over the 23 features; lower = closer): A1 0.527, A2 0.306, A3 0.304, A4 0.124, A5a 0.020, A5b 0.117. Rollover rate (bot vs pool mean): A2 0.280 vs 0.236, A3 0.280 vs 0.236, A4 0.256 vs 0.236, A5b 0.240 vs 0.236. The realism report documents attacker strength only and was not used to tune any attacker.

## E1 — open-set table (test split; D1 frozen OCSVM, D2 Isolation Forest; both attack-free)

| detector | attacker | APCER/FPR @τ5 | APCER/FPR @τ10 | BPCER/FNR @τ5 | BPCER/FNR @τ10 | ROC-AUC |
|---|---|---|---|---|---|---|
| D1 | A1 | 32.4% [31.9, 32.8] | 5.8% [5.6, 6.0] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.9541 [0.9509, 0.9567] |
| D1 | A2 | 99.4% [99.3, 99.4] | 98.0% [97.8, 98.1] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.4619 [0.4554, 0.4683] |
| D1 | A3 | 98.8% [98.7, 98.9] | 96.8% [96.6, 96.9] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.4278 [0.4218, 0.4340] |
| D1 | A4 | 92.6% [92.4, 92.9] | 86.8% [86.5, 87.2] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.4989 [0.4930, 0.5046] |
| D1 | A5a | 95.0% [94.8, 95.2] | 89.9% [89.6, 90.2] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.5060 [0.4999, 0.5119] |
| D1 | A5b | 97.2% [97.1, 97.4] | 93.4% [93.1, 93.6] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.5280 [0.5218, 0.5344] |
| D2 | A1 | 100.0% [100.0, 100.0] | 99.9% [99.9, 99.9] | 4.6% [4.1, 5.0] | 9.6% [9.0, 10.2] | 0.7964 [0.7875, 0.8050] |
| D2 | A2 | 99.6% [99.6, 99.7] | 98.5% [98.3, 98.6] | 4.6% [4.1, 5.0] | 9.6% [9.0, 10.2] | 0.3183 [0.3114, 0.3254] |
| D2 | A3 | 99.5% [99.5, 99.6] | 98.0% [97.9, 98.1] | 4.6% [4.1, 5.0] | 9.6% [9.0, 10.2] | 0.3135 [0.3065, 0.3207] |
| D2 | A4 | 96.1% [95.9, 96.3] | 90.6% [90.3, 90.9] | 4.6% [4.1, 5.0] | 9.6% [9.0, 10.2] | 0.4831 [0.4758, 0.4906] |
| D2 | A5a | 94.8% [94.6, 95.0] | 89.5% [89.2, 89.8] | 4.6% [4.1, 5.0] | 9.6% [9.0, 10.2] | 0.5087 [0.5005, 0.5160] |
| D2 | A5b | 96.8% [96.6, 97.0] | 93.1% [92.9, 93.4] | 4.6% [4.1, 5.0] | 9.6% [9.0, 10.2] | 0.4258 [0.4185, 0.4332] |

## E2 — cross-attack matrix: APCER/FPR @τ10 (rows = detector / training attack, columns = tested attack)

| trained on | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|
| D1 ← attack-free | 5.8% [5.6, 6.0] | 98.0% [97.8, 98.1] | 96.8% [96.6, 96.9] | 86.8% [86.5, 87.2] | 89.9% [89.6, 90.2] | 93.4% [93.1, 93.6] |
| D2 ← attack-free | 99.9% [99.9, 99.9] | 98.5% [98.3, 98.6] | 98.0% [97.9, 98.1] | 90.6% [90.3, 90.9] | 89.5% [89.2, 89.8] | 93.1% [92.9, 93.4] |
| D3 ← A1 | 0.0% [0.0, 0.0] | 97.9% [97.7, 98.0] | 83.3% [83.0, 83.7] | 90.0% [89.7, 90.2] | 94.7% [94.5, 94.9] | 93.1% [92.8, 93.3] |
| D3 ← A2 | 99.9% [99.8, 99.9] | 0.2% [0.1, 0.2] | 40.1% [39.6, 40.5] | 76.0% [75.6, 76.4] | 90.1% [89.8, 90.4] | 48.5% [48.1, 49.0] |
| D3 ← A3 | 0.0% [0.0, 0.0] | 68.7% [68.3, 69.2] | 1.0% [0.9, 1.1] | 77.9% [77.5, 78.3] | 90.5% [90.2, 90.7] | 68.1% [67.6, 68.5] |
| D3 ← A4 | 11.8% [11.5, 12.1] | 27.0% [26.6, 27.4] | 18.1% [17.7, 18.4] | 23.7% [23.3, 24.2] | 90.8% [90.5, 91.0] | 82.6% [82.2, 82.9] |
| D3 ← A5b | 91.2% [90.9, 91.5] | 15.0% [14.6, 15.3] | 57.4% [56.9, 57.9] | 81.7% [81.4, 82.1] | 89.0% [88.7, 89.3] | 32.8% [32.3, 33.2] |

## E2 — cross-attack matrix: ROC-AUC (rows = detector / training attack, columns = tested attack)

| trained on | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|
| D1 ← attack-free | 0.9541 [0.9509, 0.9567] | 0.4619 [0.4554, 0.4683] | 0.4278 [0.4218, 0.4340] | 0.4989 [0.4930, 0.5046] | 0.5060 [0.4999, 0.5119] | 0.5280 [0.5218, 0.5344] |
| D2 ← attack-free | 0.7964 [0.7875, 0.8050] | 0.3183 [0.3114, 0.3254] | 0.3135 [0.3065, 0.3207] | 0.4831 [0.4758, 0.4906] | 0.5087 [0.5005, 0.5160] | 0.4258 [0.4185, 0.4332] |
| D3 ← A1 | 1.0000 [0.9999, 1.0000] | 0.4859 [0.4841, 0.4877] | 0.5604 [0.5580, 0.5628] | 0.5261 [0.5241, 0.5283] | 0.5019 [0.4999, 0.5038] | 0.5101 [0.5081, 0.5121] |
| D3 ← A2 | 0.7569 [0.7509, 0.7628] | 0.9939 [0.9933, 0.9945] | 0.8510 [0.8475, 0.8545] | 0.5967 [0.5920, 0.6012] | 0.4941 [0.4894, 0.4985] | 0.8357 [0.8318, 0.8393] |
| D3 ← A3 | 0.9861 [0.9850, 0.9872] | 0.7927 [0.7883, 0.7975] | 0.9878 [0.9870, 0.9887] | 0.5492 [0.5450, 0.5537] | 0.5011 [0.4961, 0.5061] | 0.8098 [0.8056, 0.8144] |
| D3 ← A4 | 0.9589 [0.9569, 0.9611] | 0.9247 [0.9215, 0.9283] | 0.9447 [0.9424, 0.9473] | 0.9332 [0.9304, 0.9362] | 0.4970 [0.4892, 0.5051] | 0.5865 [0.5793, 0.5940] |
| D3 ← A5b | 0.8077 [0.8029, 0.8129] | 0.9563 [0.9546, 0.9580] | 0.8736 [0.8699, 0.8774] | 0.5572 [0.5527, 0.5619] | 0.5007 [0.4960, 0.5055] | 0.9198 [0.9173, 0.9226] |

## E3 — leave-one-attack-out (D3 trained on all training attackers except the tested one)

| tested attacker (held out) | detector | APCER/FPR @τ5 | APCER/FPR @τ10 | BPCER/FNR @τ5 | BPCER/FNR @τ10 | ROC-AUC |
|---|---|---|---|---|---|---|
| A1 | D3 (all-but-A1) | 11.5% [11.2, 11.8] | 3.7% [3.5, 3.9] | 4.9% [4.6, 5.2] | 10.0% [9.6, 10.5] | 0.9784 [0.9772, 0.9798] |
| A2 | D3 (all-but-A2) | 30.3% [29.9, 30.7] | 11.1% [10.8, 11.4] | 4.9% [4.6, 5.2] | 10.1% [9.6, 10.5] | 0.9587 [0.9569, 0.9607] |
| A3 | D3 (all-but-A3) | 36.6% [36.1, 37.1] | 19.0% [18.6, 19.4] | 4.9% [4.6, 5.1] | 9.8% [9.4, 10.2] | 0.9414 [0.9392, 0.9439] |
| A4 | D3 (all-but-A4) | 74.9% [74.5, 75.3] | 70.0% [69.6, 70.4] | 4.8% [4.6, 5.1] | 10.0% [9.6, 10.4] | 0.5826 [0.5783, 0.5872] |
| A5b | D3 (all-but-A5b) | 76.3% [75.9, 76.7] | 65.7% [65.2, 66.1] | 4.9% [4.5, 5.2] | 9.5% [9.0, 9.9] | 0.6944 [0.6886, 0.7009] |
| A1 | D1 (attack-free) | 32.4% [31.9, 32.8] | 5.8% [5.6, 6.0] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.9541 [0.9509, 0.9567] |
| A2 | D1 (attack-free) | 99.4% [99.3, 99.4] | 98.0% [97.8, 98.1] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.4619 [0.4554, 0.4683] |
| A3 | D1 (attack-free) | 98.8% [98.7, 98.9] | 96.8% [96.6, 96.9] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.4278 [0.4218, 0.4340] |
| A4 | D1 (attack-free) | 92.6% [92.4, 92.9] | 86.8% [86.5, 87.2] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.4989 [0.4930, 0.5046] |
| A5b | D1 (attack-free) | 97.2% [97.1, 97.4] | 93.4% [93.1, 93.6] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.5280 [0.5218, 0.5344] |
| A1 | D2 (attack-free) | 100.0% [100.0, 100.0] | 99.9% [99.9, 99.9] | 4.6% [4.1, 5.0] | 9.6% [9.0, 10.2] | 0.7964 [0.7875, 0.8050] |
| A2 | D2 (attack-free) | 99.6% [99.6, 99.7] | 98.5% [98.3, 98.6] | 4.6% [4.1, 5.0] | 9.6% [9.0, 10.2] | 0.3183 [0.3114, 0.3254] |
| A3 | D2 (attack-free) | 99.5% [99.5, 99.6] | 98.0% [97.9, 98.1] | 4.6% [4.1, 5.0] | 9.6% [9.0, 10.2] | 0.3135 [0.3065, 0.3207] |
| A4 | D2 (attack-free) | 96.1% [95.9, 96.3] | 90.6% [90.3, 90.9] | 4.6% [4.1, 5.0] | 9.6% [9.0, 10.2] | 0.4831 [0.4758, 0.4906] |
| A5b | D2 (attack-free) | 96.8% [96.6, 97.0] | 93.1% [92.9, 93.4] | 4.6% [4.1, 5.0] | 9.6% [9.0, 10.2] | 0.4258 [0.4185, 0.4332] |

## Attacker realism (mean KS statistic over features; full table in realism_report.csv)

| attacker | mean KS | max KS (feature) |
|---|---|---|
| A1 | 0.527 | 0.937 (`pp_cv`) |
| A2 | 0.306 | 0.646 (`corrections_count`) |
| A3 | 0.304 | 0.646 (`corrections_count`) |
| A4 | 0.124 | 0.270 (`hold_mean`) |
| A5a | 0.020 | 0.032 (`flight_median`) |
| A5b | 0.117 | 0.646 (`corrections_count`) |

## Generation counts

| attacker | split | bots kept | dropped (non-finite) |
|---|---|---|---|
| A1 | train | 20,000 | 0 |
| A2 | train | 20,000 | 0 |
| A3 | train | 20,000 | 0 |
| A4 | train | 20,000 | 0 |
| A5a | train | 20,000 | 0 |
| A5b | train | 20,000 | 0 |
| A1 | test | 43,248 | 0 |
| A2 | test | 43,248 | 0 |
| A3 | test | 43,248 | 0 |
| A4 | test | 43,248 | 0 |
| A5a | test | 43,248 | 0 |
| A5b | test | 43,248 | 0 |
