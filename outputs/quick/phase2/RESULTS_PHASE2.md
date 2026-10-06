# Phase 2 results

_Run mode: **quick**, seed 42, Phase 1 config hash `107e859345c78e77` (frozen). Attackers were fixed before any detector was run. APCER = FPR = attacks accepted; BPCER = FNR = humans rejected; a session is accepted iff score ≥ τ. 95% CIs: bootstrap over test users (1,000 resamples, the same resamples in every cell)._

## Paper-style summary

We evaluated three detectors on the held-out test split (4,344 human sessions from 300 users) against six non-adaptive attackers whose parameters were fitted only on disjoint attacker-pool users. The frozen one-class SVM (D1) rejected 5.0% of humans at τ5 and 9.9% at τ10 (BPCER/FNR); the Isolation Forest (D2), thresholded on validation humans in the same way, 4.4% and 8.5%. At τ10 the attack accepted most often by D1 among A1–A5b was A2 (APCER/FPR 97.2%, AUC 0.526) and the least often A1 (16.8%, AUC 0.934); for D2 the most accepted was A1 (99.9%, AUC 0.777). The verbatim replay A5a, a genuine human recording, was accepted at τ10 by D1 in 90.2% and by D2 in 89.3% of cases (AUC 0.493 and 0.520), as expected: session-level behavioural liveness cannot separate a fresh human session from a recorded one, which motivates challenge sentences and freshness checks rather than a better classifier.

A supervised Random Forest (D3) trained on one attacker accepted, at τ10, on average 9.7% of attacks from the attacker it was trained on but 63.8% of the other attackers A1–A4/A5b (cross-attack matrix); when the tested attacker was left out of training entirely (leave-one-attack-out), the mean APCER/FPR at τ10 was 35.0% (range 6.7–69.5%).

Attacker realism against held-out attacker-pool sessions (mean Kolmogorov–Smirnov statistic over the 23 features; lower = closer): A1 0.529, A2 0.316, A3 0.314, A4 0.148, A5a 0.065, A5b 0.144. Rollover rate (bot vs pool mean): A2 0.298 vs 0.248, A3 0.294 vs 0.248, A4 0.265 vs 0.248, A5b 0.236 vs 0.248. The realism report documents attacker strength only and was not used to tune any attacker.

## E1 — open-set table (test split; D1 frozen OCSVM, D2 Isolation Forest; both attack-free)

| detector | attacker | APCER/FPR @τ5 | APCER/FPR @τ10 | BPCER/FNR @τ5 | BPCER/FNR @τ10 | ROC-AUC |
|---|---|---|---|---|---|---|
| D1 | A1 | 53.4% [51.9, 54.8] | 16.8% [15.7, 17.9] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.9342 [0.9228, 0.9443] |
| D1 | A2 | 99.0% [98.7, 99.3] | 97.2% [96.6, 97.7] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.5260 [0.5068, 0.5457] |
| D1 | A3 | 97.6% [97.1, 98.1] | 94.0% [93.3, 94.6] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.4453 [0.4291, 0.4637] |
| D1 | A4 | 91.9% [91.1, 92.8] | 86.3% [85.2, 87.5] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.4966 [0.4800, 0.5143] |
| D1 | A5a | 95.1% [94.5, 95.7] | 90.2% [89.5, 91.1] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.4932 [0.4750, 0.5119] |
| D1 | A5b | 97.3% [96.8, 97.8] | 93.6% [92.7, 94.3] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.5374 [0.5192, 0.5561] |
| D2 | A1 | 100.0% [100.0, 100.0] | 99.9% [99.8, 100.0] | 4.4% [3.2, 5.7] | 8.5% [6.7, 10.4] | 0.7768 [0.7494, 0.8026] |
| D2 | A2 | 99.7% [99.5, 99.8] | 98.6% [98.2, 99.0] | 4.4% [3.2, 5.7] | 8.5% [6.7, 10.4] | 0.3147 [0.2967, 0.3328] |
| D2 | A3 | 98.4% [98.1, 98.7] | 95.8% [95.2, 96.4] | 4.4% [3.2, 5.7] | 8.5% [6.7, 10.4] | 0.3341 [0.3159, 0.3538] |
| D2 | A4 | 96.4% [95.9, 97.0] | 91.3% [90.4, 92.1] | 4.4% [3.2, 5.7] | 8.5% [6.7, 10.4] | 0.4808 [0.4596, 0.5021] |
| D2 | A5a | 94.0% [93.3, 94.8] | 89.3% [88.3, 90.3] | 4.4% [3.2, 5.7] | 8.5% [6.7, 10.4] | 0.5197 [0.4966, 0.5413] |
| D2 | A5b | 96.8% [96.3, 97.3] | 93.8% [93.0, 94.6] | 4.4% [3.2, 5.7] | 8.5% [6.7, 10.4] | 0.4122 [0.3913, 0.4326] |

## E2 — cross-attack matrix: APCER/FPR @τ10 (rows = detector / training attack, columns = tested attack)

| trained on | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|
| D1 ← attack-free | 16.8% [15.7, 17.9] | 97.2% [96.6, 97.7] | 94.0% [93.3, 94.6] | 86.3% [85.2, 87.5] | 90.2% [89.5, 91.1] | 93.6% [92.7, 94.3] |
| D2 ← attack-free | 99.9% [99.8, 100.0] | 98.6% [98.2, 99.0] | 95.8% [95.2, 96.4] | 91.3% [90.4, 92.1] | 89.3% [88.3, 90.3] | 93.8% [93.0, 94.6] |
| D3 ← A1 | 0.0% [0.0, 0.0] | 98.2% [97.8, 98.6] | 86.3% [85.3, 87.2] | 91.3% [90.5, 92.2] | 95.9% [95.4, 96.5] | 93.7% [93.0, 94.4] |
| D3 ← A2 | 100.0% [100.0, 100.0] | 0.3% [0.2, 0.5] | 41.0% [39.7, 42.4] | 77.0% [75.7, 78.1] | 91.0% [90.1, 91.8] | 53.8% [52.4, 55.3] |
| D3 ← A3 | 0.7% [0.5, 1.0] | 69.0% [67.5, 70.4] | 1.0% [0.7, 1.4] | 74.4% [73.1, 75.6] | 90.1% [89.3, 90.9] | 70.9% [69.5, 72.2] |
| D3 ← A4 | 15.8% [14.6, 16.9] | 31.4% [29.9, 32.7] | 19.2% [18.0, 20.2] | 28.6% [27.3, 29.9] | 92.3% [91.5, 93.0] | 86.4% [85.3, 87.5] |
| D3 ← A5b | 100.0% [100.0, 100.0] | 14.3% [13.3, 15.4] | 66.9% [65.4, 68.2] | 84.9% [83.8, 85.8] | 81.4% [80.2, 82.5] | 18.3% [17.3, 19.4] |

## E2 — cross-attack matrix: ROC-AUC (rows = detector / training attack, columns = tested attack)

| trained on | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|
| D1 ← attack-free | 0.9342 [0.9228, 0.9443] | 0.5260 [0.5068, 0.5457] | 0.4453 [0.4291, 0.4637] | 0.4966 [0.4800, 0.5143] | 0.4932 [0.4750, 0.5119] | 0.5374 [0.5192, 0.5561] |
| D2 ← attack-free | 0.7768 [0.7494, 0.8026] | 0.3147 [0.2967, 0.3328] | 0.3341 [0.3159, 0.3538] | 0.4808 [0.4596, 0.5021] | 0.5197 [0.4966, 0.5413] | 0.4122 [0.3913, 0.4326] |
| D3 ← A1 | 1.0000 [1.0000, 1.0000] | 0.4831 [0.4782, 0.4885] | 0.5443 [0.5379, 0.5513] | 0.5181 [0.5121, 0.5242] | 0.4947 [0.4891, 0.5006] | 0.5057 [0.5003, 0.5117] |
| D3 ← A2 | 0.5074 [0.4908, 0.5262] | 0.9931 [0.9914, 0.9949] | 0.8613 [0.8528, 0.8710] | 0.6022 [0.5900, 0.6154] | 0.5087 [0.4942, 0.5235] | 0.8368 [0.8263, 0.8486] |
| D3 ← A3 | 0.9820 [0.9776, 0.9862] | 0.8470 [0.8339, 0.8599] | 0.9900 [0.9877, 0.9922] | 0.5813 [0.5679, 0.5960] | 0.5217 [0.5061, 0.5367] | 0.8199 [0.8054, 0.8337] |
| D3 ← A4 | 0.9497 [0.9416, 0.9577] | 0.9293 [0.9196, 0.9391] | 0.9488 [0.9415, 0.9559] | 0.9324 [0.9237, 0.9414] | 0.5062 [0.4839, 0.5301] | 0.5813 [0.5578, 0.6049] |
| D3 ← A5b | 0.7546 [0.7357, 0.7752] | 0.9600 [0.9550, 0.9650] | 0.8670 [0.8555, 0.8795] | 0.5664 [0.5529, 0.5824] | 0.5410 [0.5267, 0.5559] | 0.9491 [0.9432, 0.9552] |

## E3 — leave-one-attack-out (D3 trained on all training attackers except the tested one)

| tested attacker (held out) | detector | APCER/FPR @τ5 | APCER/FPR @τ10 | BPCER/FNR @τ5 | BPCER/FNR @τ10 | ROC-AUC |
|---|---|---|---|---|---|---|
| A1 | D3 (all-but-A1) | 39.2% [37.8, 40.8] | 6.7% [5.8, 7.4] | 3.4% [2.7, 4.1] | 8.6% [7.5, 9.7] | 0.9620 [0.9559, 0.9682] |
| A2 | D3 (all-but-A2) | 28.7% [27.3, 30.0] | 10.6% [9.6, 11.5] | 3.9% [3.2, 4.6] | 8.4% [7.2, 9.5] | 0.9667 [0.9620, 0.9719] |
| A3 | D3 (all-but-A3) | 39.7% [38.3, 41.2] | 19.0% [17.7, 20.2] | 4.2% [3.5, 5.0] | 8.7% [7.6, 9.8] | 0.9474 [0.9413, 0.9541] |
| A4 | D3 (all-but-A4) | 75.4% [74.1, 76.8] | 69.2% [67.8, 70.4] | 3.8% [3.1, 4.5] | 8.4% [7.3, 9.4] | 0.6089 [0.5960, 0.6234] |
| A5b | D3 (all-but-A5b) | 81.4% [80.2, 82.5] | 69.5% [68.2, 70.8] | 3.6% [2.8, 4.3] | 7.7% [6.4, 8.9] | 0.7032 [0.6843, 0.7223] |
| A1 | D1 (attack-free) | 53.4% [51.9, 54.8] | 16.8% [15.7, 17.9] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.9342 [0.9228, 0.9443] |
| A2 | D1 (attack-free) | 99.0% [98.7, 99.3] | 97.2% [96.6, 97.7] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.5260 [0.5068, 0.5457] |
| A3 | D1 (attack-free) | 97.6% [97.1, 98.1] | 94.0% [93.3, 94.6] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.4453 [0.4291, 0.4637] |
| A4 | D1 (attack-free) | 91.9% [91.1, 92.8] | 86.3% [85.2, 87.5] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.4966 [0.4800, 0.5143] |
| A5b | D1 (attack-free) | 97.3% [96.8, 97.8] | 93.6% [92.7, 94.3] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.5374 [0.5192, 0.5561] |
| A1 | D2 (attack-free) | 100.0% [100.0, 100.0] | 99.9% [99.8, 100.0] | 4.4% [3.2, 5.7] | 8.5% [6.7, 10.4] | 0.7768 [0.7494, 0.8026] |
| A2 | D2 (attack-free) | 99.7% [99.5, 99.8] | 98.6% [98.2, 99.0] | 4.4% [3.2, 5.7] | 8.5% [6.7, 10.4] | 0.3147 [0.2967, 0.3328] |
| A3 | D2 (attack-free) | 98.4% [98.1, 98.7] | 95.8% [95.2, 96.4] | 4.4% [3.2, 5.7] | 8.5% [6.7, 10.4] | 0.3341 [0.3159, 0.3538] |
| A4 | D2 (attack-free) | 96.4% [95.9, 97.0] | 91.3% [90.4, 92.1] | 4.4% [3.2, 5.7] | 8.5% [6.7, 10.4] | 0.4808 [0.4596, 0.5021] |
| A5b | D2 (attack-free) | 96.8% [96.3, 97.3] | 93.8% [93.0, 94.6] | 4.4% [3.2, 5.7] | 8.5% [6.7, 10.4] | 0.4122 [0.3913, 0.4326] |

## Attacker realism (mean KS statistic over features; full table in realism_report.csv)

| attacker | mean KS | max KS (feature) |
|---|---|---|
| A1 | 0.529 | 0.937 (`pp_cv`) |
| A2 | 0.316 | 0.632 (`corrections_count`) |
| A3 | 0.314 | 0.632 (`corrections_count`) |
| A4 | 0.148 | 0.349 (`hold_cv`) |
| A5a | 0.065 | 0.109 (`hold_cv`) |
| A5b | 0.144 | 0.632 (`corrections_count`) |

## Generation counts

| attacker | split | bots kept | dropped (non-finite) |
|---|---|---|---|
| A1 | train | 17,143 | 0 |
| A2 | train | 17,143 | 0 |
| A3 | train | 17,143 | 0 |
| A4 | train | 17,143 | 0 |
| A5a | train | 17,143 | 0 |
| A5b | train | 17,143 | 0 |
| A1 | test | 4,344 | 0 |
| A2 | test | 4,344 | 0 |
| A3 | test | 4,344 | 0 |
| A4 | test | 4,344 | 0 |
| A5a | test | 4,344 | 0 |
| A5b | test | 4,344 | 0 |
