# Phase 3a diagnostics

_Run mode: **full**, seed 42, Phase 1 config hash `8b2fc2193f74312e`. Phases 1 and 2 are frozen (hash-guarded); nothing here changes a feature, detector, attacker or threshold, and no result was used to modify anything. Test split: 43,248 human sessions from 2,999 users; every attacker has one matched bot per human session._

## 1. Centrality ("too typical") analysis

In the frozen transformed feature space we computed each test session's Mahalanobis distance to the train-human distribution, using a robust covariance (scikit-learn `MinCovDet`, random_state = 42, fitted on 20,000 train humans). Test humans have a median distance of 6.13 (IQR 4.34–11.54). A1: median 14.51 (IQR 12.29–17.48; 2.37× the human median; KS D = 0.648, p < 1e-300; 0.0% of its sessions are closer to the centre than the human median). A2: median 5.24 (IQR 4.15–7.76; 0.85× the human median; KS D = 0.143, p < 1e-300; 62.1% of its sessions are closer to the centre than the human median). A3: median 7.29 (IQR 5.26–12.14; 1.19× the human median; KS D = 0.155, p < 1e-300; 37.5% of its sessions are closer to the centre than the human median). A4: median 6.43 (IQR 4.50–10.45; 1.05× the human median; KS D = 0.046, p 2.6e-40; 46.8% of its sessions are closer to the centre than the human median). A5a: median 6.23 (IQR 4.39–11.88; 1.02× the human median; KS D = 0.015, p 1.6e-04; 49.1% of its sessions are closer to the centre than the human median). A5b: median 6.11 (IQR 4.47–11.23; 1.00× the human median; KS D = 0.041, p 9.1e-33; 50.3% of its sessions are closer to the centre than the human median). Attackers whose median distance is below the human median (closer to the centre, 'too typical'): A2, A5b; above it (farther from the centre): A1, A3, A4, A5a.

_Estimator note._ The MCD covariance is rank-deficient in this space: 1 eigenvalue(s) below 1e-10 (condition number 4.6e+21); feature(s) with zero variance inside the MCD support subset (10,328 of 20,000 sessions): `shift_overlap_count`. scikit-learn then uses a pseudo-inverse, so such a feature does not contribute to the distance. The classical-covariance distance is therefore reported as a labelled supplement (below); it is not the specified estimator.

| group | median d | IQR (Q1–Q3) | ratio to human median | KS D vs humans | share closer than human median |
|---|---|---|---|---|---|
| **MCD (specified)** | | | | | |
| human | 6.13 | 4.34–11.54 | 1.00 | – | 50.0% |
| A1 | 14.51 | 12.29–17.48 | 2.37 | 0.648 | 0.0% |
| A2 | 5.24 | 4.15–7.76 | 0.85 | 0.143 | 62.1% |
| A3 | 7.29 | 5.26–12.14 | 1.19 | 0.155 | 37.5% |
| A4 | 6.43 | 4.50–10.45 | 1.05 | 0.046 | 46.8% |
| A5a | 6.23 | 4.39–11.88 | 1.02 | 0.015 | 49.1% |
| A5b | 6.11 | 4.47–11.23 | 1.00 | 0.041 | 50.3% |
| **classical covariance (supplement)** | | | | | |
| human | 3.96 | 3.28–5.04 | 1.00 | – | 50.0% |
| A1 | 7.43 | 6.68–8.22 | 1.88 | 0.783 | 0.0% |
| A2 | 3.56 | 3.10–4.16 | 0.90 | 0.191 | 68.6% |
| A3 | 3.86 | 3.33–4.59 | 0.97 | 0.086 | 54.3% |
| A4 | 3.88 | 3.19–4.95 | 0.98 | 0.037 | 52.5% |
| A5a | 4.01 | 3.31–5.11 | 1.01 | 0.019 | 48.5% |
| A5b | 3.98 | 3.40–4.89 | 1.00 | 0.049 | 49.5% |

Figures: `centrality_hist.png`, `centrality_ecdf.png` (MCD) and `*_classical.png` (supplement).

**Variance ratios.** Per-feature variance of each attacker's test sessions divided by the human variance in the frozen transformed space (ratio < 1 = too homogeneous): A1: median ratio 0.04 (23 of 23 features < 1; raw units: median 0.03, 23 < 1); A2: median ratio 0.19 (22 of 23 features < 1; raw units: median 0.15, 21 < 1); A3: median ratio 0.36 (22 of 23 features < 1; raw units: median 0.25, 23 < 1); A4: median ratio 0.81 (19 of 23 features < 1; raw units: median 0.80, 17 < 1); A5a: median ratio 1.02 (4 of 23 features < 1; raw units: median 1.00, 12 < 1); A5b: median ratio 0.89 (21 of 23 features < 1; raw units: median 0.85, 17 < 1). Per-feature values: `variance_ratio.csv`, `variance_ratio_heatmap.png`.

## 2. Feature-group ablation of D1

We refitted the One-Class SVM with the frozen hyper-parameters (ν = 0.1, γ = 0.3; the input transform is refitted on train for each subset, τ5/τ10 come from validation humans) on feature subsets and evaluated it on A1–A5b. For A1, the full frozen D1 reaches AUC 0.954 (APCER 5.8% at τ10); timing features only give AUC 0.971 (APCER 0.8%), timing+rhythm 0.974 (0.9%), all features except BEHAVIOUR 0.971 (2.1%) and BEHAVIOUR alone 0.774 (100.0%). Removing the BEHAVIOUR group (lev_norm, corrections, shift overlap) therefore changes the A1 AUC by +0.017 and the A1 APCER at τ10 by -3.7 pp; the timing-only detector differs from the full one by +0.017 AUC. 

For the other attackers: A2: best subset BEHAVIOUR_only (AUC 0.774; full D1 0.462); A3: best subset BEHAVIOUR_only (AUC 0.774; full D1 0.428); A4: best subset BEHAVIOUR_only (AUC 0.543; full D1 0.499); A5a: best subset all_minus_BEHAVIOUR (AUC 0.507; full D1 0.506); A5b: best subset BEHAVIOUR_only (AUC 0.768; full D1 0.528).

| subset (features) | attacker | APCER/FPR @τ5 | APCER/FPR @τ10 | BPCER/FNR @τ5 | BPCER/FNR @τ10 | ROC-AUC |
|---|---|---|---|---|---|---|
| ALL (frozen) (23) | A1 | 32.4% [31.9, 32.8] | 5.8% [5.6, 6.0] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.9541 [0.9509, 0.9567] |
| ALL (frozen) (23) | A2 | 99.4% [99.3, 99.4] | 98.0% [97.8, 98.1] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.4619 [0.4554, 0.4683] |
| ALL (frozen) (23) | A3 | 98.8% [98.7, 98.9] | 96.8% [96.6, 96.9] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.4278 [0.4218, 0.4340] |
| ALL (frozen) (23) | A4 | 92.6% [92.4, 92.9] | 86.8% [86.5, 87.2] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.4989 [0.4930, 0.5046] |
| ALL (frozen) (23) | A5a | 95.0% [94.8, 95.2] | 89.9% [89.6, 90.2] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.5060 [0.4999, 0.5119] |
| ALL (frozen) (23) | A5b | 97.2% [97.1, 97.4] | 93.4% [93.1, 93.6] | 4.8% [4.5, 5.1] | 9.9% [9.4, 10.4] | 0.5280 [0.5218, 0.5344] |
| TIMING_only (13) | A1 | 8.7% [8.5, 9.0] | 0.8% [0.8, 0.9] | 4.9% [4.6, 5.3] | 9.8% [9.3, 10.3] | 0.9714 [0.9690, 0.9737] |
| TIMING_only (13) | A2 | 98.8% [98.7, 98.9] | 97.7% [97.5, 97.8] | 4.9% [4.6, 5.3] | 9.8% [9.3, 10.3] | 0.4509 [0.4432, 0.4590] |
| TIMING_only (13) | A3 | 96.4% [96.3, 96.6] | 92.7% [92.5, 93.0] | 4.9% [4.6, 5.3] | 9.8% [9.3, 10.3] | 0.3918 [0.3858, 0.3975] |
| TIMING_only (13) | A4 | 94.4% [94.1, 94.6] | 90.2% [89.9, 90.5] | 4.9% [4.6, 5.3] | 9.8% [9.3, 10.3] | 0.4843 [0.4777, 0.4912] |
| TIMING_only (13) | A5a | 94.4% [94.2, 94.6] | 89.6% [89.4, 89.9] | 4.9% [4.6, 5.3] | 9.8% [9.3, 10.3] | 0.5051 [0.4984, 0.5113] |
| TIMING_only (13) | A5b | 94.9% [94.7, 95.1] | 90.5% [90.2, 90.8] | 4.9% [4.6, 5.3] | 9.8% [9.3, 10.3] | 0.4977 [0.4914, 0.5046] |
| TIMING_RHYTHM (16) | A1 | 6.6% [6.4, 6.8] | 0.9% [0.8, 1.0] | 4.9% [4.6, 5.3] | 9.8% [9.2, 10.3] | 0.9737 [0.9714, 0.9760] |
| TIMING_RHYTHM (16) | A2 | 98.7% [98.6, 98.8] | 97.3% [97.1, 97.4] | 4.9% [4.6, 5.3] | 9.8% [9.2, 10.3] | 0.4064 [0.3994, 0.4135] |
| TIMING_RHYTHM (16) | A3 | 96.8% [96.7, 97.0] | 93.5% [93.3, 93.7] | 4.9% [4.6, 5.3] | 9.8% [9.2, 10.3] | 0.3918 [0.3858, 0.3979] |
| TIMING_RHYTHM (16) | A4 | 91.8% [91.6, 92.1] | 86.7% [86.3, 87.0] | 4.9% [4.6, 5.3] | 9.8% [9.2, 10.3] | 0.4835 [0.4774, 0.4898] |
| TIMING_RHYTHM (16) | A5a | 94.5% [94.3, 94.7] | 89.6% [89.4, 90.0] | 4.9% [4.6, 5.3] | 9.8% [9.2, 10.3] | 0.5056 [0.4984, 0.5116] |
| TIMING_RHYTHM (16) | A5b | 94.9% [94.7, 95.1] | 90.4% [90.1, 90.6] | 4.9% [4.6, 5.3] | 9.8% [9.2, 10.3] | 0.4869 [0.4805, 0.4939] |
| all_minus_BEHAVIOUR (19) | A1 | 11.0% [10.8, 11.4] | 2.1% [1.9, 2.2] | 5.0% [4.6, 5.4] | 9.8% [9.3, 10.3] | 0.9714 [0.9690, 0.9737] |
| all_minus_BEHAVIOUR (19) | A2 | 98.4% [98.3, 98.5] | 96.6% [96.4, 96.8] | 5.0% [4.6, 5.4] | 9.8% [9.3, 10.3] | 0.3849 [0.3786, 0.3917] |
| all_minus_BEHAVIOUR (19) | A3 | 96.9% [96.8, 97.1] | 93.6% [93.3, 93.8] | 5.0% [4.6, 5.4] | 9.8% [9.3, 10.3] | 0.4357 [0.4296, 0.4418] |
| all_minus_BEHAVIOUR (19) | A4 | 92.0% [91.8, 92.3] | 87.2% [86.9, 87.5] | 5.0% [4.6, 5.4] | 9.8% [9.3, 10.3] | 0.4791 [0.4735, 0.4851] |
| all_minus_BEHAVIOUR (19) | A5a | 94.7% [94.5, 94.9] | 89.6% [89.3, 89.9] | 5.0% [4.6, 5.4] | 9.8% [9.3, 10.3] | 0.5074 [0.5009, 0.5133] |
| all_minus_BEHAVIOUR (19) | A5b | 95.1% [94.9, 95.3] | 90.9% [90.7, 91.2] | 5.0% [4.6, 5.4] | 9.8% [9.3, 10.3] | 0.4810 [0.4749, 0.4875] |
| BEHAVIOUR_only (4) | A1 | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 5.0% [4.7, 5.3] | 9.4% [9.0, 9.8] | 0.7739 [0.7700, 0.7780] |
| BEHAVIOUR_only (4) | A2 | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 5.0% [4.7, 5.3] | 9.4% [9.0, 9.8] | 0.7739 [0.7700, 0.7780] |
| BEHAVIOUR_only (4) | A3 | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 5.0% [4.7, 5.3] | 9.4% [9.0, 9.8] | 0.7739 [0.7700, 0.7780] |
| BEHAVIOUR_only (4) | A4 | 94.6% [94.4, 94.9] | 90.2% [89.9, 90.5] | 5.0% [4.7, 5.3] | 9.4% [9.0, 9.8] | 0.5429 [0.5386, 0.5476] |
| BEHAVIOUR_only (4) | A5a | 95.2% [95.0, 95.4] | 90.8% [90.5, 91.1] | 5.0% [4.7, 5.3] | 9.4% [9.0, 9.8] | 0.4980 [0.4937, 0.5023] |
| BEHAVIOUR_only (4) | A5b | 99.5% [99.4, 99.6] | 99.5% [99.4, 99.6] | 5.0% [4.7, 5.3] | 9.4% [9.0, 9.8] | 0.7676 [0.7633, 0.7717] |

_Check:_ refitting ALL 23 features with this machinery reproduces the frozen thresholds: True. Cells carry 95% user-level bootstrap CIs (same resamples as Phase 2). Figures: `ablation_heatmap_auc.png`, `ablation_heatmap_apcer_tau10.png`.

## 3. What gives each attacker away (permutation importance of D3)

For each single-attacker Random Forest (D3 trained on A1, A2, A3, A4, A5b) we measured the mean decrease of test ROC-AUC when one feature is permuted (10 repeats, 5,000 test humans + 5,000 test bots of that attacker, seeded sample; refit equals the Phase 2 detector: True). 
The most important features were — A1 (baseline AUC 1.000): `pp_cv` (0.003), `flight_median` (0.000), `pp_std` (0.000); A2 (baseline AUC 0.994): `lev_norm` (0.038), `hold_mean` (0.015), `corrections_count` (0.011); A3 (baseline AUC 0.988): `lev_norm` (0.037), `hold_mean` (0.024), `corrections_count` (0.016); A4 (baseline AUC 0.933): `hold_mean` (0.136), `hold_cv` (0.060), `rollover_rate` (0.059); A5b (baseline AUC 0.918): `corrections_count` (0.063), `correction_rate` (0.061), `lev_norm` (0.058). Importances are marginal: correlated features share credit, so they should not be summed. For A1 the detector is saturated (baseline AUC ≥ 0.999) and many features are redundant, so permuting any single feature leaves the AUC essentially unchanged: near-zero importances there do not mean that no feature gives the attacker away.

| rank | A1 | A2 | A3 | A4 | A5b |
|---|---|---|---|---|---|
| 1 | `pp_cv` (0.003) | `lev_norm` (0.038) | `lev_norm` (0.037) | `hold_mean` (0.136) | `corrections_count` (0.063) |
| 2 | `flight_median` (0.000) | `hold_mean` (0.015) | `hold_mean` (0.024) | `hold_cv` (0.060) | `correction_rate` (0.061) |
| 3 | `pp_std` (0.000) | `corrections_count` (0.011) | `corrections_count` (0.016) | `rollover_rate` (0.059) | `lev_norm` (0.058) |
| 4 | `corrections_count` (0.000) | `rollover_rate` (0.011) | `hold_cv` (0.013) | `flight_median` (0.029) | `total_duration_ms` (0.007) |
| 5 | `pp_mean` (0.000) | `correction_rate` (0.010) | `correction_rate` (0.013) | `pp_cv` (0.016) | `pp_cv` (0.007) |
| 6 | `correction_rate` (0.000) | `pp_cv` (0.007) | `rollover_rate` (0.008) | `flight_iqr` (0.013) | `pp_std` (0.006) |
| 7 | `lev_norm` (0.000) | `hold_cv` (0.006) | `flight_median` (0.006) | `hold_std` (0.012) | `wpm` (0.006) |
| 8 | `hold_mean` (0.000) | `rr_mean` (0.005) | `pp_cv` (0.006) | `rr_cv` (0.012) | `n_keys` (0.005) |
| 9 | `rr_mean` (0.000) | `shift_overlap_count` (0.005) | `rr_std` (0.005) | `pause_rate` (0.009) | `rr_cv` (0.005) |
| 10 | `flight_mean` (0.000) | `rr_cv` (0.005) | `shift_overlap_count` (0.005) | `flight_mean` (0.004) | `flight_std` (0.004) |

Figure: `importance_heatmap.png`; full table: `permutation_importance.csv`.

## 4. Per-user consistency (A2 / A3)

For every test user with at least 10 clean sessions (2,969 users, median 15 sessions) we computed the session-to-session standard deviation of `pp_mean` and `hold_mean`, and compared it with pseudo-users built from 15 random bot sessions of A2 and A3 (same number of units) and, as a control, with pseudo-users built from random human sessions of different users (real marginal distribution, no user persistence). 
`pp_mean`: real users' median per-user std is 28.9 ms (0.26 of the total std of all sessions; ICC(1) = 0.88); shuffled humans (control): 100.4 ms (0.89 of total; ICC(1) = 0.00; real users' std is lower — Mann–Whitney p = 0.0e+00); A2 pseudo-users: 34.7 ms (0.94 of total; ICC(1) = -0.00; real users' std is lower — Mann–Whitney p = 6.9e-60); A3 pseudo-users: 59.7 ms (0.95 of total; ICC(1) = -0.00; real users' std is lower — Mann–Whitney p = 0.0e+00). `hold_mean`: real users' median per-user std is 9.3 ms (0.37 of the total std of all sessions; ICC(1) = 0.80); shuffled humans (control): 24.0 ms (0.95 of total; ICC(1) = -0.00; real users' std is lower — Mann–Whitney p = 0.0e+00); A2 pseudo-users: 8.8 ms (0.90 of total; ICC(1) = -0.00; real users' std is not lower — Mann–Whitney p = 1.0e+00); A3 pseudo-users: 10.3 ms (0.91 of total; ICC(1) = -0.00; real users' std is lower — Mann–Whitney p = 7.0e-31).

Conclusion from these numbers: real users are not uniformly more self-consistent across sessions than A2/A3 pseudo-users (see the table for the exceptions); median per-unit std relative to the real users' — `pp_mean`: shuffled 3.47×, A2 1.20×, A3 2.06×; `hold_mean`: shuffled 2.59×, A2 0.95×, A3 1.11×. In absolute milliseconds the margin can be small for a generator whose sessions are globally homogeneous (see the variance ratios in section 1), which is why the std relative to the group's total std and the ICC(1) are also given. A2 and A3 draw every session independently, so nothing persists across their sessions; the shuffled-human control shows what that lack of persistence looks like with the real marginal distribution.

| group | feature | units | median std (ms) | Q1–Q3 | median std / total std | ICC(1) | real std lower? (MW p) |
|---|---|---|---|---|---|---|---|
| real users | pp_mean | 2,969 | 28.9 | 20.9–40.8 | 0.26 | 0.88 | – |
| shuffled humans (control) | pp_mean | 2,969 | 100.4 | 81.5–122.5 | 0.89 | 0.00 | 0.0e+00 |
| A2 pseudo-users | pp_mean | 2,969 | 34.7 | 29.4–40.8 | 0.94 | -0.00 | 6.9e-60 |
| A3 pseudo-users | pp_mean | 2,969 | 59.7 | 50.4–70.3 | 0.95 | -0.00 | 0.0e+00 |
| real users | hold_mean | 2,969 | 9.3 | 6.8–12.5 | 0.37 | 0.80 | – |
| shuffled humans (control) | hold_mean | 2,969 | 24.0 | 20.6–27.7 | 0.95 | -0.00 | 0.0e+00 |
| A2 pseudo-users | hold_mean | 2,969 | 8.8 | 7.4–10.8 | 0.90 | -0.00 | 1.0e+00 |
| A3 pseudo-users | hold_mean | 2,969 | 10.3 | 8.7–12.4 | 0.91 | -0.00 | 7.0e-31 |

Figure: `user_consistency.png`; per-unit values: `user_consistency_per_unit.csv`.

