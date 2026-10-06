# Phase 3a diagnostics

_Run mode: **quick**, seed 42, Phase 1 config hash `107e859345c78e77`. Phases 1 and 2 are frozen (hash-guarded); nothing here changes a feature, detector, attacker or threshold, and no result was used to modify anything. Test split: 4,344 human sessions from 300 users; every attacker has one matched bot per human session._

## 1. Centrality ("too typical") analysis

In the frozen transformed feature space we computed each test session's Mahalanobis distance to the train-human distribution, using a robust covariance (scikit-learn `MinCovDet`, random_state = 42, fitted on 17,143 train humans). Test humans have a median distance of 5.72 (IQR 4.23–10.16). A1: median 12.92 (IQR 10.95–15.34; 2.26× the human median; KS D = 0.656, p < 1e-300; 0.0% of its sessions are closer to the centre than the human median). A2: median 5.54 (IQR 4.28–8.39; 0.97× the human median; KS D = 0.086, p 2.4e-14; 52.5% of its sessions are closer to the centre than the human median). A3: median 6.87 (IQR 5.00–11.55; 1.20× the human median; KS D = 0.152, p 2.8e-44; 37.0% of its sessions are closer to the centre than the human median). A4: median 6.58 (IQR 4.56–11.19; 1.15× the human median; KS D = 0.093, p 1.3e-16; 41.1% of its sessions are closer to the centre than the human median). A5a: median 6.25 (IQR 4.32–12.53; 1.09× the human median; KS D = 0.071, p 6.5e-10; 44.7% of its sessions are closer to the centre than the human median). A5b: median 5.95 (IQR 4.36–11.03; 1.04× the human median; KS D = 0.045, p 3.4e-04; 47.4% of its sessions are closer to the centre than the human median). Attackers whose median distance is below the human median (closer to the centre, 'too typical'): A2; above it (farther from the centre): A1, A3, A4, A5a, A5b.

_Estimator note._ The MCD covariance is rank-deficient in this space: 1 eigenvalue(s) below 1e-10 (condition number 2.3e+21); feature(s) with zero variance inside the MCD support subset (8,962 of 17,143 sessions): `shift_overlap_count`. scikit-learn then uses a pseudo-inverse, so such a feature does not contribute to the distance. The classical-covariance distance is therefore reported as a labelled supplement (below); it is not the specified estimator.

| group | median d | IQR (Q1–Q3) | ratio to human median | KS D vs humans | share closer than human median |
|---|---|---|---|---|---|
| **MCD (specified)** | | | | | |
| human | 5.72 | 4.23–10.16 | 1.00 | – | 50.0% |
| A1 | 12.92 | 10.95–15.34 | 2.26 | 0.656 | 0.0% |
| A2 | 5.54 | 4.28–8.39 | 0.97 | 0.086 | 52.5% |
| A3 | 6.87 | 5.00–11.55 | 1.20 | 0.152 | 37.0% |
| A4 | 6.58 | 4.56–11.19 | 1.15 | 0.093 | 41.1% |
| A5a | 6.25 | 4.32–12.53 | 1.09 | 0.071 | 44.7% |
| A5b | 5.95 | 4.36–11.03 | 1.04 | 0.045 | 47.4% |
| **classical covariance (supplement)** | | | | | |
| human | 3.91 | 3.28–4.94 | 1.00 | – | 50.0% |
| A1 | 7.49 | 6.76–8.26 | 1.91 | 0.802 | 0.0% |
| A2 | 3.67 | 3.18–4.30 | 0.94 | 0.141 | 61.9% |
| A3 | 3.88 | 3.32–4.68 | 0.99 | 0.055 | 51.4% |
| A4 | 3.95 | 3.23–5.06 | 1.01 | 0.031 | 48.9% |
| A5a | 4.04 | 3.35–5.20 | 1.03 | 0.049 | 46.5% |
| A5b | 3.98 | 3.40–4.93 | 1.02 | 0.055 | 47.1% |

Figures: `centrality_hist.png`, `centrality_ecdf.png` (MCD) and `*_classical.png` (supplement).

**Variance ratios.** Per-feature variance of each attacker's test sessions divided by the human variance in the frozen transformed space (ratio < 1 = too homogeneous): A1: median ratio 0.04 (23 of 23 features < 1; raw units: median 0.03, 23 < 1); A2: median ratio 0.19 (21 of 23 features < 1; raw units: median 0.16, 20 < 1); A3: median ratio 0.44 (22 of 23 features < 1; raw units: median 0.26, 21 < 1); A4: median ratio 0.86 (17 of 23 features < 1; raw units: median 0.91, 16 < 1); A5a: median ratio 1.12 (6 of 23 features < 1; raw units: median 1.08, 6 < 1); A5b: median ratio 0.96 (14 of 23 features < 1; raw units: median 0.91, 18 < 1). Per-feature values: `variance_ratio.csv`, `variance_ratio_heatmap.png`.

## 2. Feature-group ablation of D1

We refitted the One-Class SVM with the frozen hyper-parameters (ν = 0.1, γ = 0.3; the input transform is refitted on train for each subset, τ5/τ10 come from validation humans) on feature subsets and evaluated it on A1–A5b. For A1, the full frozen D1 reaches AUC 0.934 (APCER 16.8% at τ10); timing features only give AUC 0.969 (APCER 3.1%), timing+rhythm 0.972 (2.2%), all features except BEHAVIOUR 0.970 (4.6%) and BEHAVIOUR alone 0.775 (100.0%). Removing the BEHAVIOUR group (lev_norm, corrections, shift overlap) therefore changes the A1 AUC by +0.036 and the A1 APCER at τ10 by -12.2 pp; the timing-only detector differs from the full one by +0.035 AUC. 

For the other attackers: A2: best subset BEHAVIOUR_only (AUC 0.775; full D1 0.526); A3: best subset BEHAVIOUR_only (AUC 0.775; full D1 0.445); A4: best subset BEHAVIOUR_only (AUC 0.545; full D1 0.497); A5a: best subset all_minus_BEHAVIOUR (AUC 0.506; full D1 0.493); A5b: best subset BEHAVIOUR_only (AUC 0.770; full D1 0.537).

| subset (features) | attacker | APCER/FPR @τ5 | APCER/FPR @τ10 | BPCER/FNR @τ5 | BPCER/FNR @τ10 | ROC-AUC |
|---|---|---|---|---|---|---|
| ALL (frozen) (23) | A1 | 53.4% [51.9, 54.8] | 16.8% [15.7, 17.9] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.9342 [0.9228, 0.9443] |
| ALL (frozen) (23) | A2 | 99.0% [98.7, 99.3] | 97.2% [96.6, 97.7] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.5260 [0.5068, 0.5457] |
| ALL (frozen) (23) | A3 | 97.6% [97.1, 98.1] | 94.0% [93.3, 94.6] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.4453 [0.4291, 0.4637] |
| ALL (frozen) (23) | A4 | 91.9% [91.1, 92.8] | 86.3% [85.2, 87.5] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.4966 [0.4800, 0.5143] |
| ALL (frozen) (23) | A5a | 95.1% [94.5, 95.7] | 90.2% [89.5, 91.1] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.4932 [0.4750, 0.5119] |
| ALL (frozen) (23) | A5b | 97.3% [96.8, 97.8] | 93.6% [92.7, 94.3] | 5.0% [4.0, 6.1] | 9.9% [8.4, 11.4] | 0.5374 [0.5192, 0.5561] |
| TIMING_only (13) | A1 | 21.6% [20.4, 22.9] | 3.1% [2.6, 3.6] | 3.6% [2.8, 4.6] | 8.7% [7.4, 10.3] | 0.9691 [0.9603, 0.9761] |
| TIMING_only (13) | A2 | 97.8% [97.4, 98.2] | 95.9% [95.4, 96.5] | 3.6% [2.8, 4.6] | 8.7% [7.4, 10.3] | 0.4708 [0.4494, 0.4933] |
| TIMING_only (13) | A3 | 94.2% [93.4, 94.8] | 91.0% [90.2, 91.8] | 3.6% [2.8, 4.6] | 8.7% [7.4, 10.3] | 0.3648 [0.3485, 0.3812] |
| TIMING_only (13) | A4 | 92.9% [92.1, 93.7] | 89.3% [88.4, 90.2] | 3.6% [2.8, 4.6] | 8.7% [7.4, 10.3] | 0.4772 [0.4584, 0.4958] |
| TIMING_only (13) | A5a | 96.0% [95.5, 96.6] | 91.6% [90.8, 92.4] | 3.6% [2.8, 4.6] | 8.7% [7.4, 10.3] | 0.5037 [0.4833, 0.5235] |
| TIMING_only (13) | A5b | 95.6% [94.9, 96.2] | 91.2% [90.4, 92.1] | 3.6% [2.8, 4.6] | 8.7% [7.4, 10.3] | 0.4985 [0.4792, 0.5195] |
| TIMING_RHYTHM (16) | A1 | 15.9% [14.7, 17.2] | 2.2% [1.8, 2.7] | 3.8% [2.9, 4.8] | 8.4% [7.0, 10.0] | 0.9715 [0.9626, 0.9784] |
| TIMING_RHYTHM (16) | A2 | 97.7% [97.2, 98.1] | 95.7% [95.1, 96.3] | 3.8% [2.9, 4.8] | 8.4% [7.0, 10.0] | 0.4409 [0.4196, 0.4621] |
| TIMING_RHYTHM (16) | A3 | 94.6% [93.9, 95.3] | 91.2% [90.4, 91.9] | 3.8% [2.9, 4.8] | 8.4% [7.0, 10.0] | 0.3692 [0.3527, 0.3850] |
| TIMING_RHYTHM (16) | A4 | 91.2% [90.4, 92.0] | 85.7% [84.7, 86.6] | 3.8% [2.9, 4.8] | 8.4% [7.0, 10.0] | 0.4811 [0.4630, 0.4984] |
| TIMING_RHYTHM (16) | A5a | 95.8% [95.2, 96.4] | 90.8% [90.0, 91.6] | 3.8% [2.9, 4.8] | 8.4% [7.0, 10.0] | 0.5045 [0.4856, 0.5245] |
| TIMING_RHYTHM (16) | A5b | 95.6% [95.0, 96.3] | 91.0% [90.1, 91.8] | 3.8% [2.9, 4.8] | 8.4% [7.0, 10.0] | 0.4869 [0.4685, 0.5064] |
| all_minus_BEHAVIOUR (19) | A1 | 22.4% [21.1, 23.6] | 4.6% [4.0, 5.2] | 3.7% [2.8, 4.7] | 8.1% [6.9, 9.5] | 0.9702 [0.9616, 0.9766] |
| all_minus_BEHAVIOUR (19) | A2 | 97.8% [97.3, 98.3] | 95.4% [94.7, 96.0] | 3.7% [2.8, 4.7] | 8.1% [6.9, 9.5] | 0.4442 [0.4259, 0.4619] |
| all_minus_BEHAVIOUR (19) | A3 | 95.0% [94.3, 95.6] | 91.3% [90.4, 92.0] | 3.7% [2.8, 4.7] | 8.1% [6.9, 9.5] | 0.4052 [0.3893, 0.4209] |
| all_minus_BEHAVIOUR (19) | A4 | 91.9% [91.1, 92.7] | 86.6% [85.6, 87.6] | 3.7% [2.8, 4.7] | 8.1% [6.9, 9.5] | 0.4841 [0.4666, 0.5004] |
| all_minus_BEHAVIOUR (19) | A5a | 95.8% [95.2, 96.4] | 90.7% [89.8, 91.5] | 3.7% [2.8, 4.7] | 8.1% [6.9, 9.5] | 0.5059 [0.4881, 0.5245] |
| all_minus_BEHAVIOUR (19) | A5b | 95.9% [95.3, 96.5] | 91.6% [90.8, 92.4] | 3.7% [2.8, 4.7] | 8.1% [6.9, 9.5] | 0.4781 [0.4601, 0.4961] |
| BEHAVIOUR_only (4) | A1 | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 6.9% [5.7, 8.2] | 10.0% [8.6, 11.5] | 0.7746 [0.7611, 0.7879] |
| BEHAVIOUR_only (4) | A2 | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 6.9% [5.7, 8.2] | 10.0% [8.6, 11.5] | 0.7746 [0.7611, 0.7879] |
| BEHAVIOUR_only (4) | A3 | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 6.9% [5.7, 8.2] | 10.0% [8.6, 11.5] | 0.7746 [0.7611, 0.7879] |
| BEHAVIOUR_only (4) | A4 | 94.3% [93.6, 95.0] | 91.5% [90.7, 92.4] | 6.9% [5.7, 8.2] | 10.0% [8.6, 11.5] | 0.5454 [0.5315, 0.5600] |
| BEHAVIOUR_only (4) | A5a | 94.4% [93.8, 95.0] | 91.3% [90.6, 92.0] | 6.9% [5.7, 8.2] | 10.0% [8.6, 11.5] | 0.5010 [0.4859, 0.5159] |
| BEHAVIOUR_only (4) | A5b | 99.7% [99.5, 99.9] | 99.7% [99.5, 99.9] | 6.9% [5.7, 8.2] | 10.0% [8.6, 11.5] | 0.7696 [0.7560, 0.7831] |

_Check:_ refitting ALL 23 features with this machinery reproduces the frozen thresholds: True. Cells carry 95% user-level bootstrap CIs (same resamples as Phase 2). Figures: `ablation_heatmap_auc.png`, `ablation_heatmap_apcer_tau10.png`.

## 3. What gives each attacker away (permutation importance of D3)

For each single-attacker Random Forest (D3 trained on A1, A2, A3, A4, A5b) we measured the mean decrease of test ROC-AUC when one feature is permuted (10 repeats, 5,000 test humans + 5,000 test bots of that attacker, seeded sample; refit equals the Phase 2 detector: True). 
The most important features were — A1 (baseline AUC 1.000): `pp_cv` (0.000), `flight_median` (0.000), `pp_std` (0.000); A2 (baseline AUC 0.993): `lev_norm` (0.034), `corrections_count` (0.014), `hold_mean` (0.014); A3 (baseline AUC 0.990): `lev_norm` (0.035), `hold_mean` (0.020), `corrections_count` (0.015); A4 (baseline AUC 0.932): `hold_mean` (0.122), `hold_cv` (0.064), `rollover_rate` (0.056); A5b (baseline AUC 0.949): `correction_rate` (0.062), `corrections_count` (0.052), `lev_norm` (0.047). Importances are marginal: correlated features share credit, so they should not be summed. For A1 the detector is saturated (baseline AUC ≥ 0.999) and many features are redundant, so permuting any single feature leaves the AUC essentially unchanged: near-zero importances there do not mean that no feature gives the attacker away.

| rank | A1 | A2 | A3 | A4 | A5b |
|---|---|---|---|---|---|
| 1 | `pp_cv` (0.000) | `lev_norm` (0.034) | `lev_norm` (0.035) | `hold_mean` (0.122) | `correction_rate` (0.062) |
| 2 | `flight_median` (0.000) | `corrections_count` (0.014) | `hold_mean` (0.020) | `hold_cv` (0.064) | `corrections_count` (0.052) |
| 3 | `pp_std` (0.000) | `hold_mean` (0.014) | `corrections_count` (0.015) | `rollover_rate` (0.056) | `lev_norm` (0.047) |
| 4 | `correction_rate` (0.000) | `rollover_rate` (0.009) | `hold_cv` (0.014) | `flight_median` (0.024) | `pp_cv` (0.019) |
| 5 | `corrections_count` (0.000) | `correction_rate` (0.008) | `correction_rate` (0.013) | `pp_cv` (0.018) | `flight_median` (0.015) |
| 6 | `hold_mean` (0.000) | `pp_cv` (0.006) | `rollover_rate` (0.009) | `hold_std` (0.013) | `hold_mean` (0.013) |
| 7 | `lev_norm` (0.000) | `hold_cv` (0.006) | `flight_median` (0.007) | `flight_iqr` (0.010) | `flight_mean` (0.012) |
| 8 | `pp_mean` (0.000) | `shift_overlap_count` (0.005) | `rr_std` (0.006) | `rr_cv` (0.008) | `flight_iqr` (0.010) |
| 9 | `rr_mean` (0.000) | `wpm` (0.004) | `shift_overlap_count` (0.005) | `pause_rate` (0.005) | `wpm` (0.010) |
| 10 | `pause_count` (0.000) | `rr_cv` (0.003) | `hold_std` (0.005) | `correction_rate` (0.005) | `rr_cv` (0.009) |

Figure: `importance_heatmap.png`; full table: `permutation_importance.csv`.

## 4. Per-user consistency (A2 / A3)

For every test user with at least 10 clean sessions (297 users, median 15 sessions) we computed the session-to-session standard deviation of `pp_mean` and `hold_mean`, and compared it with pseudo-users built from 15 random bot sessions of A2 and A3 (same number of units) and, as a control, with pseudo-users built from random human sessions of different users (real marginal distribution, no user persistence). 
`pp_mean`: real users' median per-user std is 28.4 ms (0.27 of the total std of all sessions; ICC(1) = 0.89); shuffled humans (control): 93.5 ms (0.92 of total; ICC(1) = -0.00; real users' std is lower — Mann–Whitney p = 2.1e-93); A2 pseudo-users: 31.8 ms (0.96 of total; ICC(1) = -0.00; real users' std is lower — Mann–Whitney p = 1.2e-03); A3 pseudo-users: 55.9 ms (0.94 of total; ICC(1) = 0.01; real users' std is lower — Mann–Whitney p = 1.2e-63). `hold_mean`: real users' median per-user std is 8.6 ms (0.34 of the total std of all sessions; ICC(1) = 0.81); shuffled humans (control): 24.0 ms (0.94 of total; ICC(1) = 0.01; real users' std is lower — Mann–Whitney p = 1.8e-87); A2 pseudo-users: 9.4 ms (0.83 of total; ICC(1) = 0.01; real users' std is lower — Mann–Whitney p = 1.6e-02); A3 pseudo-users: 11.4 ms (0.85 of total; ICC(1) = 0.00; real users' std is lower — Mann–Whitney p = 1.4e-18).

Conclusion from these numbers: real users are MORE self-consistent across sessions than A2/A3 pseudo-users (for both features, with p < 0.05 in every comparison); median per-unit std relative to the real users' — `pp_mean`: shuffled 3.30×, A2 1.12×, A3 1.97×; `hold_mean`: shuffled 2.78×, A2 1.09×, A3 1.32×. In absolute milliseconds the margin can be small for a generator whose sessions are globally homogeneous (see the variance ratios in section 1), which is why the std relative to the group's total std and the ICC(1) are also given. A2 and A3 draw every session independently, so nothing persists across their sessions; the shuffled-human control shows what that lack of persistence looks like with the real marginal distribution.

| group | feature | units | median std (ms) | Q1–Q3 | median std / total std | ICC(1) | real std lower? (MW p) |
|---|---|---|---|---|---|---|---|
| real users | pp_mean | 297 | 28.4 | 21.4–38.7 | 0.27 | 0.89 | – |
| shuffled humans (control) | pp_mean | 297 | 93.5 | 75.2–112.2 | 0.92 | -0.00 | 2.1e-93 |
| A2 pseudo-users | pp_mean | 297 | 31.8 | 26.5–36.8 | 0.96 | -0.00 | 1.2e-03 |
| A3 pseudo-users | pp_mean | 297 | 55.9 | 46.6–66.6 | 0.94 | 0.01 | 1.2e-63 |
| real users | hold_mean | 297 | 8.6 | 6.7–12.1 | 0.34 | 0.81 | – |
| shuffled humans (control) | hold_mean | 297 | 24.0 | 21.0–28.5 | 0.94 | 0.01 | 1.8e-87 |
| A2 pseudo-users | hold_mean | 297 | 9.4 | 7.6–12.1 | 0.83 | 0.01 | 1.6e-02 |
| A3 pseudo-users | hold_mean | 297 | 11.4 | 9.7–14.2 | 0.85 | 0.00 | 1.4e-18 |

Figure: `user_consistency.png`; per-unit values: `user_consistency_per_unit.csv`.

