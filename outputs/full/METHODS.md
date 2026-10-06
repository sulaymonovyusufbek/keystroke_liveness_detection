# Methods

_Run mode: **full**, seed 42, config hash `8b2fc2193f74312e`. Protocol: docs/PHASE1_SPEC.md incl. Revision 1._

## Paper-ready paragraph

**Data.** We used the Aalto 136M Keystrokes dataset (Dhakal et al., CHI 2018), in which each participant typed up to 15 sentences in an online typing test. The metadata file lists 168,594 participants (KEYBOARD_TYPE counts — laptop: 91,250, full: 73,759, small: 1,886, on-screen: 1,699). We kept only participants who typed on a physical keyboard (KEYBOARD_TYPE ∈ {full, laptop}), excluding small-physical and on-screen (touch) keyboards and any missing or unmapped value, which leaves 165,009 eligible participants, 165,008 of which have a keystroke file. From these we drew 20,000 participants uniformly at random (seed 42); 0 sampled files could not be parsed and 14,598,748 keystroke rows were read.

**Sessions and cleaning.** A session is one typed sentence (PARTICIPANT_ID, TEST_SECTION_ID). Of 300,000 sessions, 11,215 were dropped (0 with fewer than 10 keystrokes; 0 with missing/non-numeric times; 8,201 with a hold < 0 or > 2000 ms; 5,267 with a press-to-press gap > 5000 ms; 0 with empty SENTENCE or USER_INPUT; rules are counted independently, so one session can hit several), leaving 288,785 clean sessions.

**Split.** Users were split at the user level into train/validation/test/attacker-pool (60%/15%/15%/10%): 11,996/3,000/2,999/1,999 users and 20,000/43,291/43,248/28,854 sessions (train capped at 20,000 sessions from 173,392 by seeded uniform sampling). The attacker pool is never used for training or tuning.

**Model.** Each session is summarised by 23 timing, rhythm, behaviour and global features (FEATURES.md). Bounded rates (correction_rate, lev_norm, pause_rate, rollover_rate) are used as they are; every other feature is passed through the signed logarithm y = sign(x)·log(1 + |x|), and all features are then scaled with a RobustScaler (median / IQR) fitted on the training split only. The transformed features feed an RBF One-Class SVM trained on human sessions only. We searched ν ∈ {0.01, 0.02, 0.05, 0.1, 0.15}, γ ∈ {scale, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3} (35 configurations). Without any attack data, configurations were first screened for overfitting: a configuration is admissible iff |FNR_val,native − ν| ≤ max(0.02, 0.5·ν), where FNR_val,native is the fraction of validation humans with a negative decision value (under the ν-property of the One-Class SVM, Schölkopf et al., 2001, ν approximates the held-out outlier fraction); 32 of 35 configurations were admissible. Among these, the configuration was chosen with a reference set of M = 100,000 points drawn from the product of the marginals of the transformed training data — every feature is resampled independently, with replacement, from its own training column (cf. the artificial-outlier construction of Hempstalk, Frank & Witten, 2008, ECML PKDD) — and we picked the configuration accepting the fewest reference points at the 5%-FNR threshold (ties within 0.1 percentage points → smaller ν, then smaller γ). A guard stops the selection if reference acceptance spans less than 1 pp across the grid (observed spread 98.9 pp, from 0.8% to 99.2%) or if no configuration is admissible; the tolerance is never relaxed automatically. Chosen: ν = 0.10, γ = 0.3 (effective 0.3000); the decision threshold τ was set on the validation humans as the score quantile giving FNR = 5% (τ5 = -1.3422) and 10% (τ10 = -0.3524). A session is accepted as human iff its score ≥ τ.

**Selection rule revision.** The originally specified rule — uniform artificial outliers (Tax & Duin, 2001) drawn in the per-feature [min, max] box of the standardised training data — was abandoned because it was degenerate. In the initial quick-mode run (seed 42, 2000 users) (22 features, standard scaling), 34 of 35 configurations accepted none of the 100,000 uniform points at τ5, the most permissive accepted 0.002%, and 35 of 35 configurations were tied within the 0.1 pp tolerance, so the tie-break alone chose the model (ν = 0.01, γ = 0.001). In 22 dimensions the volume of the box is dominated by heavy-tailed features — notably flight_cv, whose denominator vanishes when rollover makes the mean flight ≈ 0 (train maximum ≈ 7.1e+07) — so uniformly drawn points almost never fall near the data and the criterion carries no information. The revision (product-of-marginals reference set, signed-log + robust transform, flight_cv replaced by flight_median and flight_iqr) was motivated by this degeneracy of the selection diagnostic, not by attack performance. The uniform-box acceptance is still computed as a diagnostic column (`uniform_accept_tau5` in grid_results.csv) and is never used for selection; under the revised transform it ranges from 0.000% to 0.248% (20 configurations at exactly 0%). Disclosure: the initial quick-mode test and A1 results had already been produced with the original rule before the revision was decided; the quick-mode test split was therefore evaluated twice (once under the original rule, once under the revised rule; `--allow-rerun`), whereas the revised protocol has not touched any full-scale test split before the freeze.

**Selection rule revision 2 (final).** With Revision 1 alone, the product-of-marginals criterion favoured the largest γ in the grid (γ = 0.3): in the quick-mode run the selected configuration (ν = 0.01) showed attack-free overfitting diagnostics — a validation FNR at the native decision boundary of 11.5% against ν = 1%, and 1,952 support vectors against 173–236 for the smoother ν = 0.01 configurations. The admissibility filter (|FNR_val,native − ν| ≤ max(0.02, 0.5·ν)) was added on the basis of those diagnostics, which do not involve attack data. Disclosure: post-hoc A1 results had already been seen when this revision was made (in that run the Revision 1 configuration ranked 20 of 35 by A1 ROC-AUC in the post-hoc sensitivity table), so Revision 2 is not independent of them, although its trigger and its formula are attack-free. Attacks A2–A5 (Phase 2) will be designed only after this freeze.

**Freeze.** Revision 2 is final: no further changes to the features, the input transform, the hyperparameter grid or the selection rule will be made in Phase 1, whatever the A1 results.

**Human test.** The test split (held out until selection was frozen) contains 43,248 sessions from 2,999 users. The false-negative rate (humans rejected) is 4.78% at τ5 (95% user-level bootstrap CI 4.45%–5.12%) and 9.90% at τ10 (9.40%–10.40%; 1000 resamples of users).

**A1 sanity check.** After freezing the model we generated one bot session per test session (43,248) with a generator that draws independent Gaussian holds N(116.24, 23.88) ms (min 30 ms) and flights N(238.656, 111.6) ms (min 60 ms). Bots accepted as human (FPR): 32.47% at τ5 and 5.86% at τ10; ROC-AUC = 0.9540.

**Post-hoc sensitivity (not used for selection).** A1 was also evaluated for all 35 grid configurations, each with its own τ5/τ10 from the validation humans (grid_attack_sensitivity.csv). The selected configuration ranks 14 of 35 by ROC-AUC (0.9540; grid range 0.7940–0.9893, median 0.9390; best: ν = 0.01, γ = scale, AUC 0.9893) and 13 of 35 by FPR at τ5 (32.47%; grid range 0.79%–100.00%). This table is post-hoc and was not used to choose the configuration.

## Results at a glance

| quantity | value |
|---|---|
| FNR (humans rejected) @ τ5, test | 4.78% |
| FNR (humans rejected) @ τ10, test | 9.90% |
| FPR (bots accepted) @ τ5, A1 | 32.47% |
| FPR (bots accepted) @ τ10, A1 | 5.86% |
| ROC-AUC (human vs A1) | 0.9540 |

## Assumptions and implementation choices

1. **HOLD_MIN = 30 ms (ASSUMPTION)** and FLIGHT_MIN = 60 ms clamp the A1 draws; A1 times are rounded to integer ms.
2. **A1 Shift handling is a reconstruction:** Shift-down → key-down delay is a flight draw; Shift is released HOLD_MIN ms after the key is released; the next character starts one flight after the Shift release. Unmapped characters get keycode 229.
3. A1 keycodes: letters (either case) use the uppercase ASCII code, digits their ASCII code, space 32; punctuation follows the US-layout table in `kl/attacks/a1_gaussian.py`.
4. Standard deviations are population standard deviations (ddof = 0); `flight_iqr` uses linearly interpolated percentiles (`np.percentile`).
5. `correction_rate` = corrections_count / n_keys; `rollover_rate` = share of the n_keys − 1 intervals with flight < 0.
6. `shift_overlap_count`: for each Shift hold, non-Shift keys with PRESS strictly inside (PRESS_shift, RELEASE_shift) minus one, floored at zero, summed over holds.
7. Keys are identified by KEYCODE only; Enter (13) is never relied upon. Sessions are sorted by PRESS_TIME with KEYSTROKE_ID as tie-break only.
8. Files are decoded as UTF-8 and fall back to latin-1 (the actual encoding of the corpus) when that fails; rows whose PARTICIPANT_ID differs from the file name are dropped and counted.
9. The user split is drawn among participants that own at least one clean session (cleaning precedes splitting).
10. Sessions with a non-finite feature (e.g. zero total duration) are dropped at the feature step and counted in features_report.json.
11. τ is the empirical `np.quantile` of the validation scores at 5% / 10%; a session is rejected iff score < τ.
12. Features with an IQR of zero on the train split (e.g. rarely non-zero counts) keep a scale of 1 in scikit-learn's RobustScaler.
13. `gamma = "scale"` is resolved to its numeric value 1 / (n_features · Var(X_transformed_train)) before fitting; ties in the selection rule use this effective gamma.
14. The reference set uses the transformed training sessions actually used to fit the SVM (after the session cap); the same reference points are used for every configuration. Reference acceptance is measured at the τ5 of each configuration.
15. Bootstrap: percentile CI over 1000 resamples of test USERS (all their sessions), seeded.
16. 0 of 1,796,262 sentence characters (0.000%) have no keycode mapping and were emitted with keycode 229.

## Selection diagnostics

- Admissible configurations: 32 of 35 (rule: |val_fnr_native - nu| <= max(0.02, 0.5 * nu)). Not admissible: ν=0.01, γ=0.3 (val FNR 10.9%); ν=0.02, γ=0.3 (val FNR 10.9%); ν=0.05, γ=0.3 (val FNR 10.9%).
- Reference acceptance at τ5 spans 0.82%–99.20% across the 35 configurations (spread 98.91 pp); among the admissible configurations the best is 0.82% and 1 of them are within 0.1 pp of it.
- Uniform-box acceptance at τ5 (diagnostic, not used): 0.000%–0.248%; 20 configuration(s) at exactly 0%.

## Coefficient-of-variation features (train split)

| feature | min | median | max |
|---|---|---|---|
| `hold_cv` | 0.1059 | 0.3608 | 5.705 |
| `pp_cv` | 0.1716 | 0.6712 | 2.771 |
| `rr_cv` | 0.1972 | 0.7335 | 4.646 |
