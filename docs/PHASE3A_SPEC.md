Phase 3a (diagnostics only). Save as docs/PHASE3A_SPEC.md with a CLAUDE.md pointer. Phases 1
and 2 are FROZEN (keep the hash guard). No new features, no new detectors, no changes to
attackers.

1. Centrality ("too typical") analysis, TEST split:
   - In the frozen transformed feature space, compute each session's Mahalanobis distance to the
     train-human distribution (covariance estimated on train humans with the MinCovDet from
     sklearn.covariance, random_state=seed).
   - Compare the distributions for test humans vs A1–A5b: median, IQR, a KS test, and overlaid
     histograms/ECDFs.
   - Also report each attacker's per-feature variance divided by the human variance. A ratio
     < 1 means "too homogeneous".

2. Feature-group ablation for D1 (answers Reviewer 1):
   - Refit the One-Class SVM with the FROZEN hyperparameters (ν=0.10, γ=0.3; τ from val humans)
     on each feature subset: TIMING only; TIMING+RHYTHM; all minus BEHAVIOUR; BEHAVIOUR only.
   - Evaluate on A1–A5b: APCER@τ5/τ10, BPCER, AUC, with bootstrap CIs by user.
   - Key question: how much of D1's A1 detection comes from BEHAVIOUR (lev_norm, corrections)
     vs timing.

3. What gives each attacker away (from D3, supervised):
   - For each single-attacker D3 (trained on A1, A2, A3, A4, A5b): permutation importance on a
     held-out TEST sample (sklearn.inspection.permutation_importance, scoring="roc_auc",
     n_repeats=10). Report the top 10 features per attacker as a table plus a heatmap
     (features × attackers).

4. Per-user consistency check (for A2/A3):
   - For test humans, compute each user's session-to-session std of pp_mean and hold_mean
     across their ~15 sessions.
   - Compare with "pseudo-users" built from 15 random A2/A3 bot sessions.
   - Report whether real users are more self-consistent across sessions than bots.

Outputs go to outputs/full/phase3a/: CSV tables, figures, and DIAGNOSTICS.md, a paper-style
paragraph per analysis with the numbers. Work order: implement → run the tests → quick run → full
run → report the tables. Don't change anything based on the results.
