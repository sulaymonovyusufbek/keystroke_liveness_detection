# Legacy protocol vs clean protocol

_Run mode: **full**. Legacy numbers are IN-SAMPLE by construction (the same 5,000 human sessions are used to fit the model and to test it, and the configuration is chosen by F1 on humans + A1)._

| | legacy protocol | clean protocol |
|---|---|---|
| training humans | same sessions as the test | train split (disjoint users) |
| test humans | same sessions as training | test split (disjoint users, touched once) |
| hyperparameter selection | best F1 on humans + A1 (uses attack data) | attack-free: admissibility filter + product-of-marginals reference set (Revision 2) |
| chosen (nu, gamma) | (0.1, 0.1) | (0.1, 0.3) |
| decision threshold | native OCSVM boundary (score >= 0) | tau5 / tau10 set on validation humans |
| FNR (humans rejected) | 10.02% | 4.78% @ tau5 [4.45%, 5.12%]; 9.90% @ tau10 |
| FPR (bots accepted) | 2.94% | 32.47% @ tau5; 5.86% @ tau10 |
| ROC-AUC (human vs A1) | 0.9578 | 0.9540 |
| F1 (bots = positive) | 0.9374 | not used for selection |

Notes: the legacy replica uses the same features and input transform (signed log + RobustScaler) as the clean pipeline, so only the protocol differs. F1 uses bots as the positive class (assumption; F1 with humans positive is in legacy_grid.csv). If the clean columns show n/a, run steps 05-07 first.
