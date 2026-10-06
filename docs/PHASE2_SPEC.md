Phase 2. First save this message verbatim as docs/PHASE2_SPEC.md and add a pointer in CLAUDE.md.
Then implement it. Phase 1 is FROZEN: do not modify features.py, model.py, selection.py,
config protocol keys, or anything in outputs/full/model/. Before any evaluation, assert that the
config hash matches the frozen one.

## Threat model (write this into ATTACKS.md)
Non-adaptive attackers with population-level typing statistics. Every attacker parameter is
fitted ONLY on attacker_pool sessions (users disjoint from train/val/test). Attackers never
query the detector. Each attacker types the SAME target sentence as the matched human session
(one bot session per human session per split), and all bot output goes through the
unchanged features.py. Keycode conventions match A1.

## Attackers (src/kl/attacks/), same interface: generate(sentence, rng) -> Aalto-schema DataFrame
A1 existing (independent Gaussians). Unchanged.

A2 per-digraph empirical:
  - From attacker_pool, build empirical distributions of pp (press-to-press) per digraph
    (lower-cased char pair) and of hold per key.
  - Backoff when a digraph has < 30 samples: per second character → global.
  - Sample independently per keystroke from those empirical distributions (negative flights
    arise naturally when hold > pp).

A3 conditional/temporal:
  - Session speed factor: draw one session-level log-mean pp from the attacker_pool distribution
    of session means.
  - log pp_i = μ_digraph + s + φ·(log pp_{i−1} − μ_prev − s) + ε, with ε ~ N(0, σ²).
    Fit φ and σ on attacker_pool residuals.
  - Hold: per-key empirical, scaled by the session speed factor using the pool's hold–speed
    regression.

A4 realistic imitator (A3 plus behaviour, all rates drawn per session from the attacker_pool
  empirical distributions):
  - Pauses: insert extra flight > 500 ms, more likely at word boundaries (fit the boundary vs
    within-word pause rates on the pool).
  - Errors: substitution typos using adjacent QWERTY keys. Each error is corrected via Backspace
    (keycode 8) either immediately or after 1–3 more characters (fit the correction-delay
    distribution on the pool where possible; otherwise document the assumption). A fraction
    stays uncorrected, so the distribution of lev_norm matches the pool.
  - Shift overlap: with the pool's per-session probability, release Shift after the next key
    press.
  - Rollover: already produced by the A3 timing; verify that rollover_rate is similar to the pool.

A5 replay:
  - A5a verbatim: replay a random attacker_pool session exactly (its own sentence and timings).
    This is a real human recording, and we EXPECT the detector to accept it. Report it as the
    boundary of session-level behavioural liveness (it motivates challenge sentences and
    freshness checks). Do not treat it as a detector failure to fix.
  - A5b transplanted: take a random attacker_pool session's hold/flight sequence and apply it in
    order to the target sentence. If the target is longer, continue by resampling a contiguous
    chunk of another pool session; if shorter, truncate. USER_INPUT = target sentence.

Realism report: for each attacker, compare every feature's distribution against held-out
attacker_pool sessions (KS statistic and the mean difference). Save realism_report.csv. This
documents attacker strength; it must NOT be used to tune attackers against the detector.

## Detectors
D1 One-Class SVM: the frozen Phase 1 model and τ5/τ10. Load it; never refit.
D2 Isolation Forest: pre-declared hyperparameters (n_estimators=300, max_samples=256,
   random_state=seed), same transform, trained on train humans. τ5/τ10 set on val humans only.
   No tuning.
D3 Random Forest (supervised): n_estimators=300, class_weight="balanced", other settings default,
   same transform. Trained on train humans + train bots from ONE attacker A_i. τ5/τ10 set on val
   humans only (so that its FNR matches D1/D2).

## Experiments
E1 Open-set table: D1 and D2 against A1, A2, A3, A4, A5a, A5b on the TEST split.
E2 Cross-attack matrix: D3 trained on A_i (rows: A1–A4, A5b), tested on A_j (columns: A1–A4,
   A5a, A5b). Add a D1 row and a D2 row (attack-free) in the same table for direct comparison.
E3 Leave-one-attack-out: D3 trained on all attackers except A_j, tested on A_j.
Metrics per cell:
  - APCER (= FPR, attacks accepted) at τ5 and τ10.
  - BPCER (= FNR, humans rejected) at τ5 and τ10.
  - ROC-AUC.
  - 95% bootstrap CIs (resampling by user, 1000 reps).
  Report both naming conventions (APCER/BPCER and FPR/FNR) in the headers.

## Outputs (outputs/full/phase2/)
attack_matrix.csv, open_set_table.csv, loao_table.csv, realism_report.csv,
heatmap figures (APCER@τ10 and AUC), ROC curves per attacker for D1, ATTACKS.md (a precise
description of every generator with all fitted parameter values, so it is reproducible), and a
RESULTS_PHASE2.md paragraph in paper style.

## Tests
Schema validity and determinism for every attacker. A2 reproduces the digraph means on 10k
samples within 5%. A4's correction and lev_norm distributions are within tolerance of the pool.
A5a output is byte-identical to the source session's timings. A5b preserves order and handles
both length cases. Assert that attacker_pool users never appear in any train/val/test bot
generation input except as the attacker's statistics source.

Work order: implement → run the tests → a --quick run → then the full run → report the tables to
me. Do not change any attacker after seeing detector results. If something is broken (a bug,
not "too easy/too hard"), fix it, then say what changed and why.
