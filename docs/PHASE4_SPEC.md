Phase 4: revision analyses requested by the supervisor (Part A). Save this spec as
docs/PHASE4_SPEC.md. Everything from Phases 1, 2, 3a, 3b and 3c is FROZEN: do not retrain,
re-select or change any existing model, threshold, attacker or split. Use the saved weights,
saved thresholds and saved score files. If something needed below was not saved, recompute it
ONLY from frozen models and VALIDATION humans, and say so in the report.

Before writing any analysis code:
- Compute the SHA-256 hash of docs/PHASE4_SPEC.md and append it with a timestamp to the same
  hash record file used in earlier phases (no git).
- Re-verify the hashes of the Phase 1–3c frozen configs; stop and tell me if any changed.

Project root: /home/nitro/PycharmProjects/keystroke_liveness. Outputs go to outputs/phase4/.
The final report goes to docs/PHASE4_RESULTS.md.

=====================================================================
A1. Bootstrap confidence intervals for the main results
=====================================================================
Goal: 95% CIs for the headline numbers, reflecting uncertainty from the TEST POPULATION.
Report seed variation SEPARATELY (these are different things).

Resampling unit = TEST USER (cluster bootstrap), 1,000 replicates, seed 42.
- In each replicate, draw test users with replacement. Each drawn user brings ALL of their
  human test sessions AND every synthetic session matched to those sessions (the matched bot
  for each attacker shares the human session's target sentence).
- Check how A5a (replay) sessions are linked to test sessions. If each A5a session is matched
  to a test human session, resample it with that session. If not, resample A5a sessions
  independently within the attacker, and DOCUMENT exactly which rule was used.
- Thresholds stay FIXED at the frozen validation values (primary analysis).
- Sensitivity check, D4-primary only: also resample VALIDATION users in each replicate and
  recompute tau5/tau10 inside the replicate. Report these CIs next to the primary ones.

Seeds: for sequence models (D4, D5), in each replicate compute the metric for each of the 3
seeds and average over seeds; the CI is taken over these averages. Separately report the
across-seed standard deviation of the point estimate (no bootstrap).

Metrics with point estimate + 95% percentile CI:
- Human-only detectors D1, D2, D4-primary, D4-20k, D4-nokey, against A1, A2, A3, A4, A5a, A5b:
  APCER at tau10, APCER at tau5, BPCER at tau10, BPCER at tau5, ROC-AUC.
- D3 and D5 leave-one-attacker-out: APCER at tau10 and tau5 for the held-out attacker, and the
  BPCER of each model.
- D3 and D5 cross-attack matrix (trained on one, tested on all): point estimates at tau10 AND
  tau5, plus CIs only for the off-diagonal averages (excluding A5a) of 61.4% (D3) and
  37.7% (D5).

=====================================================================
A2. APCER at 5% BPCER (tau5)
=====================================================================
Covered by A1. Make sure tau5 exists for every detector and seed. If tau5 was not saved for
D4 or D5, compute it now from frozen scores of VALIDATION humans only (quantile so that 5% of
validation humans are rejected), save it, and note this in the report.

=====================================================================
A3. Identical-key subset (Table VIII of the manuscript)
=====================================================================
Using the identical-key human–bot pairs from Phase 3c (same pairs, do not rebuild them):
- Report D4-primary BPCER at tau10 and tau5 on the human sessions of these pairs, EXACTLY,
  per attacker (A2, A3, A4, A5b), mean over seeds + bootstrap CI (resampling unit = test user).
- Report the same for D5, and state EXACTLY which D5 variant(s) were used in Phase 3c for
  this table (trained on the same attacker? leave-one-out? which seeds?).
- Report the number of pairs per attacker and explain why A4 has fewer pairs (how the subset
  was selected).

=====================================================================
A4. New detector D6: simple key-pair timing model (pre-registered, humans only)
=====================================================================
Purpose: separate the value of KEY CONDITIONING from the value of the GRU in D4.
All choices below are fixed now. No tuning. No attack data in fitting or selection.

Data:
- Fit on ALL training-human sessions (the same 173,392 sessions used by D4), using the same
  key vocabulary, the same truncation (first 94 keys), and the same per-step timing as D4:
  h_i = log(1 + hold_i), p_i = log(1 + pp_i) in ms, with p masked on the last key.
- Use the RAW log values (no standardisation needed; D6 is a lookup model).
- Assert in code that no validation, test or attacker-pool user is used for fitting.

Contexts and back-off:
- Hold h_i is conditioned on the current key k_i.
  Back-off: key k_i -> global.
- Press-to-press p_i is conditioned on the key pair (k_i, k_{i+1}).
  Back-off: pair -> next key k_{i+1} -> global.
- A context is used only if it has at least n_min = 50 training observations; otherwise back
  off to the next level.

Three variants (all pre-registered; D6-gauss is primary):
1. D6-gauss (PRIMARY): a 1-D Gaussian on log time for each context (mean and SD, SD floor
   0.05). h and p are scored independently. Step NLL = NLL(h_i) + NLL(p_i) (only NLL(h_i) on
   the last key).
2. D6-hist: an empirical histogram on log time for each context: 60 equal-width bins over the
   global 0.1%–99.9% range of the training values, add-one (Laplace) smoothing, values outside
   the range clipped to the edge bins; density = bin probability / bin width.
3. D6-gauss-speed: the same as D6-gauss, but first compute standardised residuals
   z_i = (x_i − mu_ctx) / sd_ctx, subtract the session's own MEAN residual (separately for h
   and p), and score the NLL of the centred residuals under N(0, 1). This removes each
   session's overall speed, so it tests key conditioning plus a simple per-session speed
   adjustment, without a neural network.

Session score = mean step NLL over the same steps that D4 scores (higher = less human-like).
Thresholds tau5 and tau10 come from VALIDATION humans only.

Evaluate each variant ONCE on the test split against A1, A2, A3, A4, A5a and A5b. Report:
- APCER at tau10 and tau5, BPCER, and ROC-AUC, with bootstrap CIs exactly as in A1;
- the identical-key subset results (AUC, APCER and BPCER at tau10) exactly as in A3;
- side by side with D4-primary and D4-nokey in one table.
Label D6 in the report as "pre-registered, added after the main evaluation".

=====================================================================
Tests (pytest) — must pass before any full run
=====================================================================
- Bootstrap: with the identity resample (all users exactly once), every metric equals the
  saved point estimate.
- Bootstrap: resampling is by user (all of a user's sessions and matched bots move together).
- Thresholds: tau5 and tau10 use validation humans only; on validation humans the rejection
  rate is 5% / 10% within ±0.2 percentage points.
- D6-gauss: the NLL of one step equals the analytic scipy.stats.norm value within 1e-9.
- D6 back-off: a pair with fewer than 50 observations uses the next-key level; an unseen key
  uses the global level.
- D6-hist: bin probabilities sum to 1 for every context.
- D6 fitting data: asserts that the user IDs are a subset of the train users only.
- Point estimates of D1, D4 and D5 reproduce the frozen numbers in the manuscript tables
  (for example D4-primary APCER at tau10 on A2 = 0.6%, D1 on A1 = 5.8% or 5.9%). If D1 on A1
  gives two different values in different saved files, report BOTH, the file each comes from,
  and the reason they differ. Do not change either.

=====================================================================
Work order and report
=====================================================================
1. Write the spec and hash it.
2. Implement.
3. Run the tests.
4. --quick smoke run on a small subset (change nothing based on its numbers).
5. Full run.
   Estimate the runtime first; if it exceeds 3 hours, tell me before starting.

docs/PHASE4_RESULTS.md must contain:
(a) Table: human-only detectors x attackers, APCER at tau10 and tau5 with CIs, AUC with CI.
(b) Table: D3/D5 leave-one-out with CIs; the cross-attack averages with CIs.
(c) Table: identical-key subset, including D4 and D5 BPCER, and the D5 variant used.
(d) Table: D6 variants vs D4-primary vs D4-nokey.
(e) The across-seed SD for D4 and D5.
(f) The bootstrap rules actually used (especially for A5a and the paired sessions).
(g) Any deviation from this spec, with the reason.
(h) Runtime.
Do not interpret beyond one short paragraph. Do not edit the manuscript.
