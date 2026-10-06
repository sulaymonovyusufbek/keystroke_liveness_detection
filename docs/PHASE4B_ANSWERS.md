# Phase 4, Part B — answers from the frozen code, configs and docs

_Written 2026-10-06. Read-only: nothing was modified, rerun or retrained to answer these questions. Every answer cites
`file:line`. Two answers also use read-only facts from saved outputs or file timestamps, and say so where they do. The
manuscript itself is not in the project folder, so the final section lists what to check against it rather than quoting
it._

---

## B1. Attacker A1 (`src/kl/attacks/a1_gaussian.py`)

**a) Standard deviation or variance?** It is the **standard deviation**.
- The parameters are named `hold_std_ms: float = 23.88` and `flight_std_ms: float = 111.6` (`a1_gaussian.py:66-71`).
  The config has the same names (`config.yaml:57-63`: `hold_std_ms: 23.88`, `flight_std_ms: 111.6`).
- They are passed as the `scale` argument of `rng.normal(mean, std)` (`a1_gaussian.py:89`, `:93`). In numpy, `scale`
  is the standard deviation.
- The flight mean is 238.656 ms (`config.yaml:61`, `a1_gaussian.py:69`). "238.66" is that value rounded.

**b) How are invalid values handled?** They are **clipped (clamped), never resampled.**
- Hold: `int(round(max(hold_min_ms, rng.normal(116.24, 23.88))))` (`a1_gaussian.py:87-89`), with `hold_min_ms = 30`
  (`config.yaml:60`, marked `# ASSUMPTION`). A negative or very small draw therefore becomes exactly 30 ms.
- Flight: `int(round(max(flight_min_ms, rng.normal(238.656, 111.6))))` (`a1_gaussian.py:91-93`), with
  `flight_min_ms = 60` (`config.yaml:63`). Any draw below 60 ms becomes exactly 60 ms.
- The clamp is applied first, then the result is rounded to integer ms (`a1_gaussian.py:9`).
- Both floors are documented as assumptions, because the original generator was lost (`a1_gaussian.py:7-9`).
- Consequence, computed from the stated distributions with `scipy.stats.norm.cdf` (not read from code):
  - P(flight draw < 60) = **5.47%**, so about 1 in 18 A1 flights is exactly 60 ms.
  - P(hold draw < 30) = 0.015%.
  - The saved A1 test bots have a minimum hold of exactly 30 ms (`outputs/full/phase3b/interim/bot_A1_test.npz`).

**c) Is rollover possible in A1?**
- **Between two characters: no.** The next press is `PRESS_next = RELEASE_prev + flight` with flight ≥ 60 ms
  (`a1_gaussian.py:99-101`, `:111-114`). The next key is always pressed at least 60 ms after the previous one was
  released.
- **But rollover as the features define it does occur, from Shift nesting.**
  - A shifted character is typed as: Shift down, then the key down after a flight draw, the key up, and Shift up
    `hold_min_ms` = 30 ms after the key is released (`a1_gaussian.py:103-110`; docstring `:10-13`).
  - The features sort rows by press and define `flight_i = PRESS_{i+1} − RELEASE_i`, with `< 0` meaning rollover
    (`src/kl/features.py:4-7`, `:129`, `:143`).
  - The Shift row is followed by the shifted key, which is pressed before Shift is released. So that flight is
    negative and counts in `rollover_rate`.
  - Read-only check on the saved bots (`outputs/full/phase2/bots_A1_test.parquet` together with
    `phase3b/interim/bot_A1_test.npz`): in **every** one of the 43,248 A1 test bots, the number of negative flights
    equals the number of Shift presses exactly. 43,189 bots have at least one.
  - `shift_overlap_count` stays 0, because exactly one key is pressed inside each Shift hold, and the feature counts
    "(keys inside) − 1" (`features.py:64-65`, `:152-156`).

---

## B2. Attacker A5b (transplant) (`src/kl/attacks/a5_replay.py`)

**Source session.** It is chosen **uniformly at random from the attacker-pool FIT sessions, for every bot independently.**
- It is **not** from the same user, **not** matched on length and **not** matched on sentence.
  - `A5bTransplant(bank, fit_idx)` (`src/kl/phase2/generate.py:41`).
  - The constructor keeps the FIT sessions with ≥ 2 rows (`a5_replay.py:43-45`).
  - The source is `src = source_idx[rng.integers(len(source_idx))]` (`a5_replay.py:49`).
- The FIT sessions are those of the pool users who are not in the 20% held-out group (`scripts/10_fit_attackers.py:57-62`):
  23,066 sessions (`outputs/full/phase2/pool_stats_summary.json`, `n_fit_sessions`).

**Which timings are transplanted.**
- `hold_flight(i)` returns the holds and flights of **all** rows of the source session, in press order
  (`src/kl/attacks/pool.py:114-118`). This includes the source's Shift, Backspace and other non-character rows: the
  source keys are ignored, only their timing sequence is used.
- The target keystroke list is the target sentence's characters, with a Shift row inserted before each shifted
  character (`a5_replay.py:63-68`).

**Target of a different length** (`a5_replay.py:47-59`, docstring `:6-9`):
- **Target shorter than the source:** the source sequence is **truncated**. The bot takes the first `n` holds and the
  first `n − 1` flights (`:51`).
- **Target longer than the source:** the source is used completely and then **continued with contiguous chunks of
  other randomly drawn FIT sessions** (`:52-58`).
  - Each new session is again drawn uniformly at random (`:53`), with no exclusion of sessions already used.
  - Each chunk starts at a random position `start ∈ {1, …, len − 1}` (`rng.integers(1, len(oh))`, `:55`).
  - The chunk is joined by that session's own flight just before `start` (`flight[start − 1]`, `:58`).
  - Chunks are added until `n` holds exist.
- It is **not** cycling or repeating the same source, and **not** restricted to sessions of equal length.

**Assembly.** Keystroke `k` gets hold `k`, and press `k+1` = release `k` + flight `k` (`a5_replay.py:70-76`). The frame
is then sorted by press time (`src/kl/attacks/common.py:38-41`). Negative source flights (rollover) can therefore
change the key order. Phase 4 counted 875 A5b test bots whose key sequence differs from the canonical one for this
reason (`docs/PHASE4_RESULTS.md`, section c).

For comparison, A5a replays one uniformly random FIT session verbatim and ignores the target sentence
(`a5_replay.py:3-5`, `:33-35`).

---

## B3. Attackers A2–A4 (one line each, for Table II)

All statistics are fitted on the 23,066 attacker-pool FIT sessions (`pool.py:333-494`). Threshold parameters:
`config_phase2.yaml:7-9`. Fitted values: `outputs/full/phase2/pool_stats_summary.json` and `outputs/full/phase2/ATTACKS.md`.

- **A2 (back-off):**
  - Keys are lower-cased characters (`pool.py:33-36`, `:195-213`, `:364-372`; `a2_digraph.py:27-39`).
  - A press-to-press interval is resampled from the empirical distribution of its digraph (previous char, current
    char) if that digraph has ≥ 30 samples (`min_samples: 30`, `config_phase2.yaml:7`). Otherwise from the second
    character's distribution, again if it has ≥ 30 samples. Otherwise from the global distribution.
  - A hold is resampled per key (≥ 30 samples), otherwise from the global distribution.
  - Fitted: 553 digraph distributions, 51 second-character distributions, 51 key-hold distributions.
  - All draws are independent.
- **A3 (speed factor and autoregression):**
  - **Speed:** one speed offset `s` per session, drawn uniformly from the pool's empirical per-session offsets
    (`pool.py:223-225`, `:383-388`, `:478`). An offset is the mean log-pp residual of a pool session with ≥ 10
    non-pause char→char pairs (`config_phase2.yaml:8`). There are 20,274 such offsets: mean 0.0125, sd 0.322,
    5–95% range −0.50 to 0.55 in log-ms, i.e. a press-to-press factor of exp(s) ≈ 0.61–1.73 (`ATTACKS.md:63`).
  - **Autoregression:** the AR(1) residual is `r_i = φ·r_{i−1} + ε_i`, with ε ~ N(0, σ²), and
    `log pp_i = μ_digraph + s + r_i` (`a3_temporal.py:39-43`).
    - **φ = −0.00049** and σ = 0.406, from pooled least squares over 709,899 adjacent residual pairs
      (`pool.py:389-395`; `ATTACKS.md:57`).
    - φ ≈ 0, so the fitted AR structure is effectively absent.
  - Holds are scaled by `exp(0.136·(s − 0.0125))` (`pool.py:227-233`, `:396-402`).
- **A4 (rates of pauses, typos and corrections):** each bot draws its own rates (`a4_imitator.py:3-8`, `:98-101`;
  `pool.py:240-251`, `:404-476`).
  - **One joint draw from one uniformly random pool session** (20,196 eligible sessions) gives four rates:
    - `c`: Backspace presses per character;
    - `u`: the session's lev_norm;
    - `p_boundary` and `p_within`: the share of char→char transitions that are pauses (> 500 ms), after a space and
      inside a word.
  - **Shift overlap:** `p_so` is drawn from a **separate** random pool session that has Shift holds (18,198 sessions).
  - **Corrected typos:** `stochastic_round(c·L / 1.641)` events (`a4_imitator.py:53`; 1.641 = mean Backspaces per
    event).
  - **Uncorrected typos:** `stochastic_round(u·L)` (`a4_imitator.py:54`).
  - **Correction delay:** each corrected typo waits `d` further characters, with d ∈ {0, 1, 2, 3} and
    P(d) = (0.572, 0.266, 0.109, 0.052). The distribution comes from pooled Backspace-run lengths ≤ 4
    (`pool.py:424-434`; longer runs, 5.0%, are excluded).
  - **Pauses:** each char→char transition is a pause with probability `p_boundary` (after a space) or `p_within`,
    with the pause flight resampled from the pool's pause flights (`a4_imitator.py:112-114`).

---

## B4. D4 input and masking (`src/kl/phase3b/nets.py`; pre-registered in `docs/PHASE3B_PREREGISTRATION.md:96-111`)

**a) First key.** The inputs are **zeros, not a learned start value.**
- The previous-timing input is `prev = [cat(0, h[:, :-1]), cat(0, p[:, :-1])]` (`nets.py:111-112`). At step 0 the
  inputs h₋₁ and p₋₁ are therefore 0.
- The pre-registration says the same: `h_{−1} = p_{−1} = 0` (`PHASE3B_PREREGISTRATION.md:98`).
- h and p are standardised with the train-human mean and sd (`src/kl/phase3b/sequences.py:248`, `:250`), so 0 means
  "the training mean of log1p(time)".
- **The first step is scored.** `valid = arange(L) < lens` includes position 0 (`nets.py:110`). The step's NLL is
  kept (`:129`) and enters the loss (`:131-135`) and the session score (`:138-141`). The first step predicts
  (h₀, p₀) from key₀, the next key and the zero inputs.

**b) Last key.**
- The "next key" input is the **zero vector**. `e_next` is the key embeddings shifted by one (`nets.py:116`) and
  multiplied by 0 for every position ≥ `len − 1` (`:118`). This equals the PAD embedding, since
  `padding_idx = 0` (`:101`; `PHASE3B_PREREGISTRATION.md:100`: "`key_L` = PAD = zero vector").
- **Only the hold is scored on the last key. Confirmed.** `pmask = ar < (lens − 1)` is False on the last kept key
  (`sequences.py:246`). The p values there are stored as 0 (`:250`). `mixture_nll` adds the p term only where
  `pmask` is True (`nets.py:82`), so the last step uses the exact marginal of h (`PHASE3B_PREREGISTRATION.md:104-105`).
- The same applies to the last kept key of a truncated session (`PHASE3B_PREREGISTRATION.md:74-76`; truncation
  `sequences.py:242`).

**c) Padding.** It is masked **both ways**.
- **Packed sequences:** the GRU runs on `pack_padded_sequence(..., enforce_sorted=False)` and outputs are unpacked
  to the full length (`nets.py:87-90`). Padding never enters the recurrence.
- **Loss mask:** the step NLL is set to 0 where `valid` is False (`nets.py:129`).
  - Training loss = masked sum / number of valid steps (`:131-135`).
  - Session score = masked sum / valid steps per session (`:141`).
- Batches are also trimmed to their longest sequence (`sequences.py:283-287`).
- Padded positions carry key 0 (PAD) and h = p = 0 (`sequences.py:247-250`).
- (D5 for comparison: packed BiGRU, then a masked mean over the real steps, `nets.py:153-159`. `pad_packed_sequence`
  returns zeros beyond each length.)

---

## B5. D1 reference points (`src/kl/selection.py`)

- **Data:** the reference set is built from the **20,000 capped training sessions after the transform.**
  - `run_grid` takes `train` = `features_train.parquet` (`scripts/05_select_hyperparams.py:45`). This has 20,000 rows:
    the seeded cap of the train split (`config.yaml:29`, `max_train_sessions: 20000`; cap in `src/kl/split.py:29-36`,
    rng `default_rng([42, 1])`). `selection.json`: `n_train = 20000`.
  - It fits the transform on that matrix (signed log1p on non-bounded features, then RobustScaler; `selection.py:180`,
    `src/kl/model.py:42-67`) and transforms it (`selection.py:181`).
  - It then draws the reference set from the **transformed** matrix: `x_ref = product_of_marginals(x_tr, n_reference,
    seed)` (`selection.py:182`).
  - Each of the 23 coordinates of each of the 100,000 points (`config.yaml:44`) is resampled **independently, with
    replacement**, from that feature's transformed training column: `idx = rng.integers(0, n_train, size=(n, d))`,
    `x_train_scaled[idx, arange(d)]` (`selection.py:114-118`).
  - So the marginals are the empirical marginals of the 20,000 transformed (scaled) training sessions. Nothing is
    estimated parametrically.
- **Seed:** `np.random.default_rng([seed, 5])` with the global `seed: 42` (`selection.py:117`; `config.yaml:4`), i.e.
  `default_rng([42, 5])`. The uniform-box diagnostic uses `default_rng([42, 2])` (`selection.py:111`).

---

## B6. Pre-registration timeline

Sources: hash records (`docs/phase3b_prereg_hash.txt`, `docs/phase4_prereg_hash.txt`, `docs/phase*_frozen_hashes.json`,
`docs/MANIFEST_SHA256.txt` header line 1). Where no hash record exists, file modification times (`stat`) are given and
labelled as such. Modification times record the last write, not the first: a file that was later edited, rerun or
copied shows the later time. All times are CEST.

| phase | document | SHA-256 | hash first recorded | spec file last modified | first surviving result output of the phase |
|---|---|---|---|---|---|
| 1 | `PHASE1_SPEC.md` (incl. Revisions 1, 2) | `f5020b…481bdf` | **no record before results**; first in `MANIFEST_SHA256.txt`, 2026-09-26 13:58:11 | 2026-09-24 11:07:08 | quick: `results/grid_results_original_rule.csv` 2026-09-24 **10:41:41**; full: `logs/01_sample_and_parse.log` 11:11:43 |
| 2 | `PHASE2_SPEC.md` | `44102e…b17315` | no record before results; first in MANIFEST, 2026-09-26 13:58:11 | 2026-09-24 11:35:15 | quick: `phase2/pool.joblib` 11:45:10; full 11:48:25 |
| 3a | `PHASE3A_SPEC.md` | `f90f5f…30dab1` | no record before results; first in MANIFEST, 2026-09-26 13:58:11 | 2026-09-24 12:56:51 | quick: `phase3a/centrality_summary.csv` 13:01:17; full 13:05:07 |
| 3b | `PHASE3B_PREREGISTRATION.md` (+ `config_phase3b.yaml` `3229e2…1172f9`) | `d3e8ec…c968ca` | **2026-09-25 13:55:34** (`phase3b_prereg_hash.txt`) | 2026-09-25 13:55:24 | quick: `phase3b/interim/encoder.json` 2026-09-25 14:01:36; first test-split use (ledger) quick 14:11:16, full 16:32:58 |
| 3b | `PHASE3B_SPEC.md` (the request; the pre-registration is the binding version) | `b95b28…e41e4a` | no separate record; first in MANIFEST, 2026-09-26 13:58:11 | 2026-09-25 13:53:12 | as above |
| 3c | `PHASE3C_SPEC.md` | `eb6fe3…0de99b` | **after results**: 2026-09-26 13:54:47 (`phase3c_frozen_hashes.json`) | 2026-09-26 13:05:21 | quick: `phase3c/realism_summary.csv` 2026-09-26 13:11:48; full 13:14:09 |
| 4 | `PHASE4_SPEC.md` | `bebccb…ef190d` | **2026-10-06 14:02:58** (`phase4_prereg_hash.txt`) | 2026-10-06 14:02:52 | first code `src/kl/phase4_guard.py` 14:07:18; first output (quick) 14:13:45 |

Freeze records of sources and outputs (written after each phase, not pre-registrations):
- `phase1_frozen_hashes.json`: "written at the start of Phase 2 (2026-09-24)" (its `_note`). No time is recorded.
- `phase2_frozen_hashes.json`: "written at the start of Phase 3a (2026-09-24)". No time.
- `phase3a_frozen_hashes.json`: `recorded_at 2026-09-25`.
- `phase3b_frozen_hashes.json`: 2026-09-26 13:10:10.
- `phase3c_frozen_hashes.json`: 2026-09-26 13:54:47.

What the table shows:
- **Hash-verified before any result:** Phase 3b (pre-registration recorded 6 min before the first output and 2 h 37 min
  before the first full test-split use) and Phase 4 (spec recorded before any Phase 4 code or output).
- **Phases 2 and 3a:** the spec file's last modification precedes the phase's first surviving output (by 10 and
  4 min). There is **no hash record from that time**, so this rests on file timestamps only.
- **Phase 1:** a quick-mode result (10:41:41) is **older** than the spec's last modification (11:07:08). This is
  expected: Revisions 1 and 2 were appended to the spec after quick-mode results (see below). The time at which the
  original (pre-revision) Phase 1 spec was fixed **cannot be determined** from the files.
- **Phase 3c:** spec saved 6 min before its first output, but its hash was recorded only afterwards (the phase is
  post hoc by design).

**When the D1 selection rule was fixed, relative to the first A1 results:**
- **Original rule (uniform box):** run in quick mode first. That run produced test **and A1 results**
  (`outputs/full/METHODS.md:15`, "Disclosure: the initial quick-mode test and A1 results had already been produced
  with the original rule before the revision was decided"; archived numbers in
  `docs/revision1_original_rule_diagnostics.json`; surviving file `outputs/quick/results/grid_results_original_rule.csv`,
  10:41:41).
- **Revision 1 (product of marginals), 2026-09-24** (`PHASE1_SPEC.md:151-200`): decided **after** those first quick A1
  results. Its stated trigger is the degeneracy of the uniform rule, not attack performance (`METHODS.md:15`).
  Revision 1 also added the post-hoc A1 sensitivity table (`PHASE1_SPEC.md:184-188`).
- **Revision 2 (ν-property admissibility filter), FINAL, 2026-09-24** (`PHASE1_SPEC.md:203-232`): decided **after**
  the Revision 1 quick run **and its post-hoc A1 results had been seen**. `METHODS.md:17`: "Disclosure: post-hoc A1
  results had already been seen when this revision was made (in that run the Revision 1 configuration ranked 20 of 35
  by A1 ROC-AUC …), so Revision 2 is not independent of them, although its trigger and its formula are attack-free".
  The archived Revision 1 quick files (`grid_results_revision1.csv`, `grid_attack_sensitivity_revision1.csv`) carry
  11:07:08, the same second as the spec's last modification. That is consistent with archiving them when Revision 2
  was written.
- **Full mode** started after Revision 2 (`outputs/full/logs/01_sample_and_parse.log` 11:11:43). The full test split
  and full A1 were evaluated only under the final rule (`06_final_test.log` 11:14:45, `07_a1_sanity_check.log`
  11:16:53).
- **Exact clock times** of the two revisions are not recorded. Only the date 2026-09-24 is in the spec.

**Pre-registered vs post hoc:**
- **Pre-registered by hash before results:**
  - Phase 3b, everything in `PHASE3B_PREREGISTRATION.md`: D4 and its variants D4-20k / D4-nokey / D4-typicality, D5,
    and E4–E6.
  - Phase 4: A1–A3 analyses and D6.
- **Specified before results but not hash-recorded at the time:** Phase 1 (original spec; then revised twice after
  quick results, see above), Phase 2 (attackers A1–A5b, D2, D3, E1–E3), Phase 3a (diagnostics).
- **Post hoc:**
  - Phase 1 `grid_attack_sensitivity.csv` ("post-hoc; not used for selection", `PHASE1_SPEC.md:184-188`).
  - Phase 1 Revision 2, which was decided with post-hoc A1 results known.
  - Phase 3a classical-covariance centrality, labelled a supplement (`src/kl/phase3a/report.py:59`;
    `scripts/20_run_diagnostics.py:69-73`).
  - **All of Phase 3c** (`PHASE3C_SPEC.md:1-3`, "post-hoc artefact check"). Within it, the **identical-key control was
    an addition beyond the 3c spec** (`scripts/40_run_phase3c.py:49-50`, `:98`, "ADDITION (not in the spec)").
  - The D5 off-diagonal average **excluding A5a (37.7%)**, first computed for the handover summary
    (`PROJECT_STATE.md`, Phase 3b E5 note; `RESULTS_PHASE3B.md` reports the 25-cell 48.1%).
  - Phase 4 A1–A3: re-analyses of results that were already known, specified before the Phase 4 computation.
  - Phase 4 D6: pre-registered before its own evaluation, but **designed after the D4 / D4-nokey / Phase 3c results
    were known**. It is labelled "pre-registered, added after the main evaluation".

---

## B7. ICC pseudo-users (Phase 3a; `src/kl/phase3a/analyses.py`, `config_phase3a.yaml:20-24`)

**How the pseudo-users are grouped:**
- Each pseudo-user is **15 distinct bot sessions drawn at random** (without replacement within the unit) from all
  test bots of that attacker (`analyses.py:181-186`; `pseudo_user_sessions: 15`, `config_phase3a.yaml:23`).
- Each pseudo-user is drawn **independently**, so a bot session can appear in more than one pseudo-user.
- There are as many pseudo-users as real test users with ≥ 10 sessions: 2,969 (`analyses.py:200-206`;
  `config_phase3a.yaml:22`; `DIAGNOSTICS.md:97`).
- RNG: `default_rng([42, 42, attacker_code])` (`analyses.py:206`). Attackers analysed: A2 and A3
  (`config_phase3a.yaml:21`).
- The **bots' matched `user_id` is not used for grouping.** Bots are not grouped by the human they were generated for.
- The control "shuffled humans" is built the same way from random human test sessions, with `default_rng([42, 41])`
  (`analyses.py:203-204`).

**Do attackers keep persistent per-person parameters?** **No. Every synthetic session draws its parameters
independently.**
- The attacker interface is `generate(sentence, rng)`: it receives no user or person identity (`src/kl/attacks/base.py:28`).
- Each bot gets its own RNG, `default_rng([seed, attacker_code, split_code, i])` (`src/kl/phase2/generate.py:51`). The
  Phase 3b keystroke regeneration uses the identical generator (`src/kl/phase3b/regen.py:30`).
- Per-session draws:
  - A3 draws its speed offset `s` per session (`a3_temporal.py:35`).
  - A4 draws `s`, the rates `(c, u, p_boundary, p_within)` and `p_so` per session (`a4_imitator.py:99-101`).
  - A5a / A5b draw their source session(s) per bot (`a5_replay.py:35`, `:49`).
  - A1 and A2 have no session-level parameter at all.
- So the attackers' ICC ≈ 0 holds **by construction**. The analysis shows that real users have persistence the
  attackers lack, not that some per-user attacker failed to reproduce it.

---

## B8. Bootstrap rules (for the manuscript)

Proposed text:

> All confidence intervals are 95% percentile intervals over 1,000 cluster-bootstrap replicates in which the test
> users are resampled with replacement (seed 42). The draw is stored as one count per user, and every session carries
> the count of its user. Every synthetic session, including the A5a replays, was generated for one specific test human
> session and carries that session's identifier and user, so each drawn user brings all of their human test sessions
> together with every bot matched to them. Thresholds τ5 / τ10 are fixed at the values set on validation humans and are
> not re-estimated within replicates, except in a sensitivity analysis for D4, in which validation users are also
> resampled and τ is re-set inside every replicate. For the multi-seed sequence models, Phase 3b reported the CI of
> seed 42 only, whereas Phase 4 computes the metric for each of the three seeds within every replicate and takes the
> CI over the seed averages.

Sources:
- Replicates: `src/kl/phase2/evaluation.py:24-26` (`user_weights`, multinomial, `default_rng(seed)`), `:61`, `:69`.
- User index and matched bots: `scripts/12_run_experiments.py:52-57`. Bots keep the matched human's
  `session_id` / `user_id`: `src/kl/phase2/generate.py:54`, `scripts/11_generate_bots.py:50-55`.
- Phase 3b reuses the same replicates (`src/kl/phase3b/runner.py:139`). Its CI is for seed 42 only
  (`src/kl/phase3b/evaluation.py:3-5`, `:114-130`).
- Phase 4: `src/kl/phase4_bootstrap.py` and `docs/PHASE4_RESULTS.md` section f.
- `bootstrap_reps: 1000`, `ci_level: 0.95`, `seed: 42` (`config.yaml:4`, `:53-55`).

**Exception, Phase 1 test FNR:** the Phase 1 human-FNR CI used a different but equivalent user bootstrap
(`src/kl/evaluate.py:38-55`). It draws user indices with `default_rng(seed).integers`, with seed 42 for τ5 and 43 for
τ10 (`scripts/06_final_test.py:49`). Its replicates therefore differ from Phase 2's: [4.45, 5.12] in
`results/test_results.json` vs [4.47, 5.14] in `phase2/open_set_table.csv`, for the same 4.78%. Phase 1's A1 result
(5.86%) has no CI.

---

## Things that may contradict the manuscript (check each against the current text)

The manuscript is not in the project folder, so these are the points where the frozen code differs from what was
quoted in the questions or from common shorthand descriptions:

1. **A3's autoregression is effectively zero.** The fitted φ = −0.00049 (`ATTACKS.md:57`). A description of A3 as
   adding "temporal (AR) structure" overstates it. What A3 really adds over A2 is the per-session speed offset, the
   speed-scaled holds, and a parametric log-normal press-to-press time around the digraph means.
2. **A1 values are clamped, not resampled.** About 5.5% of A1 flights sit exactly at 60 ms (a point mass). A1 has no
   character-to-character rollover, but every capital produces one negative flight (Shift nesting), which
   `rollover_rate` counts.
3. **D1 on A1 has two values:** 5.86% (Phase 1, `results/a1_results.json`) and 5.77% (Phase 2,
   `phase2/open_set_table.csv`). Same model and τ10; different A1 bot draws. The bootstrap CI exists for 5.77% only.
4. **Phase 3b CIs are seed-42 CIs.** In `RESULTS_PHASE3B.md` / `e4–e6.csv`, the brackets are the bootstrap CI of seed 42,
   not of the seed mean. If the manuscript prints them next to seed means, replace them with the Phase 4 seed-averaged
   CIs.
5. **Table VIII's D5 is the E5 model trained on the same attacker it is tested on (`D5<-A`), seeds 42–44.** It is not
   the leave-one-out model.
6. **A5b transplants the source's full timing sequence**, including Backspace / Shift rows, truncated or continued
   with random chunks of other sessions. It is not a character-matched transplant, and 875 A5b bots end up with a
   reordered key sequence.
7. **Pre-registration:**
   - Only Phase 3b and Phase 4 were hash-recorded before results.
   - Phases 1, 2 and 3a specs have no contemporaneous hash.
   - The Phase 1 selection rule was revised twice after quick-mode results; Revision 2 was decided with post-hoc A1
     results known (disclosed in `METHODS.md:17`).
   - Phase 3c, including the identical-key control, is post hoc.
   - D6 was designed after D4's results.

   Any wording such as "all analyses were pre-registered" would contradict this.
8. **Phase 3a pseudo-users are random sets of 15 bot sessions**, and no attacker has per-person parameters. The
   attackers' ICC ≈ 0 is by construction.
9. **The D1 reference marginals** are the empirical marginals of the 20,000 **transformed** (signed-log + RobustScaler)
   training sessions, seed `[42, 5]`. They are not the raw features and not all 173,392 training sessions.
10. **Internal inconsistency in a frozen output:** `outputs/full/METHODS.md:3` says "Protocol: docs/PHASE1_SPEC.md
    incl. Revision 1", although Revision 2 (FINAL) was also applied (`METHODS.md:17-19`). Not changed, because the
    file is frozen.
