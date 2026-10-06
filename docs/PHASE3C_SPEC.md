# Phase 3c spec — post-hoc artefact check (user's message, 2026-09-26, verbatim)

Phase 3c: a post-hoc artefact check. Run it now. Save this message as docs/PHASE3C_SPEC.md
and add a pointer in CLAUDE.md. Everything from Phases 1–3b is frozen (keep the hash guards).
This is diagnostic only: no retraining, no changes to any model, attacker, threshold or split.

Question: are D4's and D5's detections of A2–A5b driven by timing given the keys (a real signal),
or by differences in the KEY SEQUENCES that the bots generate (an implementation artefact)?

1. Keycode realism: for test humans vs each attacker (A1–A5b), report:
   - the JSD of the keycode frequency distribution;
   - the JSD of the keycode-bigram distribution;
   - the % of UNK tokens;
   - the Shift pattern (Shift presses per capital letter; the position of Shift relative to the
     capital; the Shift hold duration);
   - the punctuation keycodes used;
   - the Backspace/Delete frequency;
   - the session length in keys.
   List every keycode or bigram that is at least 10× more frequent in bots than in humans (or the
   reverse) and appears in at least 0.1% of sessions.

2. Swap test, on the TEST split, frozen weights, frozen τ10.
   Detectors: D4-primary (3 seeds) and D5 single-attacker models from E5 (3 seeds, each tested
   only on the attacker it was trained on).
   For each matched human–bot pair (same target sentence):
     S1 = human keys + human timing (control = test humans)
     S2 = bot keys   + bot timing   (= the original attack)
     S3 = human keys + bot timing   (the bot's hold/pp sequence applied to the human's keys)
     S4 = bot keys   + human timing (the human's hold/pp sequence applied to the bot's keys)
   Align by position and truncate to the shorter length. Report how many pairs had a length
   mismatch and by how much.
   Report APCER@τ10 and AUC vs S1 for S2, S3 and S4, per attacker and detector
   (mean ± sd over seeds).
   Reading:
   - S4 flagged about as strongly as S2 → the detector reacts to key sequences (artefact).
   - S3 flagged about as strongly as S2 and S4 near S1 → the detector reacts to timing (real
     signal).
   - Both partly flagged → mixed. Quantify the share of the effect explained by each.

3. Letters-only robustness (D4-primary): recompute the session score using only steps where both
   the current and the next key are letters a–z or space; mask all other steps. Recompute τ10 on
   val humans with this same masked score, then report APCER@τ10 and AUC for A1–A5b.

4. Write DIAGNOSTICS_PHASE3C.md in outputs/full/phase3c/, clearly labelled POST-HOC, with all
   the tables and a one-paragraph verdict per detector (D4, D5): signal, artefact, or mixed.
   If an artefact is found, name the exact cause (which keycodes, which pattern) and which
   attacker generator produces it. Do NOT fix any generator; just report it.

Work order: tests for the swap and masking logic → quick run → full run → report the tables to
me.
