_Written by hand after reading the full-run tables above (post-hoc interpretation)._

**How to read the swap test here.** The per-cell rule labels every cell *mixed*. That label is not informative, because
the swap as specified has a confound: positional alignment. Only 51% of aligned positions carry the same key for
A1/A2/A3/A5b (37% for A4, 9% for A5a), because the human corrected typos and the bot did not. S3 and S4 therefore pair
timings with keys they were not typed for. The A5a row is the control: both of its halves are real humans typing the
same sentence, yet D4 flags its S3 and S4 at AUC 0.967, *more* than any bot's S4 (0.763–0.861). So a large S4 effect
measures the damage the swap does to the key–timing link, not the bots' keys. Timing shares above 1 (S3 flagged more
than S2) show the same thing. Two cleaner checks answer the question: (i) the identical-key-sequence pairs, where
the keys are equal by construction, and (ii) the letters-only score.

**D4 (one-class, humans only): SIGNAL.**
- **Same keys, different results.** A1, A2, A3 and A5b emit *byte-identical* key sequences (JSD to humans 0.0315 for
  all four), yet D4 accepts them at very different rates: 64.0%, 0.6%, 3.0% and 7.5%. The key sequence cannot be
  what separates them.
- **Identical-key pairs.** In the 9,670 pairs (3,502 for A4) where the human typed exactly the bot's keys, D4 still
  separates A2/A3/A4/A5b from their humans. AUC is 0.993, 0.983, 0.986 and 0.975, and APCER@τ10 is 0.9–9.0%, with
  these humans rejected at only 6.8–7.6%.
- **Letters-only score.** With every step that touches Shift, punctuation, Backspace or any other non-letter key
  masked, and τ10 re-set on val humans, performance barely moves. AUC changes by at most 0.008, and APCER@τ10 rises
  by at most 2.3 pp (A5b: 7.5% → 9.8%).

D4's detections of A2–A5b are therefore driven by timing *given* the keys. The residual A1 weakness (about 60%
accepted) is also a timing property, and is unchanged under both checks.

**D5 (supervised BiGRU, E5): MIXED — the bot detections are timing-driven, but D5 also uses an error-free-key-sequence
cue that comes from the attacker generators.**
- **Bot detections: timing.** In the identical-key pairs D5 still catches its own attacker almost perfectly: AUC
  0.988–1.000, APCER@τ10 0.0–0.6%. So its detections of A2–A5b do not need any key difference.
- **The cue: error-free humans look like bots.** The same humans who typed error-free are rejected at 24–33% at the
  frozen τ10 (A4 model: 19%), against 10.3–10.7% for all test humans. D4 rejects these humans at only 6.8–7.6%, so
  their *timing* is typical. The excess rejection is therefore most plausibly a *key-sequence* cue D5 learned from
  its training bots: "no correction, no CapsLock, no arrow key → bot". This is not proven here; the evidence is
  indirect.
- **Swap results are consistent with this.** D5 flags S4 (bot keys + human timing) at AUC 0.71–0.94 (A5b model 0.935,
  key share 0.87). S3 is often flagged less than S4 for A2, A3 and A5b. Part of this is the alignment confound above.
- **Consequence.** The cue does not inflate the reported bot detection; bots are caught with the keys held fixed. It
  costs false rejections of error-free humans, and it would let a bot that inserts corrections look more human to D5.

**Exact cause (artefact in the generators — reported, not fixed).**
- **A1, A2, A3, A5b** type the target sentence verbatim: 0 Backspace/Delete (humans 5.28 per 100 keys, 62.7% of
  sessions), 0 CapsLock (humans 0.51/100, 8.1% of sessions), no arrow, Ctrl, Home or Insert keys, and no stray
  punctuation typos (`;`, `=`, `\`, `` ` ``, `[`, keypad codes 96–110). They press exactly one Shift per
  Shift-requiring character (1.000 vs humans 0.954). As a result they are 5 keys shorter on average (43.2 vs 48.4).
  The key sequences come from `a1_gaussian.char_to_key` + the per-character Shift rows in `a1_gaussian` / `common.py`
  / `a5_replay` (A5b).
- **A4** (`a4_imitator`) adds substitution typos with Backspace (4.65/100, 55.6% of sessions). But it still has no
  CapsLock or arrow keys, and its typo model produces 77 keycode bigrams that are ≥ 10× more frequent than in humans,
  e.g. `Z → Backspace`, `Q → Backspace`, `H → D`, `F → K` (each in ≤ 2.4% of sessions).
- **No keycode is over-represented in any bot set** (every ≥ 10× keycode is *more frequent in humans*). No bot emits
  UNK or the unmapped code 229 on the test sentences.

**Shift timing** (a timing property, not a key artefact):
- A1 always releases Shift after the capital is released (100% vs humans 31%) and holds it longer (median 384 vs
  248 ms).
- A5b releases Shift before the next key press in 38.8% of cases (humans 5.5%), with a short hold (median 159 ms).
- A2–A4 never release Shift before the next press (humans 5.5%).

The identical-key-pair results include these Shift timings, so they are part of the timing signal D4 and D5 use.
