#!/usr/bin/env python
"""Phase 4 report: docs/PHASE4_RESULTS.md (full) or outputs/phase4/quick/PHASE4_RESULTS_quick.md, built only from
outputs/phase4/<mode>/ files (no number is typed by hand). The short interpretation paragraph is read from
outputs/phase4/<mode>/interpretation.md if that file exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kl.config import ROOT, load_settings  # noqa: E402
from kl.phase4_guard import assert_frozen_p4, read_record  # noqa: E402

ATK = ["A1", "A2", "A3", "A4", "A5a", "A5b"]
TRAIN_ATK = ["A1", "A2", "A3", "A4", "A5b"]
HUMAN_ONLY = ["D1", "D2", "D4-primary", "D4-20k", "D4-nokey"]


def pct(v: float) -> str:
    return "–" if pd.isna(v) else f"{100 * v:.1f}%"


def pci(r: pd.Series, m: str) -> str:
    """``value [lo, hi]`` (percent, or AUC with 3 decimals)."""
    v, lo, hi = r[m], r[f"{m}_ci_lo"], r[f"{m}_ci_hi"]
    if m == "auc":
        return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
    return f"{100 * v:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def sd(v: float, auc: bool = False) -> str:
    if pd.isna(v):
        return "–"
    return f"{v:.4f}" if auc else f"{100 * v:.2f} pp"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pair_diagnostics(st) -> dict[str, int]:
    """Why the identical-key pair counts differ: canonical (= A1) key sequences among humans, A4 and A5b bots."""
    from kl.phase3b.sequences import SeqSet
    from kl.phase3c.swap import identical_keys

    d = st.out_dir / "phase3b" / "interim"
    H = SeqSet.load(d / "human_test.npz")
    B = {a: SeqSet.load(d / f"bot_{a}_test.npz") for a in ("A1", "A4", "A5b")}
    i = np.arange(H.n_sessions)
    h_canon = identical_keys(H, i, B["A1"], i)
    a4 = identical_keys(B["A1"], i, B["A4"], i)
    a5 = identical_keys(B["A1"], i, B["A5b"], i)
    a4_pairs = identical_keys(H, i, B["A4"], i)
    return {"n_humans": int(H.n_sessions), "h_canon": int(h_canon.sum()), "a4_canon": int(a4.sum()), "a4_canon_and_h": int((a4 & h_canon).sum()),
            "a4_noncanon_pairs": int((a4_pairs & ~a4).sum()), "a5b_noncanon": int((~a5).sum()), "a5b_lost": int((h_canon & ~a5).sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    st = load_settings(None, args.quick)
    assert_frozen_p4(st)
    d = ROOT / "outputs" / "phase4" / ("quick" if args.quick else "full")
    res = pd.read_csv(d / "results_all.csv")
    avg = pd.read_csv(d / "results_cross_attack_averages.csv")
    meta = json.loads((d / "run_meta.json").read_text())
    d6m = json.loads((d / "d6_meta.json").read_text())
    rp = pd.read_csv(d / "reproduction_check.csv")
    vt = pd.read_csv(d / "thresholds_val_rejection.csv")
    rec = read_record()
    report_seconds = meta["timings_seconds"]

    def get(analysis: str, det: str, tested: str, trained: str | None = None, subset: str = "all") -> pd.Series:
        q = (res.analysis == analysis) & (res.detector == det) & (res.tested_on == tested) & (res.subset == subset)
        if trained is not None:
            q &= res.trained_on == trained
        sel = res[q]
        if len(sel) != 1:
            raise KeyError((analysis, det, tested, trained, subset, len(sel)))
        return sel.iloc[0]

    L: list[str] = []
    mode = meta["mode"]
    L.append(f"# Phase 4 results — revision analyses, Part A ({mode} run)\n")
    L.append(f"_Spec: `docs/PHASE4_SPEC.md` (SHA-256 `{rec['spec_sha256']}`, recorded {rec['recorded_at']} in "
             f"`docs/phase4_prereg_hash.txt`). Every number below is read from `outputs/phase4/{mode}/` by "
             "`scripts/phase4_report.py`. Phases 1–3c are frozen; nothing earlier was retrained, re-selected or re-thresholded "
             "(the D3 exception is in section g)._\n")
    L.append("Conventions: APCER = attacks accepted (score ≥ τ); BPCER = humans rejected (score < τ); τ5 / τ10 = thresholds "
             "set on validation humans so that 5% / 10% of them are rejected. Brackets = 95% percentile CI over "
             f"{meta['bootstrap_reps']:,} test-user cluster-bootstrap replicates (seed {meta['seed']}). D4 / D5: point = mean over "
             f"seeds {', '.join(map(str, meta['seq_seeds']))}; each replicate is averaged over the seeds before the CI is taken. "
             f"Test split: {meta['n_test_human_sessions']:,} human sessions from {meta['n_test_users']:,} users; one matched bot "
             "per human session per attacker.\n")

    # ---- (a) ----------------------------------------------------------------------------------------------------------
    L.append("## (a) Human-only detectors × attackers (A1, A2)\n")
    rows = []
    for det in HUMAN_ONLY:
        for a in ATK:
            r = get("human_only", det, a)
            rows.append([det, a, pci(r, "apcer_fpr_tau10"), pci(r, "apcer_fpr_tau5"), pci(r, "auc")])
    L.append(table(["detector", "attacker", "APCER @τ10", "APCER @τ5", "ROC-AUC"], rows))
    L.append("\nBPCER (humans rejected) of each detector. It is the same for every attacker, because the human sessions and "
             "replicates are the same:\n")
    rows = []
    for det in HUMAN_ONLY + ["D6-gauss", "D6-hist", "D6-gauss-speed"]:
        r = get("human_only", det, "A1")
        rows.append([det, pci(r, "bpcer_fnr_tau10"), pci(r, "bpcer_fnr_tau5")])
    L.append(table(["detector", "BPCER @τ10", "BPCER @τ5"], rows))
    L.append("\nD1 is the frozen OCSVM; its bot sessions were re-scored with the frozen model, which reproduces the Phase 2 "
             "point estimates and CIs exactly. D2 (Isolation Forest) was not saved and is deterministic, so its rows are "
             "copied from `phase2/open_set_table.csv`, whose CIs were computed on the same test-user replicates.\n")
    L.append("**Sensitivity check (D4-primary only): τ5 / τ10 re-set inside every replicate on resampled validation users** "
             "(test users and validation users resampled independently):\n")
    rows = []
    for a in ATK:
        p, s = get("human_only", "D4-primary", a), get("human_only_sensitivity", "D4-primary", a)
        rows.append([a, pci(p, "apcer_fpr_tau10"), pci(s, "apcer_fpr_tau10"), pci(p, "apcer_fpr_tau5"), pci(s, "apcer_fpr_tau5")])
    p, s = get("human_only", "D4-primary", "A1"), get("human_only_sensitivity", "D4-primary", "A1")
    rows.append(["BPCER (humans)", pci(p, "bpcer_fnr_tau10"), pci(s, "bpcer_fnr_tau10"), pci(p, "bpcer_fnr_tau5"), pci(s, "bpcer_fnr_tau5")])
    L.append(table(["attacker", "APCER @τ10, τ fixed", "APCER @τ10, τ re-set", "APCER @τ5, τ fixed", "APCER @τ5, τ re-set"], rows))
    L.append("\n**A2 (τ5 for every detector and seed).** τ5 was saved for D1, for every D4 variant and seed (`tau5_primary`), "
             "and for every D5 model and seed. Nothing had to be computed from scratch. As a check, τ5 / τ10 were recomputed "
             "from validation humans only. For D4 the saved validation scores were used. For D5 the validation logits had not "
             "been saved, so they were recomputed from the frozen weights, on validation humans only. Validation rejection at "
             "the saved thresholds:")
    vt_s = vt.copy()
    L.append(f"τ5 {100 * vt_s['val_reject_tau5'].min():.2f}–{100 * vt_s['val_reject_tau5'].max():.2f}%, "
             f"τ10 {100 * vt_s['val_reject_tau10'].min():.2f}–{100 * vt_s['val_reject_tau10'].max():.2f}% "
             f"over {len(vt_s)} detector/seed combinations (`thresholds_val_rejection.csv`).\n")

    # ---- (b) ----------------------------------------------------------------------------------------------------------
    L.append("## (b) D3 / D5 leave-one-attacker-out and cross-attack\n")
    rows = []
    for a in TRAIN_ATK:
        for det in ("D3", "D5"):
            r = get("loao", det, a, f"all-but-{a}")
            rows.append([a, det, pci(r, "apcer_fpr_tau10"), pci(r, "apcer_fpr_tau5"), pci(r, "bpcer_fnr_tau10"), pci(r, "bpcer_fnr_tau5"), pci(r, "auc")])
    L.append(table(["held-out attacker", "detector", "APCER @τ10", "APCER @τ5", "BPCER @τ10 (model)", "BPCER @τ5 (model)", "ROC-AUC"], rows))
    L.append("\nD3 is a single deterministic model per row. Its LOAO rows are copied from `phase2/loao_table.csv`, which "
             "used the same replicates. D5 rows are new seed-averaged bootstrap CIs.\n")
    for m, lab in (("apcer_fpr_tau10", "τ10"), ("apcer_fpr_tau5", "τ5")):
        L.append(f"**Cross-attack point estimates, APCER @{lab}** (rows = trained on, columns = tested on; cell = D5 mean over "
                 "seeds / D3):\n")
        rows = []
        for tr in TRAIN_ATK:
            rows.append([tr] + [f"{pct(get('cross_attack_cell', 'D5', a, tr)[m])} / {pct(get('cross_attack_cell', 'D3', a, tr)[m])}" for a in ATK])
        L.append(table(["trained on"] + ATK, rows) + "\n")
    L.append("**Cross-attack averages with CIs** (the average is formed inside every replicate, so the CI accounts for the "
             "correlation between cells):\n")
    rows = []
    for _, r in avg.iterrows():
        rows.append([r["detector"], r["average"], "τ10" if r["metric"].endswith("tau10") else "τ5", int(r["n_cells"]),
                     f"{100 * r['point']:.1f}% [{100 * r['ci_lo']:.1f}, {100 * r['ci_hi']:.1f}]", sd(r["seed_sd"])])
    L.append(table(["detector", "average", "τ", "cells", "APCER [95% CI]", "across-seed sd"], rows))
    L.append(f"\nD3 refit reproduced Phase 2 exactly: **{meta['d3_refit_reproduces_phase2']}** (section g).\n")

    # ---- (c) ----------------------------------------------------------------------------------------------------------
    L.append("## (c) Identical-key subset (A3; Table VIII of the manuscript)\n")
    L.append("These are the Phase 3c pairs. The pair lists themselves were not saved. They were regenerated by calling the "
             "frozen Phase 3c function `kl.phase3c.swap.identical_keys` on the frozen Phase 3b sequence files, the same "
             "call that `scripts/40_run_phase3c.py` makes, and the pair counts and every per-seed Phase 3c value match "
             "exactly. A pair is a test human session and the bot generated for it (same `session_id`, same target "
             "sentence) whose keycode sequences are identical: same length and same key at every position. Spec A3 asks "
             "for A2, A3, A4 and A5b; A1 is included for completeness.\n")
    rows = []
    for a in TRAIN_ATK:
        r4, r5, rk = get("identical_keys", "D4-primary", a, subset="identical keys"), get("identical_keys", "D5 (E5, own attacker)", a, subset="identical keys"), \
            get("identical_keys", "D4-nokey", a, subset="identical keys")
        rows.append([a, int(r4["n_humans"]), pci(r4, "bpcer_fnr_tau10"), pci(r4, "bpcer_fnr_tau5"), pci(r4, "apcer_fpr_tau10"), pci(r4, "auc"),
                     pci(r5, "bpcer_fnr_tau10"), pci(r5, "bpcer_fnr_tau5"), pci(r5, "apcer_fpr_tau10"), pci(r5, "auc")])
    L.append(table(["attacker", "pairs", "D4 BPCER @τ10", "D4 BPCER @τ5", "D4 APCER @τ10", "D4 AUC",
                    "D5 BPCER @τ10", "D5 BPCER @τ5", "D5 APCER @τ10", "D5 AUC"], rows))
    L.append("\nD4-nokey (timing-only control) on the same pairs:\n")
    rows = []
    for a in TRAIN_ATK:
        rk = get("identical_keys", "D4-nokey", a, subset="identical keys")
        rows.append([a, pci(rk, "bpcer_fnr_tau10"), pci(rk, "bpcer_fnr_tau5"), pci(rk, "apcer_fpr_tau10"), pci(rk, "auc")])
    L.append(table(["attacker", "BPCER @τ10", "BPCER @τ5", "APCER @τ10", "AUC"], rows))
    ip = meta["identical_key_pairs"]
    L.append(f"""
**D5 variant used in Phase 3c (and here).** `D5 (E5, own attacker)` is the **E5 single-attacker model trained on the
same attacker it is tested on**: `D5<-A` (e.g. `D5<-A2`, trained on train humans + A2 train bots, early stopping on
validation humans + A2 validation bots). It is **not** the leave-one-out (E6) model. All three seeds (42, 43, 44) are
used, each with its own saved τ5 / τ10; the table shows the mean over seeds. Because every attacker has its own D5
model, D5's human scores, and therefore its BPCER on the same human sessions, differ between attackers. D4-primary is
one model per seed, and its BPCER differs between attackers only because the pair subsets differ.

**Pair counts and why A4 has fewer.** Pairs: {', '.join(f'{a} {ip[a]:,}' for a in ip)}.
- A1, A2 and A3 always type the target sentence's canonical keycode sequence: no errors, and Shift before each
  capital. So they share one set of pairs: the human sessions in which the person typed exactly that sequence, without
  any correction.
- A4 inserts substitution typos with Backspace corrections. Only the A4 bots that happened to make no typo have the
  canonical sequence, so a pair needs both an error-free human and a typo-free A4 bot (or, rarely, the same typo
  corrected in the same way).
- A5b transplants a real pool session's timing onto the target keys. The keystrokes are then ordered by press time,
  as for every session. Where the transplanted timings overlap, two keys swap order (e.g. space pressed before the
  preceding letter), so some A5b bots leave the canonical sequence.
""")
    pc = pair_diagnostics(st)
    L.append(f"""Counts from the frozen test sequence files (descriptive only):
- {pc['n_humans']:,} test human sessions, of which {pc['h_canon']:,} are canonical (= the A1/A2/A3 pairs).
- A4: {pc['a4_canon']:,} of {pc['n_humans']:,} bots are typo-free (canonical). {pc['a4_canon_and_h']:,} of those coincide with a canonical human,
  and {pc['a4_noncanon_pairs']:,} pair(s) are humans who made exactly the same correction as their A4 bot.
- A5b: {pc['a5b_noncanon']:,} bots differ from the canonical sequence (keys reordered by overlapping transplanted timings);
  {pc['a5b_lost']:,} of those fall on canonical humans, hence {pc['h_canon']:,} − {pc['a5b_lost']:,} = {pc['h_canon'] - pc['a5b_lost']:,} pairs.
""")

    # ---- (d) ----------------------------------------------------------------------------------------------------------
    L.append("## (d) D6 (pre-registered, added after the main evaluation) vs D4-primary vs D4-nokey\n")
    L.append("D6 = key-pair timing lookup model, fitted on training humans only and thresholded on validation humans only. Each "
             "variant was evaluated once on the test split (ledger `outputs/phase4/full/test_access_phase4.json`). D6 has "
             "no seeds.\n")
    dets = ["D6-gauss", "D6-hist", "D6-gauss-speed", "D4-primary", "D4-nokey"]
    rows = []
    for a in ATK:
        for det in dets:
            r = get("human_only", det, a)
            rows.append([a, det, pci(r, "apcer_fpr_tau10"), pci(r, "apcer_fpr_tau5"), pci(r, "auc")])
    L.append(table(["attacker", "detector", "APCER @τ10", "APCER @τ5", "ROC-AUC"], rows))
    L.append("\nBPCER: see the second table in (a).\n")
    L.append("**Identical-key subset** (same pairs as in (c)):\n")
    rows = []
    for a in TRAIN_ATK:
        for det in dets:
            r = get("identical_keys", det, a, subset="identical keys")
            rows.append([a, det, int(r["n_humans"]), pci(r, "auc"), pci(r, "apcer_fpr_tau10"), pci(r, "bpcer_fnr_tau10")])
    L.append(table(["attacker", "detector", "pairs", "AUC", "APCER @τ10", "BPCER @τ10"], rows))
    cu = d6m["context_usage_val"]
    L.append(f"""
D6 fitting set: {d6m['n_train_sessions']:,} training-human sessions ({d6m['n_train_users']:,} users), {d6m['n_train_steps']:,}
hold steps, {d6m['n_train_press_steps']:,} press-to-press steps; vocabulary {d6m['vocab_size']} (incl. PAD, UNK), max_len
{d6m['max_len']}. Contexts with ≥ 50 observations: {d6m['n_hold_contexts_used']} keys (hold), {d6m['n_pair_contexts_used']:,} key
pairs and {d6m['n_next_key_contexts_used']} next keys (press-to-press). On validation humans, holds were scored at the key level for
{100 * cu['hold_key']:.2f}% of steps (global {100 * cu['hold_global']:.2f}%), and press-to-press times at the pair / next-key /
global level for {100 * cu['press_pair']:.2f}% / {100 * cu['press_next_key']:.2f}% / {100 * cu['press_global']:.2f}%. Histogram
range (log1p ms): hold [{d6m['edges_h'][0]:.3f}, {d6m['edges_h'][1]:.3f}], press-to-press [{d6m['edges_p'][0]:.3f}, {d6m['edges_p'][1]:.3f}].
Thresholds: {', '.join(f"{v}: τ5 {d6m['variants'][v]['tau5']:.4f}, τ10 {d6m['variants'][v]['tau10']:.4f}" for v in d6m['variants'])}
(on −score, i.e. higher = more human).
""")

    # ---- (e) ----------------------------------------------------------------------------------------------------------
    L.append("## (e) Across-seed standard deviation (D4, D5; point estimates, no bootstrap, ddof = 1, 3 seeds)\n")
    rows = []
    for det in ("D4-primary", "D4-20k", "D4-nokey"):
        for a in ATK:
            r = get("human_only", det, a)
            rows.append([det, "attack-free", a, sd(r["apcer_fpr_tau10_seed_sd"]), sd(r["apcer_fpr_tau5_seed_sd"]),
                         sd(r["bpcer_fnr_tau10_seed_sd"]), sd(r["bpcer_fnr_tau5_seed_sd"]), sd(r["auc_seed_sd"], True)])
    for a in TRAIN_ATK:
        r = get("loao", "D5", a, f"all-but-{a}")
        rows.append(["D5 (LOAO)", f"all-but-{a}", a, sd(r["apcer_fpr_tau10_seed_sd"]), sd(r["apcer_fpr_tau5_seed_sd"]),
                     sd(r["bpcer_fnr_tau10_seed_sd"]), sd(r["bpcer_fnr_tau5_seed_sd"]), sd(r["auc_seed_sd"], True)])
    for a in TRAIN_ATK:
        r = get("identical_keys", "D4-primary", a, subset="identical keys")
        r5 = get("identical_keys", "D5 (E5, own attacker)", a, subset="identical keys")
        rows.append(["D4-primary (identical keys)", "attack-free", a, sd(r["apcer_fpr_tau10_seed_sd"]), sd(r["apcer_fpr_tau5_seed_sd"]),
                     sd(r["bpcer_fnr_tau10_seed_sd"]), sd(r["bpcer_fnr_tau5_seed_sd"]), sd(r["auc_seed_sd"], True)])
        rows.append(["D5 E5 own attacker (identical keys)", a, a, sd(r5["apcer_fpr_tau10_seed_sd"]), sd(r5["apcer_fpr_tau5_seed_sd"]),
                     sd(r5["bpcer_fnr_tau10_seed_sd"]), sd(r5["bpcer_fnr_tau5_seed_sd"]), sd(r5["auc_seed_sd"], True)])
    L.append(table(["model", "trained on", "tested on", "APCER @τ10", "APCER @τ5", "BPCER @τ10", "BPCER @τ5", "AUC"], rows))
    L.append("\nD5 cross-attack (E5), across-seed sd of APCER @τ10 (rows = trained on):\n")
    rows = [[tr] + [sd(get("cross_attack_cell", "D5", a, tr)["apcer_fpr_tau10_seed_sd"]) for a in ATK] for tr in TRAIN_ATK]
    L.append(table(["trained on"] + ATK, rows))
    L.append("\nThe across-seed sd of the cross-attack averages is the last column of the averages table in (b).\n")

    # ---- (f) ----------------------------------------------------------------------------------------------------------
    L.append("## (f) Bootstrap rules actually used\n")
    L.append(f"""- **Unit:** the test user (cluster bootstrap). Each replicate draws the {meta['n_test_users']:,} test users with
  replacement. This is stored as a draw count per user and applied as a weight to every session of that user.
- **Replicates:** {meta['bootstrap_reps']:,}, seed {meta['seed']}. They are generated by `{meta['test_weights']}`, i.e. **the same
  replicates as the Phase 2 and Phase 3b CIs**. The D1 CIs recomputed here equal the saved Phase 2 CIs to ≤ 1e-15.
- **Matched bots:** for every attacker, including **A5a**, the test bot file holds exactly one bot per test human
  session. The bot carries that session's `session_id` and `user_id`. For A1–A4 and A5b it was generated for the
  session's target sentence. A5a (verbatim replay) ignores the sentence and replays a random attacker-pool recording,
  but it is still generated for, and recorded against, one test human session. **So A5a is resampled with its test
  session**, like every other attacker, and the "resample A5a independently" fallback was not needed. A drawn user
  therefore brings all their human test sessions and every bot matched to them, for every attacker.
  Rule per attacker: {'; '.join(f'{a}: {v}' for a, v in meta['match_rule'].items())}.
- **Thresholds:** fixed at the frozen validation values (primary analysis).
- **Seeds:** for D4 / D5, each replicate computes the metric for seeds 42, 43 and 44 and averages them; the CI is the
  2.5% / 97.5% percentile of these averages. The point estimate is the mean over seeds of the full-sample metric. The
  across-seed sd (section e) is separate.
- **Sensitivity check (D4-primary):** in each replicate the validation users ({meta['n_val_users']:,}; {meta['n_val_human_sessions']:,}
  sessions) are also resampled, independently of the test users ({meta['val_weights']}). τ5 / τ10 are then recomputed with
  the frozen rule (`np.quantile` of the validation-human scores, as if the drawn sessions had been repeated), for each
  seed in that replicate.
- **Averages over cells** (cross-attack): the mean over cells is taken inside each replicate, then the percentile CI.
- **Identical-key subset:** the same replicates; only the paired human and bot sessions enter. A human session and its
  paired bot belong to the same user and move together.
- **AUC:** humans positive, ties counted ½, computed with each session weighted by its user's draw count.
""")

    # ---- (g) ----------------------------------------------------------------------------------------------------------
    n_fail = int((~rp["ok"]).sum())
    d3row = rp[rp.check.str.startswith("D3")]
    L.append("## (g) Deviations from the spec, with reasons\n")
    L.append(f"""1. **Hash record file.** The spec says to append the spec's hash to "the same hash record file used in earlier
   phases". That file, `docs/phase3b_prereg_hash.txt`, is frozen and read-only, and editing it would break the Phase 3b
   freeze guard. The hash was instead written, with a timestamp, to a new file in the same format,
   `docs/phase4_prereg_hash.txt`. It is checked by `kl.phase4_guard` before and after every Phase 4 step.
2. **D3 refit (approved by the user before implementation).** Phase 2 saved neither D3's per-session scores nor the
   fitted forests, and the CI of D3's off-diagonal cross-attack average needs per-session scores. The five
   single-attacker D3 forests (E2) were therefore refitted with the frozen code (`kl.phase2.detectors.fit_d3`), frozen
   data, pre-declared hyper-parameters and seed 42. They were to be used only if they reproduced Phase 2 exactly. Result:
   all 10 thresholds identical, and all 30 cells × 5 metrics × (point, CI lo, CI hi) = {len(d3row) - 10} saved values reproduced, with
   max |diff| {d3row['abs_diff'].max():.1e}. Reproduced exactly: **{meta['d3_refit_reproduces_phase2']}**. The refitted forests were not saved and
   were used for nothing else. The D3 LOAO models were not refitted: their CIs were already saved.
3. **D2 and D3 LOAO CIs copied, not recomputed.** Their per-session scores were not saved. The saved Phase 2 CIs used
   the identical test-user replicates (shown by the exact D1 reproduction), and D2 / D3 are single deterministic
   models, so no seed averaging applies.
4. **D5 validation logits recomputed (check only).** They were not saved. They were recomputed from the frozen D5
   weights, on validation humans only, to confirm the saved τ5 / τ10; the saved thresholds are the ones used. They are
   saved in `d5_val_logits_recomputed.npz`.
5. **Identical-key pairs regenerated with the frozen Phase 3c function** (they were not saved as a list). The same
   frozen function was called on the same frozen inputs, so the pairs are the same. The counts and every per-seed Phase 3c
   value (BPCER, APCER, AUC at τ10) are reproduced exactly.
6. **Additions beyond the spec:**
   - A1 in the identical-key tables.
   - D4-nokey on the identical-key subset.
   - Cross-attack averages including A5a (25 cells) and the diagonal averages.
   - The AUC CI for the LOAO rows.
7. **Implementation choices the spec left open (fixed before any D6 test evaluation):**
   - D6 Gaussian SD = population SD (ddof = 0), then floored at 0.05.
   - UNK (vocabulary id 1, the keycodes D4 also maps to UNK) is a key context like any other.
   - The histogram range is computed separately for hold and press-to-press values.
   - D6-gauss-speed uses the D6-gauss contexts, subtracts the session mean of the residuals over that session's scored
     h steps and p steps, and scores the centred residual with the N(0, 1) density in z units.
   - Validation-user replicates for the sensitivity check come from an independent stream derived from seed 42.
   - Outputs go to `outputs/phase4/full/` and `outputs/phase4/quick/` (the spec says `outputs/phase4/`).

**D1 on A1: two saved values (both reported, neither changed).** `outputs/full/results/a1_results.json` (Phase 1,
step 07): APCER @τ10 = **5.86%** (2,536 / 43,248). `outputs/full/phase2/open_set_table.csv` (Phase 2, step 12):
**5.77%** (2,496 / 43,248), which rounds to 5.8%. Same frozen D1, same τ10 and same test humans, but **different A1 bot
sessions**. Phase 1 drew all A1 bots from one sequential random stream, `default_rng([42, 3])`
(`kl.evaluate.build_bot_features`). Phase 2 regenerated A1 with one independent stream per bot,
`default_rng([42, attacker, split, i])` (`kl.phase2.generate`), so its results do not depend on chunking or
parallelism. Same generator and parameters, different random draws. Only the Phase 2 bots were saved, so the CIs
here refer to 5.77%.

**Reproduction checks:** {len(rp)} comparisons with saved Phase 1–3c values (`reproduction_check.csv`); failed: **{n_fail}**.
""")

    # ---- (h) ----------------------------------------------------------------------------------------------------------
    L.append("## (h) Runtime\n")
    rows = [[k, f"{v:.0f} s"] for k, v in report_seconds.items()]
    L.append(table(["step", "wall time"], rows))
    L.append("\nCPU for everything except the D5 validation check (GPU, NVIDIA RTX 3050 Laptop). The quick smoke run took "
             "about 70 s.\n")

    # ---- interpretation ---------------------------------------------------------------------------------------------
    ip_path = d / "interpretation.md"
    L.append("## Short interpretation\n")
    L.append(ip_path.read_text(encoding="utf-8").strip() + "\n" if ip_path.exists() else "_(not written yet)_\n")

    target = ROOT / "docs" / "PHASE4_RESULTS.md" if not args.quick else d / "PHASE4_RESULTS_quick.md"
    target.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
