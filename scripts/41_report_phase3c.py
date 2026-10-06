#!/usr/bin/env python
"""Phase 3c report: <out>/phase3c/DIAGNOSTICS_PHASE3C.md (POST-HOC) from the CSVs of 40_run_phase3c.py.

The per-cell labels follow the rule stated in the report (fixed before the full run). The verdict paragraphs are
read from <out>/phase3c/verdict.md (written by hand after reading the full-run tables) and included verbatim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kl.config import load_settings  # noqa: E402
from kl.phase3c.guard import assert_frozen_p1234  # noqa: E402

SIGNAL, ARTEFACT = 0.75, 0.25      # label rule on the AUC-effect shares (see _rule text)


def pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{100 * x:.1f}%"


def ms(g: pd.Series, f=pct) -> str:
    m, s = g.mean(), g.std(ddof=1)
    if pd.isna(m):
        return "—"
    return f"{f(m)} ± {f(s) if f is not pct else f'{100 * s:.1f}'}" if len(g) > 1 else f(m)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def label(timing: float, keys: float) -> str:
    if pd.isna(timing) or pd.isna(keys):
        return "n/a (S2 not detected)"
    if timing >= SIGNAL and keys <= ARTEFACT:
        return "signal"
    if keys >= SIGNAL and timing <= ARTEFACT:
        return "artefact"
    return "mixed"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    assert_frozen_p1234(st)
    d = st.out_dir / "phase3c"
    meta = json.loads((d / "phase3c_meta.json").read_text())
    r = lambda n: pd.read_csv(d / n)
    summ, jsd_t, ek, eb, other = r("realism_summary.csv"), r("realism_jsd.csv"), r("realism_enriched_keys.csv"), r("realism_enriched_bigrams.csv"), r("realism_other_keys.csv")
    mism, per, shares, ident, let, repro = (r("swap_length_mismatch.csv"), r("swap_per_seed.csv"), r("swap_shares_per_seed.csv"),
                                             r("identical_keys_per_seed.csv"), r("letters_only_per_seed.csv"), r("reproduction_check.csv"))
    L = [f"# Phase 3c — artefact check of D4 / D5 (**POST-HOC**, diagnostic only)",
         "",
         f"_Run mode **{meta['mode']}**. **POST-HOC**: designed after the Phase 3b results were seen; nothing here was "
         "pre-registered. No model, attacker, threshold or split was changed and nothing was retrained. The TEST split is "
         "re-used post hoc with the frozen Phase 3b weights and the frozen τ10 (letters-only: τ10 recomputed on val "
         "humans for the new masked score). Freeze guards (Phases 1, 2, 3a, 3b pre-registration and "
         "`docs/phase3b_frozen_hashes.json`) pass before and after the run. Re-scoring S1/S2 reproduces the stored Phase "
         f"3b scores: max |diff| = {repro['max_abs_diff_vs_phase3b'].max():.3g}. Device {meta['device']} ({meta.get('gpu')})._",
         "",
         "Question: are D4's and D5's detections of A2–A5b driven by timing given the keys (a real signal), or by "
         "differences in the key sequences the bots generate (an implementation artefact)?", ""]

    # ---- 1. realism ------------------------------------------------------------------------------------------------
    L += ["## 1. Keycode realism (test humans vs test bots, raw sequences before truncation)", ""]
    t = summ.assign(**{"sessions": summ["n_sessions"].map("{:,}".format),
                       "length mean / median [p5, p95]": [f"{a:.1f} / {b:.0f} [{c:.0f}, {e:.0f}]" for a, b, c, e in zip(summ.len_mean, summ.len_median, summ.len_p5, summ.len_p95)],
                       "UNK %": summ.unk_pct.map("{:.3f}".format),
                       "Backspace / 100 keys": summ.backspace_per_100_keys.map("{:.2f}".format),
                       "Delete / 100 keys": summ.delete_per_100_keys.map("{:.3f}".format),
                       "sessions with Bksp/Del": summ.sessions_with_backspace_or_delete.map(pct),
                       "CapsLock / 100 keys": summ.capslock_per_100_keys.map("{:.2f}".format),
                       "non-letter/space share": summ.non_letter_space_share.map(pct)})
    L += [md_table(t[["group", "sessions", "length mean / median [p5, p95]", "UNK %", "Backspace / 100 keys", "Delete / 100 keys",
                      "sessions with Bksp/Del", "CapsLock / 100 keys", "non-letter/space share"]]), ""]
    L += ["**JSD** (Jensen–Shannon divergence, base 2, 0 = identical, 1 = disjoint) of the keycode and keycode-bigram "
          "distributions, test humans vs each attacker:", "", md_table(jsd_t.round(4)), ""]
    sh = summ.assign(**{"Shift per capital": summ.shift_shift_per_capital.map("{:.3f}".format),
                        "Shift per Shift-char": summ.shift_shift_per_shift_char.map("{:.3f}".format),
                        "next key a letter": summ.shift_next_is_letter.map(pct),
                        "Shift released before next press": summ.shift_released_before_next_press.map(pct),
                        "released during next key": summ.shift_released_during_next_key.map(pct),
                        "released after next key": summ.shift_released_after_next_release.map(pct),
                        "Shift→key lead (median ms)": summ.shift_lead_ms_median.map("{:.0f}".format),
                        "Shift hold median [p10, p90] ms": [f"{a:.0f} [{b:.0f}, {c:.0f}]" for a, b, c in zip(summ.shift_hold_ms_median, summ.shift_hold_ms_p10, summ.shift_hold_ms_p90)]})
    L += ["**Shift pattern.** Capitals are counted in the target sentence (both sides type the same sentence); "
          "'Shift-char' also counts punctuation that needs Shift. Position = the next non-Shift key after each Shift press.", "",
          md_table(sh[["group", "Shift per capital", "Shift per Shift-char", "next key a letter", "Shift released before next press",
                       "released during next key", "released after next key", "Shift→key lead (median ms)", "Shift hold median [p10, p90] ms"]]), ""]
    oth = other.copy()
    for c in oth.columns[2:]:
        oth[c] = oth[c].map("{:.2f}".format)
    L += ["**Keys other than letters and space** (punctuation, digits, control keys), per 1,000 keystrokes; keycodes present "
          "in ≥ 0.1% of the sessions of at least one group:", "", md_table(oth), ""]
    L += ["**Keycodes ≥ 10× more frequent in one group** (frequency = share of all keystrokes; present in ≥ 0.1% of the "
          "sessions of the group where it is more frequent; ∞ = absent from the other group):", ""]
    fmt = lambda df: df.assign(freq_humans=df.freq_humans.map(lambda v: f"{100 * v:.3f}%"), freq_bots=df.freq_bots.map(lambda v: f"{100 * v:.3f}%"),
                               ratio=df.ratio.map(lambda v: "∞" if np.isinf(v) else f"{v:.1f}"),
                               sessions_humans=df.sessions_humans.map(pct), sessions_bots=df.sessions_bots.map(pct))
    cols = ["attacker", "more_in", "name", "freq_humans", "freq_bots", "ratio", "sessions_humans", "sessions_bots"]
    L += [md_table(fmt(ek)[cols]) if len(ek) else "_none_", ""]
    nb = eb.groupby(["attacker", "more_in"]).size().rename("n").reset_index()
    L += [f"**Keycode bigrams ≥ 10× more frequent in one group** (same rule): {len(eb)} bigrams in total "
          "(`realism_enriched_bigrams.csv` lists all). Counts per attacker and direction:", "", md_table(nb), "",
          "The 15 most frequent (in the group where they are enriched) per attacker and direction:", ""]
    top = (eb.assign(f=np.where(eb.more_in == "bots", eb.freq_bots, eb.freq_humans)).sort_values("f", ascending=False)
             .groupby(["attacker", "more_in"], sort=False).head(15).sort_values(["attacker", "more_in", "f"], ascending=[True, True, False]))
    L += [md_table(fmt(top)[cols]) if len(top) else "_none_", ""]

    # ---- 2. swap ---------------------------------------------------------------------------------------------------
    L += ["## 2. Swap test (TEST split, frozen weights, frozen τ10)", "",
          "Each test bot is paired with the test human it was generated for (same session, so the same target sentence). "
          "S1 = human keys + human timing (all test humans), S2 = bot keys + bot timing (the original attack), S3 = human "
          "keys + bot timing, S4 = bot keys + human timing; S3/S4 are aligned by position and truncated to the shorter "
          "length. Controls (not in the spec): S1t / S2t = human / bot truncated to the same pair length.", "",
          "**Length mismatch and alignment:**", ""]
    m = mism.assign(**{"pairs": mism.n_pairs.map("{:,}".format), "length mismatch": [f"{a:,} ({pct(b)})" for a, b in zip(mism.n_mismatch, mism.frac_mismatch)],
                       "bot longer / shorter": [f"{a:,} / {b:,}" for a, b in zip(mism.n_bot_longer, mism.n_bot_shorter)],
                       "|diff| mean / median / p95 / max (mismatched)": [f"{a:.1f} / {b:.0f} / {c:.0f} / {e}" for a, b, c, e in zip(mism.abs_diff_mean_mismatched, mism.abs_diff_median_mismatched, mism.abs_diff_p95_mismatched, mism.abs_diff_max)],
                       "keys dropped human / bot": [f"{pct(a)} / {pct(b)}" for a, b in zip(mism.frac_keys_dropped_human, mism.frac_keys_dropped_bot)],
                       "aligned positions with the same key": mism.frac_aligned_positions_same_key.map(pct),
                       "identical key sequences": mism.n_identical_key_sequences.map("{:,}".format)})
    L += [md_table(m[["attacker", "pairs", "length mismatch", "bot longer / shorter", "|diff| mean / median / p95 / max (mismatched)",
                      "keys dropped human / bot", "aligned positions with the same key", "identical key sequences"]]), "",
          "**Caveat (found in the quick run, before the full run):** positional alignment pairs each timing with the key at "
          "the same POSITION. After the first place where the human corrected a typo (or the replayed human typed "
          "differently), the positions no longer carry the same character. The 'aligned positions with the same key' "
          "column shows how often they do. S3 and S4 therefore combine the keys of one sequence with timings that "
          "belong to OTHER keys. A detector that models timing *given the key* will flag that combination whoever made "
          "the keys. A5a (a verbatim replay of another human typing the same sentence) is the built-in control: its "
          "S3 and S4 are human keys + human timing from two different people.", ""]
    for det in per.detector.unique():
        g = per[per.detector == det]
        rows = []
        for a in g.attacker.unique():
            x = g[g.attacker == a]
            row = {"attacker": a, "S1 BPCER@τ10": ms(x[x.set == "S1"].flagged)}
            for k in ("S2", "S3", "S4", "S2t", "S1t"):
                row[f"{k} APCER"] = ms(x[x.set == k].apcer_tau10)
            for k in ("S2", "S3", "S4", "S2t", "S1t"):
                row[f"{k} AUC"] = ms(x[x.set == k].auc_vs_S1, lambda v: f"{v:.3f}")
            rows.append(row)
        L += [f"**{det}**: APCER@τ10 (attacks accepted) and AUC vs S1; mean ± sd over seeds 42/43/44"
              + (" (each D5 model tested only on the attacker it was trained on)" if det.startswith("D5") else "") + ":", "",
              md_table(pd.DataFrame(rows)), ""]
    L += ["**Share of the S2 effect reproduced by each swap.** Effect(S) = AUC(S1 vs S) − 0.5 (and, second, "
          "flag rate(S) − flag rate(S1)); share = Effect(S3) / Effect(S2) (timing) and Effect(S4) / Effect(S2) (keys), "
          "per seed, then mean ± sd. Undefined when |Effect(S2)| < 0.05. The shares need not add to 1 (interaction, "
          "misalignment). Label rule, fixed before the full run: *signal* if the timing share ≥ 0.75 and the key share ≤ 0.25; "
          "*artefact* if the key share ≥ 0.75 and the timing share ≤ 0.25; *mixed* otherwise.", ""]
    rows = []
    for (det, a), x in shares.groupby(["detector", "attacker"], sort=False):
        f2 = lambda v: f"{v:.2f}"
        rows.append({"detector": det, "attacker": a, "Effect(S2) AUC": ms(x.effect_auc_S2, f2),
                     "timing share (S3) AUC": ms(x.share_auc_S3, f2), "key share (S4) AUC": ms(x.share_auc_S4, f2),
                     "timing share (S3) flag": ms(x.share_flag_S3, f2), "key share (S4) flag": ms(x.share_flag_S4, f2),
                     "truncation control (S2t) AUC": ms(x.share_auc_S2t, f2),
                     "label (AUC shares)": label(x.share_auc_S3.mean(), x.share_auc_S4.mean())})
    L += [md_table(pd.DataFrame(rows)), ""]
    ir = []
    for (det, a), x in ident.groupby(["detector", "attacker"], sort=False):
        ir.append({"detector": det, "attacker": a, "identical pairs": f"{int(x.n_pairs.iloc[0]):,}",
                   "humans BPCER@τ10": ms(x.bpcer_tau10_humans), "bots APCER@τ10": ms(x.apcer_tau10_bots),
                   "AUC": ms(x.auc, lambda v: f"{v:.3f}")})
    L += ["**Identical-key-sequence pairs (addition, not in the spec).** These are the pairs where the human typed exactly the bot's keycode "
          "sequence: same length, same key at every position (so no correction, the same Shift use). Here the keys are "
          "identical by construction, so any separation of the bot from its human can only come from timing. S1 and S2 "
          "are restricted to these pairs:", "", md_table(pd.DataFrame(ir)), ""]

    # ---- 3. letters-only -------------------------------------------------------------------------------------------
    base = per[(per.detector == "D4-primary") & (per.set == "S2")]
    hum = let[let.attacker == "(humans)"]
    L += ["## 3. Letters-only D4-primary", "",
          "Session score = mean per-step NLL over the steps where both the current and the next key are a–z or space "
          "(all other steps masked; the recurrent state still sees the whole history). τ10 recomputed on val humans with the same "
          f"masked score (10% of val humans rejected). Test humans: BPCER@τ10 {ms(hum.bpcer_tau10)}; letter steps are "
          f"{pct(hum.letter_steps_share.mean())} of human steps; sessions without any letter step: test humans "
          f"{int(hum.n_no_letter_steps.max())}, val humans {int(hum.val_n_no_letter_steps.max())} (excluded).", ""]
    rows = []
    for a in [x for x in let.attacker.unique() if x != "(humans)"]:
        x, b = let[let.attacker == a], base[base.attacker == a]
        rows.append({"attacker": a, "APCER@τ10 original": ms(b.apcer_tau10), "APCER@τ10 letters-only": ms(x.apcer_tau10),
                     "AUC original": ms(b.auc_vs_S1, lambda v: f"{v:.3f}"), "AUC letters-only": ms(x.auc, lambda v: f"{v:.3f}"),
                     "letter-step share": pct(x.letter_steps_share.mean()), "sessions without letter steps": int(x.n_no_letter_steps.max())})
    L += [md_table(pd.DataFrame(rows)), ""]

    # ---- 4. verdict -------------------------------------------------------------------------------------------------
    L += ["## 4. Verdict", ""]
    vf = d / "verdict.md"
    L += [vf.read_text(encoding="utf-8").strip() if vf.exists() else "_Verdict not written yet (see `verdict.md`)._", ""]
    L += ["## Files", "", "`realism_*.csv`, `swap_length_mismatch.csv`, `swap_per_seed.csv`, `swap_shares_per_seed.csv`, "
          "`identical_keys_per_seed.csv`, `letters_only_per_seed.csv`, `reproduction_check.csv`, `phase3c_meta.json`; code "
          "`src/kl/phase3c/`, `scripts/40_run_phase3c.py`, `scripts/41_report_phase3c.py`; spec `docs/PHASE3C_SPEC.md`.", ""]
    (d / "DIAGNOSTICS_PHASE3C.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {d / 'DIAGNOSTICS_PHASE3C.md'}")


if __name__ == "__main__":
    main()
