"""DIAGNOSTICS.md: one paper-style paragraph (with the numbers) plus the key table per Phase 3a analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..config import Settings, read_json


def _pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def _rowv(df: pd.DataFrame, **kw: Any) -> pd.Series:
    m = df
    for k, v in kw.items():
        m = m[m[k] == v]
    return m.iloc[0]


def write_diagnostics_md(st: Settings, p2: Mapping[str, Any], p3: Mapping[str, Any], out: Path) -> None:
    """Write ``DIAGNOSTICS.md`` from the CSV / JSON outputs of ``scripts/20_run_diagnostics.py``."""
    res = read_json(out / "diagnostics_results.json", {}) or {}
    tests = p2["experiments"]["test_attackers"]
    atk = [a for a in tests]
    L = ["# Phase 3a diagnostics", "",
         f"_Run mode: **{'quick' if st.quick else 'full'}**, seed {st.seed}, Phase 1 config hash `{st.config_hash}`. Phases 1 and 2 are frozen (hash-guarded); nothing here changes a feature, "
         "detector, attacker or threshold, and no result was used to modify anything. Test split: "
         f"{res.get('n_test_human_sessions', 0):,} human sessions from {res.get('n_test_users', 0):,} users; every attacker has one matched bot per human session._", ""]

    # ------------------------------------------------------------------ 1
    if (out / "centrality_summary.csv").exists():
        c = pd.read_csv(out / "centrality_summary.csv")
        vr = pd.read_csv(out / "variance_ratio.csv")
        m = c[c["covariance"] == "mcd"]
        h = _rowv(m, group="human")
        cinfo = (res.get("centrality") or {}).get("info", {})
        info = cinfo.get("mcd", {})
        L += ["## 1. Centrality (\"too typical\") analysis", ""]
        parts = [f"In the frozen transformed feature space we computed each test session's Mahalanobis distance to the train-human distribution, using a robust covariance "
                 f"(scikit-learn `MinCovDet`, random_state = {st.seed}, fitted on {res.get('n_train_human_sessions', 0):,} train humans). Test humans have a median distance of {h['median_d']:.2f} "
                 f"(IQR {h['q25_d']:.2f}–{h['q75_d']:.2f}). "]
        for a in atk:
            r = _rowv(m, group=a)
            parts.append(f"{a}: median {r['median_d']:.2f} (IQR {r['q25_d']:.2f}–{r['q75_d']:.2f}; {r['median_ratio_to_human']:.2f}× the human median; KS D = {r['ks_stat_vs_human']:.3f}, "
                         f"p {'< 1e-300' if r['ks_pvalue_vs_human'] == 0 else format(r['ks_pvalue_vs_human'], '.1e')}; {_pct(r['frac_below_human_median'])} of its sessions are closer to the centre than the human median). ")
        closer = [a for a in atk if _rowv(m, group=a)["median_ratio_to_human"] < 1]
        farther = [a for a in atk if _rowv(m, group=a)["median_ratio_to_human"] > 1]
        parts.append(f"Attackers whose median distance is below the human median (closer to the centre, 'too typical'): {', '.join(closer) if closer else 'none'}; "
                     f"above it (farther from the centre): {', '.join(farther) if farther else 'none'}.")
        L += ["".join(parts), ""]
        if info:
            L += [f"_Estimator note._ The MCD covariance is rank-deficient in this space: {info['n_eigenvalues_below_1e-10']} eigenvalue(s) below 1e-10 (condition number {info['condition_number']:.1e}); "
                  f"feature(s) with zero variance inside the MCD support subset ({info['support_size']:,} of {cinfo['n_train']:,} sessions): "
                  f"{', '.join('`%s`' % f for f in info['zero_variance_features_in_support']) or 'none'}. scikit-learn then uses a pseudo-inverse, so such a feature does not contribute to the distance. "
                  "The classical-covariance distance is therefore reported as a labelled supplement (below); it is not the specified estimator.", ""]
        L += ["| group | median d | IQR (Q1–Q3) | ratio to human median | KS D vs humans | share closer than human median |", "|---|---|---|---|---|---|"]
        for cov, ttl in (("mcd", "MCD (specified)"), ("classical", "classical covariance (supplement)")):
            mm = c[c["covariance"] == cov]
            if len(mm):
                L.append(f"| **{ttl}** | | | | | |")
                for _, r in mm.iterrows():
                    ks = "–" if np.isnan(r["ks_stat_vs_human"]) else f"{r['ks_stat_vs_human']:.3f}"
                    L.append(f"| {r['group']} | {r['median_d']:.2f} | {r['q25_d']:.2f}–{r['q75_d']:.2f} | {r['median_ratio_to_human']:.2f} | {ks} | {_pct(r['frac_below_human_median'])} |")
        L += ["", "Figures: `centrality_hist.png`, `centrality_ecdf.png` (MCD) and `*_classical.png` (supplement).", ""]
        s = []
        for a in atk:
            v = vr[vr["attacker"] == a]["var_ratio_transformed"]
            vraw = vr[vr["attacker"] == a]["var_ratio_raw"]
            s.append(f"{a}: median ratio {v.median():.2f} ({int((v < 1).sum())} of {len(v)} features < 1; raw units: median {vraw.median():.2f}, {int((vraw < 1).sum())} < 1)")
        L += ["**Variance ratios.** Per-feature variance of each attacker's test sessions divided by the human variance in the frozen transformed space (ratio < 1 = too homogeneous): " + "; ".join(s)
              + ". Per-feature values: `variance_ratio.csv`, `variance_ratio_heatmap.png`.", ""]

    # ------------------------------------------------------------------ 2
    if (out / "ablation_table.csv").exists():
        ab = pd.read_csv(out / "ablation_table.csv")
        meta = (res.get("ablation") or {})
        order = ["ALL (frozen)"] + [k for k in p3["ablation"]["subsets"]]
        a1 = {s: _rowv(ab, trained_on=s, tested_on="A1") for s in order}
        L += ["## 2. Feature-group ablation of D1", "",
              f"We refitted the One-Class SVM with the frozen hyper-parameters (ν = {meta.get('nu')}, γ = {meta.get('gamma')}; the input transform is refitted on train for each subset, τ5/τ10 come from validation humans) "
              "on feature subsets and evaluated it on A1–A5b. "
              f"For A1, the full frozen D1 reaches AUC {a1['ALL (frozen)']['auc']:.3f} (APCER {_pct(a1['ALL (frozen)']['apcer_fpr_tau10'])} at τ10); "
              f"timing features only give AUC {a1['TIMING_only']['auc']:.3f} (APCER {_pct(a1['TIMING_only']['apcer_fpr_tau10'])}), timing+rhythm {a1['TIMING_RHYTHM']['auc']:.3f} ({_pct(a1['TIMING_RHYTHM']['apcer_fpr_tau10'])}), "
              f"all features except BEHAVIOUR {a1['all_minus_BEHAVIOUR']['auc']:.3f} ({_pct(a1['all_minus_BEHAVIOUR']['apcer_fpr_tau10'])}) and BEHAVIOUR alone {a1['BEHAVIOUR_only']['auc']:.3f} ({_pct(a1['BEHAVIOUR_only']['apcer_fpr_tau10'])}). "
              f"Removing the BEHAVIOUR group (lev_norm, corrections, shift overlap) therefore changes the A1 AUC by {a1['all_minus_BEHAVIOUR']['auc'] - a1['ALL (frozen)']['auc']:+.3f} "
              f"and the A1 APCER at τ10 by {100 * (a1['all_minus_BEHAVIOUR']['apcer_fpr_tau10'] - a1['ALL (frozen)']['apcer_fpr_tau10']):+.1f} pp; the timing-only detector differs from the full one by "
              f"{a1['TIMING_only']['auc'] - a1['ALL (frozen)']['auc']:+.3f} AUC. ", ""]
        others = []
        for a in atk[1:]:
            best = max(order, key=lambda s: _rowv(ab, trained_on=s, tested_on=a)["auc"])
            others.append(f"{a}: best subset {best} (AUC {_rowv(ab, trained_on=best, tested_on=a)['auc']:.3f}; full D1 {_rowv(ab, trained_on='ALL (frozen)', tested_on=a)['auc']:.3f})")
        L += ["For the other attackers: " + "; ".join(others) + ".", "",
              "| subset (features) | attacker | " + " | ".join(["APCER/FPR @τ5", "APCER/FPR @τ10", "BPCER/FNR @τ5", "BPCER/FNR @τ10", "ROC-AUC"]) + " |", "|---|---|---|---|---|---|---|"]
        from ..phase2.evaluation import fmt_cell
        for s in order:
            nf = 23 if s == "ALL (frozen)" else (meta.get("subsets", {}).get(s, {}).get("n_features", "?"))
            for a in atk:
                r = _rowv(ab, trained_on=s, tested_on=a)
                L.append(f"| {s} ({nf}) | {a} | " + " | ".join(fmt_cell(r, k) for k in ("apcer_fpr_tau5", "apcer_fpr_tau10", "bpcer_fnr_tau5", "bpcer_fnr_tau10", "auc")) + " |")
        ok = meta.get("refit_all_matches_frozen_thresholds")
        L += ["", f"_Check:_ refitting ALL 23 features with this machinery reproduces the frozen thresholds: {ok}. Cells carry 95% user-level bootstrap CIs (same resamples as Phase 2). "
              "Figures: `ablation_heatmap_auc.png`, `ablation_heatmap_apcer_tau10.png`.", ""]

    # ------------------------------------------------------------------ 3
    if (out / "permutation_importance.csv").exists():
        imp = pd.read_csv(out / "permutation_importance.csv")
        top = pd.read_csv(out / "top10_features_per_attacker.csv")
        im = res.get("importance") or {}
        attackers = list(dict.fromkeys(imp["attacker"]))
        L += ["## 3. What gives each attacker away (permutation importance of D3)", "",
              f"For each single-attacker Random Forest (D3 trained on {', '.join(attackers)}) we measured the mean decrease of test ROC-AUC when one feature is permuted "
              f"({im.get('n_repeats')} repeats, {im.get('n_per_class'):,} test humans + {im.get('n_per_class'):,} test bots of that attacker, seeded sample; refit equals the Phase 2 detector: {im.get('d3_refit_matches_phase2')}). "]
        parts = []
        for a in attackers:
            t3 = imp[(imp["attacker"] == a)].nsmallest(3, "rank")
            parts.append(f"{a} (baseline AUC {t3['baseline_auc'].iloc[0]:.3f}): " + ", ".join(f"`{r.feature}` ({r.importance_mean:.3f})" for r in t3.itertuples()))
        sat = [a for a in attackers if imp[imp["attacker"] == a]["baseline_auc"].iloc[0] >= 0.999]
        note = (f" For {', '.join(sat)} the detector is saturated (baseline AUC ≥ 0.999) and many features are redundant, so permuting any single feature leaves the AUC essentially unchanged: "
                "near-zero importances there do not mean that no feature gives the attacker away." if sat else "")
        L += ["The most important features were — " + "; ".join(parts) + ". Importances are marginal: correlated features share credit, so they should not be summed." + note, "",
              "| rank | " + " | ".join(attackers) + " |", "|---|" + "---|" * len(attackers)]
        k = int(top["rank"].max())
        for r_ in range(1, k + 1):
            cells = []
            for a in attackers:
                x = top[(top["attacker"] == a) & (top["rank"] == r_)]
                cells.append(f"`{x['feature'].iloc[0]}` ({x['importance_mean'].iloc[0]:.3f})" if len(x) else "")
            L.append(f"| {r_} | " + " | ".join(cells) + " |")
        L += ["", "Figure: `importance_heatmap.png`; full table: `permutation_importance.csv`.", ""]

    # ------------------------------------------------------------------ 4
    if (out / "user_consistency_summary.csv").exists():
        cs = pd.read_csv(out / "user_consistency_summary.csv")
        cc = res.get("consistency") or {}
        feats = list(dict.fromkeys(cs["feature"]))
        groups = list(dict.fromkeys(cs["group"]))
        real = {f: _rowv(cs, group="real users", feature=f) for f in feats}
        L += ["## 4. Per-user consistency (A2 / A3)", "",
              f"For every test user with at least {cc.get('min_user_sessions')} clean sessions ({int(real[feats[0]]['n_units']):,} users, median {real[feats[0]]['sessions_per_unit_median']:.0f} sessions) we computed the session-to-session standard deviation "
              f"of {' and '.join('`%s`' % f for f in feats)}, and compared it with pseudo-users built from {cc.get('pseudo_user_sessions')} random bot sessions of A2 and A3 (same number of units) and, as a control, "
              "with pseudo-users built from random human sessions of different users (real marginal distribution, no user persistence). "]
        verdicts = []
        for f in feats:
            txt = [f"`{f}`: real users' median per-user std is {real[f]['median_std']:.1f} ms ({real[f]['median_std_over_total_std']:.2f} of the total std of all sessions; ICC(1) = {real[f]['icc1']:.2f})"]
            for g in groups[1:]:
                r = _rowv(cs, group=g, feature=f)
                txt.append(f"{g}: {r['median_std']:.1f} ms ({r['median_std_over_total_std']:.2f} of total; ICC(1) = {r['icc1']:.2f}; real users' std is {'lower' if r['ratio_median_std_vs_real'] > 1 else 'not lower'} — Mann–Whitney p = {r['mannwhitney_real_less_p']:.1e})")
            verdicts.append("; ".join(txt) + ".")
        L += [" ".join(verdicts), ""]
        bots = [g for g in groups if "pseudo-users" in g]
        more = all(_rowv(cs, group=g, feature=f)["ratio_median_std_vs_real"] > 1 and _rowv(cs, group=g, feature=f)["mannwhitney_real_less_p"] < 0.05 for g in bots for f in feats)
        ratios = "; ".join(f"`{f}`: " + ", ".join(f"{g.split()[0]} {_rowv(cs, group=g, feature=f)['ratio_median_std_vs_real']:.2f}×" for g in groups[1:]) for f in feats)
        L += [f"Conclusion from these numbers: real users are {'MORE' if more else 'not uniformly more'} self-consistent across sessions than A2/A3 pseudo-users "
              f"({'for both features, with p < 0.05 in every comparison' if more else 'see the table for the exceptions'}); median per-unit std relative to the real users' — {ratios}. "
              "In absolute milliseconds the margin can be small for a generator whose sessions are globally homogeneous (see the variance ratios in section 1), which is why the std relative to the group's total std and the ICC(1) are also given. "
              "A2 and A3 draw every session independently, so nothing persists across their sessions; the shuffled-human control shows what that lack of persistence looks like with the real marginal distribution.", "",
              "| group | feature | units | median std (ms) | Q1–Q3 | median std / total std | ICC(1) | real std lower? (MW p) |", "|---|---|---|---|---|---|---|---|"]
        for f in feats:
            for g in groups:
                r = _rowv(cs, group=g, feature=f)
                mw = "–" if np.isnan(r["mannwhitney_real_less_p"]) else f"{r['mannwhitney_real_less_p']:.1e}"
                L.append(f"| {g} | {f} | {int(r['n_units']):,} | {r['median_std']:.1f} | {r['q25_std']:.1f}–{r['q75_std']:.1f} | {r['median_std_over_total_std']:.2f} | {r['icc1']:.2f} | {mw} |")
        L += ["", "Figure: `user_consistency.png`; per-unit values: `user_consistency_per_unit.csv`.", ""]
    (out / "DIAGNOSTICS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
