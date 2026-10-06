#!/usr/bin/env python
"""Phase 3a — diagnostics only (Phases 1 and 2 frozen; nothing is tuned and nothing feeds back).

  1. centrality    Mahalanobis distance (MinCovDet, random_state=seed) of test humans vs A1-A5b in the frozen transformed space
                   + per-feature variance ratios                                  -> centrality_*.csv/png, variance_ratio*.csv/png
  2. ablation      D1 refitted with the frozen nu / gamma on feature-group subsets -> ablation_table.csv, ablation_heatmap_*.png
  3. importance    permutation importance of single-attacker D3 models on TEST     -> permutation_importance.csv, top10_features_per_attacker.csv, importance_heatmap.png
  4. consistency   per-user session-to-session std of pp_mean / hold_mean          -> user_consistency*.csv/png
Results and fit notes are also stored in diagnostics_results.json (input of 21_report_diagnostics.py).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from kl.config import ROOT, load_settings, read_json, setup_logging, write_json  # noqa: E402
from kl.features import FEATURE_NAMES  # noqa: E402
from kl.phase2.config import load_phase2_config, phase2_dir  # noqa: E402
from kl.phase2.detectors import fit_d3  # noqa: E402
from kl.phase2.evaluation import plot_heatmap, user_weights  # noqa: E402
from kl.phase3a import analyses as an  # noqa: E402
from kl.phase3a.guard import assert_frozen_p12  # noqa: E402
from kl.phase3a.plots import plot_centrality, plot_consistency, plot_matrix  # noqa: E402

STEPS = ("centrality", "ablation", "importance", "consistency")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", choices=STEPS, help="run only these analyses (default: all)")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    out = st.out_dir / "phase3a"
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(st.out_dir, "20_run_diagnostics")
    model = assert_frozen_p12(st)
    p2 = load_phase2_config()
    cfg = yaml.safe_load((ROOT / "config_phase3a.yaml").read_text(encoding="utf-8"))
    p2out = phase2_dir(st)
    todo = args.only or STEPS
    cols, seed = model.feature_names, st.seed
    tests = p2["experiments"]["test_attackers"]

    train = pd.read_parquet(st.interim_dir / "features_train.parquet")
    val = pd.read_parquet(st.interim_dir / "features_val.parquet")
    test = pd.read_parquet(st.interim_dir / "features_test.parquet")
    bots_te = {a: pd.read_parquet(p2out / f"bots_{a}_test.parquet") for a in tests}
    results_path = out / "diagnostics_results.json"
    results = read_json(results_path, {}) or {}
    results.update({"seed": seed, "config_hash": st.config_hash, "n_test_human_sessions": len(test), "n_test_users": int(test["user_id"].nunique()),
                    "n_train_human_sessions": len(train), "features": cols})
    t0 = time.time()

    # ---------------------------------------------------------------- 1. centrality
    if "centrality" in todo:
        t = time.time()
        x_tr, x_h = an.transformed(model, train), an.transformed(model, test)
        x_b = {a: an.transformed(model, b) for a, b in bots_te.items()}
        table, dists, info = an.centrality(x_tr, x_h, x_b, cols, seed, bool(cfg["centrality"]["supplementary_classical_covariance"]))
        table.to_csv(out / "centrality_summary.csv", index=False)
        plot_centrality(dists["mcd"], out / "centrality_hist.png", out / "centrality_ecdf.png", "Robust (MCD) Mahalanobis distance, test split")
        if "classical" in dists:
            plot_centrality(dists["classical"], out / "centrality_hist_classical.png", out / "centrality_ecdf_classical.png", "Classical-covariance Mahalanobis distance (supplement), test split")
        vr = an.variance_ratios(test, bots_te, x_h, x_b, cols)
        vr.to_csv(out / "variance_ratio.csv", index=False)
        mat = np.log2(vr.pivot(index="feature", columns="attacker", values="var_ratio_transformed").loc[cols, tests])
        plot_matrix(mat, out / "variance_ratio_heatmap.png", "log2(bot variance / human variance), frozen transformed space (< 0: too homogeneous)", "log2 ratio", "RdBu_r", -3, 3, "{:.1f}")
        results["centrality"] = {"info": info}
        log.info("centrality: MCD info %s (%.0fs)", info["mcd"], time.time() - t)
        log.info("\n%s", table[["covariance", "group", "median_d", "iqr_d", "median_ratio_to_human", "ks_stat_vs_human", "frac_below_human_median"]].round(3).to_string(index=False))

    # ---------------------------------------------------------------- 2. ablation
    if "ablation" in todo:
        t = time.time()
        nu, gamma = float(model.params["nu"]), float(model.params["gamma_effective"])
        ev = st.cfg["evaluation"]
        users = np.sort(test["user_id"].unique())
        weights = user_weights(len(users), int(ev["bootstrap_reps"]), seed)
        subsets = an.ablation_subsets(cfg["ablation"]["subsets"])
        # sanity check of the machinery: refitting ALL features must reproduce the frozen model's thresholds
        chk_tab, chk_meta = an.ablation(train, val, test, {tests[0]: bots_te[tests[0]]}, {"ALL_refit": list(cols)}, nu, gamma, weights[:2], users, float(ev["ci_level"]), 1)
        refit_ok = bool(np.allclose([chk_meta["ALL_refit"]["thresholds"]["tau5"], chk_meta["ALL_refit"]["thresholds"]["tau10"]], [model.thresholds["tau5"], model.thresholds["tau10"]], rtol=1e-6, atol=1e-9))
        table, meta = an.ablation(train, val, test, bots_te, subsets, nu, gamma, weights, users, float(ev["ci_level"]), st.n_jobs)
        # the ALL row is the frozen D1 itself (scores and thresholds of the frozen model)
        from kl.phase2.evaluation import cell_row
        idx_of = pd.Index(users).get_indexer
        hs = model.score(test)
        all_rows = [cell_row("D1[ALL (frozen)]", "ALL (frozen)", a, hs, idx_of(test["user_id"]), model.score(b), idx_of(b["user_id"]), model.thresholds, weights, float(ev["ci_level"])) for a, b in bots_te.items()]
        table = pd.concat([pd.DataFrame(all_rows), table], ignore_index=True)
        table.to_csv(out / "ablation_table.csv", index=False)
        order = ["ALL (frozen)"] + list(subsets)
        for metric, fname, ttl, pct, vmin, vmax, cmap in (("apcer_fpr_tau10", "ablation_heatmap_apcer_tau10.png", "D1 ablation: APCER / FPR at τ10", True, 0.0, 1.0, "YlOrRd"),
                                                           ("auc", "ablation_heatmap_auc.png", "D1 ablation: ROC-AUC", False, 0.3, 1.0, "YlGnBu")):
            mat = pd.DataFrame({a: [table[(table.trained_on == s) & (table.tested_on == a)][metric].iloc[0] for s in order] for a in tests}, index=order)
            plot_heatmap(mat, out / fname, ttl, metric, pct, vmin, vmax, cmap)
        results["ablation"] = {"nu": nu, "gamma": gamma, "subsets": meta, "refit_all_matches_frozen_thresholds": refit_ok}
        log.info("ablation: nu=%s gamma=%s; refit(ALL) matches frozen thresholds: %s (%.0fs)", nu, gamma, refit_ok, time.time() - t)
        log.info("\n%s", table[table.tested_on == "A1"][["trained_on", "apcer_fpr_tau5", "apcer_fpr_tau10", "bpcer_fnr_tau10", "auc"]].round(4).to_string(index=False))

    # ---------------------------------------------------------------- 3. permutation importance
    if "importance" in todo:
        t = time.time()
        icfg, hp = cfg["importance"], p2["detectors"]["d3"]
        x_tr, x_h = an.transformed(model, train), an.transformed(model, test)
        frames, d3_equiv = [], None
        for a in icfg["attackers"]:
            tb = pd.read_parquet(p2out / f"bots_{a}_train.parquet")
            rf = an.fit_single_attacker_d3(x_tr, an.transformed(model, tb), seed, hp, st.n_jobs)
            if d3_equiv is None:  # the refit must equal the Phase 2 D3 (same data, same seed): compare scores on a test slice
                ref = fit_d3(train, tb, model.transform, cols, seed, hp, val, st.n_jobs)
                sl = test.iloc[:2000]
                d3_equiv = bool(np.allclose(ref.score(sl), rf.predict_proba(an.transformed(model, sl))[:, list(rf.classes_).index(1.0)]))
                log.info("single-attacker D3 (%s) refit equals the Phase 2 detector: %s", a, d3_equiv)
            df = an.importance_table(rf, x_h, an.transformed(model, bots_te[a]), cols, int(icfg["n_per_class"]), int(icfg["n_repeats"]), seed, a)
            frames.append(df)
            log.info("importance %-4s: baseline AUC %.4f; top: %s (%.0fs)", a, df["baseline_auc"].iloc[0], df.head(3)[["feature", "importance_mean"]].round(4).values.tolist(), time.time() - t)
            del rf
        imp = pd.concat(frames, ignore_index=True)
        imp.to_csv(out / "permutation_importance.csv", index=False)
        k = int(icfg["top_k"])
        top = imp[imp["rank"] <= k].sort_values(["attacker", "rank"])[["attacker", "rank", "feature", "importance_mean", "importance_std", "baseline_auc"]]
        top.to_csv(out / "top10_features_per_attacker.csv", index=False)
        mat = imp.pivot(index="feature", columns="attacker", values="importance_mean").loc[cols, icfg["attackers"]]
        mat = mat.loc[mat.max(axis=1).sort_values(ascending=False).index]
        plot_matrix(mat, out / "importance_heatmap.png", "Permutation importance of single-attacker D3 (mean drop of test ROC-AUC)", "AUC drop", "YlOrRd", 0.0, float(np.nanmax(mat.to_numpy())), "{:.3f}")
        results["importance"] = {"n_per_class": int(icfg["n_per_class"]), "n_repeats": int(icfg["n_repeats"]), "d3_refit_matches_phase2": d3_equiv}

    # ---------------------------------------------------------------- 4. per-user consistency
    if "consistency" in todo:
        t = time.time()
        ccfg = cfg["consistency"]
        summ, per_unit = an.consistency(test, bots_te, ccfg["attackers"], ccfg["features"], int(ccfg["min_user_sessions"]), int(ccfg["pseudo_user_sessions"]), seed)
        summ.to_csv(out / "user_consistency_summary.csv", index=False)
        per_unit.to_csv(out / "user_consistency_per_unit.csv", index=False)
        plot_consistency(per_unit, list(ccfg["features"]), out / "user_consistency.png", "Session-to-session std per user / pseudo-user (test split)")
        results["consistency"] = {"min_user_sessions": int(ccfg["min_user_sessions"]), "pseudo_user_sessions": int(ccfg["pseudo_user_sessions"])}
        log.info("consistency (%.0fs):\n%s", time.time() - t, summ[["group", "feature", "n_units", "median_std", "median_std_over_total_std", "ratio_median_std_vs_real", "icc1"]].round(3).to_string(index=False))

    write_json(results_path, results)
    assert_frozen_p12(st)
    log.info("done in %.0fs -> %s", time.time() - t0, out)


if __name__ == "__main__":
    main()
