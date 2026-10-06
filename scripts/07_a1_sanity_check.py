#!/usr/bin/env python
"""Step 7 — A1 sanity check (only after selection is frozen).

One A1 bot session per TEST human session (same sentence), pushed through the same features.py path
as humans and scored by the frozen model. FNR = humans rejected; FPR = bots accepted.

Also writes grid_attack_sensitivity.csv: A1 evaluated for ALL grid configurations (each with its own
tau5/tau10 from validation). This table is POST-HOC and NOT used for selection.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kl.attacks.a1_gaussian import A1Gaussian, A1Params, UNMAPPED_KEYCODE, char_to_key  # noqa: E402
from kl.config import load_settings, read_json, setup_logging, write_json  # noqa: E402
from kl.evaluate import SENSITIVITY_NOTE, build_bot_features, confusion_matrix, grid_attack_sensitivity, plot_roc, roc_auc  # noqa: E402
from kl.features import FEATURE_NAMES, FeatureParams  # noqa: E402
from kl.model import LivenessModel  # noqa: E402
from kl.report import write_methods_md  # noqa: E402


def run_sensitivity(st, model: LivenessModel, test: pd.DataFrame, bots: pd.DataFrame, log) -> dict:
    """Post-hoc A1 evaluation of every grid config -> grid_attack_sensitivity.csv; returns the summary for METHODS.md."""
    mcfg = st.cfg["model"]
    cols = model.feature_names
    train = pd.read_parquet(st.interim_dir / "features_train.parquet")
    val = pd.read_parquet(st.interim_dir / "features_val.parquet")
    tf = model.transform.transform
    mat = lambda df: tf(df[cols].to_numpy(dtype="float64"))  # noqa: E731
    sens = grid_attack_sensitivity(mat(train), mat(val), mat(test), mat(bots), mcfg["nu_grid"], mcfg["gamma_grid"], mcfg["target_fnrs"], st.n_jobs)
    grid = pd.read_csv(st.results_dir / "grid_results.csv", dtype={"gamma": str})
    sens = sens.merge(grid[["nu", "gamma", "ref_accept_tau5", "admissible", "selected"]], on=["nu", "gamma"], how="left")
    sens["rank_by_auc"] = sens["auc"].rank(ascending=False, method="min").astype(int)
    sens["rank_by_fpr_tau5"] = sens["fpr_tau5"].rank(ascending=True, method="min").astype(int)
    sens["rank_by_fpr_tau10"] = sens["fpr_tau10"].rank(ascending=True, method="min").astype(int)
    sens = sens[["nu", "gamma", "gamma_effective", "admissible", "selected", "rank_by_auc", "rank_by_fpr_tau5", "rank_by_fpr_tau10", "fpr_tau5", "fpr_tau10", "auc",
                 "test_fnr_tau5", "test_fnr_tau10", "tau5", "tau10", "ref_accept_tau5", "note"]]
    sens.to_csv(st.results_dir / "grid_attack_sensitivity.csv", index=False)
    sel = sens[sens["selected"]].iloc[0]
    refit_ok = bool(np.isclose(sel["tau5"], model.thresholds["tau5"], rtol=1e-6, atol=1e-9))
    if not refit_ok:
        log.warning("refitted selected config gives tau5=%.6f but the frozen model has %.6f", sel["tau5"], model.thresholds["tau5"])
    best = sens.sort_values("auc", ascending=False).iloc[0]
    log.info("[%s] selected config ranks %d/%d by AUC (%.4f; best %.4f at nu=%s gamma=%s), %d/%d by FPR@tau5",
             SENSITIVITY_NOTE, sel["rank_by_auc"], len(sens), sel["auc"], best["auc"], best["nu"], best["gamma"], sel["rank_by_fpr_tau5"], len(sens))
    return {
        "note": SENSITIVITY_NOTE, "n_configs": int(len(sens)), "refit_matches_frozen": refit_ok,
        "selected": {k: (v.item() if hasattr(v, "item") else v) for k, v in sel.items() if k not in ("note", "selected", "admissible")},
        "auc_min": float(sens["auc"].min()), "auc_median": float(sens["auc"].median()), "auc_max": float(sens["auc"].max()),
        "fpr_tau5_min": float(sens["fpr_tau5"].min()), "fpr_tau5_median": float(sens["fpr_tau5"].median()), "fpr_tau5_max": float(sens["fpr_tau5"].max()),
        "best_by_auc": {"nu": float(best["nu"]), "gamma": str(best["gamma"]), "auc": float(best["auc"])},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    log = setup_logging(st.out_dir, "07_a1_sanity_check")
    if read_json(st.results_dir / "selection.json") is None:
        sys.exit("selection.json not found: freeze the model with 05_select_hyperparams.py before running the attack.")
    model = LivenessModel.load(st.model_dir)
    if model.config_hash != st.config_hash:
        sys.exit("config hash differs from the frozen model's: the protocol config changed after selection.")

    test = pd.read_parquet(st.interim_dir / "features_test.parquet").sort_values("session_id").reset_index(drop=True)
    index = pd.read_parquet(st.interim_dir / "session_index.parquet", columns=["session_id", "SENTENCE"]).set_index("session_id")["SENTENCE"]
    sentences = index.loc[test["session_id"]]
    attacker = A1Gaussian(A1Params.from_config(st.cfg["a1"]))
    bots = build_bot_features(sentences, attacker, FeatureParams.from_config(st.cfg["features"]), st.seed)
    finite = np.isfinite(bots[FEATURE_NAMES].to_numpy(dtype="float64")).all(axis=1)
    n_bad = int((~finite).sum())
    bots = bots.loc[finite]

    h_scores, b_scores = model.score(test), model.score(bots)
    auc = roc_auc(h_scores, b_scores)
    out: dict = {"n_human_sessions": int(len(test)), "n_bot_sessions": int(len(bots)), "bot_sessions_dropped_nonfinite": n_bad,
                 "auc": auc, "convention": "FNR = humans rejected, FPR = bots accepted; accept iff score >= tau"}
    for key, tau in model.thresholds.items():
        cm = confusion_matrix(h_scores, b_scores, tau)
        out[key] = cm
        log.info("%s=%.4f  FNR(humans rejected)=%.2f%%  FPR(bots accepted)=%.2f%%", key, tau, 100 * cm["FNR_humans_rejected"], 100 * cm["FPR_bots_accepted"])
    log.info("ROC-AUC (human vs A1) = %.4f", auc)
    chars = "".join(sentences.tolist())
    n_unmapped = sum(char_to_key(c)[0] == UNMAPPED_KEYCODE for c in chars)
    out["notes"] = [f"{n_unmapped:,} of {len(chars):,} sentence characters ({100 * n_unmapped / max(len(chars), 1):.3f}%) have no keycode mapping and were emitted with keycode {UNMAPPED_KEYCODE}."]
    out["bot_feature_medians"] = {k: float(bots[k].median()) for k in FEATURE_NAMES}
    out["human_feature_medians"] = {k: float(test[k].median()) for k in FEATURE_NAMES}
    out["grid_sensitivity"] = run_sensitivity(st, model, test, bots, log)
    write_json(st.results_dir / "a1_results.json", out)
    plot_roc(h_scores, b_scores, st.results_dir / "a1_roc.png", f"Human vs A1 (AUC = {auc:.3f})")
    write_methods_md(st)


if __name__ == "__main__":
    main()
