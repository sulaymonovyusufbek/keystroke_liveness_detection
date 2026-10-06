#!/usr/bin/env python
"""Step 99 — replica of the legacy protocol (for the response letter only; NOT part of the clean pipeline).

Legacy protocol: fit AND "test" on the SAME human sessions, add one A1 bot per session, and select
(nu, gamma) by the best F1 on humans + A1 (bots = positive class, decision = OCSVM native boundary
``predict == -1``). Writes <out>/results/legacy_grid.csv, legacy_results.json and legacy_vs_clean.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from kl.attacks.a1_gaussian import A1Gaussian, A1Params  # noqa: E402
from kl.config import load_settings, read_json, setup_logging, write_json  # noqa: E402
from kl.evaluate import build_bot_features, roc_auc  # noqa: E402
from kl.features import FeatureParams, feature_columns  # noqa: E402
from kl.model import fit_ocsvm, fit_transform_on_train  # noqa: E402

def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    log = setup_logging(st.out_dir, "99_legacy_protocol")
    lcfg = st.cfg["legacy"]
    cols = feature_columns(st.cfg["features"]["groups"])

    humans = pd.read_parquet(st.interim_dir / "features_train.parquet")
    n = min(int(lcfg["n_sessions"]), len(humans))
    pick = np.sort(np.random.default_rng([st.seed, 4]).choice(len(humans), size=n, replace=False))
    humans = humans.iloc[pick].sort_values("session_id").reset_index(drop=True)
    index = pd.read_parquet(st.interim_dir / "session_index.parquet", columns=["session_id", "SENTENCE"]).set_index("session_id")["SENTENCE"]
    bots = build_bot_features(index.loc[humans["session_id"]], A1Gaussian(A1Params.from_config(st.cfg["a1"])), FeatureParams.from_config(st.cfg["features"]), st.seed + 100)
    bots = bots[np.isfinite(bots[cols].to_numpy(dtype="float64")).all(axis=1)]
    log.info("legacy sample: %d human sessions + %d A1 bot sessions (same sessions used to fit and to test)", len(humans), len(bots))

    xh_raw, xb_raw = humans[cols].to_numpy(dtype="float64"), bots[cols].to_numpy(dtype="float64")
    transform = fit_transform_on_train(xh_raw, cols)  # same features + transform as the clean pipeline: only the PROTOCOL differs
    xh, xb = transform.transform(xh_raw), transform.transform(xb_raw)
    y = np.r_[np.zeros(len(xh)), np.ones(len(xb))]  # bots = positive class
    rows = []
    for nu in lcfg["nu_grid"]:
        for gamma in lcfg["gamma_grid"]:
            svm, g_eff = fit_ocsvm(xh, nu, gamma)
            pred_h_out, pred_b_out = svm.predict(xh) == -1, svm.predict(xb) == -1
            pred = np.r_[pred_h_out, pred_b_out].astype(int)
            hs, bs = svm.decision_function(xh), svm.decision_function(xb)
            rows.append({"nu": nu, "gamma": str(gamma), "gamma_effective": g_eff, "f1_bot_positive": f1_score(y, pred),
                         "f1_human_positive": f1_score(1 - y, 1 - pred),
                         "FNR_humans_rejected": float(pred_h_out.mean()), "FPR_bots_accepted": float(1 - pred_b_out.mean()),
                         "auc": roc_auc(hs, bs), "n_support_vectors": int(svm.support_vectors_.shape[0])})
    grid = pd.DataFrame(rows)
    grid.to_csv(st.results_dir / "legacy_grid.csv", index=False)
    best = grid.sort_values("f1_bot_positive", ascending=False, kind="mergesort").iloc[0]
    log.info("legacy best: nu=%s gamma=%s F1=%.4f FNR=%.2f%% FPR=%.2f%% AUC=%.4f", best["nu"], best["gamma"], best["f1_bot_positive"],
             100 * best["FNR_humans_rejected"], 100 * best["FPR_bots_accepted"], best["auc"])
    legacy = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
    write_json(st.results_dir / "legacy_results.json", {"n_human_sessions": len(humans), "n_bot_sessions": len(bots), "best": legacy})

    test = read_json(st.results_dir / "test_results.json") or {}
    a1 = read_json(st.results_dir / "a1_results.json") or {}
    sel = read_json(st.results_dir / "selection.json") or {}
    t5, t10 = test.get("tau5", {}), test.get("tau10", {})
    a5, a10 = a1.get("tau5", {}), a1.get("tau10", {})
    ch = sel.get("chosen", {})
    md = [
        "# Legacy protocol vs clean protocol", "",
        f"_Run mode: **{'quick' if st.quick else 'full'}**. Legacy numbers are IN-SAMPLE by construction "
        f"(the same {len(humans):,} human sessions are used to fit the model and to test it, and the configuration is chosen by F1 on humans + A1)._", "",
        "| | legacy protocol | clean protocol |", "|---|---|---|",
        "| training humans | same sessions as the test | train split (disjoint users) |",
        "| test humans | same sessions as training | test split (disjoint users, touched once) |",
        "| hyperparameter selection | best F1 on humans + A1 (uses attack data) | attack-free: admissibility filter + product-of-marginals reference set (Revision 2) |",
        f"| chosen (nu, gamma) | ({legacy['nu']}, {legacy['gamma']}) | ({ch.get('nu', 'n/a')}, {ch.get('gamma', 'n/a')}) |",
        "| decision threshold | native OCSVM boundary (score >= 0) | tau5 / tau10 set on validation humans |",
        f"| FNR (humans rejected) | {_pct(legacy['FNR_humans_rejected'])} | {_pct(t5.get('fnr'))} @ tau5 [{_pct(t5.get('ci_lo'))}, {_pct(t5.get('ci_hi'))}]; {_pct(t10.get('fnr'))} @ tau10 |",
        f"| FPR (bots accepted) | {_pct(legacy['FPR_bots_accepted'])} | {_pct(a5.get('FPR_bots_accepted'))} @ tau5; {_pct(a10.get('FPR_bots_accepted'))} @ tau10 |",
        f"| ROC-AUC (human vs A1) | {legacy['auc']:.4f} | {a1.get('auc', float('nan')):.4f} |",
        f"| F1 (bots = positive) | {legacy['f1_bot_positive']:.4f} | not used for selection |", "",
        "Notes: the legacy replica uses the same features and input transform (signed log + RobustScaler) as the clean pipeline, so only the protocol differs. F1 uses bots as the positive class (assumption; F1 with humans positive is in legacy_grid.csv). "
        "If the clean columns show n/a, run steps 05-07 first.", "",
    ]
    (st.results_dir / "legacy_vs_clean.md").write_text("\n".join(md), encoding="utf-8")
    log.info("wrote %s", st.results_dir / "legacy_vs_clean.md")


if __name__ == "__main__":
    main()
