#!/usr/bin/env python
"""Step 5 — attack-free hyperparameter selection; freezes the model.

Revision 2 (FINAL) rule: among ADMISSIBLE configs (|val_fnr_native - nu| <= max(0.02, 0.5 nu)) pick the lowest
acceptance of the product-of-marginals reference set at tau5 (ties: smaller nu, then smaller gamma).

Reads features_train / features_val only. Writes <out>/results/grid_results.csv,
<out>/results/selection.json and <out>/model/{model.joblib,thresholds.json}.
If reference acceptance does not discriminate across the grid (spread < min_reference_spread) or no config is
admissible, nothing is selected: grid_results.csv and selection_stopped.json are written, stale frozen outputs are
removed, and the script exits with status 2 (the tolerance is never relaxed automatically).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from kl.config import load_settings, setup_logging, write_json  # noqa: E402
from kl.features import feature_columns  # noqa: E402
from kl.model import LivenessModel  # noqa: E402
from kl.report import write_methods_md  # noqa: E402
from kl.selection import (DegenerateSelectionError, NoAdmissibleConfigError, admissible_mask,  # noqa: E402
                          run_grid, select_best, tau_key)

CSV_COLUMNS = ["nu", "gamma", "fit_time", "val_fnr_native", "tau5", "tau10",
               "ref_accept_tau5", "ref_accept_tau10",            # selection criterion
               "uniform_accept_tau5", "uniform_accept_tau10",    # diagnostic only (old Tax & Duin box), not used to select
               "n_support_vectors", "gamma_effective", "admissible", "selected"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    log = setup_logging(st.out_dir, "05_select_hyperparams")
    mcfg, scfg = st.cfg["model"], st.cfg["selection"]
    cols = feature_columns(st.cfg["features"]["groups"])
    train = pd.read_parquet(st.interim_dir / "features_train.parquet")
    val = pd.read_parquet(st.interim_dir / "features_val.parquet")
    log.info("train %d sessions, val %d sessions, %d features", len(train), len(val), len(cols))

    transform, results = run_grid(
        train, val, cols, mcfg["nu_grid"], mcfg["gamma_grid"], mcfg["target_fnrs"],
        int(scfg["n_reference_points"]), int(scfg["n_uniform_outliers"]), st.seed, st.n_jobs,
    )
    grid = pd.DataFrame([r.record for r in results])
    adm_cfg = scfg["admissibility"]
    grid["admissible"] = admissible_mask(grid, float(adm_cfg["abs_tol"]), float(adm_cfg["rel_tol"]))
    grid["selected"] = False
    ref = grid["ref_accept_tau5"]
    log.info("admissible configurations: %d of %d (|val_fnr_native - nu| <= max(%g, %g*nu))", int(grid["admissible"].sum()), len(grid),
             adm_cfg["abs_tol"], adm_cfg["rel_tol"])
    log.info("reference acceptance at tau5 across the grid: min %.3f%%  max %.3f%%  spread %.3f pp",
             100 * ref.min(), 100 * ref.max(), 100 * (ref.max() - ref.min()))
    log.info("(diagnostic) uniform-box acceptance at tau5: min %.4f%%  max %.4f%%",
             100 * grid["uniform_accept_tau5"].min(), 100 * grid["uniform_accept_tau5"].max())
    try:
        pos = select_best(grid, float(scfg["tie_tolerance"]), float(scfg["min_reference_spread"]), primary=tau_key(mcfg["target_fnrs"][0]),
                          abs_tol=float(adm_cfg["abs_tol"]), rel_tol=float(adm_cfg["rel_tol"]))
    except (DegenerateSelectionError, NoAdmissibleConfigError) as exc:
        grid[CSV_COLUMNS].to_csv(st.results_dir / "grid_results.csv", index=False)
        write_json(st.results_dir / "selection_stopped.json", {
            "reason": "degenerate" if isinstance(exc, DegenerateSelectionError) else "no_admissible_config",
            "message": str(exc), "grid_size": int(len(grid)), "n_admissible": int(grid["admissible"].sum()),
            "min_ref_accept_tau5": float(ref.min()), "max_ref_accept_tau5": float(ref.max()),
        })
        for stale in (st.results_dir / "selection.json", st.model_dir / "model.joblib", st.model_dir / "thresholds.json"):
            stale.unlink(missing_ok=True)
        write_methods_md(st)
        log.error("%s", exc)
        sys.exit(2)
    grid.loc[grid.index[pos], "selected"] = True
    grid[CSV_COLUMNS].to_csv(st.results_dir / "grid_results.csv", index=False)
    (st.results_dir / "selection_stopped.json").unlink(missing_ok=True)

    chosen = grid.iloc[pos]
    adm_ref = ref[grid["admissible"]]
    best_val = float(adm_ref.min())
    n_tied = int((adm_ref <= best_val + float(scfg["tie_tolerance"])).sum())
    thresholds = {tau_key(f): float(chosen[tau_key(f)]) for f in mcfg["target_fnrs"]}
    model = LivenessModel(
        transform=transform, svm=results[pos].svm, feature_names=cols, thresholds=thresholds,
        params={"nu": float(chosen["nu"]), "gamma": str(chosen["gamma"]), "gamma_effective": float(chosen["gamma_effective"])},
        config_hash=st.config_hash,
    )
    model.save(st.model_dir)
    write_json(st.results_dir / "selection.json", {
        "chosen": {k: (v.item() if hasattr(v, "item") else v) for k, v in chosen.drop("selected").items()},
        "n_admissible": int(grid["admissible"].sum()),
        "admissibility_rule": f"|val_fnr_native - nu| <= max({adm_cfg['abs_tol']}, {adm_cfg['rel_tol']} * nu)",
        "not_admissible": [{"nu": float(r.nu), "gamma": str(r.gamma), "val_fnr_native": float(r.val_fnr_native)}
                           for r in grid[~grid["admissible"]].itertuples()],
        "min_ref_accept_tau5": best_val,                      # over admissible configs
        "grid_min_ref_accept_tau5": float(ref.min()),         # over the whole grid
        "max_ref_accept_tau5": float(ref.max()),
        "ref_accept_spread_tau5": float(ref.max() - ref.min()),
        "n_configs_tied_with_best": n_tied,
        "uniform_diagnostic": {"min_tau5": float(grid["uniform_accept_tau5"].min()), "max_tau5": float(grid["uniform_accept_tau5"].max()),
                               "n_configs_zero_tau5": int((grid["uniform_accept_tau5"] == 0).sum())},
        "grid_size": int(len(grid)), "n_train": int(len(train)), "n_val": int(len(val)),
        "n_features": len(cols), "feature_cols": cols, "n_reference": int(scfg["n_reference_points"]),
        "n_uniform": int(scfg["n_uniform_outliers"]),
        "config_hash": st.config_hash,
    })
    write_methods_md(st)
    log.info("chosen: nu=%s gamma=%s (eff %.4g); ref_accept@tau5=%.3f%%; %d admissible configs tied within %.2g of the best",
             chosen["nu"], chosen["gamma"], chosen["gamma_effective"], 100 * chosen["ref_accept_tau5"], n_tied, scfg["tie_tolerance"])
    log.info("thresholds: %s", thresholds)


if __name__ == "__main__":
    main()
