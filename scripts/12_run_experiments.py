#!/usr/bin/env python
"""Phase 2, step 12 — detectors D1/D2/D3 and experiments E1 (open set), E2 (cross-attack matrix), E3 (leave-one-attack-out).

D1 = the FROZEN Phase 1 OCSVM (never refitted). D2 = Isolation Forest, D3 = Random Forest with the pre-declared
hyper-parameters of config_phase2.yaml, same input transform, thresholds tau5/tau10 from VALIDATION HUMANS only.
No attacker is modified here; results never feed back into generation.

Writes <out>/phase2/{open_set_table,attack_matrix,loao_table}.csv, heatmap_apcer_tau10.png, heatmap_auc.png,
roc_d1_per_attacker.png, detectors_summary.json, experiments_meta.json.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from kl.config import load_settings, setup_logging, write_json  # noqa: E402
from kl.features import FEATURE_NAMES  # noqa: E402
from kl.phase2.config import load_phase2_config, phase2_dir  # noqa: E402
from kl.phase2.detectors import Detector, d1_from_model, fit_d2, fit_d3  # noqa: E402
from kl.phase2.evaluation import cell_row, plot_heatmap, plot_roc_per_attacker, user_weights  # noqa: E402
from kl.phase2.guard import assert_frozen  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    log = setup_logging(st.out_dir, "12_run_experiments")
    model = assert_frozen(st)
    p2, out = load_phase2_config(), phase2_dir(st)
    ev = st.cfg["evaluation"]
    reps, ci = int(ev["bootstrap_reps"]), float(ev["ci_level"])
    tests, train_atk, loao_atk = p2["experiments"]["test_attackers"], p2["experiments"]["train_attackers"], p2["experiments"]["loao_attackers"]
    cols = model.feature_names
    t0 = time.time()

    train = pd.read_parquet(st.interim_dir / "features_train.parquet")
    val = pd.read_parquet(st.interim_dir / "features_val.parquet")
    test = pd.read_parquet(st.interim_dir / "features_test.parquet")
    bots_tr = {a: pd.read_parquet(out / f"bots_{a}_train.parquet") for a in set(train_atk)}
    bots_te = {a: pd.read_parquet(out / f"bots_{a}_test.parquet") for a in tests}
    users = np.sort(test["user_id"].unique())
    idx_of = pd.Index(users).get_indexer
    h_users = idx_of(test["user_id"])
    b_users = {a: idx_of(b["user_id"]) for a, b in bots_te.items()}
    assert (h_users >= 0).all() and all((u >= 0).all() for u in b_users.values()), "bot matched to a non-test user"
    weights = user_weights(len(users), reps, st.seed)
    log.info("test: %d human sessions, %d users; bots per attacker: %s", len(test), len(users), {a: len(b) for a, b in bots_te.items()})

    # ---- detectors ------------------------------------------------------------------------------------------
    d1 = d1_from_model(model)
    d2 = fit_d2(train, model.transform, cols, st.seed, p2["detectors"]["d2"], val)
    log.info("D1 (frozen) thresholds %s; D2 thresholds %s", d1.thresholds, d2.thresholds)
    summary = {"D1": {"thresholds": d1.thresholds, "description": d1.description}, "D2": {"thresholds": d2.thresholds, "description": d2.description}, "D3": {}}

    def scored(det: Detector) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return det.score(test), {a: det.score(b) for a, b in bots_te.items()}

    def cells(det: Detector, trained_on: str, attackers: list[str], sc: tuple) -> list:
        hs, bs = sc
        return [delayed(cell_row)(det.name, trained_on, a, hs, h_users, bs[a], b_users[a], det.thresholds, weights, ci) for a in attackers]

    jobs, roc_scores = [], {}
    sc1, sc2 = scored(d1), scored(d2)
    roc_scores = (sc1[0], sc1[1])
    jobs += cells(d1, "attack-free", tests, sc1) + cells(d2, "attack-free", tests, sc2)
    n_attack_free = len(jobs)

    def train_d3(train_on: list[str], name: str) -> Detector:
        bots = pd.concat([bots_tr[a] for a in train_on], ignore_index=True)
        t = time.time()
        det = fit_d3(train, bots, model.transform, cols, st.seed, p2["detectors"]["d3"], val, st.n_jobs, name)
        log.info("%s trained on %s (%d bots): thresholds %s (%.0fs)", name, train_on, len(bots), det.thresholds, time.time() - t)
        summary["D3"][name] = {"trained_on": train_on, "n_train_humans": len(train), "n_train_bots": len(bots), "thresholds": det.thresholds, "description": det.description}
        return det

    # E2: D3 trained on one attacker
    e2_labels = []
    for a in train_atk:
        det = train_d3([a], f"D3<-{a}")
        det.name = "D3"
        jobs += cells(det, a, tests, scored(det))
    # E3: D3 trained on all training attackers except A_j, tested on A_j
    loao_start = len(jobs)
    for a in loao_atk:
        det = train_d3([x for x in train_atk if x != a], f"D3<-all-but-{a}")
        det.name = "D3"
        jobs += cells(det, "all-but-" + a, [a], scored(det))
    log.info("evaluating %d cells with %d bootstrap resamples ...", len(jobs), reps)
    rows = Parallel(n_jobs=st.n_jobs)(jobs)
    df = pd.DataFrame(rows)
    open_set = df.iloc[:n_attack_free].reset_index(drop=True)
    matrix = df.iloc[:loao_start].reset_index(drop=True)                       # E2 (includes the D1 / D2 rows)
    loao_d3 = df.iloc[loao_start:].reset_index(drop=True)
    ref = open_set[open_set["tested_on"].isin(loao_atk)]                       # D1 / D2 on the same held-out attacker, for comparison
    loao = pd.concat([loao_d3, ref], ignore_index=True)
    open_set.to_csv(out / "open_set_table.csv", index=False)
    matrix.to_csv(out / "attack_matrix.csv", index=False)
    loao.to_csv(out / "loao_table.csv", index=False)

    # ---- figures ----------------------------------------------------------------------------------------------
    row_order = [("D1", "attack-free", "D1 (frozen OCSVM)"), ("D2", "attack-free", "D2 (Isolation Forest)")] + [("D3", a, f"D3 trained on {a}") for a in train_atk]
    for metric, fname, title, cbar, pct, vmin, vmax, cmap in (
        ("apcer_fpr_tau10", "heatmap_apcer_tau10.png", "APCER / FPR at τ10 (attacks accepted)", "APCER", True, 0.0, 1.0, "YlOrRd"),
        ("auc", "heatmap_auc.png", "ROC-AUC (human vs attack)", "AUC", False, 0.5, 1.0, "YlGnBu"),
    ):
        mat = pd.DataFrame({t: [matrix[(matrix.detector == d) & (matrix.trained_on == tr) & (matrix.tested_on == t)][metric].iloc[0] for d, tr, _ in row_order] for t in tests},
                           index=[lab for _, _, lab in row_order])
        plot_heatmap(mat, out / fname, title, cbar, pct, vmin, vmax, cmap)
    plot_roc_per_attacker(roc_scores[0], roc_scores[1], out / "roc_d1_per_attacker.png", "D1 (frozen OCSVM): human vs attacker, test split")

    write_json(out / "detectors_summary.json", summary)
    write_json(out / "experiments_meta.json", {
        "seed": st.seed, "config_hash": st.config_hash, "bootstrap_reps": reps, "ci_level": ci, "n_test_human_sessions": len(test), "n_test_users": len(users),
        "n_train_human_sessions": len(train), "n_val_human_sessions": len(val), "n_bots_test": {a: len(b) for a, b in bots_te.items()},
        "n_bots_train": {a: len(b) for a, b in bots_tr.items()}, "d1_frozen": True, "runtime_seconds": time.time() - t0,
        "features": FEATURE_NAMES,
    })
    assert_frozen(st)
    log.info("done in %.0fs; wrote tables and figures to %s", time.time() - t0, out)


if __name__ == "__main__":
    main()
