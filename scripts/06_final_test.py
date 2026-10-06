#!/usr/bin/env python
"""Step 6 — final human test. Touches the test split ONCE.

Reports test FNR (humans rejected) at tau5 / tau10 with user-level bootstrap CIs and a score
histogram. Refuses to run twice (guard = existing test_results.json) unless --allow-rerun is given,
in which case the result is flagged ``"rerun": true``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kl.config import load_settings, read_json, setup_logging, write_json  # noqa: E402
from kl.evaluate import bootstrap_reject_rate_by_user, plot_score_hist  # noqa: E402
from kl.model import LivenessModel  # noqa: E402
from kl.report import write_methods_md  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--allow-rerun", action="store_true", help="evaluate the test split again (recorded in the result)")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    log = setup_logging(st.out_dir, "06_final_test")
    result_path = st.results_dir / "test_results.json"
    if result_path.exists() and not args.allow_rerun:
        sys.exit(f"The test split was already evaluated ({result_path}). Refusing to touch it again; use --allow-rerun to override.")
    if read_json(st.results_dir / "selection.json") is None:
        sys.exit("selection.json not found: run 05_select_hyperparams.py first (the model must be frozen before testing).")
    model = LivenessModel.load(st.model_dir)
    if model.config_hash != st.config_hash:
        sys.exit(f"config hash {st.config_hash} differs from the frozen model's {model.config_hash}: the protocol config changed after selection.")

    test = pd.read_parquet(st.interim_dir / "features_test.parquet")
    assert set(test["split"].unique()) == {"test"}
    scores = model.score(test)
    ev = st.cfg["evaluation"]
    out: dict = {"rerun": bool(args.allow_rerun and result_path.exists()), "n_sessions": int(len(test)),
                 "n_users": int(test["user_id"].nunique()), "thresholds": model.thresholds}
    for i, (key, tau) in enumerate(model.thresholds.items()):
        res = bootstrap_reject_rate_by_user(scores, test["user_id"].to_numpy(), tau, int(ev["bootstrap_reps"]), st.seed + i, float(ev["ci_level"]))
        res["tau"] = tau
        out[key] = res
        log.info("%s = %.4f: test FNR (humans rejected) = %.2f%%  [%.2f%%, %.2f%%]", key, tau, 100 * res["fnr"], 100 * res["ci_lo"], 100 * res["ci_hi"])
    out["score_quantiles"] = {q: float(np.quantile(scores, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}
    write_json(result_path, out)
    pd.DataFrame({"session_id": test["session_id"], "user_id": test["user_id"], "score": scores}).to_parquet(st.interim_dir / "test_scores.parquet", index=False)
    plot_score_hist(scores, model.thresholds, st.results_dir / "test_score_hist.png", "Human test sessions")
    write_methods_md(st)


if __name__ == "__main__":
    main()
