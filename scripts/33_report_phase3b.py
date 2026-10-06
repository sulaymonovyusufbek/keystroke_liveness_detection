#!/usr/bin/env python
"""Phase 3b, step 33 — figures and RESULTS_PHASE3B.md from whatever parts (E4 / E5 / E6) have finished.

Reads only the Phase 3b tables, score files and logs; trains and evaluates nothing, so it can run after every part.
Writes <out>/phase3b/{training_curves_d4.png, training_curves_d5_e5.png, training_curves_d5_e6.png,
d4_primary_score_hist.png, e5_heatmaps.png, RESULTS_PHASE3B.md}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from kl.config import load_settings, setup_logging  # noqa: E402
from kl.phase3b.config import load_phase3b_config, phase3b_dir  # noqa: E402
from kl.phase3b.guard import assert_frozen_p123  # noqa: E402
from kl.phase3b.report import plot_d4_hist, plot_e5_heatmaps, plot_training_curves, write_results_md  # noqa: E402
from kl.phase3b.runner import load_scores, score_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    log = setup_logging(st.out_dir, "33_report_phase3b")
    assert_frozen_p123(st)
    p3, out = load_phase3b_config(), phase3b_dir(st)
    attackers = list(p3["experiments"]["test_attackers"])

    cpath = out / "training_curves.csv"
    if cpath.exists():
        curves = pd.read_csv(cpath)
        d4 = curves[curves["detector"] != "D5"]
        plot_training_curves(d4, out / "training_curves_d4.png", "D4 variants: token-level mean NLL (train dashed, val solid; dot = best epoch)", "mean NLL")
        d5 = curves[curves["detector"] == "D5"]
        is_e6 = d5["trained_on"].astype(str).str.startswith("all-but-")
        plot_training_curves(d5[~is_e6], out / "training_curves_d5_e5.png", "D5 (E5, one training attacker): balanced BCE (train dashed, val solid)", "BCE")
        plot_training_curves(d5[is_e6], out / "training_curves_d5_e6.png", "D5 (E6, leave one attacker out): balanced BCE (train dashed, val solid)", "BCE")
    s42 = score_path(out, "D4/seed42")
    if s42.exists():
        plot_d4_hist(load_scores(s42), attackers, out / "d4_primary_score_hist.png", "D4-primary (seed 42): mean per-step NLL, test humans vs each attacker")
    if (out / "e5_cross_attack.csv").exists():
        e5 = pd.read_csv(out / "e5_cross_attack.csv")
        if (e5["detector"] == "D5").any():
            plot_e5_heatmaps(e5, out / "e5_heatmaps.png")
    write_results_md(st, out)
    assert_frozen_p123(st)
    log.info("wrote figures and RESULTS_PHASE3B.md to %s", out)


if __name__ == "__main__":
    main()
