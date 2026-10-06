#!/usr/bin/env python
"""Phase 2, step 13 — RESULTS_PHASE2.md (paper-style paragraph + tables) from the experiment CSVs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kl.config import load_settings, setup_logging  # noqa: E402
from kl.phase2.config import load_phase2_config, phase2_dir  # noqa: E402
from kl.phase2.guard import assert_frozen  # noqa: E402
from kl.phase2.report import write_results_md  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    log = setup_logging(st.out_dir, "13_report_phase2")
    assert_frozen(st)
    out = phase2_dir(st)
    write_results_md(st, load_phase2_config(), out)
    log.info("wrote %s", out / "RESULTS_PHASE2.md")


if __name__ == "__main__":
    main()
