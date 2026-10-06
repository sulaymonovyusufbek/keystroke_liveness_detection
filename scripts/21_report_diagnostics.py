#!/usr/bin/env python
"""Phase 3a — DIAGNOSTICS.md (paper-style paragraph per analysis) from the CSV / JSON outputs of 20_run_diagnostics.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from kl.config import ROOT, load_settings, setup_logging  # noqa: E402
from kl.phase2.config import load_phase2_config  # noqa: E402
from kl.phase3a.guard import assert_frozen_p12  # noqa: E402
from kl.phase3a.report import write_diagnostics_md  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    log = setup_logging(st.out_dir, "21_report_diagnostics")
    assert_frozen_p12(st)
    out = st.out_dir / "phase3a"
    write_diagnostics_md(st, load_phase2_config(), yaml.safe_load((ROOT / "config_phase3a.yaml").read_text(encoding="utf-8")), out)
    log.info("wrote %s", out / "DIAGNOSTICS.md")


if __name__ == "__main__":
    main()
