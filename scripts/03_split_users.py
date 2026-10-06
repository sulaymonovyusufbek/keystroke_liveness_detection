#!/usr/bin/env python
"""Step 3 — user-level train/val/test/attacker_pool split + manifest (<out>/results/split_manifest.json)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from kl.config import load_settings, setup_logging, write_json  # noqa: E402
from kl.report import write_methods_md  # noqa: E402
from kl.split import build_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    log = setup_logging(st.out_dir, "03_split_users")
    index = pd.read_parquet(st.interim_dir / "session_index.parquet", columns=["session_id", "PARTICIPANT_ID"])
    scfg = st.cfg["split"]
    manifest = build_manifest(index, scfg["fractions"], scfg["max_train_sessions"], st.seed)
    write_json(st.results_dir / "split_manifest.json", manifest)
    write_methods_md(st)
    log.info("users:    %s", manifest["counts"]["users"])
    log.info("sessions: %s (train before cap: %d)", manifest["counts"]["sessions"], manifest["counts"]["train_sessions_before_cap"])


if __name__ == "__main__":
    main()
