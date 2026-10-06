#!/usr/bin/env python
"""Step 2 — sessionize and clean.

Reads  <out>/interim/keystrokes.parquet ; writes <out>/interim/sessions.parquet (keystroke rows of
clean sessions + ``session_id``), <out>/interim/session_index.parquet (one row per clean session)
and the counts in <out>/results/cleaning_report.json (section ``step2_cleaning``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from tqdm import tqdm  # noqa: E402

from kl.config import load_settings, setup_logging, update_json_section  # noqa: E402
from kl.io_aalto import KEYSTROKE_SCHEMA  # noqa: E402
from kl.report import write_methods_md  # noqa: E402
from kl.sessions import CleaningParams, clean_sessions, empty_counts, merge_counts, session_index  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    log = setup_logging(st.out_dir, "02_clean_and_sessionize")
    params = CleaningParams.from_config(st.cfg["cleaning"])
    src = pq.ParquetFile(st.interim_dir / "keystrokes.parquet")
    schema = KEYSTROKE_SCHEMA.append(pa.field("session_id", pa.string()))
    counts = empty_counts()
    index_parts: list[pd.DataFrame] = []
    with pq.ParquetWriter(st.interim_dir / "sessions.parquet", schema) as writer:
        for i in tqdm(range(src.num_row_groups), desc="clean"):
            df = src.read_row_group(i).to_pandas()
            kept, c = clean_sessions(df, params)
            counts = merge_counts(counts, c)
            if len(kept):
                writer.write_table(pa.Table.from_pandas(kept, schema=schema, preserve_index=False), row_group_size=len(kept))
                index_parts.append(session_index(kept))
    index = pd.concat(index_parts, ignore_index=True) if index_parts else pd.DataFrame()
    index.to_parquet(st.interim_dir / "session_index.parquet", index=False)
    ordered = {k: counts[k] for k in empty_counts()}
    ordered["thresholds"] = st.cfg["cleaning"]
    update_json_section(st.results_dir / "cleaning_report.json", "step2_cleaning", ordered)
    write_methods_md(st)
    log.info("sessions: total %d, dropped %d, kept %d", counts["sessions_total"], counts["dropped_total"], counts["sessions_kept"])
    for k in ordered:
        if k.startswith("drop_"):
            log.info("  %-28s %d", k, ordered[k])


if __name__ == "__main__":
    main()
