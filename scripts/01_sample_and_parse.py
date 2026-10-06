#!/usr/bin/env python
"""Step 1 — keyboard-type eligibility, seeded user sampling, robust parsing.

Writes  <out>/interim/keystrokes.parquet  (one row group per chunk of participants),
        <out>/interim/sampled_participants.csv,
        <out>/results/cleaning_report.json  (section ``step1_sampling``).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from tqdm import tqdm  # noqa: E402

from kl.config import load_settings, setup_logging, update_json_section  # noqa: E402
from kl.io_aalto import (KEYSTROKE_SCHEMA, eligibility, list_participant_files,  # noqa: E402
                         load_metadata, parse_chunk)
from kl.report import write_methods_md  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true", help="small run (sampling.n_users overridden by quick_overrides)")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    log = setup_logging(st.out_dir, "01_sample_and_parse")
    scfg = st.cfg["sampling"]

    files = list_participant_files(st.raw_dir, st.cfg["paths"]["participant_glob"])
    log.info("participant files found: %d in %s", len(files), st.raw_dir)
    meta = load_metadata(st.metadata_path)
    log.info("metadata rows: %d (raw %s)", len(meta), meta.attrs.get("rows_raw"))
    eligible, rep = eligibility(meta, files.keys(), scfg["keyboard_types_include"], scfg["keyboard_types_exclude"])
    log.info("unique KEYBOARD_TYPE values: %s", rep["keyboard_type_counts"])
    log.info("unique LAYOUT values: %s", meta["LAYOUT"].value_counts().to_dict())
    if rep["unmapped_values"]:
        log.warning("KEYBOARD_TYPE values in neither include nor exclude (excluded!): %s", rep["unmapped_values"])
    log.info("eligible: %d, eligible with file: %d", rep["eligible"], rep["eligible_with_file"])

    ids = np.sort(eligible["PARTICIPANT_ID"].to_numpy())
    n = min(int(scfg["n_users"]), len(ids))
    rng = np.random.default_rng(st.seed)
    sampled = np.sort(rng.choice(ids, size=n, replace=False))
    layout = dict(zip(eligible["PARTICIPANT_ID"], eligible["LAYOUT"]))
    ktype = dict(zip(eligible["PARTICIPANT_ID"], eligible["KEYBOARD_TYPE"]))
    pd.DataFrame({"PARTICIPANT_ID": sampled, "LAYOUT": [layout[i] for i in sampled],
                  "KEYBOARD_TYPE": [ktype[i] for i in sampled]}).to_csv(st.interim_dir / "sampled_participants.csv", index=False)
    log.info("sampled %d users (seed %d)", n, st.seed)

    chunk = int(scfg["parse_chunk_files"])
    items = [(int(i), str(files[int(i)]), layout[i]) for i in sampled]
    chunks = [items[k:k + chunk] for k in range(0, len(items), chunk)]
    out_path = st.interim_dir / "keystrokes.parquet"
    tot = Counter()
    failed: list[dict] = []
    enc = Counter()
    with pq.ParquetWriter(out_path, KEYSTROKE_SCHEMA) as writer:
        gen = Parallel(n_jobs=st.n_jobs, return_as="generator")(delayed(parse_chunk)(c) for c in chunks)
        for table, stats in tqdm(gen, total=len(chunks), desc="parse"):
            if table is not None:
                writer.write_table(table, row_group_size=max(table.num_rows, 1))
            for s in stats:
                tot["rows_read"] += s["rows_read"]
                tot["lines_not_parsed"] += s["lines_not_parsed"]
                tot["rows_bad_ids"] += s["rows_bad_ids"]
                enc[s["encoding"] or "none"] += 1
                if s["error"]:
                    failed.append({"file": s["file"], "error": s["error"]})
                    log.warning("failed file %s: %s", s["file"], s["error"])
    section = {
        "n_users_requested": int(scfg["n_users"]),
        **rep,
        "sampled": int(n),
        "files_failed": len(failed),
        "failed_files_examples": failed[:50],
        "rows_read": int(tot["rows_read"]),
        "lines_not_parsed_approx": int(tot["lines_not_parsed"]),
        "rows_dropped_bad_ids": int(tot["rows_bad_ids"]),
        "files_by_encoding": dict(enc),
        "keystrokes_parquet": str(out_path),
    }
    update_json_section(st.results_dir / "cleaning_report.json", "step1_sampling", section)
    write_methods_md(st)
    log.info("done: %d files parsed, %d failed, %d rows -> %s", len(items) - len(failed), len(failed), tot["rows_read"], out_path)


if __name__ == "__main__":
    main()
