#!/usr/bin/env python
"""Step 4 — session features (one code path for humans and bots) -> <out>/interim/features_{split}.parquet.

Also writes FEATURES.md and <out>/results/features_report.json. Sessions with a non-finite feature
are dropped and counted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from tqdm import tqdm  # noqa: E402

from kl.config import load_settings, read_json, setup_logging, write_json  # noqa: E402
from kl.features import FEATURE_NAMES, FeatureParams, extract_features_frame  # noqa: E402
from kl.report import write_features_md, write_methods_md  # noqa: E402
from kl.split import SPLITS  # noqa: E402

READ_COLS = ["session_id", "PARTICIPANT_ID", "SENTENCE", "USER_INPUT", "PRESS_TIME", "RELEASE_TIME", "KEYSTROKE_ID", "KEYCODE", "LAYOUT"]


def _work(path: str, row_group: int, split_of: dict[str, str], params: FeatureParams) -> pd.DataFrame:
    """Extract features for the manifest sessions of one row group."""
    df = pq.ParquetFile(path).read_row_group(row_group, columns=READ_COLS).to_pandas()
    df = df[df["session_id"].isin(split_of)]
    out = extract_features_frame(df, params)
    out.insert(2, "split", out["session_id"].map(split_of))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    log = setup_logging(st.out_dir, "04_extract_features")
    manifest = read_json(st.results_dir / "split_manifest.json")
    split_of = {sid: s for s in SPLITS for sid in manifest["sessions"][s]}
    params = FeatureParams.from_config(st.cfg["features"])
    path = str(st.interim_dir / "sessions.parquet")
    n_groups = pq.ParquetFile(path).num_row_groups
    parts = Parallel(n_jobs=st.n_jobs)(delayed(_work)(path, i, split_of, params) for i in tqdm(range(n_groups), desc="features"))
    feats = pd.concat([p for p in parts if len(p)], ignore_index=True)
    dup = int(feats["session_id"].duplicated().sum())
    if dup:
        raise RuntimeError(f"{dup} sessions were split across parquet row groups")
    finite = np.isfinite(feats[FEATURE_NAMES].to_numpy(dtype="float64")).all(axis=1)
    dropped = feats.loc[~finite]
    feats = feats.loc[finite]
    report = {"n_features": len(FEATURE_NAMES), "feature_names": FEATURE_NAMES,
              "dropped_nonfinite_total": int((~finite).sum()),
              "dropped_nonfinite_by_split": dropped["split"].value_counts().to_dict(),
              "sessions_per_split": {}, "users_per_split": {}}
    train = feats[feats["split"] == "train"]
    cv_diag = {c: {"min": float(train[c].min()), "median": float(train[c].median()), "max": float(train[c].max())}
               for c in FEATURE_NAMES if c.endswith("_cv")}
    report["cv_diagnostics_train"] = cv_diag
    warn_at = float(st.cfg["features"]["cv_warn_threshold"])
    for c, d in cv_diag.items():
        flag = "  <-- exceeds %g: candidate for removal" % warn_at if d["max"] > warn_at else ""
        (log.warning if flag else log.info)("train %s: min %.4g  median %.4g  max %.4g%s", c, d["min"], d["median"], d["max"], flag)
    for s in SPLITS:
        part = feats[feats["split"] == s].sort_values("session_id").reset_index(drop=True)
        part.to_parquet(st.interim_dir / f"features_{s}.parquet", index=False)
        report["sessions_per_split"][s] = int(len(part))
        report["users_per_split"][s] = int(part["user_id"].nunique())
        log.info("features_%s: %d sessions, %d users", s, len(part), part["user_id"].nunique())
    write_json(st.results_dir / "features_report.json", report)
    write_features_md(st.out_dir / "FEATURES.md")
    write_methods_md(st)
    log.info("dropped for non-finite features: %d", report["dropped_nonfinite_total"])


if __name__ == "__main__":
    main()
