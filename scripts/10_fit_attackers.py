#!/usr/bin/env python
"""Phase 2, step 10 — fit attacker statistics on attacker_pool FIT users (Phase 1 must be frozen).

Reads the attacker_pool sessions of <out>/interim/sessions.parquet, splits the pool's USERS into FIT (parameters of
A2-A5) and HELD-OUT (realism report only), fits :class:`kl.attacks.pool.PoolStats` on the FIT sessions and writes
<out>/phase2/{pool.joblib, pool_split.json, pool_stats_summary.json, digraph_table.csv, ATTACKS.md}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from kl.attacks.pool import SessionBank, fit_pool_stats  # noqa: E402
from kl.config import load_settings, read_json, setup_logging, write_json  # noqa: E402
from kl.phase2.config import load_phase2_config, phase2_dir, pool_params  # noqa: E402
from kl.phase2.guard import assert_frozen  # noqa: E402
from kl.phase2.report import write_attacks_md  # noqa: E402

ROW_COLS = ["session_id", "PARTICIPANT_ID", "SENTENCE", "USER_INPUT", "KEYSTROKE_ID", "PRESS_TIME", "RELEASE_TIME", "LETTER", "KEYCODE"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    log = setup_logging(st.out_dir, "10_fit_attackers")
    assert_frozen(st)
    p2, out = load_phase2_config(), phase2_dir(st)

    manifest = read_json(st.results_dir / "split_manifest.json")
    pool_sessions = set(manifest["sessions"]["attacker_pool"])
    pool_users = sorted(manifest["users"]["attacker_pool"])
    src = pq.ParquetFile(st.interim_dir / "sessions.parquet")
    rows, meta = [], []
    for i in range(src.num_row_groups):
        df = src.read_row_group(i, columns=ROW_COLS).to_pandas()
        df = df[df["session_id"].isin(pool_sessions)]
        if len(df):
            rows.append(df[["session_id", "KEYSTROKE_ID", "PRESS_TIME", "RELEASE_TIME", "LETTER", "KEYCODE"]])
            meta.append(df.drop_duplicates("session_id")[["session_id", "PARTICIPANT_ID", "SENTENCE", "USER_INPUT"]])
    bank = SessionBank.from_frames(pd.concat(rows, ignore_index=True), pd.concat(meta, ignore_index=True))
    assert set(bank.user_ids.tolist()) <= set(pool_users), "non attacker_pool user found in the pool bank"
    log.info("attacker_pool bank: %d users, %d sessions, %d keystroke rows", len(set(bank.user_ids.tolist())), bank.n_sessions, len(bank.press))

    rng = np.random.default_rng([st.seed, 7])
    users = np.array(sorted(set(bank.user_ids.tolist())))
    n_hold = int(round(float(p2["pool"]["holdout_user_fraction"]) * len(users)))
    heldout_users = set(int(u) for u in rng.permutation(users)[:n_hold])
    is_held = np.array([int(u) in heldout_users for u in bank.user_ids])
    fit_idx, held_idx = np.flatnonzero(~is_held), np.flatnonzero(is_held)
    params = pool_params(st, p2)
    stats = fit_pool_stats(bank, fit_idx, params)
    desc = stats.describe()
    split_info = {"n_pool_users": len(users), "n_pool_sessions": bank.n_sessions, "n_fit_users": len(users) - n_hold, "n_fit_sessions": len(fit_idx),
                  "n_heldout_users": n_hold, "n_heldout_sessions": len(held_idx)}
    log.info("split: %s", split_info)
    log.info("A3: phi=%.4f sigma=%.4f hold_slope=%.4f; A4: delay_p=%s, mean bs/event=%.3f", stats.phi, stats.sigma, stats.hold_slope,
             np.round(stats.delay_p, 3).tolist(), stats.mean_bs_per_event)

    joblib.dump({"bank": bank, "fit_idx": fit_idx, "heldout_idx": held_idx, "stats": stats, "pool_users": [int(u) for u in users]}, out / "pool.joblib")
    write_json(out / "pool_split.json", {"fit_users": sorted(int(u) for u in users if int(u) not in heldout_users), "heldout_users": sorted(heldout_users), **split_info})
    write_json(out / "pool_stats_summary.json", desc)
    stats.digraph_table().to_csv(out / "digraph_table.csv", index=False)
    write_attacks_md(st, p2, desc, split_info, out / "ATTACKS.md")
    assert_frozen(st)
    log.info("wrote %s", out / "ATTACKS.md")


if __name__ == "__main__":
    main()
