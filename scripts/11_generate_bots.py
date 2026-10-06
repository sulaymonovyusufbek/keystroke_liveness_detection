#!/usr/bin/env python
"""Phase 2, step 11 — bot sessions for every attacker (one bot per matched human session) + realism report.

Bots for the train and test splits are featurised by the unchanged features.py and written to
<out>/phase2/bots_<attacker>_<split>.parquet. For the realism report every attacker also types the sentences of the
HELD-OUT attacker_pool sessions; each feature is compared with those sessions (KS statistic, mean difference) in
<out>/phase2/realism_report.csv. The realism report is documentation only and never feeds back into any attacker.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

from kl.config import load_settings, setup_logging, write_json  # noqa: E402
from kl.features import FEATURE_NAMES, FEATURE_GROUPS, FeatureParams  # noqa: E402
from kl.phase2.config import load_phase2_config, phase2_dir  # noqa: E402
from kl.phase2.generate import assert_pool_disjoint, build_attackers, generate_bot_features  # noqa: E402
from kl.phase2.guard import assert_frozen  # noqa: E402

GROUP_OF = {n: g for g, names in FEATURE_GROUPS.items() for n in names}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    log = setup_logging(st.out_dir, "11_generate_bots")
    assert_frozen(st)
    p2, out = load_phase2_config(), phase2_dir(st)
    pool = joblib.load(out / "pool.joblib")
    bank, stats = pool["bank"], pool["stats"]
    fparams = FeatureParams.from_config(st.cfg["features"])
    names = p2["bots"]["attackers"]
    attackers = build_attackers(stats, bank, pool["fit_idx"], st.cfg["a1"], names)
    sentences = pd.read_parquet(st.interim_dir / "session_index.parquet", columns=["session_id", "SENTENCE"]).set_index("session_id")["SENTENCE"]
    report: dict[str, dict] = {}
    t0 = time.time()

    for split in p2["bots"]["splits"]:
        feats = pd.read_parquet(st.interim_dir / f"features_{split}.parquet", columns=["session_id", "user_id"])
        matched = feats.assign(sentence=sentences.loc[feats["session_id"]].to_numpy())
        assert_pool_disjoint(pool["pool_users"], matched["user_id"].unique(), f"{split} bot generation")
        for name in names:
            t = time.time()
            bots = generate_bot_features(attackers[name], matched, split, st.seed, fparams, st.n_jobs, int(p2["bots"]["chunk_size"]))
            bots.to_parquet(out / f"bots_{name}_{split}.parquet", index=False)
            report[f"{name}_{split}"] = {"attacker": name, "split": split, "n_matched": len(matched), "n_kept": len(bots), "n_dropped": bots.attrs["n_dropped"]}
            log.info("%-4s %-5s: %6d bots (%d dropped) in %.1fs", name, split, len(bots), bots.attrs["n_dropped"], time.time() - t)
    write_json(out / "generation_report.json", report)

    # ---- realism report: bots typing the sentences of HELD-OUT pool sessions vs those sessions' features --------
    held = pool["heldout_idx"]
    matched = pd.DataFrame({"session_id": [str(bank.session_ids[i]) for i in held], "user_id": [int(bank.user_ids[i]) for i in held],
                            "sentence": [bank.sentence(i) for i in held]})
    pool_feats = pd.read_parquet(st.interim_dir / "features_attacker_pool.parquet")
    pool_feats = pool_feats[pool_feats["session_id"].isin(set(matched["session_id"]))]
    rows = []
    for name in names:
        bots = generate_bot_features(attackers[name], matched, "pool_heldout", st.seed, fparams, st.n_jobs, int(p2["bots"]["chunk_size"]))
        for f in FEATURE_NAMES:
            h, b = pool_feats[f].to_numpy(dtype="float64"), bots[f].to_numpy(dtype="float64")
            rows.append({"attacker": name, "feature": f, "group": GROUP_OF[f], "ks": float(ks_2samp(h, b).statistic), "mean_diff": float(b.mean() - h.mean()),
                         "mean_diff_in_pool_std": float((b.mean() - h.mean()) / h.std()) if h.std() > 0 else float("nan"),
                         "pool_mean": float(h.mean()), "bot_mean": float(b.mean()), "pool_median": float(pd.Series(h).median()), "bot_median": float(pd.Series(b).median()),
                         "n_pool": len(h), "n_bot": len(b)})
        log.info("realism %-4s: mean KS %.3f", name, pd.DataFrame(rows).query("attacker == @name")["ks"].mean())
    pd.DataFrame(rows).to_csv(out / "realism_report.csv", index=False)
    assert_frozen(st)
    log.info("done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
