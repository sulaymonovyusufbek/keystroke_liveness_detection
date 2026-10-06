#!/usr/bin/env python
"""Phase 3b, step 30 — keystroke sequences for humans and (regenerated) bots, the encoder, and the bot verification.

Humans: train_all (every clean session of a TRAIN user), train_20k (= D1's training sessions), val, test, taken from the
Phase 1 sessions.parquet and checked against the Phase 1 feature files. Bots: the Phase 2 train/test bots regenerated
with the FROZEN attackers and seeds and verified against the stored Phase 2 features (1e-9; exit code 3 and nothing
else on a mismatch), plus val bots of the training attackers (D5 early stopping only).

Writes <out>/phase3b/interim/{human_*,bot_*}.npz, encoder.json, bot_verification.json, truncation_report.csv,
sequences_meta.json.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kl.config import load_settings, read_json, setup_logging, write_json  # noqa: E402
from kl.features import FeatureParams  # noqa: E402
from kl.phase2.config import load_phase2_config, phase2_dir  # noqa: E402
from kl.phase2.generate import assert_pool_disjoint, build_attackers  # noqa: E402
from kl.phase3b.config import load_phase3b_config, phase3b_dir  # noqa: E402
from kl.phase3b.guard import assert_frozen_p123  # noqa: E402
from kl.phase3b.regen import regenerate_bots, verify_against_phase2  # noqa: E402
from kl.phase3b.sequences import Encoder, SeqSet, build_human_sets, check_against_features  # noqa: E402

FEATURE_FILE = {"train_20k": "train", "val": "val", "test": "test"}


def trunc_row(s: SeqSet, max_len: int) -> dict:
    n_tr = int((s.lengths > max_len).sum())
    return {"set": s.name, "split": s.split, "kind": s.kind, "attacker": s.attacker or "", "n_sessions": s.n_sessions, "n_truncated": n_tr,
            "frac_truncated": n_tr / s.n_sessions if s.n_sessions else float("nan"), "max_len": max_len,
            "max_n_keys": int(s.lengths.max()) if s.n_sessions else 0, "min_hold_ms": float(s.hold.min()) if len(s.hold) else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    log = setup_logging(st.out_dir, "30_build_sequences")
    assert_frozen_p123(st)
    p2, p3 = load_phase2_config(), load_phase3b_config()
    out, p2out = phase3b_dir(st), phase2_dir(st)
    interim = out / "interim"
    t0 = time.time()
    meta: dict = {"seed": st.seed, "config_hash": st.config_hash}

    # ---- humans -----------------------------------------------------------------------------------------------
    manifest = read_json(st.results_dir / "split_manifest.json")
    humans = build_human_sets(st.interim_dir / "sessions.parquet", manifest, st.n_jobs)
    checks = {}
    for name, split in FEATURE_FILE.items():
        feats = pd.read_parquet(st.interim_dir / f"features_{split}.parquet", columns=["session_id", "user_id", "n_keys", "hold_mean", "pp_mean"])
        checks[name] = check_against_features(humans[name], feats)
    log.info("human sets: %s; agree with the Phase 1 feature files (max |diff| %s)", {k: v.n_sessions for k, v in humans.items()}, checks)
    enc = Encoder.fit(humans["train_all"], int(p3["sequences"]["min_key_count"]), float(p3["sequences"]["max_len_percentile"]))
    write_json(interim / "encoder.json", enc.to_json())
    log.info("encoder: max_len %d (p%g of train_all lengths), vocabulary %d (%d kept keycodes + PAD + UNK); h ~ %.4f±%.4f, p ~ %.4f±%.4f",
             enc.max_len, enc.info["max_len_percentile"], enc.vocab_size, len(enc.keycodes), enc.mu_h, enc.sd_h, enc.mu_p, enc.sd_p)
    trunc = []
    for name, s in humans.items():
        s.save(interim / f"human_{name}.npz")
        trunc.append(trunc_row(s, enc.max_len))
    meta["human_sets"] = {k: {"n_sessions": v.n_sessions, "n_users": int(len(np.unique(v.user_ids))), "n_keystrokes": int(len(v.hold))} for k, v in humans.items()}
    meta["human_feature_checks"] = checks
    unk = {f"human_{k}": float(np.mean(enc.key_ids(v.keycode) == 1)) for k, v in humans.items()}
    del humans

    # ---- bots: regenerate + verify (train, test), val bots for early stopping ----------------------------------
    pool = joblib.load(p2out / "pool.joblib")
    names = p2["bots"]["attackers"]
    attackers = build_attackers(pool["stats"], pool["bank"], pool["fit_idx"], st.cfg["a1"], names)
    fparams = FeatureParams.from_config(st.cfg["features"])
    sentences = pd.read_parquet(st.interim_dir / "session_index.parquet", columns=["session_id", "SENTENCE"]).set_index("session_id")["SENTENCE"]
    chunk = int(p2["bots"]["chunk_size"])
    verification, all_ok = {}, True
    plan = [(split, a) for split in p2["bots"]["splits"] for a in names] + [("val", a) for a in p3["sequences"]["val_bot_attackers"]]
    for split, a in plan:
        t = time.time()
        feats = pd.read_parquet(st.interim_dir / f"features_{split}.parquet", columns=["session_id", "user_id"])
        matched = feats.assign(sentence=sentences.loc[feats["session_id"]].to_numpy())
        assert_pool_disjoint(pool["pool_users"], matched["user_id"].unique(), f"{split} bot regeneration")
        bf, seqs = regenerate_bots(attackers[a], matched, split, st.seed, fparams, st.n_jobs, chunk)
        stored_path = p2out / f"bots_{a}_{split}.parquet"
        if stored_path.exists():
            rep = verify_against_phase2(bf, pd.read_parquet(stored_path))
            all_ok &= rep["ok"]
        else:
            rep = {"ok": None, "n_regenerated": len(bf), "reason": "Phase 2 stored no bots for this split (val bots are new; used for D5 early stopping only)"}
        rep["n_dropped_nonfinite"] = bf.attrs["n_dropped"]
        check_against_features(seqs, bf)                                    # sequences == the featurised sessions
        verification[f"{a}_{split}"] = rep
        seqs.save(interim / f"bot_{a}_{split}.npz")
        trunc.append(trunc_row(seqs, enc.max_len))
        unk[f"bot_{a}_{split}"] = float(np.mean(enc.key_ids(seqs.keycode) == 1))
        log.info("%-4s %-5s: %6d bots regenerated, verified=%s max|diff|=%s (%.1fs)", a, split, len(bf), rep["ok"], rep.get("max_abs_diff"), time.time() - t)
        if rep["ok"] is False:
            log.error("MISMATCH for %s %s: %s", a, split, {k: v for k, v in rep.items() if k != "max_abs_diff_per_feature"})
    write_json(out / "bot_verification.json", {"all_verified": bool(all_ok), "tolerance": 1e-9, "bots": verification})
    if not all_ok:
        log.error("regenerated bots do NOT reproduce the Phase 2 features: stopping (see bot_verification.json)")
        assert_frozen_p123(st)
        sys.exit(3)

    tr = pd.DataFrame(trunc)
    tr.to_csv(out / "truncation_report.csv", index=False)
    meta.update({"encoder": {k: v for k, v in enc.to_json().items() if k != "keycodes"}, "unk_fraction": unk, "runtime_seconds": time.time() - t0})
    write_json(out / "sequences_meta.json", meta)
    log.info("truncation (max_len %d):\n%s", enc.max_len, tr[["set", "n_sessions", "n_truncated", "frac_truncated", "max_n_keys", "min_hold_ms"]].to_string(index=False))
    assert_frozen_p123(st)
    log.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
