#!/usr/bin/env python
"""Phase 3c (POST-HOC, diagnostic only) — are D4 / D5 detections driven by timing given the keys, or by the key
sequences the bots generate? See docs/PHASE3C_SPEC.md.

1. keycode realism of test bots vs test humans (raw sequences);
2. swap test S1-S4 (+ truncated controls) on the TEST split with FROZEN weights and FROZEN tau10: D4-primary (3 seeds,
   every attacker) and the E5 D5 models (3 seeds, each on the attacker it was trained on);
3. letters-only D4-primary score (tau10 recomputed on val humans with the same masked score).

Nothing is trained or refitted; no Phase 1-3b file is written (the Phase 3b test ledger is not touched: this is a
post-hoc re-use of the test split, labelled as such). Writes <out>/phase3c/*.csv, phase3c_meta.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kl.phase3b import nets  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG before torch touches CUDA)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from kl.config import load_settings, setup_logging  # noqa: E402
from kl.model import tau_for_fnr  # noqa: E402
from kl.phase3b.config import load_phase3b_config  # noqa: E402
from kl.phase3b.runner import load_encoder, load_scores, score_path  # noqa: E402
from kl.phase3b.sequences import SeqSet  # noqa: E402
from kl.phase3c.guard import PHASE3B_RECORD, assert_frozen_p1234, record_phase3b  # noqa: E402
from kl.phase3c.realism import realism_tables  # noqa: E402
from kl.phase3c.swap import d4_letters_scores, identical_keys, letter_ids, pair_index, swap_sets  # noqa: E402

ATTACKERS = ["A1", "A2", "A3", "A4", "A5a", "A5b"]
D5_ATTACKERS = ["A1", "A2", "A3", "A4", "A5b"]           # E5 training attackers (A5a is never a training attack)
SETS = ["S1", "S2", "S3", "S4", "S1t", "S2t"]
MIN_EFFECT = 0.05                                       # shares are undefined when S2 itself is barely detected


def auc_vs(h: np.ndarray, x: np.ndarray) -> float:
    """ROC-AUC of human-scores ``h`` (positive) vs ``x`` (higher = more human; never flipped)."""
    return float(roc_auc_score(np.r_[np.ones(len(h)), np.zeros(len(x))], np.r_[h, x]))


def identical_row(det: str, a: str, seed: int, h: np.ndarray, b: np.ndarray, idx: tuple[np.ndarray, np.ndarray], tau: float) -> dict:
    """ADDITION: S1 vs S2 restricted to pairs whose key sequences are identical (keys equal by construction)."""
    hi, bi = idx
    row = {"detector": det, "attacker": a, "seed": seed, "n_pairs": len(hi)}
    if len(hi) < 2:
        return {**row, "bpcer_tau10_humans": float("nan"), "apcer_tau10_bots": float("nan"), "auc": float("nan")}
    return {**row, "bpcer_tau10_humans": float(np.mean(h[hi] < tau)), "apcer_tau10_bots": float(np.mean(b[bi] >= tau)), "auc": auc_vs(h[hi], b[bi])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--record-phase3b-freeze", action="store_true", help="write docs/phase3b_frozen_hashes.json (once) and exit")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    if args.record_phase3b_freeze:
        rec = record_phase3b()
        print(f"recorded {PHASE3B_RECORD}: {len(rec['source_files'])} sources, " + ", ".join(f"{m}: {len(v['output_files'])} outputs" for m, v in rec["modes"].items()))
        return
    log = setup_logging(st.out_dir, "40_run_phase3c")
    assert_frozen_p1234(st)
    t0 = time.time()
    p3 = load_phase3b_config()
    mc, ebs = p3["model"], int(p3["training"]["eval_batch_size"])
    seeds = [int(s) for s in p3["training"]["seeds"]]
    in3b = st.out_dir / "phase3b"
    out = st.out_dir / "phase3c"
    out.mkdir(parents=True, exist_ok=True)
    device = nets.pick_device()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    log.info("POST-HOC Phase 3c, mode %s, device %s", "quick" if st.quick else "full", device)

    enc = load_encoder(in3b)
    humans = SeqSet.load(in3b / "interim" / "human_test.npz")
    bots = {a: SeqSet.load(in3b / "interim" / f"bot_{a}_test.npz") for a in ATTACKERS}
    val_h = enc.encode(SeqSet.load(in3b / "interim" / "human_val.npz"))
    sentence_of = pd.read_parquet(st.interim_dir / "session_index.parquet", columns=["session_id", "SENTENCE"]).set_index("session_id")["SENTENCE"].to_dict()

    # ---- 1. keycode realism --------------------------------------------------------------------------------------
    tabs = realism_tables(humans, bots, sentence_of, enc.key_ids)
    for k, df in tabs.items():
        df.to_csv(out / f"realism_{k}.csv", index=False)
    log.info("realism:\n%s\n%s", tabs["jsd"].round(4).to_string(index=False),
             tabs["summary"][["group", "len_mean", "unk_pct", "backspace_per_100_keys", "shift_shift_per_capital", "shift_hold_ms_median"]].round(3).to_string(index=False))

    # ---- 2. swap sets ------------------------------------------------------------------------------------------------
    enc_sets: dict[str, dict[str, object]] = {}
    ident: dict[str, tuple[np.ndarray, np.ndarray]] = {}       # ADDITION (not in the spec): identical-key-sequence pairs
    mism = []
    h_enc = enc.encode(humans)
    for a in ATTACKERS:
        sets, summ = swap_sets(humans, bots[a])
        mism.append({"attacker": a, **summ})
        hi, bi = pair_index(humans, bots[a])
        same = identical_keys(humans, hi, bots[a], bi)
        ident[a] = (hi[same], bi[same])
        enc_sets[a] = {"S1": h_enc, "S2": enc.encode(bots[a]), **{k: enc.encode(v) for k, v in sets.items()}}
    pd.DataFrame(mism).to_csv(out / "swap_length_mismatch.csv", index=False)
    log.info("length mismatch:\n%s", pd.DataFrame(mism).round(3).to_string(index=False))

    rows, repro, id_rows = [], [], []
    letters = letter_ids(enc.key_ids)
    let_rows = []
    for s in seeds:
        key = f"D4/seed{s}"
        z = load_scores(score_path(in3b, key))
        model = nets.D4Model(enc.vocab_size, int(mc["embedding_dim"]), int(mc["hidden"]), int(mc["num_layers"]), float(mc["dropout"]),
                             int(mc["mixture_components"]), float(mc["sigma_floor"]), True).to(device)
        model.load_state_dict(torch.load(in3b / "models" / f"D4_seed{s}.pt", map_location=device))
        model.eval()
        tau = float(z["tau10_primary"])
        s1 = -nets.d4_session_scores(model, h_enc, ebs, device)
        repro.append({"detector": "D4-primary", "seed": s, "set": "S1 (test humans)", "max_abs_diff_vs_phase3b": float(np.max(np.abs(-s1 - z["test_m"])))})
        for a in ATTACKERS:
            for name in SETS:
                x = s1 if name == "S1" else -nets.d4_session_scores(model, enc_sets[a][name], ebs, device)
                if name == "S2":
                    repro.append({"detector": "D4-primary", "seed": s, "set": f"S2 ({a})", "max_abs_diff_vs_phase3b": float(np.max(np.abs(-x - z[f"bot_m_{a}"])))})
                rows.append({"detector": "D4-primary", "attacker": a, "seed": s, "set": name, "n": len(x),
                             "flagged": float(np.mean(x < tau)), "apcer_tau10": float(np.mean(x >= tau)), "auc_vs_S1": auc_vs(s1, x), "tau10": tau})
                if name == "S2":
                    id_rows.append(identical_row("D4-primary", a, s, s1, x, ident[a], tau))
        # ---- 3. letters-only D4 (tau10 from val humans, same masked score) ------------------------------------------
        v, vn = d4_letters_scores(model, val_h, letters, ebs, device)
        tau_l = tau_for_fnr(-v[np.isfinite(v)], 0.10)
        th, thn = d4_letters_scores(model, h_enc, letters, ebs, device)
        hl = -th[np.isfinite(th)]
        let_rows.append({"seed": s, "attacker": "(humans)", "n": len(th), "n_no_letter_steps": int((thn == 0).sum()),
                         "val_n_no_letter_steps": int((vn == 0).sum()), "tau10": tau_l, "bpcer_tau10": float(np.mean(hl < tau_l)),
                         "apcer_tau10": float("nan"), "auc": float("nan"), "letter_steps_share": float(thn.sum() / h_enc.lens.sum())})
        for a in ATTACKERS:
            b = enc_sets[a]["S2"]
            tb, tbn = d4_letters_scores(model, b, letters, ebs, device)
            bl = -tb[np.isfinite(tb)]
            let_rows.append({"seed": s, "attacker": a, "n": len(tb), "n_no_letter_steps": int((tbn == 0).sum()), "val_n_no_letter_steps": int((vn == 0).sum()),
                             "tau10": tau_l, "bpcer_tau10": float(np.mean(hl < tau_l)), "apcer_tau10": float(np.mean(bl >= tau_l)),
                             "auc": auc_vs(hl, bl), "letter_steps_share": float(tbn.sum() / b.lens.sum())})
        del model
        log.info("D4 seed %d done (%.0fs)", s, time.time() - t0)

    for s in seeds:
        for a in D5_ATTACKERS:
            key = f"D5<-{a}/seed{s}"
            sp = score_path(in3b, key)
            z = load_scores(sp)
            model = nets.D5Model(enc.vocab_size, int(mc["embedding_dim"]), int(mc["hidden"]), int(mc["num_layers"]), float(mc["dropout"])).to(device)
            model.load_state_dict(torch.load(in3b / "models" / (sp.stem + ".pt"), map_location=device))
            model.eval()
            tau = float(z["tau10"])
            s1 = nets.d5_logits(model, h_enc, ebs, device)
            repro.append({"detector": f"D5<-{a}", "seed": s, "set": "S1 (test humans)", "max_abs_diff_vs_phase3b": float(np.max(np.abs(s1 - z["test_h"])))})
            for name in SETS:
                x = s1 if name == "S1" else nets.d5_logits(model, enc_sets[a][name], ebs, device)
                if name == "S2":
                    repro.append({"detector": f"D5<-{a}", "seed": s, "set": f"S2 ({a})", "max_abs_diff_vs_phase3b": float(np.max(np.abs(x - z[f"bot_{a}"])))})
                rows.append({"detector": "D5 (E5, own attacker)", "attacker": a, "seed": s, "set": name, "n": len(x),
                             "flagged": float(np.mean(x < tau)), "apcer_tau10": float(np.mean(x >= tau)), "auc_vs_S1": auc_vs(s1, x), "tau10": tau})
                if name == "S2":
                    id_rows.append(identical_row("D5 (E5, own attacker)", a, s, s1, x, ident[a], tau))
            del model
        log.info("D5 seed %d done (%.0fs)", s, time.time() - t0)

    pd.DataFrame(id_rows).to_csv(out / "identical_keys_per_seed.csv", index=False)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "swap_per_seed.csv", index=False)
    pd.DataFrame(let_rows).to_csv(out / "letters_only_per_seed.csv", index=False)
    rp = pd.DataFrame(repro)
    rp.to_csv(out / "reproduction_check.csv", index=False)
    log.info("reproduction of the stored Phase 3b scores: max |diff| %.3g", rp["max_abs_diff_vs_phase3b"].max())

    # ---- shares (per seed, then mean / sd) ---------------------------------------------------------------------------
    sh = []
    for (d, a, s), g in per_seed.groupby(["detector", "attacker", "seed"]):
        m = g.set_index("set")
        e_auc = {k: m.loc[k, "auc_vs_S1"] - 0.5 for k in SETS}
        e_flag = {k: m.loc[k, "flagged"] - m.loc["S1", "flagged"] for k in SETS}
        row = {"detector": d, "attacker": a, "seed": s, "effect_auc_S2": e_auc["S2"], "effect_flag_S2": e_flag["S2"]}
        for k in ("S3", "S4", "S2t"):
            row[f"share_auc_{k}"] = e_auc[k] / e_auc["S2"] if abs(e_auc["S2"]) >= MIN_EFFECT else float("nan")
            row[f"share_flag_{k}"] = e_flag[k] / e_flag["S2"] if abs(e_flag["S2"]) >= MIN_EFFECT else float("nan")
        sh.append(row)
    pd.DataFrame(sh).to_csv(out / "swap_shares_per_seed.csv", index=False)

    meta = {"mode": "quick" if st.quick else "full", "post_hoc": True, "device": str(device), "gpu": nets.device_info(device).get("gpu_name"),
            "seeds": seeds, "min_effect_for_shares": MIN_EFFECT, "seconds": time.time() - t0,
            "letters_keycodes": "space (32) + A-Z (65-90)", "max_len": enc.max_len,
            "note": "Test split re-used post hoc with frozen weights and frozen tau10 (letters-only: tau10 recomputed on val humans)."}
    (out / "phase3c_meta.json").write_text(json.dumps(meta, indent=2))
    assert_frozen_p1234(st)
    log.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
