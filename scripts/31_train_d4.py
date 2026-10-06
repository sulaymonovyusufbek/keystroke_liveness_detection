#!/usr/bin/env python
"""Phase 3b, step 31 — D4 (autoregressive human-timing model) and its controls; experiment E4 (open set).

For every variant of config_phase3b.yaml ``d4_variants`` (D4, D4-20k, D4-nokey) and every training seed: train on
humans only (early stopping on the val-human NLL), set tau5/tau10 on VAL humans, score the TEST split once. Scores:
D4-primary = mean per-step NLL, D4-typicality = |mean NLL - median val-human mean NLL| (variant D4 only); the human
score used for thresholds / metrics is the NEGATED anomaly (higher = more human-like, as in Phases 1-2).

A finished (variant, seed) is skipped on a re-run (its score file exists), so an interrupted run resumes.
Writes <out>/phase3b/{e4_open_set.csv, training_curves.csv, runtime_log.json, models/, scores/}.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kl.phase3b import nets  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG before torch touches CUDA)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from kl.config import load_settings, setup_logging  # noqa: E402
from kl.phase2.config import phase2_dir  # noqa: E402
from kl.phase3b.config import load_phase3b_config, phase3b_dir  # noqa: E402
from kl.phase3b.evaluation import Scored, ordered_columns, phase2_rows, thresholds_from_val_humans  # noqa: E402
from kl.phase3b.guard import assert_frozen_p123  # noqa: E402
from kl.phase3b.thermal import ThermalGuard  # noqa: E402
from kl.phase3b.runner import (append_curves, build_table, gpu_lock, load_encoded, load_encoder, load_scores,  # noqa: E402
                               load_test_split, run_state, save_scores, score_path, update_runtime)

def d4_required(attackers: list[str]) -> list[str]:
    """Arrays a complete D4 score file must hold."""
    return ["val_m", "val_median", "test_m", "tau5_primary", "tau10_primary", "tau5_typicality", "tau10_typicality"] + [f"bot_m_{a}" for a in attackers]


#: E4 rows: (row label, variant, score kind)
E4_ROWS = [("D4-primary", "D4", "primary"), ("D4-typicality", "D4", "typicality"), ("D4-20k", "D4-20k", "primary"), ("D4-nokey", "D4-nokey", "primary")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--allow-rerun", action="store_true", help="retrain finished runs and re-use the test split")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    log = setup_logging(st.out_dir, "31_train_d4")
    assert_frozen_p123(st)
    p3, out = load_phase3b_config(), phase3b_dir(st)
    lock = gpu_lock(out)  # noqa: F841 - held until exit: one training process at a time (also while SIGSTOP-paused)
    thermal = ThermalGuard(out, log)
    mc, tc = p3["model"], p3["training"]
    seeds, bs, ebs = [int(s) for s in tc["seeds"]], int(tc["batch_size"]), int(tc["eval_batch_size"])
    attackers = list(p3["experiments"]["test_attackers"])
    device = nets.pick_device()
    dinfo = nets.device_info(device)
    print(f"device: {device} ({dinfo['gpu_name'] or 'CPU'}; torch {dinfo['torch_version']}, CUDA {dinfo['cuda_version']})", flush=True)
    log.info("device info: %s", dinfo)
    update_runtime(out, "device", "31_train_d4", dinfo)
    t0 = time.time()

    enc = load_encoder(out)
    sets = {"train_all": load_encoded(out, enc, "human_train_all"), "train_20k": load_encoded(out, enc, "human_train_20k")}
    val = load_encoded(out, enc, "human_val")
    test = load_test_split(out, enc, attackers, args.allow_rerun)
    log.info("train_all %d, train_20k %d, val %d sessions; max_len %d, vocabulary %d", len(sets["train_all"]), len(sets["train_20k"]), len(val), enc.max_len, enc.vocab_size)

    for variant, vcfg in p3["d4_variants"].items():
        for seed in seeds:
            key = f"{variant}/seed{seed}"
            spath = score_path(out, key)
            state = run_state(out, key, out / "models" / f"{variant}_seed{seed}.pt", d4_required(attackers))
            if state == "complete" and not args.allow_rerun:
                log.info("%s: COMPLETE earlier (%s), skipped", key, spath.name)
                continue
            if state == "interrupted-after-test-access" and not args.allow_rerun:
                raise SystemExit(f"{key}: used the test split but has no complete result; retraining it re-uses the test split: "
                                 "rerun with --allow-rerun after checking why")
            log.info("%s: state %s -> training from scratch", key, state)
            train = sets[vcfg["train_set"]]
            nets.set_determinism(seed)
            gen = torch.Generator().manual_seed(seed)
            model = nets.D4Model(enc.vocab_size, int(mc["embedding_dim"]), int(mc["hidden"]), int(mc["num_layers"]), float(mc["dropout"]),
                                 int(mc["mixture_components"]), float(mc["sigma_floor"]), bool(vcfg["use_keys"])).to(device)
            log.info("== %s: training on %s (%d sessions), keys=%s, %d parameters", key, vcfg["train_set"], len(train), vcfg["use_keys"],
                     sum(p.numel() for p in model.parameters()))
            res = nets.fit(model, nets.d4_epoch_batches(model, train, bs, gen, device), lambda: nets.d4_mean_nll(model, val, ebs, device),
                           float(tc["lr"]), int(tc["max_epochs"]), int(tc["patience"]), float(tc["grad_clip_norm"]), lambda m: log.info("%s %s", key, m),
                           between_epochs=lambda e: thermal.check(key, e))
            torch.save(model.state_dict(), out / "models" / f"{variant}_seed{seed}.pt")
            ts = time.time()
            val_m = nets.d4_session_scores(model, val, ebs, device)
            med = float(np.median(val_m))
            taus_primary = thresholds_from_val_humans(Scored(-val_m, "val", "human", "val humans"))
            taus_typ = thresholds_from_val_humans(Scored(-np.abs(val_m - med), "val", "human", "val humans"))
            humans_te, bots_te = test.for_detector(key)
            arrays = {"val_m": val_m, "val_median": med, "test_m": nets.d4_session_scores(model, humans_te, ebs, device),
                      "tau5_primary": taus_primary["tau5"], "tau10_primary": taus_primary["tau10"],
                      "tau5_typicality": taus_typ["tau5"], "tau10_typicality": taus_typ["tau10"]}
            for a, b in bots_te.items():
                arrays[f"bot_m_{a}"] = nets.d4_session_scores(model, b, ebs, device)
            save_scores(spath, **arrays)
            append_curves(out, {"detector": variant, "trained_on": vcfg["train_set"], "seed": seed}, res.curve)
            update_runtime(out, "runs", key, {"train_set": vcfg["train_set"], "n_train": len(train), "use_keys": bool(vcfg["use_keys"]),
                                               "epochs_run": res.epochs_run, "best_epoch": res.best_epoch, "best_val_nll": res.best_val,
                                               "stopped_early": res.stopped_early, "train_seconds": res.seconds, "score_seconds": time.time() - ts,
                                               "seconds_per_epoch": res.seconds / max(res.epochs_run, 1), "thermal_paused_seconds": res.paused_seconds})
            log.info("%s: best epoch %d / %d (val NLL %.5f), %.0fs; val median mean-NLL %.4f; taus primary %s", key, res.best_epoch,
                     res.epochs_run, res.best_val, res.seconds, med, taus_primary)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- E4 table ---------------------------------------------------------------------------------------------
    humans_users, bot_users = test.users()
    specs = []
    for label, variant, kind in E4_ROWS:
        loaded = {s: load_scores(score_path(out, f"{variant}/seed{s}")) for s in seeds if score_path(out, f"{variant}/seed{s}").exists()}
        if len(loaded) < len(seeds):
            log.warning("%s: only %d of %d seeds finished; row skipped", label, len(loaded), len(seeds))
            continue
        for a in attackers:
            per_seed = {}
            for s, z in loaded.items():
                if kind == "primary":
                    h, b = -z["test_m"], -z[f"bot_m_{a}"]
                else:
                    med = float(z["val_median"])
                    h, b = -np.abs(z["test_m"] - med), -np.abs(z[f"bot_m_{a}"] - med)
                per_seed[s] = {"h": h, "b": b, "taus": {"tau5": float(z[f"tau5_{kind}"]), "tau10": float(z[f"tau10_{kind}"])}}
            specs.append({"detector": label, "trained_on": "attack-free", "tested_on": a, "per_seed": per_seed})
    ev = st.cfg["evaluation"]
    table = build_table(specs, humans_users, bot_users, int(ev["bootstrap_reps"]), st.seed,
                        float(ev["ci_level"]), st.n_jobs) if specs else pd.DataFrame()
    p2 = phase2_rows(pd.read_csv(phase2_dir(st) / "open_set_table.csv"), ["D1", "D2"])
    e4 = ordered_columns(pd.concat([p2, table], ignore_index=True), seeds)
    e4.to_csv(out / "e4_open_set.csv", index=False)
    update_runtime(out, "parts", "E4 (31_train_d4)", {"seconds": time.time() - t0})
    show = e4[["detector", "tested_on", "apcer_fpr_tau10_mean", "apcer_fpr_tau10_sd", "bpcer_fnr_tau10_mean", "auc_mean", "auc_sd"]]
    log.info("E4 open set (TEST):\n%s", show.round(4).to_string(index=False))
    assert_frozen_p123(st)
    log.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
