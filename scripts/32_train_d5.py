#!/usr/bin/env python
"""Phase 3b, step 32 — D5 (supervised bidirectional GRU) for E5 (trained on ONE attacker) or E6 (leave one attacker out).

Training sets exactly as for D3: the 20k train humans + the (regenerated, verified) Phase 2 train bots of the training
attacker(s); class-balanced epochs; early stopping on val humans + val bots of the TRAINING attacker(s) only;
tau5/tau10 on VAL humans; the TEST split is scored once per (run, seed). Finished runs are skipped on a re-run.
Writes <out>/phase3b/{e5_cross_attack.csv | e6_loao.csv}, training_curves.csv, runtime_log.json, models/, scores/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kl.phase3b import nets  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG before torch touches CUDA)
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


def plan(experiment: str, p3: dict) -> list[tuple[str, list[str], list[str]]]:
    """``(trained_on label, training attackers, tested attackers)`` per run."""
    ex = p3["experiments"]
    if experiment == "e5":
        return [(a, [a], list(ex["test_attackers"])) for a in ex["e5_train_attackers"]]
    runs = []
    for held in ex["e6_heldout_attackers"]:
        train_on = [a for a in ex["e5_train_attackers"] if a != held]
        assert held not in train_on, "E6: the held-out attacker must not be a training attacker"
        runs.append((f"all-but-{held}", train_on, [held]))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--experiment", choices=("e5", "e6"), required=True)
    ap.add_argument("--allow-rerun", action="store_true", help="retrain finished runs and re-use the test split")
    args = ap.parse_args()
    st = load_settings(args.config, args.quick)
    log = setup_logging(st.out_dir, f"32_train_d5_{args.experiment}")
    assert_frozen_p123(st)
    p3, out = load_phase3b_config(), phase3b_dir(st)
    lock = gpu_lock(out)  # noqa: F841 - held until exit: one training process at a time (also while SIGSTOP-paused)
    thermal = ThermalGuard(out, log)
    mc, tc = p3["model"], p3["training"]
    seeds, bs, ebs = [int(s) for s in tc["seeds"]], int(tc["batch_size"]), int(tc["eval_batch_size"])
    runs = plan(args.experiment, p3)
    device = nets.pick_device()
    dinfo = nets.device_info(device)
    print(f"device: {device} ({dinfo['gpu_name'] or 'CPU'}; torch {dinfo['torch_version']}, CUDA {dinfo['cuda_version']})", flush=True)
    log.info("device info: %s", dinfo)
    update_runtime(out, "device", f"32_train_d5_{args.experiment}", dinfo)
    t0 = time.time()

    enc = load_encoder(out)
    humans_tr, val = load_encoded(out, enc, "human_train_20k"), load_encoded(out, enc, "human_val")
    train_atk = sorted({a for _, tr, _ in runs for a in tr})
    bots_tr = {a: load_encoded(out, enc, f"bot_{a}_train") for a in train_atk}
    bots_val = {a: load_encoded(out, enc, f"bot_{a}_val") for a in train_atk}
    tested = sorted({a for _, _, te in runs for a in te})
    test = load_test_split(out, enc, tested, args.allow_rerun)
    log.info("%s: %d runs x %d seeds; train humans %d, val humans %d", args.experiment.upper(), len(runs), len(seeds), len(humans_tr), len(val))

    for trained_on, train_on, test_on in runs:
        # training bots and early-stopping val bots come from `train_on` only; in E6 the held-out attacker is never among them
        if args.experiment == "e6" and set(test_on) & set(train_on):
            raise AssertionError(f"E6 run {trained_on}: held-out attacker among the training attackers")
        bots = nets.concat_encoded([bots_tr[a] for a in train_on], "train bots " + "+".join(train_on), "+".join(train_on))
        vb = nets.concat_encoded([bots_val[a] for a in train_on], "val bots " + "+".join(train_on), "+".join(train_on))
        for seed in seeds:
            key = f"D5<-{trained_on}/seed{seed}"
            spath = score_path(out, key)
            state = run_state(out, key, out / "models" / (spath.stem + ".pt"), ["test_h", "tau5", "tau10"] + [f"bot_{a}" for a in test_on])
            if state == "complete" and not args.allow_rerun:
                log.info("%s: COMPLETE earlier (%s), skipped", key, spath.name)
                continue
            if state == "interrupted-after-test-access" and not args.allow_rerun:
                raise SystemExit(f"{key}: used the test split but has no complete result; retraining it re-uses the test split: "
                                 "rerun with --allow-rerun after checking why")
            log.info("%s: state %s -> training from scratch", key, state)
            nets.set_determinism(seed)
            gen = torch.Generator().manual_seed(seed)
            model = nets.D5Model(enc.vocab_size, int(mc["embedding_dim"]), int(mc["hidden"]), int(mc["num_layers"]), float(mc["dropout"])).to(device)
            log.info("== %s: %d train humans + %d train bots (%s), early stopping on %d val humans + %d val bots", key, len(humans_tr), len(bots),
                     "+".join(train_on), len(val), len(vb))
            res = nets.fit(model, nets.d5_epoch_batches(model, humans_tr, bots, bs, gen, device), lambda: nets.d5_val_loss(model, val, vb, ebs, device),
                           float(tc["lr"]), int(tc["max_epochs"]), int(tc["patience"]), float(tc["grad_clip_norm"]), lambda m: log.info("%s %s", key, m),
                           between_epochs=lambda e: thermal.check(key, e))
            torch.save(model.state_dict(), out / "models" / (spath.stem + ".pt"))
            ts = time.time()
            taus = thresholds_from_val_humans(Scored(nets.d5_logits(model, val, ebs, device), "val", "human", "val humans"))
            humans_te, bots_te = test.for_detector(key)
            arrays = {"test_h": nets.d5_logits(model, humans_te, ebs, device), "tau5": taus["tau5"], "tau10": taus["tau10"]}
            for a in test_on:
                arrays[f"bot_{a}"] = nets.d5_logits(model, bots_te[a], ebs, device)
            save_scores(spath, **arrays)
            append_curves(out, {"detector": "D5", "trained_on": trained_on, "seed": seed}, res.curve)
            update_runtime(out, "runs", key, {"experiment": args.experiment, "train_attackers": train_on, "n_train_humans": len(humans_tr),
                                               "n_train_bots": len(bots), "n_val_bots": len(vb), "epochs_run": res.epochs_run, "best_epoch": res.best_epoch,
                                               "best_val_loss": res.best_val, "stopped_early": res.stopped_early, "train_seconds": res.seconds,
                                               "score_seconds": time.time() - ts, "seconds_per_epoch": res.seconds / max(res.epochs_run, 1), "thermal_paused_seconds": res.paused_seconds})
            log.info("%s: best epoch %d / %d (val loss %.5f), %.0fs; taus %s", key, res.best_epoch, res.epochs_run, res.best_val, res.seconds, taus)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- table ------------------------------------------------------------------------------------------------
    specs = []
    for trained_on, _, test_on in runs:
        paths = {s: score_path(out, f"D5<-{trained_on}/seed{s}") for s in seeds}
        if not all(p.exists() for p in paths.values()):
            log.warning("D5<-%s: not all seeds finished; rows skipped", trained_on)
            continue
        loaded = {s: load_scores(p) for s, p in paths.items()}
        for a in test_on:
            per_seed = {s: {"h": z["test_h"], "b": z[f"bot_{a}"], "taus": {"tau5": float(z["tau5"]), "tau10": float(z["tau10"])}} for s, z in loaded.items()}
            specs.append({"detector": "D5", "trained_on": trained_on, "tested_on": a, "per_seed": per_seed})
    ev = st.cfg["evaluation"]
    h_users, b_users = test.users()
    table = build_table(specs, h_users, b_users, int(ev["bootstrap_reps"]), st.seed, float(ev["ci_level"]), st.n_jobs) if specs else pd.DataFrame()
    p2out = phase2_dir(st)
    if args.experiment == "e5":
        ref = phase2_rows(pd.read_csv(p2out / "attack_matrix.csv"), ["D3"])
        name = "e5_cross_attack.csv"
    else:
        ref = phase2_rows(pd.read_csv(p2out / "loao_table.csv"), ["D3", "D1", "D2"])
        name = "e6_loao.csv"
    tab = ordered_columns(pd.concat([table, ref], ignore_index=True), seeds)
    tab.to_csv(out / name, index=False)
    update_runtime(out, "parts", f"{args.experiment.upper()} (32_train_d5)", {"seconds": time.time() - t0})
    show = tab[["detector", "trained_on", "tested_on", "apcer_fpr_tau10_mean", "apcer_fpr_tau10_sd", "bpcer_fnr_tau10_mean", "auc_mean", "auc_sd"]]
    log.info("%s (TEST):\n%s", name, show.round(4).to_string(index=False))
    assert_frozen_p123(st)
    log.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
