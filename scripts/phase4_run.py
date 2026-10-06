#!/usr/bin/env python
"""Phase 4 (docs/PHASE4_SPEC.md, Part A): bootstrap CIs (A1, A2), identical-key subset (A3) and detector D6 (A4).

Everything from Phases 1-3c is FROZEN. Saved scores, weights and thresholds are used as they are; nothing earlier is
retrained, re-selected or re-thresholded. Two documented exceptions (see docs/PHASE4_RESULTS.md, section g):
* the five single-attacker D3 forests (E2) are REFITTED deterministically (frozen code, data, hyper-parameters, seed)
  because their per-session scores were never saved; they are used only if every threshold and all 30 saved
  cross-attack cells (point estimates AND CIs) are reproduced, and only for the CI of D3's off-diagonal average;
* D5 validation-human logits were not saved: they are recomputed from the frozen D5 weights (validation humans only)
  to check the saved tau5 / tau10; the saved thresholds are the ones used.

D6 (pre-registered, added after the main evaluation) is fitted on training humans only, thresholded on validation
humans only and evaluated ONCE on the test split (ledger ``test_access_phase4.json``).

Writes outputs/phase4/<mode>/: results_*.csv, reproduction_check.csv, d6_*, run_meta.json, logs/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kl.config import ROOT, load_settings, setup_logging, write_json  # noqa: E402
from kl.phase2.config import load_phase2_config  # noqa: E402
from kl.phase2.detectors import d1_from_model, fit_d3  # noqa: E402
from kl.phase3b.config import load_phase3b_config  # noqa: E402
from kl.phase3b.runner import load_encoder, load_scores, score_path  # noqa: E402
from kl.phase3b.sequences import SeqSet  # noqa: E402
from kl.phase3c.swap import identical_keys, pair_index  # noqa: E402
from kl.phase4_bootstrap import (METRICS, RATE_METRICS, BootResult, SeedScores, ValScores, average_cells,  # noqa: E402
                                 evaluate_cell, sensitivity_cell, test_weights, thresholds, user_index, val_weights)
from kl.phase4_d6 import VARIANTS, D6Model, assert_train_only, steps_from_seqset  # noqa: E402
from kl.phase4_guard import assert_frozen_p4  # noqa: E402

ATTACKERS = ["A1", "A2", "A3", "A4", "A5a", "A5b"]
TRAIN_ATK = ["A1", "A2", "A3", "A4", "A5b"]
IDENT_ATK = ["A1", "A2", "A3", "A4", "A5b"]        # spec A3 asks for A2, A3, A4, A5b; A1 is reported as well
D4_VARIANTS = {"D4-primary": "D4", "D4-20k": "D4-20k", "D4-nokey": "D4-nokey"}
TOL = 1e-9


def phase4_dir(quick: bool) -> Path:
    d = ROOT / "outputs" / "phase4" / ("quick" if quick else "full")
    d.mkdir(parents=True, exist_ok=True)
    return d


class Ledger:
    """Single test-split evaluation per D6 variant (a second run is refused unless ``allow_rerun``)."""

    def __init__(self, path: Path, allow_rerun: bool) -> None:
        self.path, self.allow = path, allow_rerun

    def claim(self, key: str) -> None:
        led = json.loads(self.path.read_text()) if self.path.exists() else {}
        if key in led and not self.allow:
            raise SystemExit(f"{key} was already evaluated on the test split at {led[key]}; pass --allow-rerun to override")
        led.setdefault(key, []).append(time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.path.write_text(json.dumps(led, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--allow-rerun", action="store_true", help="allow a second test-split evaluation of D6")
    ap.add_argument("--skip-d5-val-check", action="store_true", help="skip the GPU recomputation of D5 validation logits")
    args = ap.parse_args()
    st = load_settings(None, args.quick)
    out = phase4_dir(args.quick)
    log = setup_logging(out, "phase4_run")
    model = assert_frozen_p4(st)
    t_all = time.time()
    timings: dict[str, float] = {}
    p3 = load_phase3b_config()
    seeds = [int(s) for s in p3["training"]["seeds"]]
    reps, level, seed = int(st.cfg["evaluation"]["bootstrap_reps"]), float(st.cfg["evaluation"]["ci_level"]), st.seed
    in3b, in2 = st.out_dir / "phase3b", st.out_dir / "phase2"
    manifest = json.loads((st.results_dir / "split_manifest.json").read_text())
    log.info("Phase 4, mode %s: %d replicates, CI %.2f, seed %d, D4/D5 seeds %s", "quick" if st.quick else "full", reps, level, seed, seeds)

    # ---- test population, user index and replicates (identical to Phases 2 / 3b) -----------------------------------
    t0 = time.time()
    H = SeqSet.load(in3b / "interim" / "human_test.npz")
    B = {a: SeqSet.load(in3b / "interim" / f"bot_{a}_test.npz") for a in ATTACKERS}
    feats_te = pd.read_parquet(st.interim_dir / "features_test.parquet")
    users = np.sort(feats_te["user_id"].unique())
    if not np.array_equal(users, np.unique(H.user_ids)) or list(feats_te["session_id"]) != list(H.session_ids):
        raise AssertionError("Phase 2 and Phase 3b test sets differ")
    if set(map(int, users)) != set(map(int, manifest["users"]["test"])):
        raise AssertionError("test users differ from the split manifest")
    h_u = user_index(users, H.user_ids)
    match_rule = {}
    for a, b in B.items():          # every bot (A5a included) carries the session id and user of ONE test human session
        hi, bi = pair_index(H, b)
        if not (np.array_equal(b.session_ids, H.session_ids) and np.array_equal(b.user_ids, H.user_ids) and len(bi) == H.n_sessions):
            raise AssertionError(f"{a}: bots are not matched one-to-one to the test human sessions")
        bp = pd.read_parquet(in2 / f"bots_{a}_test.parquet", columns=["session_id", "user_id"])
        if list(bp["session_id"]) != list(H.session_ids) or not np.array_equal(bp["user_id"].to_numpy(), H.user_ids):
            raise AssertionError(f"{a}: Phase 2 bot features are not in the test-session order")
        match_rule[a] = f"one bot per test human session ({len(bi)} of {H.n_sessions}); bot.session_id = human session_id, bot.user_id = human user_id"
    b_u = h_u                       # bots are in the same order as the humans and belong to the same users
    W = test_weights(len(users), reps, seed)
    log.info("test: %d human sessions, %d users; bots matched one-to-one for %s", H.n_sessions, len(users), ATTACKERS)

    rows: list[dict] = []
    repro: list[dict] = []
    keep: dict[tuple, BootResult] = {}

    def add(analysis: str, det: str, trained_on: str, tested_on: str, res: BootResult, source: str, subset: str = "all", **extra) -> None:
        keep[(analysis, det, trained_on, tested_on, subset)] = res
        rows.append({"analysis": analysis, "detector": det, "trained_on": trained_on, "tested_on": tested_on, "subset": subset,
                     "source": source, **extra, **res.row(level)})

    def check(what: str, got: float, want: float, tol: float = TOL) -> None:
        d = abs(float(got) - float(want))
        repro.append({"check": what, "phase4": float(got), "saved": float(want), "abs_diff": d, "ok": bool(d <= tol)})

    # ---- D1 (frozen OCSVM; bots scored with the frozen model, never refitted) ---------------------------------------
    d1 = d1_from_model(model)
    h1 = d1.score(feats_te)
    saved_h1 = pd.read_parquet(st.interim_dir / "test_scores.parquet")
    check("D1 test-human scores vs interim/test_scores.parquet (max |diff|)", float(np.max(np.abs(h1 - saved_h1["score"].to_numpy()))), 0.0, 1e-9)
    val_feats = pd.read_parquet(st.interim_dir / "features_val.parquet")
    v1 = d1.score(val_feats)
    for k, t in thresholds(v1).items():
        check(f"D1 {k} recomputed on val humans vs frozen thresholds.json", t, d1.thresholds[k], 1e-12)
    vt: list[dict] = [{"detector": "D1", "seed": "-", **{f"val_reject_{k}": float(np.mean(v1 < t)) for k, t in d1.thresholds.items()}}]
    open2 = pd.read_csv(in2 / "open_set_table.csv")
    d1_bot_scores = {}
    for a in ATTACKERS:
        bf = pd.read_parquet(in2 / f"bots_{a}_test.parquet")
        d1_bot_scores[a] = d1.score(bf)
        res = evaluate_cell([SeedScores(h1, d1_bot_scores[a], d1.thresholds)], h_u, b_u, W, seeds=[0])
        add("human_only", "D1", "attack-free", a, res, "frozen D1 re-scored; Phase 2 replicates")
        ref = open2[(open2.detector == "D1") & (open2.tested_on == a)].iloc[0]
        for m in METRICS:
            check(f"D1 {a} {m} point vs open_set_table.csv", res.point[m], ref[m])
            lo, hi = res.ci(m, level)
            check(f"D1 {a} {m} CI lo vs open_set_table.csv", lo, ref[f"{m}_ci_lo"])
            check(f"D1 {a} {m} CI hi vs open_set_table.csv", hi, ref[f"{m}_ci_hi"])
    # ---- D2: deterministic single model whose scores were not saved; its Phase 2 CIs used these same replicates ------
    for a in ATTACKERS:
        r = open2[(open2.detector == "D2") & (open2.tested_on == a)].iloc[0]
        row = {"analysis": "human_only", "detector": "D2", "trained_on": "attack-free", "tested_on": a, "subset": "all",
               "source": "copied from phase2/open_set_table.csv (same test-user replicates, seed 42, 1,000 reps)",
               "n_humans": int(r["n_humans"]), "n_attacks": int(r["n_attacks"]), "n_seeds": 1}
        for m in METRICS:
            row.update({m: r[m], f"{m}_ci_lo": r[f"{m}_ci_lo"], f"{m}_ci_hi": r[f"{m}_ci_hi"], f"{m}_seed_sd": float("nan")})
        rows.append(row)
    timings["D1_D2"] = time.time() - t0
    log.info("D1 / D2 done (%.0fs)", timings["D1_D2"])

    # ---- D4 variants (saved scores; 3 seeds) + threshold sensitivity for D4-primary ---------------------------------
    t0 = time.time()
    e4 = pd.read_csv(in3b / "e4_open_set.csv")
    Hval = SeqSet.load(in3b / "interim" / "human_val.npz")
    val_users = np.unique(Hval.user_ids)
    if set(map(int, val_users)) != set(map(int, manifest["users"]["val"])):
        raise AssertionError("validation users differ from the split manifest")
    val_u = user_index(val_users, Hval.user_ids)
    Wv = val_weights(len(val_users), reps, seed)
    d4 = {}
    for name, stem in D4_VARIANTS.items():
        d4[name] = {s: load_scores(score_path(in3b, f"{stem}/seed{s}")) for s in seeds}
        for s, z in d4[name].items():
            taus = thresholds(-z["val_m"])
            for k in ("tau5", "tau10"):
                check(f"{name} seed{s} {k} recomputed on val humans vs saved", taus[k], float(z[f"{k}_primary"]), 1e-12)
            vt.append({"detector": name, "seed": s, **{f"val_reject_{k}": float(np.mean(-z["val_m"] < float(z[f"{k}_primary"]))) for k in ("tau5", "tau10")}})
        for a in ATTACKERS:
            per = [SeedScores(-z["test_m"], -z[f"bot_m_{a}"], {"tau5": float(z["tau5_primary"]), "tau10": float(z["tau10_primary"])}) for z in d4[name].values()]
            res = evaluate_cell(per, h_u, b_u, W, seeds=seeds)
            add("human_only", name, "attack-free", a, res, "phase3b saved scores")
            ref = e4[(e4.detector == name) & (e4.tested_on == a)].iloc[0]
            for m in METRICS:
                check(f"{name} {a} {m} mean over seeds vs e4_open_set.csv", res.point[m], ref[f"{m}_mean"])
            if name == "D4-primary":
                vals = [ValScores(-z["val_m"]) for z in d4[name].values()]
                sens = sensitivity_cell(per, vals, val_u, Wv, h_u, b_u, W)
                sres = BootResult({m: res.point[m] for m in RATE_METRICS}, {m: res.per_seed[m] for m in RATE_METRICS}, sens,
                                  res.n_humans, res.n_attacks, seeds)
                add("human_only_sensitivity", name, "attack-free", a, sres, "phase3b saved scores; tau re-set on resampled val users in every replicate")
    timings["D4"] = time.time() - t0
    log.info("D4 variants done (%.0fs)", timings["D4"])

    # ---- D5 (saved scores; 3 seeds): leave-one-out (E6) and cross-attack (E5) ---------------------------------------
    t0 = time.time()
    e5, e6 = pd.read_csv(in3b / "e5_cross_attack.csv"), pd.read_csv(in3b / "e6_loao.csv")
    d5e6 = {a: {s: load_scores(score_path(in3b, f"D5<-all-but-{a}/seed{s}")) for s in seeds} for a in TRAIN_ATK}
    d5e5 = {a: {s: load_scores(score_path(in3b, f"D5<-{a}/seed{s}")) for s in seeds} for a in TRAIN_ATK}
    for a in TRAIN_ATK:
        per = [SeedScores(z["test_h"], z[f"bot_{a}"], {"tau5": float(z["tau5"]), "tau10": float(z["tau10"])}) for z in d5e6[a].values()]
        res = evaluate_cell(per, h_u, b_u, W, seeds=seeds)
        add("loao", "D5", f"all-but-{a}", a, res, "phase3b saved scores")
        ref = e6[(e6.detector == "D5") & (e6.tested_on == a)].iloc[0]
        for m in METRICS:
            check(f"D5 LOAO {a} {m} mean over seeds vs e6_loao.csv", res.point[m], ref[f"{m}_mean"])
    loao2 = pd.read_csv(in2 / "loao_table.csv")
    for a in TRAIN_ATK:
        r = loao2[(loao2.detector == "D3") & (loao2.tested_on == a)].iloc[0]
        row = {"analysis": "loao", "detector": "D3", "trained_on": f"all-but-{a}", "tested_on": a, "subset": "all",
               "source": "copied from phase2/loao_table.csv (same test-user replicates)", "n_humans": int(r["n_humans"]),
               "n_attacks": int(r["n_attacks"]), "n_seeds": 1}
        for m in METRICS:
            row.update({m: r[m], f"{m}_ci_lo": r[f"{m}_ci_lo"], f"{m}_ci_hi": r[f"{m}_ci_hi"], f"{m}_seed_sd": float("nan")})
        rows.append(row)
    for tr in TRAIN_ATK:
        for a in ATTACKERS:
            per = [SeedScores(z["test_h"], z[f"bot_{a}"], {"tau5": float(z["tau5"]), "tau10": float(z["tau10"])}) for z in d5e5[tr].values()]
            res = evaluate_cell(per, h_u, b_u, W, metrics=RATE_METRICS, seeds=seeds)
            add("cross_attack_cell", "D5", tr, a, res, "phase3b saved scores")
            ref = e5[(e5.detector == "D5") & (e5.trained_on == tr) & (e5.tested_on == a)].iloc[0]
            for m in RATE_METRICS:
                check(f"D5 E5 {tr}->{a} {m} mean over seeds vs e5_cross_attack.csv", res.point[m], ref[f"{m}_mean"])
    timings["D5"] = time.time() - t0
    log.info("D5 done (%.0fs)", timings["D5"])

    # ---- D3 cross-attack: deterministic refit, used only if Phase 2 is reproduced exactly ----------------------------
    t0 = time.time()
    p2 = load_phase2_config()
    feats_tr = pd.read_parquet(st.interim_dir / "features_train.parquet")
    summ2 = json.loads((in2 / "detectors_summary.json").read_text())
    matrix2 = pd.read_csv(in2 / "attack_matrix.csv")
    bot_feats = {a: pd.read_parquet(in2 / f"bots_{a}_test.parquet") for a in ATTACKERS}
    d3_ok, d3_cells = True, {}
    for tr in TRAIN_ATK:
        bots_tr = pd.read_parquet(in2 / f"bots_{tr}_train.parquet")
        det = fit_d3(feats_tr, bots_tr, model.transform, model.feature_names, st.seed, p2["detectors"]["d3"], val_feats, st.n_jobs, f"D3<-{tr}")
        for k in ("tau5", "tau10"):
            want = summ2["D3"][f"D3<-{tr}"]["thresholds"][k]
            check(f"D3<-{tr} refit {k} vs detectors_summary.json", det.thresholds[k], want, 0.0)
            d3_ok &= det.thresholds[k] == want
        hs = det.score(feats_te)
        for a in ATTACKERS:
            res = evaluate_cell([SeedScores(hs, det.score(bot_feats[a]), det.thresholds)], h_u, b_u, W, seeds=[0])
            d3_cells[(tr, a)] = res
            ref = matrix2[(matrix2.detector == "D3") & (matrix2.trained_on == tr) & (matrix2.tested_on == a)].iloc[0]
            for m in METRICS:
                lo, hi = res.ci(m, level)
                for what, got, want in ((m, res.point[m], ref[m]), (f"{m} CI lo", lo, ref[f"{m}_ci_lo"]), (f"{m} CI hi", hi, ref[f"{m}_ci_hi"])):
                    check(f"D3 refit {tr}->{a} {what} vs attack_matrix.csv", got, want)
                    d3_ok &= abs(got - want) <= TOL
        log.info("D3<-%s refitted (%.0fs)", tr, time.time() - t0)
    for (tr, a), res in d3_cells.items():
        add("cross_attack_cell", "D3", tr, a, res, "D3 deterministic refit (reproduces attack_matrix.csv)" if d3_ok else "D3 refit FAILED reproduction: not used")
    timings["D3_refit"] = time.time() - t0
    log.info("D3 refit: Phase 2 reproduced exactly = %s (%.0fs)", d3_ok, timings["D3_refit"])

    # ---- cross-attack averages ----------------------------------------------------------------------------------
    avg_rows = []
    for det in ("D3", "D5"):
        if det == "D3" and not d3_ok:
            continue
        for label, cols in (("off-diagonal, excluding A5a (20 cells)", TRAIN_ATK), ("off-diagonal, including A5a (25 cells)", ATTACKERS)):
            cells = [keep[("cross_attack_cell", det, tr, a, "all")] for tr in TRAIN_ATK for a in cols if a != tr]
            diag = [keep[("cross_attack_cell", det, tr, tr, "all")] for tr in TRAIN_ATK]
            for m in ("apcer_fpr_tau10", "apcer_fpr_tau5"):
                for what, cl in ((label, cells), ("diagonal (own attacker, 5 cells)", diag)):
                    if what.startswith("diagonal") and cols is ATTACKERS:
                        continue
                    r = average_cells(cl, m)
                    lo, hi = r.ci(m, level)
                    avg_rows.append({"detector": det, "average": what, "metric": m, "n_cells": len(cl), "point": r.point[m],
                                     "ci_lo": lo, "ci_hi": hi, "seed_sd": r.seed_sd(m), "seeds": len(r.per_seed[m])})
    pd.DataFrame(avg_rows).to_csv(out / "results_cross_attack_averages.csv", index=False)

    # ---- A3: identical-key subset (the Phase 3c pairs: frozen kl.phase3c.swap.identical_keys on the frozen sets) -----
    t0 = time.time()
    id3c = pd.read_csv(st.out_dir / "phase3c" / "identical_keys_per_seed.csv")
    ident = {}
    for a in IDENT_ATK:
        hi, bi = pair_index(H, B[a])
        same = identical_keys(H, hi, B[a], bi)
        ident[a] = hi[same]                  # hi == bi == arange (one-to-one, same order; asserted above)
        n3c = id3c[(id3c.attacker == a)]["n_pairs"].unique()
        check(f"identical-key pairs {a}: count vs phase3c identical_keys_per_seed.csv", len(ident[a]), float(n3c[0]) if len(n3c) == 1 else -1, 0)

    def ident_cell(per: list[SeedScores], a: str) -> BootResult:
        sel = ident[a]
        return evaluate_cell([SeedScores(p.h[sel], p.b[sel], p.taus) for p in per], h_u[sel], b_u[sel], W, seeds=seeds[:len(per)] if len(per) > 1 else [0])

    for a in IDENT_ATK:
        per = [SeedScores(-z["test_m"], -z[f"bot_m_{a}"], {"tau5": float(z["tau5_primary"]), "tau10": float(z["tau10_primary"])}) for z in d4["D4-primary"].values()]
        res = ident_cell(per, a)
        add("identical_keys", "D4-primary", "attack-free", a, res, "phase3b saved scores; phase3c pairs", subset="identical keys")
        per5 = [SeedScores(z["test_h"], z[f"bot_{a}"], {"tau5": float(z["tau5"]), "tau10": float(z["tau10"])}) for z in d5e5[a].values()]
        res5 = ident_cell(per5, a)
        add("identical_keys", "D5 (E5, own attacker)", a, a, res5, "phase3b saved scores of D5<-A (E5) tested on A; phase3c pairs", subset="identical keys")
        perk = [SeedScores(-z["test_m"], -z[f"bot_m_{a}"], {"tau5": float(z["tau5_primary"]), "tau10": float(z["tau10_primary"])}) for z in d4["D4-nokey"].values()]
        add("identical_keys", "D4-nokey", "attack-free", a, ident_cell(perk, a), "phase3b saved scores; phase3c pairs", subset="identical keys")
        for det, r in (("D4-primary", res), ("D5 (E5, own attacker)", res5)):
            ref = id3c[(id3c.detector == det) & (id3c.attacker == a)].set_index("seed")
            for s, v in zip(seeds, r.per_seed["bpcer_fnr_tau10"]):
                check(f"identical keys {det} {a} seed{s} BPCER@tau10 vs phase3c", v, ref.loc[s, "bpcer_tau10_humans"])
            for s, v in zip(seeds, r.per_seed["apcer_fpr_tau10"]):
                check(f"identical keys {det} {a} seed{s} APCER@tau10 vs phase3c", v, ref.loc[s, "apcer_tau10_bots"])
            for s, v in zip(seeds, r.per_seed["auc"]):
                check(f"identical keys {det} {a} seed{s} AUC vs phase3c", v, ref.loc[s, "auc"])
    timings["identical_keys"] = time.time() - t0

    # ---- A4: D6 (fit on TRAIN humans only; tau on VAL humans only; test split once) --------------------------------
    t0 = time.time()
    enc = load_encoder(in3b)
    Htr = SeqSet.load(in3b / "interim" / "human_train_all.npz")
    if Htr.kind != "human" or Htr.split != "train":
        raise AssertionError("D6 must be fitted on training humans")
    others = [u for s in ("val", "test", "attacker_pool") for u in manifest["users"][s]]
    assert_train_only(np.unique(Htr.user_ids), manifest["users"]["train"], others)
    tr_steps = steps_from_seqset(Htr, enc)
    d6 = D6Model.fit(tr_steps, enc.vocab_size)
    val_steps = steps_from_seqset(Hval, enc)
    d6_meta = {"n_train_sessions": Htr.n_sessions, "n_train_users": int(len(np.unique(Htr.user_ids))), "n_train_steps": int(len(tr_steps.key)),
               "n_train_press_steps": int(tr_steps.has_p.sum()), "vocab_size": enc.vocab_size, "max_len": enc.max_len,
               "edges_h": d6.edges_h, "edges_p": d6.edges_p, "context_usage_train": d6.context_usage(tr_steps),
               "context_usage_val": d6.context_usage(val_steps), "n_hold_contexts_used": int((d6.n_h[:-1] >= 50).sum()),
               "n_pair_contexts_used": int((d6.n_p[:enc.vocab_size ** 2] >= 50).sum()),
               "n_next_key_contexts_used": int((d6.n_p[enc.vocab_size ** 2:-1] >= 50).sum()), "variants": {}}
    np.savez_compressed(out / "d6_model.npz", n_h=d6.n_h, n_p=d6.n_p, mu_h=d6.mu_h, sd_h=d6.sd_h, mu_p=d6.mu_p, sd_p=d6.sd_p,
                        edges_h=np.array(d6.edges_h), edges_p=np.array(d6.edges_p), hist_h=d6.hist_h, hist_p=d6.hist_p)
    d6_val = {v: d6.session_scores(val_steps, v) for v in VARIANTS}
    d6_taus = {v: thresholds(-d6_val[v]) for v in VARIANTS}
    for v in VARIANTS:
        vt.append({"detector": f"D6-{v}", "seed": "-", **{f"val_reject_{k}": float(np.mean(-d6_val[v] < t)) for k, t in d6_taus[v].items()}})
        d6_meta["variants"][v] = {"tau5": d6_taus[v]["tau5"], "tau10": d6_taus[v]["tau10"]}
    write_json(out / "d6_thresholds.json", d6_taus)                       # saved BEFORE the test split is touched
    ledger = Ledger(out / "test_access_phase4.json", args.allow_rerun)
    te_steps = steps_from_seqset(H, enc)
    bot_steps = {a: steps_from_seqset(B[a], enc) for a in ATTACKERS}
    d6_scores = {}
    for v in VARIANTS:
        ledger.claim(f"D6-{v}")
        d6_scores[v] = {"test_h": d6.session_scores(te_steps, v), **{f"bot_{a}": d6.session_scores(bot_steps[a], v) for a in ATTACKERS}}
    np.savez_compressed(out / "d6_scores.npz", **{f"{v}__{k}": x for v in VARIANTS for k, x in d6_scores[v].items()},
                        **{f"{v}__val_h": d6_val[v] for v in VARIANTS})
    for v in VARIANTS:
        name = f"D6-{v}"
        for a in ATTACKERS:
            per = [SeedScores(-d6_scores[v]["test_h"], -d6_scores[v][f"bot_{a}"], d6_taus[v])]
            add("human_only", name, "attack-free", a, evaluate_cell(per, h_u, b_u, W, seeds=[0]), "Phase 4 D6 (pre-registered, added after the main evaluation)")
        for a in IDENT_ATK:
            per = [SeedScores(-d6_scores[v]["test_h"], -d6_scores[v][f"bot_{a}"], d6_taus[v])]
            add("identical_keys", name, "attack-free", a, ident_cell(per, a), "Phase 4 D6; phase3c pairs", subset="identical keys")
    write_json(out / "d6_meta.json", d6_meta)
    timings["D6"] = time.time() - t0
    log.info("D6 done (%.0fs); taus %s", timings["D6"], d6_taus)

    # ---- D5 thresholds: recompute validation-human logits from the frozen weights (check only) ---------------------
    if not args.skip_d5_val_check:
        t0 = time.time()
        vt += d5_val_check(st, in3b, enc, Hval, seeds, p3, check, out)
        timings["D5_val_check"] = time.time() - t0
        log.info("D5 validation check done (%.0fs)", timings["D5_val_check"])

    # ---- write ------------------------------------------------------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(out / "results_all.csv", index=False)
    pd.DataFrame(vt).to_csv(out / "thresholds_val_rejection.csv", index=False)
    rp = pd.DataFrame(repro)
    rp.to_csv(out / "reproduction_check.csv", index=False)
    log.info("reproduction checks: %d, failed: %d", len(rp), int((~rp["ok"]).sum()))
    if (~rp["ok"]).any():
        log.warning("FAILED checks:\n%s", rp[~rp["ok"]].to_string(index=False))
    timings["total"] = time.time() - t_all
    write_json(out / "run_meta.json", {
        "mode": "quick" if st.quick else "full", "bootstrap_reps": reps, "ci_level": level, "seed": seed, "seq_seeds": seeds,
        "n_test_users": int(len(users)), "n_test_human_sessions": int(H.n_sessions), "n_val_users": int(len(val_users)),
        "n_val_human_sessions": int(Hval.n_sessions), "match_rule": match_rule, "d3_refit_reproduces_phase2": bool(d3_ok),
        "identical_key_pairs": {a: int(len(v)) for a, v in ident.items()},
        "test_weights": "kl.phase2.evaluation.user_weights(n_test_users, reps, seed) over np.sort(unique test user ids) (= Phase 2 / 3b replicates)",
        "val_weights": "multinomial, np.random.default_rng(SeedSequence(seed, spawn_key=(4,))) over np.unique(val user ids)",
        "timings_seconds": timings, "n_checks": int(len(rp)), "n_failed_checks": int((~rp["ok"]).sum()),
    })
    assert_frozen_p4(st)
    log.info("done in %.0fs; wrote %s", timings["total"], out)


def d5_val_check(st, in3b: Path, enc, Hval: SeqSet, seeds: list[int], p3: dict, check, out: Path) -> list[dict]:
    """Recompute D5 validation-human logits with the frozen weights; compare tau5 / tau10 with the saved values."""
    from kl.phase3b import nets  # noqa: PLC0415  (sets CUBLAS_WORKSPACE_CONFIG before torch touches CUDA)
    import torch  # noqa: PLC0415

    device = nets.pick_device()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    mc, ebs = p3["model"], int(p3["training"]["eval_batch_size"])
    val = enc.encode(Hval)
    rows, saved = [], {}
    for run in [f"D5<-{a}" for a in TRAIN_ATK] + [f"D5<-all-but-{a}" for a in TRAIN_ATK]:
        for s in seeds:
            sp = score_path(in3b, f"{run}/seed{s}")
            z = load_scores(sp)
            m = nets.D5Model(enc.vocab_size, int(mc["embedding_dim"]), int(mc["hidden"]), int(mc["num_layers"]), float(mc["dropout"])).to(device)
            m.load_state_dict(torch.load(in3b / "models" / (sp.stem + ".pt"), map_location=device))
            m.eval()
            v = nets.d5_logits(m, val, ebs, device)
            saved[f"{sp.stem}"] = v
            taus = thresholds(v)
            for k in ("tau5", "tau10"):
                check(f"{run} seed{s} {k} recomputed on val humans vs saved", taus[k], float(z[k]), 1e-5)
            rows.append({"detector": run, "seed": s, **{f"val_reject_{k}": float(np.mean(v < float(z[k]))) for k in ("tau5", "tau10")}})
            del m
    np.savez_compressed(out / "d5_val_logits_recomputed.npz", **saved)
    return rows


if __name__ == "__main__":
    main()
