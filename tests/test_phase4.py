"""Phase 4 tests (docs/PHASE4_SPEC.md, "Tests"): bootstrap, thresholds, D6, and reproduction of frozen numbers.

The synthetic tests always run; the tests marked ``needs_full`` / ``needs_outputs`` read the frozen outputs and are
skipped when they are absent.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from kl.phase2.evaluation import cell_metrics, user_weights
from kl.phase4_bootstrap import (METRICS, SeedScores, ValScores, average_cells, evaluate_cell, identity_weights, point_auc,
                                 quantile_counts, sensitivity_cell, test_weights, thresholds, user_index, val_weights)
from kl.phase4_d6 import N_BINS, N_MIN, SD_FLOOR, D6DataError, D6Model, Steps, assert_train_only, gauss_nll

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "outputs" / "full"
# Saved scores, sequences and model weights are not in the public repository (only the CSV / JSON reports are), so the
# skip conditions check for the binary files these tests actually read, not for the reports.
_NEEDED = ("phase3b/e4_open_set.csv", "phase3b/scores/D4_seed42.npz", "phase3b/interim/human_test.npz",
           "phase3b/interim/human_train_all.npz", "results/split_manifest.json")
_NEEDED_FULL = _NEEDED + ("model/model.joblib", "phase2/bots_A1_test.parquet")


def _has(base: Path, files) -> bool:
    return all((base / f).exists() for f in files)


MODE_DIR = FULL if _has(FULL, _NEEDED) else ROOT / "outputs" / "quick"
needs_outputs = pytest.mark.skipif(not _has(MODE_DIR, _NEEDED), reason="no saved Phase 3b scores / sequences")
needs_full = pytest.mark.skipif(not _has(FULL, _NEEDED_FULL), reason="no full-mode saved scores / model")
SEEDS = [42, 43, 44]


def _synthetic(n_users: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    per_user = rng.integers(1, 6, n_users)
    users = np.repeat(np.arange(n_users), per_user)
    h = rng.normal(1.0, 1.0, len(users))
    b = rng.normal(0.0, 1.0, len(users))          # one matched bot per human session (same user)
    return users, h, b


# ------------------------------------------------------------------------------------------------ bootstrap
def test_identity_resample_equals_point_estimate():
    users, h, b = _synthetic()
    per = [SeedScores(h + d, b, {"tau5": -0.5, "tau10": 0.0}) for d in (0.0, 0.1, -0.2)]
    res = evaluate_cell(per, users, users, identity_weights(users.max() + 1), seeds=SEEDS)
    for m in METRICS:
        assert res.boot[m][0] == pytest.approx(res.point[m], abs=1e-12), m


def test_same_replicates_and_cis_as_phase2():
    users, h, b = _synthetic(60, 1)
    w = test_weights(60, 200, 42)
    assert np.array_equal(w, user_weights(60, 200, 42))
    taus = {"tau5": -0.5, "tau10": 0.0}
    res = evaluate_cell([SeedScores(h, b, taus)], users, users, w)
    ref = cell_metrics(h, users, b, users, taus, w, 0.95)
    for m in METRICS:
        assert res.point[m] == pytest.approx(ref[m], abs=1e-12)
        lo, hi = res.ci(m)
        assert (lo, hi) == pytest.approx((ref[f"{m}_ci_lo"], ref[f"{m}_ci_hi"]), abs=1e-12)


def test_resampling_is_by_user_humans_and_matched_bots_move_together():
    users, h, b = _synthetic(10, 2)
    w = np.zeros((1, 10))
    w[0, 3], w[0, 7] = 2, 1                       # user 3 drawn twice, user 7 once, everyone else not drawn
    taus = {"tau5": -0.5, "tau10": 0.0}
    res = evaluate_cell([SeedScores(h, b, taus)], users, users, w)
    sel = np.r_[np.flatnonzero(users == 3), np.flatnonzero(users == 3), np.flatnonzero(users == 7)]
    explicit = evaluate_cell([SeedScores(h[sel], b[sel], taus)], users[sel], users[sel], identity_weights(10))
    for m in METRICS:
        assert res.boot[m][0] == pytest.approx(explicit.point[m], abs=1e-12), m


def test_seed_average_inside_each_replicate():
    users, h, b = _synthetic(30, 3)
    w = test_weights(30, 50, 42)
    per = [SeedScores(h + d, b, {"tau5": -0.5, "tau10": 0.0}) for d in (0.0, 0.3)]
    both = evaluate_cell(per, users, users, w)
    single = [evaluate_cell([p], users, users, w) for p in per]
    for m in METRICS:
        assert np.allclose(both.boot[m], (single[0].boot[m] + single[1].boot[m]) / 2)
    avg = average_cells(single, "apcer_fpr_tau10")
    assert np.allclose(avg.boot["apcer_fpr_tau10"], (single[0].boot["apcer_fpr_tau10"] + single[1].boot["apcer_fpr_tau10"]) / 2)


def test_point_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    _, h, b = _synthetic(50, 4)
    h = np.round(h, 1)                            # ties
    b = np.round(b, 1)
    assert point_auc(h, b) == pytest.approx(roc_auc_score(np.r_[np.ones(len(h)), np.zeros(len(b))], np.r_[h, b]), abs=1e-12)


def test_user_index_rejects_unknown_users():
    with pytest.raises(ValueError):
        user_index(np.array([1, 2, 3]), np.array([2, 4]))


# ------------------------------------------------------------------------------------------------ thresholds
def test_thresholds_reject_5_and_10_percent_of_val_humans():
    v = np.random.default_rng(5).normal(size=43291)
    taus = thresholds(v)
    assert abs(np.mean(v < taus["tau5"]) - 0.05) <= 0.002
    assert abs(np.mean(v < taus["tau10"]) - 0.10) <= 0.002


def test_quantile_counts_equals_repeated_quantile():
    rng = np.random.default_rng(6)
    x = np.sort(rng.normal(size=500))
    c = rng.integers(0, 4, 500)
    for q in (0.05, 0.10, 0.5):
        assert quantile_counts(x, c, q) == pytest.approx(float(np.quantile(np.repeat(x, c), q)), abs=1e-12)


def test_sensitivity_uses_resampled_validation_thresholds():
    users, h, b = _synthetic(20, 7)
    vu = np.repeat(np.arange(15), 4)
    v = np.random.default_rng(8).normal(1.0, 1.0, len(vu))
    wv = val_weights(15, 30, 42)
    w = test_weights(20, 30, 42)
    out = sensitivity_cell([SeedScores(h, b, {})], [ValScores(v)], vu, wv, users, users, w)
    for r in (0, 7, 29):
        vr = np.repeat(v, wv[r][vu].astype(int))
        t = float(np.quantile(vr, 0.10))
        hr, br = np.repeat(h, w[r][users].astype(int)), np.repeat(b, w[r][users].astype(int))
        assert out["bpcer_fnr_tau10"][r] == pytest.approx(np.mean(hr < t), abs=1e-12)
        assert out["apcer_fpr_tau10"][r] == pytest.approx(np.mean(br >= t), abs=1e-12)
    assert not np.array_equal(wv, test_weights(15, 30, 42))      # an independent stream


@needs_outputs
def test_saved_d4_thresholds_come_from_val_humans():
    for s in SEEDS:
        z = np.load(MODE_DIR / "phase3b" / "scores" / f"D4_seed{s}.npz")
        taus = thresholds(-z["val_m"])
        assert taus["tau5"] == float(z["tau5_primary"]) and taus["tau10"] == float(z["tau10_primary"])
        assert abs(np.mean(-z["val_m"] < taus["tau5"]) - 0.05) <= 0.002
        assert abs(np.mean(-z["val_m"] < taus["tau10"]) - 0.10) <= 0.002


# ------------------------------------------------------------------------------------------------ D6
def _steps(keys, nxt, h, p, sess=None) -> Steps:
    keys, nxt = np.asarray(keys, dtype="int64"), np.asarray(nxt, dtype="int64")
    sess = np.zeros(len(keys), dtype="int64") if sess is None else np.asarray(sess, dtype="int64")
    return Steps(sess, keys, nxt, np.asarray(h, dtype="float64"), np.asarray(p, dtype="float64"), int(sess.max()) + 1)


def _toy_model(V: int = 6) -> tuple[D6Model, Steps]:
    rng = np.random.default_rng(9)
    # key 2 -> key 3: 200 pairs; key 4 -> key 3: 30 pairs (rare pair, but next key 3 has 230 >= 50); key 5: 10 holds only
    k = np.r_[np.full(200, 2), np.full(30, 4), np.full(10, 5)]
    n = np.r_[np.full(200, 3), np.full(30, 3), np.full(10, -1)]
    h = rng.normal(4.5, 0.3, len(k))
    p = np.where(n >= 0, rng.normal(5.3, 0.4, len(k)), np.nan)
    st = _steps(k, n, h, p, np.arange(len(k)))
    return D6Model.fit(st, V), st


def test_d6_gauss_step_nll_equals_scipy():
    m, _ = _toy_model()
    one = _steps([2], [3], [4.9], [5.0])
    nh, np_ = m.step_nll(one, "gauss")
    ch, cp = m.hold_context(one.key)[0], m.press_context(one.key, one.nxt)[0]
    want_h = -norm.logpdf(4.9, loc=m.mu_h[ch], scale=m.sd_h[ch])
    want_p = -norm.logpdf(5.0, loc=m.mu_p[cp], scale=m.sd_p[cp])
    assert abs(nh[0] - want_h) <= 1e-9 and abs(np_[0] - want_p) <= 1e-9
    assert abs(m.session_scores(one, "gauss")[0] - (want_h + want_p)) <= 1e-9
    last = _steps([2], [-1], [4.9], [np.nan])                     # last key: only NLL(h)
    assert abs(m.session_scores(last, "gauss")[0] - want_h) <= 1e-9
    assert gauss_nll(np.array([1.0]), np.array([0.0]), np.array([SD_FLOOR]))[0] == pytest.approx(-norm.logpdf(1.0, 0.0, SD_FLOOR), abs=1e-9)


def test_d6_backoff_levels():
    V = 6
    m, _ = _toy_model(V)
    # frequent pair (2, 3) -> pair level; rare pair (4, 3) (30 < 50) -> next-key level; pair (1, 1) with unseen next key -> global
    cp = m.press_context(np.array([2, 4, 1]), np.array([3, 3, 1]))
    assert cp[0] == 2 * V + 3
    assert m.n_p[4 * V + 3] == 30 and cp[1] == V * V + 3
    assert cp[2] == V * V + V
    # hold: seen key 2 -> key level; key 5 has 10 < 50 holds -> global; unseen key 0 -> global
    ch = m.hold_context(np.array([2, 5, 0]))
    assert ch[0] == 2 and ch[1] == V and ch[2] == V
    assert m.n_h[V] == m.n_h[:V].sum() and m.n_p[V * V + V] == 230 and N_MIN == 50


def test_d6_hist_probabilities_sum_to_one():
    m, st = _toy_model()
    for tab in (m.hist_h, m.hist_p):
        assert tab.shape[1] == N_BINS
        assert np.allclose(tab.sum(1), 1.0, atol=1e-12)
    nh, _ = m.step_nll(st, "hist")
    assert np.isfinite(nh).all()


def test_d6_speed_variant_removes_a_constant_shift():
    m, _ = _toy_model()
    a = _steps([2, 2, 2], [3, 3, -1], [4.4, 4.6, 4.5], [5.2, 5.4, np.nan])
    b = _steps([2, 2, 2], [3, 3, -1], [4.4 + 0.3, 4.6 + 0.3, 4.5 + 0.3], [5.2 - 0.5, 5.4 - 0.5, np.nan])
    assert m.session_scores(a, "gauss-speed")[0] == pytest.approx(m.session_scores(b, "gauss-speed")[0], abs=1e-12)
    assert m.session_scores(a, "gauss")[0] != pytest.approx(m.session_scores(b, "gauss")[0])


def test_d6_fitting_users_must_be_train_users():
    assert_train_only([1, 2], [1, 2, 3], [7, 8])
    with pytest.raises(D6DataError):
        assert_train_only([1, 9], [1, 2, 3], [7, 8])
    with pytest.raises(D6DataError):
        assert_train_only([1, 7], [1, 2, 3, 7], [7, 8])


@needs_outputs
def test_d6_real_fitting_set_is_train_users_only():
    man = json.loads((MODE_DIR / "results" / "split_manifest.json").read_text())
    with np.load(MODE_DIR / "phase3b" / "interim" / "human_train_all.npz") as z:
        users = np.unique(z["user_ids"])
    others = [u for s in ("val", "test", "attacker_pool") for u in man["users"][s]]
    assert_train_only(users, man["users"]["train"], others)


# ------------------------------------------------------------------------------------------------ frozen numbers
@needs_outputs
def test_identity_resample_reproduces_saved_phase3b_point_estimates():
    e4 = pd.read_csv(MODE_DIR / "phase3b" / "e4_open_set.csv")
    with np.load(MODE_DIR / "phase3b" / "interim" / "human_test.npz") as z:
        uid = z["user_ids"]
    users = np.unique(uid)
    hu = user_index(users, uid)
    zs = [np.load(MODE_DIR / "phase3b" / "scores" / f"D4_seed{s}.npz") for s in SEEDS]
    for a in ("A1", "A2", "A5a"):
        per = [SeedScores(-z["test_m"], -z[f"bot_m_{a}"], {"tau5": float(z["tau5_primary"]), "tau10": float(z["tau10_primary"])}) for z in zs]
        res = evaluate_cell(per, hu, hu, identity_weights(len(users)), seeds=SEEDS)
        ref = e4[(e4.detector == "D4-primary") & (e4.tested_on == a)].iloc[0]
        for m in METRICS:
            assert res.boot[m][0] == pytest.approx(ref[f"{m}_mean"], abs=1e-9), (a, m)
            assert res.point[m] == pytest.approx(ref[f"{m}_mean"], abs=1e-9), (a, m)
    e6 = pd.read_csv(MODE_DIR / "phase3b" / "e6_loao.csv")
    zs = [np.load(MODE_DIR / "phase3b" / "scores" / f"D5_from_all-but-A4_seed{s}.npz") for s in SEEDS]
    per = [SeedScores(z["test_h"], z["bot_A4"], {"tau5": float(z["tau5"]), "tau10": float(z["tau10"])}) for z in zs]
    res = evaluate_cell(per, hu, hu, identity_weights(len(users)), seeds=SEEDS)
    ref = e6[(e6.detector == "D5") & (e6.tested_on == "A4")].iloc[0]
    for m in METRICS:
        assert res.boot[m][0] == pytest.approx(ref[f"{m}_mean"], abs=1e-9), m


@needs_full
def test_manuscript_numbers_d4_d5():
    e4 = pd.read_csv(FULL / "phase3b" / "e4_open_set.csv")
    with np.load(FULL / "phase3b" / "interim" / "human_test.npz") as z:
        uid = z["user_ids"]
    zs = [np.load(FULL / "phase3b" / "scores" / f"D4_seed{s}.npz") for s in SEEDS]
    per = [SeedScores(-z["test_m"], -z["bot_m_A2"], {"tau5": float(z["tau5_primary"]), "tau10": float(z["tau10_primary"])}) for z in zs]
    users = np.unique(uid)
    res = evaluate_cell(per, user_index(users, uid), user_index(users, uid), identity_weights(len(users)), seeds=SEEDS)
    assert round(100 * res.point["apcer_fpr_tau10"], 1) == 0.6                          # D4-primary on A2: 0.6%
    assert e4[(e4.detector == "D4-primary") & (e4.tested_on == "A2")]["apcer_fpr_tau10_mean"].iloc[0] == pytest.approx(res.point["apcer_fpr_tau10"], abs=1e-12)
    e6 = pd.read_csv(FULL / "phase3b" / "e6_loao.csv")
    assert round(100 * e6[(e6.detector == "D5") & (e6.tested_on == "A2")]["apcer_fpr_tau10_mean"].iloc[0], 1) == 1.2  # D5 LOAO A2: 1.2%
    e5 = pd.read_csv(FULL / "phase3b" / "e5_cross_attack.csv")
    off = e5[(e5.detector == "D5") & (e5.trained_on != e5.tested_on) & (e5.tested_on != "A5a")]
    assert len(off) == 20 and round(100 * off["apcer_fpr_tau10_mean"].mean(), 1) == 37.7


@needs_full
def test_manuscript_numbers_d1_on_a1_both_files():
    """D1 on A1: 5.86% (Phase 1, results/a1_results.json) and 5.77% (Phase 2, phase2/open_set_table.csv).

    Both use the same frozen D1 and tau10; they differ because the A1 bots were drawn twice with different random
    streams (Phase 1 step 07: one stream default_rng([42, 3]); Phase 2 step 11: default_rng([42, code, split, i]) per bot).
    """
    from kl.model import LivenessModel
    a1 = json.loads((FULL / "results" / "a1_results.json").read_text())
    assert round(100 * a1["tau10"]["FPR_bots_accepted"], 1) == 5.9                       # 2,536 / 43,248 = 5.86%
    model = LivenessModel.load(FULL / "model")
    bots = pd.read_parquet(FULL / "phase2" / "bots_A1_test.parquet")
    acc = float(np.mean(model.score(bots) >= model.thresholds["tau10"]))
    assert round(100 * acc, 1) == 5.8                                                     # 2,496 / 43,248 = 5.77%
    ref = pd.read_csv(FULL / "phase2" / "open_set_table.csv")
    assert ref[(ref.detector == "D1") & (ref.tested_on == "A1")]["apcer_fpr_tau10"].iloc[0] == pytest.approx(acc, abs=1e-12)
    assert math.isclose(a1["tau10"]["tau"], model.thresholds["tau10"])
