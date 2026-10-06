"""Phase 3a diagnostics: ablation subsets, Mahalanobis, variance ratios, permutation importance, per-user consistency, freeze guards."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from kl.features import FEATURE_GROUPS, FEATURE_NAMES
from kl.phase2.guard import FrozenViolation, sha256_file, verify_files
from kl.phase3a import analyses as an


# ---------------------------------------------------------------- ablation subsets
def test_ablation_subsets_are_the_requested_feature_groups():
    subsets = an.ablation_subsets({"TIMING_only": ["TIMING"], "TIMING_RHYTHM": ["TIMING", "RHYTHM"],
                                   "all_minus_BEHAVIOUR": ["TIMING", "RHYTHM", "GLOBAL"], "BEHAVIOUR_only": ["BEHAVIOUR"]})
    assert subsets["TIMING_only"] == FEATURE_GROUPS["TIMING"] and len(subsets["TIMING_only"]) == 13
    assert set(subsets["TIMING_RHYTHM"]) == set(FEATURE_GROUPS["TIMING"]) | set(FEATURE_GROUPS["RHYTHM"])
    assert set(subsets["all_minus_BEHAVIOUR"]) == set(FEATURE_NAMES) - set(FEATURE_GROUPS["BEHAVIOUR"]) and len(subsets["all_minus_BEHAVIOUR"]) == 19
    assert subsets["BEHAVIOUR_only"] == ["corrections_count", "correction_rate", "lev_norm", "shift_overlap_count"]
    assert all(c in FEATURE_NAMES for cols in subsets.values() for c in cols)


def test_ablation_refit_with_all_features_is_the_frozen_recipe():
    rng = np.random.default_rng(0)
    cols = ["hold_mean", "pp_mean", "lev_norm"]
    mk = lambda n: pd.DataFrame(np.abs(rng.normal(100, 20, size=(n, 3))), columns=cols).assign(user_id=rng.integers(0, 20, n))  # noqa: E731
    train, val, test = mk(400), mk(300), mk(200)
    bots = {"A1": mk(200)}
    users = np.sort(test["user_id"].unique())
    from kl.phase2.evaluation import user_weights
    tab, meta = an.ablation(train, val, test, bots, {"x": cols[:2], "y": cols}, 0.1, 0.3, user_weights(len(users), 20, 1), users, 0.95, 1)
    assert set(tab["trained_on"]) == {"x", "y"} and meta["x"]["n_features"] == 2 and meta["y"]["gamma_effective"] == 0.3
    assert 0 <= tab["auc"].min() and tab["auc"].max() <= 1
    assert meta["x"]["thresholds"]["tau5"] <= meta["x"]["thresholds"]["tau10"]
    tab2, meta2 = an.ablation(train, val, test, bots, {"y": cols}, 0.1, 0.3, user_weights(len(users), 20, 1), users, 0.95, 1)
    assert meta2["y"]["thresholds"] == meta["y"]["thresholds"]                     # deterministic refit


# ---------------------------------------------------------------- centrality
def test_mahalanobis_orders_typical_before_atypical_and_is_seeded():
    rng = np.random.default_rng(0)
    x_tr = rng.normal(size=(3000, 5))
    human = rng.normal(size=(1000, 5))
    typical = rng.normal(scale=0.4, size=(1000, 5))                                # 'too typical': tighter than humans
    far = rng.normal(loc=3.0, size=(1000, 5))
    names = list("abcde")
    tab, dists, info = an.centrality(x_tr, human, {"typ": typical, "far": far}, names, seed=1)
    m = tab[tab.covariance == "mcd"].set_index("group")
    assert m.loc["typ", "median_d"] < m.loc["human", "median_d"] < m.loc["far", "median_d"]
    assert m.loc["typ", "frac_below_human_median"] > 0.9 and m.loc["far", "ks_stat_vs_human"] > 0.9
    assert m.loc["human", "median_d"] == pytest.approx(np.sqrt(4.35), rel=0.1)     # chi2_5 median = 4.35
    assert info["mcd"]["zero_variance_features_in_support"] == [] and "classical" in dists
    tab2, _, _ = an.centrality(x_tr, human, {"typ": typical, "far": far}, names, seed=1)
    pd.testing.assert_frame_equal(tab, tab2)


def test_mcd_reports_zero_variance_features_in_the_support():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, 4))
    x[:, 3] = (rng.random(2000) > 0.95) * rng.normal(size=2000)                    # 95% zeros -> constant inside the support subset
    tab, _, info = an.centrality(x, x[:500], {"b": x[500:1000]}, ["a", "b", "c", "z"], seed=0)
    assert "z" in info["mcd"]["zero_variance_features_in_support"] and info["mcd"]["n_eigenvalues_below_1e-10"] >= 1
    assert np.isfinite(tab["median_d"]).all()


def test_variance_ratio_exact():
    rng = np.random.default_rng(2)
    h = pd.DataFrame({"f1": rng.normal(0, 2.0, 5000), "f2": rng.normal(0, 1.0, 5000)})
    b = pd.DataFrame({"f1": rng.normal(0, 1.0, 5000), "f2": rng.normal(0, 3.0, 5000)})
    vr = an.variance_ratios(h, {"A": b}, h.to_numpy(), {"A": b.to_numpy()}, ["f1", "f2"]).set_index("feature")
    assert vr.loc["f1", "var_ratio_transformed"] == pytest.approx(b["f1"].var(ddof=0) / h["f1"].var(ddof=0))
    assert vr.loc["f1", "var_ratio_transformed"] < 1 < vr.loc["f2", "var_ratio_transformed"]        # too homogeneous vs too dispersed
    assert vr.loc["f1", "var_ratio_raw"] == pytest.approx(vr.loc["f1", "var_ratio_transformed"])


# ---------------------------------------------------------------- permutation importance
def test_permutation_importance_finds_the_discriminating_feature():
    rng = np.random.default_rng(3)
    n = 3000
    hum = rng.normal(size=(n, 4))
    bot = rng.normal(size=(n, 4))
    bot[:, 2] += 3.0                                                               # only feature 2 gives the bot away
    rf = an.fit_single_attacker_d3(hum[:2000], bot[:2000], 0, {"n_estimators": 60, "class_weight": "balanced"})
    imp = an.importance_table(rf, hum[2000:], bot[2000:], ["a", "b", "c", "d"], 500, 5, 0, "A2")
    assert imp.iloc[0]["feature"] == "c" and imp.iloc[0]["rank"] == 1 and imp.iloc[0]["importance_mean"] > 0.2
    assert imp[imp.feature != "c"]["importance_mean"].abs().max() < 0.03
    assert imp["baseline_auc"].iloc[0] > 0.95 and len(imp) == 4
    imp2 = an.importance_table(rf, hum[2000:], bot[2000:], ["a", "b", "c", "d"], 500, 5, 0, "A2")
    pd.testing.assert_frame_equal(imp, imp2)                                       # seeded sample + seeded permutations


# ---------------------------------------------------------------- per-user consistency
def _sessions(n_users=60, k=15, persistent=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        base_pp, base_h = rng.normal(200, 40), rng.normal(100, 15)
        for _ in range(k):
            rows.append({"user_id": u, "pp_mean": (base_pp if persistent else rng.normal(200, 40)) + rng.normal(0, 8),
                         "hold_mean": (base_h if persistent else rng.normal(100, 15)) + rng.normal(0, 3)})
    return pd.DataFrame(rows)


def test_icc1_high_for_persistent_users_and_near_zero_without_persistence():
    hi, lo = _sessions(persistent=True), _sessions(persistent=False)
    assert an.icc1(hi["pp_mean"].to_numpy(), hi["user_id"].to_numpy()) > 0.9
    assert abs(an.icc1(lo["pp_mean"].to_numpy(), lo["user_id"].to_numpy())) < 0.1


def test_real_users_are_more_consistent_than_pseudo_users_and_construction_is_seeded():
    test = _sessions(persistent=True)
    bots = {"A2": _sessions(persistent=False, seed=5).assign(user_id=lambda d: d["user_id"]), "A3": _sessions(persistent=False, seed=6)}
    summ, per_unit = an.consistency(test, bots, ["A2", "A3"], ["pp_mean", "hold_mean"], min_sessions=10, k=15, seed=1)
    real = summ[summ.group == "real users"].set_index("feature")
    for f in ("pp_mean", "hold_mean"):
        for g in ("A2 pseudo-users", "A3 pseudo-users", "shuffled humans (control)"):
            r = summ[(summ.group == g) & (summ.feature == f)].iloc[0]
            assert r["median_std"] > 2 * real.loc[f, "median_std"] and r["mannwhitney_real_less_p"] < 1e-6 and r["ratio_median_std_vs_real"] > 2
            assert r["icc1"] < 0.15 < real.loc[f, "icc1"]
    assert real.loc["pp_mean", "n_units"] == 60 and (per_unit["n_sessions"] >= 10).all()
    s2, _ = an.consistency(test, bots, ["A2", "A3"], ["pp_mean", "hold_mean"], 10, 15, 1)
    pd.testing.assert_frame_equal(summ, s2)


def test_pseudo_users_have_k_distinct_sessions_each():
    df = pd.DataFrame({"x": np.arange(200)})
    p = an.pseudo_users(df, 30, 15, np.random.default_rng(0))
    assert len(p) == 450 and p.groupby("unit")["x"].nunique().eq(15).all()


# ---------------------------------------------------------------- freeze guards
def test_phase2_sources_still_match_the_freeze_record():
    rec = json.loads(Path("docs/phase2_frozen_hashes.json").read_text())
    verify_files(rec["source_files"], Path("."))
    assert any(p.endswith("a4_imitator.py") for p in rec["source_files"]) and any(p.endswith("detectors.py") for p in rec["source_files"])


def test_verify_files_detects_a_modified_phase2_file(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1")
    good = {"a.py": sha256_file(f)}
    f.write_text("x = 2")
    with pytest.raises(FrozenViolation):
        verify_files(good, tmp_path)
