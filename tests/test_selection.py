"""Transform, attack-free selection (reference set, guard, tie-break, degeneracy), model round trip, bootstrap."""
import numpy as np
import pandas as pd
import pytest

from kl.attacks.a1_gaussian import A1Gaussian
from kl.evaluate import bootstrap_reject_rate_by_user, confusion_matrix, roc_auc
from kl.features import BOUNDED_RATE_FEATURES
from kl.model import FeatureTransform, LivenessModel, signed_log1p, tau_for_fnr
from kl.selection import (AttackDataError, DegenerateSelectionError, NoAdmissibleConfigError, admissible_mask, guard_no_attack,
                          product_of_marginals, run_grid, select_best, uniform_outliers)

COLS = ["a", "b", "c"]


def _frame(n, split, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 3)), columns=COLS)
    df["split"] = split
    return df


def _grid_args(**kw):
    return dict(feature_cols=COLS, nu_grid=[0.1], gamma_grid=["scale"], target_fnrs=[0.05, 0.1], n_reference=100, n_uniform=100, seed=0, **kw)


# ---------------------------------------------------------------- transform
def test_signed_log1p_is_odd_monotone():
    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    y = signed_log1p(x)
    assert np.allclose(y, [-np.log1p(1000), -np.log(2), 0, np.log(2), np.log1p(1000)]) and np.all(np.diff(y) > 0)


def test_transform_skips_log_for_bounded_rates_and_uses_train_stats_only():
    names = ["pause_rate", "hold_mean", "lev_norm"]
    assert {"pause_rate", "lev_norm"} <= BOUNDED_RATE_FEATURES
    rng = np.random.default_rng(0)
    train = np.column_stack([rng.uniform(0, 1, 500), rng.lognormal(5, 1, 500), rng.uniform(0, 0.3, 500)])
    tf = FeatureTransform(names).fit(train)
    assert list(tf.log_mask) == [False, True, False]
    pre = tf._pre(train)
    assert np.allclose(pre[:, 0], train[:, 0]) and np.allclose(pre[:, 1], np.log1p(train[:, 1]))
    out = tf.transform(train)
    assert np.allclose(np.median(out, axis=0), 0, atol=1e-9)                      # median-centred
    q75, q25 = np.percentile(out, [75, 25], axis=0)
    assert np.allclose(q75 - q25, 1.0)                                               # IQR-scaled
    other = train[:10] * 3 + 1                                                       # a different (val/test-like) matrix
    assert np.allclose(tf.transform(other), FeatureTransform(names).fit(train).transform(other))
    assert not np.allclose(FeatureTransform(names).fit(other).transform(other), tf.transform(other))  # fitting elsewhere differs


def test_transform_handles_negative_flights_and_zero_iqr_columns():
    names = ["flight_mean", "shift_overlap_count"]
    x = np.column_stack([np.linspace(-100, 400, 200), np.r_[np.zeros(190), np.arange(10)]])
    out = FeatureTransform(names).fit_transform(x)
    assert np.isfinite(out).all()  # zero-IQR column is left with scale 1, no division by zero


# ---------------------------------------------------------------- guard
def test_guard_rejects_attack_inputs():
    with pytest.raises(AttackDataError):
        guard_no_attack(x=A1Gaussian())
    with pytest.raises(AttackDataError):
        guard_no_attack(x=_frame(5, "train", 0).assign(is_attack=True))
    with pytest.raises(AttackDataError):
        guard_no_attack(x=_frame(5, "attacker_pool", 0))
    tagged = _frame(5, "train", 0)
    tagged.attrs["kind"] = "attack"
    with pytest.raises(AttackDataError):
        guard_no_attack(x=tagged)
    guard_no_attack(x=_frame(5, "train", 0), y=_frame(5, "val", 1))  # humans are fine


def test_run_grid_raises_on_attack_or_wrong_split():
    with pytest.raises(AttackDataError):
        run_grid(_frame(50, "train", 0).assign(is_attack=False), _frame(20, "val", 1), **_grid_args())
    with pytest.raises(AttackDataError):
        run_grid(_frame(50, "train", 0), _frame(20, "test", 1), **_grid_args())
    with pytest.raises(AttackDataError):
        run_grid(_frame(50, "attacker_pool", 0), _frame(20, "val", 1), **_grid_args())


# ---------------------------------------------------------------- reference set
def test_product_of_marginals_resamples_each_column_independently():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(2000, 1))
    x = np.column_stack([z[:, 0], 2 * z[:, 0] + 0.01 * rng.normal(size=2000), rng.normal(size=2000)])  # cols 0 and 1 almost perfectly correlated
    ref = product_of_marginals(x, 5000, seed=3)
    assert ref.shape == (5000, 3)
    for j in range(3):
        assert np.isin(ref[:, j], x[:, j]).all()                                     # values come from the train column (with replacement)
    assert np.corrcoef(x[:, 0], x[:, 1])[0, 1] > 0.99 and abs(np.corrcoef(ref[:, 0], ref[:, 1])[0, 1]) < 0.1  # dependence destroyed
    assert np.array_equal(ref, product_of_marginals(x, 5000, seed=3)) and not np.array_equal(ref, product_of_marginals(x, 5000, seed=4))


def test_uniform_diagnostic_is_seeded_box():
    x = np.random.default_rng(0).normal(size=(100, 3))
    u = uniform_outliers(x, 500, 1)
    assert (u >= x.min(axis=0)).all() and (u <= x.max(axis=0)).all() and np.array_equal(u, uniform_outliers(x, 500, 1))


# ---------------------------------------------------------------- selection rule
def test_select_best_tie_rule_uses_reference_acceptance():
    g = pd.DataFrame({
        "nu": [0.05, 0.01, 0.02, 0.01], "gamma_effective": [0.1, 0.3, 0.001, 0.001],
        "ref_accept_tau5": [0.0300, 0.0305, 0.0500, 0.0304],
        "uniform_accept_tau5": [0.0, 0.0, 0.0, 0.0],  # diagnostic column must be ignored
        "val_fnr_native": [0.05, 0.01, 0.02, 0.01],   # all admissible (== nu)
    })
    # min = 0.0300; within 0.001 -> rows 0, 1, 3 tied; smallest nu = 0.01 (rows 1, 3); then smaller gamma -> row 3
    assert select_best(g, 0.001, min_spread=0.01) == 3
    g2 = g.assign(ref_accept_tau5=[0.0100, 0.0305, 0.0500, 0.0304])
    assert select_best(g2, 0.001, min_spread=0.01) == 0  # clear winner outside the tie band


def test_selection_ignores_uniform_diagnostic():
    g = pd.DataFrame({"nu": [0.05, 0.01], "gamma_effective": [0.1, 0.1], "ref_accept_tau5": [0.02, 0.20],
                      "uniform_accept_tau5": [0.9, 0.0], "val_fnr_native": [0.05, 0.01]})
    assert select_best(g, 0.001) == 0  # lowest REFERENCE acceptance wins even though its uniform acceptance is worst


def test_degeneracy_guard_stops_instead_of_selecting():
    g = pd.DataFrame({"nu": [0.01, 0.02], "gamma_effective": [0.1, 0.1], "ref_accept_tau5": [0.010, 0.019], "val_fnr_native": [0.01, 0.02]})  # spread 0.9 pp < 1 pp
    with pytest.raises(DegenerateSelectionError) as ei:
        select_best(g, 0.001, min_spread=0.01)
    assert ei.value.spread == pytest.approx(0.009) and "refusing to select" in str(ei.value)
    ok = g.assign(ref_accept_tau5=[0.010, 0.021])  # spread 1.1 pp -> selects
    assert select_best(ok, 0.001, min_spread=0.01) == 0


def test_select_best_refuses_attack_frame():
    with pytest.raises(AttackDataError):
        select_best(pd.DataFrame({"nu": [0.1], "gamma_effective": [1.0], "ref_accept_tau5": [0.0], "val_fnr_native": [0.1], "is_attack": [True]}), 0.001)


# ---------------------------------------------------------------- admissibility (Revision 2)
def test_admissible_mask_formula():
    g = pd.DataFrame({
        "nu":             [0.01,  0.01,  0.05,  0.05,  0.10,  0.10,  0.15,  0.15],
        "val_fnr_native": [0.030, 0.0311, 0.075, 0.0751, 0.150, 0.1501, 0.225, 0.2251],
    })
    # tol = max(0.02, 0.5 nu): 0.02, 0.02, 0.025, 0.025, 0.05, 0.05, 0.075, 0.075 (boundaries are inclusive)
    assert list(admissible_mask(g)) == [True, False, True, False, True, False, True, False]
    assert list(admissible_mask(g.assign(val_fnr_native=[0.01, 0.0, 0.05, 0.0, 0.10, 0.0, 0.15, 0.30]))) == [True, True, True, False, True, False, True, False]


def test_admissibility_filters_overfit_config_even_if_reference_criterion_prefers_it():
    g = pd.DataFrame({
        "nu": [0.01, 0.01, 0.01], "gamma_effective": [0.03, 0.1, 0.3], "ref_accept_tau5": [0.49, 0.04, 0.003],
        "val_fnr_native": [0.0099, 0.0185, 0.1145],   # gamma=0.3 is the overfit one (val FNR 11.5% vs nu = 1%)
    })
    assert select_best(g, 0.001) == 1                    # lowest reference acceptance among ADMISSIBLE configs, not overall
    assert list(admissible_mask(g)) == [True, True, False]


def test_admissible_tie_break_still_smaller_nu_then_gamma():
    g = pd.DataFrame({
        "nu": [0.05, 0.02, 0.02, 0.15], "gamma_effective": [0.1, 0.3, 0.1, 0.1], "ref_accept_tau5": [0.100, 0.1005, 0.1004, 0.90],
        "val_fnr_native": [0.05, 0.02, 0.02, 0.15],
    })
    assert select_best(g, 0.001) == 2  # all within 0.1 pp of the admissible best -> nu = 0.02 (rows 1, 2), then smaller gamma (row 2)


def test_no_admissible_config_stops_and_reports():
    g = pd.DataFrame({"nu": [0.01, 0.02], "gamma_effective": [0.1, 0.3], "ref_accept_tau5": [0.2, 0.5], "val_fnr_native": [0.20, 0.30]})
    with pytest.raises(NoAdmissibleConfigError) as ei:
        select_best(g, 0.001)
    assert "refusing to relax" in str(ei.value)


# ---------------------------------------------------------------- grid + model
def test_grid_smoke_and_model_roundtrip(tmp_path):
    train, val = _frame(300, "train", 0), _frame(100, "val", 1)
    tf, res = run_grid(train, val, COLS, [0.05, 0.1], ["scale", 0.1], [0.05, 0.1], 500, 500, 7, n_jobs=1)
    assert len(res) == 4
    rec = res[0].record
    assert {"ref_accept_tau5", "ref_accept_tau10", "uniform_accept_tau5", "uniform_accept_tau10"} <= set(rec)
    assert rec["tau5"] <= rec["tau10"]
    scores = res[0].svm.decision_function(tf.transform(val[COLS].to_numpy()))
    assert np.mean(scores < rec["tau5"]) <= 0.06
    m = LivenessModel(tf, res[0].svm, COLS, {"tau5": rec["tau5"], "tau10": rec["tau10"]}, {"nu": 0.05}, "abc")
    m.save(tmp_path)
    m2 = LivenessModel.load(tmp_path)
    assert np.allclose(m.score(val), m2.score(val)) and m2.thresholds == m.thresholds and m2.config_hash == "abc"
    assert isinstance(m2.transform, FeatureTransform) and tau_for_fnr(np.arange(101.0), 0.05) == pytest.approx(5.0)
    tf2, res2 = run_grid(train, val, COLS, [0.05, 0.1], ["scale", 0.1], [0.05, 0.1], 500, 500, 7, n_jobs=1)  # deterministic
    assert [r.record["ref_accept_tau5"] for r in res] == [r.record["ref_accept_tau5"] for r in res2]


def test_bootstrap_by_user_and_metrics():
    scores = np.r_[np.full(50, 1.0), np.full(50, -1.0)]
    users = np.repeat(np.arange(20), 5)  # first 10 users all accepted, last 10 all rejected
    r = bootstrap_reject_rate_by_user(scores, users, 0.0, 500, 1)
    assert r["fnr"] == 0.5 and r["ci_lo"] < 0.5 < r["ci_hi"] and r["n_users"] == 20
    assert r == bootstrap_reject_rate_by_user(scores, users, 0.0, 500, 1)  # seeded
    assert roc_auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    cm = confusion_matrix(np.array([1.0, -1.0, 1.0, 1.0]), np.array([-1.0, 1.0]), 0.0)
    assert cm["matrix"] == [[3, 1], [1, 1]] and cm["FNR_humans_rejected"] == 0.25 and cm["FPR_bots_accepted"] == 0.5
