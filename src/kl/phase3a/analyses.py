"""The four Phase 3a diagnostics. Pure functions of frozen inputs; nothing here feeds back into any detector or attacker."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from ..features import feature_columns
from ..model import FeatureTransform, LivenessModel, fit_ocsvm, tau_for_fnr
from ..phase2.evaluation import cell_row

ATTACKER_CODES = {"A1": 1, "A2": 2, "A3": 3, "A4": 4, "A5a": 5, "A5b": 6}


def transformed(model: LivenessModel, df: pd.DataFrame) -> np.ndarray:
    """Frozen-transform matrix of the sessions in ``df`` (the model's feature order)."""
    return model.transform.transform(df[model.feature_names].to_numpy(dtype="float64"))


# ------------------------------------------------------------------------------ 1. centrality
def centrality(
    x_train: np.ndarray, x_human: np.ndarray, x_bots: Mapping[str, np.ndarray], feature_names: Sequence[str], seed: int,
    with_classical: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Mahalanobis distances to the train-human distribution in the frozen transformed space.

    The primary estimator is ``MinCovDet(random_state=seed)`` fitted on TRAIN humans. Because it can be rank-deficient
    (a zero-inflated feature is constant inside its support subset and scikit-learn then uses a pseudo-inverse), the
    classical covariance is optionally reported as a labelled supplement.

    Returns ``(table, distances, info)``: ``table`` has one row per (covariance, group) with median / IQR of the distance
    d = sqrt(d^2), a two-sample KS test against the test humans and the share of the group closer than the human median /
    25th percentile ("too typical"); ``distances[cov][group]`` are the raw distances; ``info`` documents the fit.
    """
    estimators: dict[str, Any] = {"mcd": MinCovDet(random_state=seed).fit(x_train)}
    if with_classical:
        estimators["classical"] = EmpiricalCovariance().fit(x_train)
    groups = {"human": x_human, **x_bots}
    rows, dists = [], {}
    info: dict[str, Any] = {"n_train": int(len(x_train))}
    for cov_name, est in estimators.items():
        d = {g: np.sqrt(np.maximum(est.mahalanobis(x), 0.0)) for g, x in groups.items()}
        dists[cov_name] = d
        h = d["human"]
        h_med, h_q25 = float(np.median(h)), float(np.quantile(h, 0.25))
        for g, v in d.items():
            q25, med, q75 = np.quantile(v, [0.25, 0.5, 0.75])
            ks = ks_2samp(h, v) if g != "human" else None
            rows.append({
                "covariance": cov_name, "group": g, "n": len(v), "median_d": float(med), "q25_d": float(q25), "q75_d": float(q75), "iqr_d": float(q75 - q25),
                "median_ratio_to_human": float(med / h_med), "ks_stat_vs_human": float(ks.statistic) if ks else np.nan,
                "ks_pvalue_vs_human": float(ks.pvalue) if ks else np.nan,
                "frac_below_human_median": float(np.mean(v < h_med)), "frac_below_human_q25": float(np.mean(v < h_q25)),
            })
        if cov_name == "mcd":
            cov = est.covariance_
            var0 = [feature_names[j] for j in np.flatnonzero(np.diag(cov) < 1e-12)]
            eig = np.linalg.eigvalsh(cov)
            info["mcd"] = {"support_size": int(est.support_.sum()), "support_fraction_used": float(est.support_.mean()), "n_eigenvalues_below_1e-10": int((eig < 1e-10).sum()),
                           "zero_variance_features_in_support": var0, "condition_number": float(np.linalg.cond(cov))}
    return pd.DataFrame(rows), dists, info


def variance_ratios(
    human_raw: pd.DataFrame, bots_raw: Mapping[str, pd.DataFrame], human_t: np.ndarray, bots_t: Mapping[str, np.ndarray], feature_names: Sequence[str],
) -> pd.DataFrame:
    """Per-feature variance of each attacker divided by the human variance (< 1 = too homogeneous), in the frozen transformed
    space (primary) and in raw feature units (secondary). Human = test humans, attacker = its test bots, population variance."""
    hv_t, hv_r = human_t.var(axis=0), human_raw[list(feature_names)].to_numpy(dtype="float64").var(axis=0)
    rows = []
    for a in bots_t:
        bv_t = bots_t[a].var(axis=0)
        bv_r = bots_raw[a][list(feature_names)].to_numpy(dtype="float64").var(axis=0)
        for j, f in enumerate(feature_names):
            rows.append({"attacker": a, "feature": f, "human_var_transformed": float(hv_t[j]), "bot_var_transformed": float(bv_t[j]),
                         "var_ratio_transformed": float(bv_t[j] / hv_t[j]) if hv_t[j] > 0 else np.nan,
                         "human_var_raw": float(hv_r[j]), "bot_var_raw": float(bv_r[j]),
                         "var_ratio_raw": float(bv_r[j] / hv_r[j]) if hv_r[j] > 0 else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ 2. feature-group ablation
def ablation_subsets(config_subsets: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """Feature columns of every ablation subset (features.py order; group names resolved with ``feature_columns``)."""
    return {name: feature_columns(groups) for name, groups in config_subsets.items()}


def ablation(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, bots_te: Mapping[str, pd.DataFrame], subsets: Mapping[str, Sequence[str]],
    nu: float, gamma: float, weights: np.ndarray, users: np.ndarray, ci_level: float, n_jobs: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Refit the One-Class SVM with the FROZEN (nu, gamma) on each feature subset and evaluate on every attacker.

    Per subset: input transform fitted on TRAIN (same signed-log + RobustScaler code as the frozen model), SVM fitted on
    TRAIN humans, tau5 / tau10 from VAL humans. Cells carry user-bootstrap CIs (same resamples as Phase 2).
    """
    idx_of = pd.Index(users).get_indexer
    h_users = idx_of(test["user_id"])
    b_users = {a: idx_of(b["user_id"]) for a, b in bots_te.items()}
    jobs, meta = [], {}
    for name, cols in subsets.items():
        tf = FeatureTransform(cols).fit(train[list(cols)].to_numpy(dtype="float64"))
        svm, g_eff = fit_ocsvm(tf.transform(train[list(cols)].to_numpy(dtype="float64")), nu, gamma)
        score = lambda df, tf=tf, svm=svm, cols=cols: svm.decision_function(tf.transform(df[list(cols)].to_numpy(dtype="float64")))  # noqa: E731
        vs = score(val)
        taus = {"tau5": tau_for_fnr(vs, 0.05), "tau10": tau_for_fnr(vs, 0.10)}
        hs = score(test)
        meta[name] = {"n_features": len(cols), "features": list(cols), "gamma_effective": float(g_eff), "n_support_vectors": int(svm.support_vectors_.shape[0]), "thresholds": taus,
                      "val_fnr_native": float(np.mean(vs < 0))}
        for a, b in bots_te.items():
            jobs.append(delayed(cell_row)(f"D1[{name}]", name, a, hs, h_users, score(b), b_users[a], taus, weights, ci_level))
    rows = Parallel(n_jobs=n_jobs)(jobs)
    return pd.DataFrame(rows), meta


# ------------------------------------------------------------------------------ 3. permutation importance
def fit_single_attacker_d3(x_human_t: np.ndarray, x_bot_t: np.ndarray, seed: int, hp: Mapping[str, Any], n_jobs: int = 1) -> RandomForestClassifier:
    """Random Forest on transformed TRAIN humans (1) + TRAIN bots of ONE attacker (0), exactly the Phase 2 D3 recipe."""
    x = np.vstack([x_human_t, x_bot_t])
    y = np.r_[np.ones(len(x_human_t)), np.zeros(len(x_bot_t))]
    rf = RandomForestClassifier(n_estimators=int(hp["n_estimators"]), class_weight=str(hp["class_weight"]), random_state=seed, n_jobs=n_jobs)
    return rf.fit(x, y)


def importance_table(
    rf: RandomForestClassifier, x_human_t: np.ndarray, x_bot_t: np.ndarray, feature_names: Sequence[str], n_per_class: int, n_repeats: int,
    seed: int, attacker: str,
) -> pd.DataFrame:
    """Permutation importance (mean decrease of ROC-AUC over ``n_repeats``) on a seeded held-out TEST sample.

    The sample holds ``n_per_class`` test humans and ``n_per_class`` test bots of ``attacker``. Correlated features share
    credit (permuting one leaves its correlated partners intact), so importances are marginal, not additive.
    """
    rng = np.random.default_rng([seed, 31, ATTACKER_CODES[attacker]])
    ih = rng.choice(len(x_human_t), size=min(n_per_class, len(x_human_t)), replace=False)
    ib = rng.choice(len(x_bot_t), size=min(n_per_class, len(x_bot_t)), replace=False)
    x = np.vstack([x_human_t[ih], x_bot_t[ib]])
    y = np.r_[np.ones(len(ih)), np.zeros(len(ib))]
    pi = permutation_importance(rf, x, y, scoring="roc_auc", n_repeats=n_repeats, random_state=seed)
    df = pd.DataFrame({"attacker": attacker, "feature": list(feature_names), "importance_mean": pi.importances_mean, "importance_std": pi.importances_std})
    df["baseline_auc"] = _auc(rf, x, y)
    df["rank"] = df["importance_mean"].rank(ascending=False, method="first").astype(int)
    return df.sort_values("rank").reset_index(drop=True)


def _auc(rf: RandomForestClassifier, x: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, rf.predict_proba(x)[:, list(rf.classes_).index(1.0)]))


# ------------------------------------------------------------------------------ 4. per-user consistency
def per_unit_std(df: pd.DataFrame, unit: str, cols: Sequence[str], min_sessions: int) -> pd.DataFrame:
    """Session-to-session standard deviation (ddof=1) of ``cols`` within every unit (user / pseudo-user) with >= ``min_sessions`` sessions."""
    g = df.groupby(unit)
    out = g[list(cols)].std(ddof=1)
    out["n_sessions"] = g.size()
    return out[out["n_sessions"] >= min_sessions].reset_index()


def icc1(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way random-effects ICC(1): share of the variance that is between units (unbalanced-design formula)."""
    codes, inv = np.unique(groups, return_inverse=True)
    n_i = np.bincount(inv).astype("float64")
    m, N = len(codes), len(values)
    sums = np.bincount(inv, weights=values)
    means = sums / n_i
    grand = values.mean()
    ssb = float(np.sum(n_i * (means - grand) ** 2))
    ssw = float(np.sum((values - means[inv]) ** 2))
    msb, msw = ssb / (m - 1), ssw / (N - m)
    k0 = (N - np.sum(n_i ** 2) / N) / (m - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def pseudo_users(sessions: pd.DataFrame, n_units: int, k: int, rng: np.random.Generator) -> pd.DataFrame:
    """``n_units`` pseudo-users, each made of ``k`` distinct sessions drawn at random from ``sessions`` (independently per unit)."""
    idx = np.stack([rng.choice(len(sessions), size=k, replace=False) for _ in range(n_units)])
    out = sessions.iloc[idx.ravel()].reset_index(drop=True).copy()
    out["unit"] = np.repeat(np.arange(n_units), k)
    return out


def consistency(
    test: pd.DataFrame, bots_te: Mapping[str, pd.DataFrame], attackers: Sequence[str], cols: Sequence[str], min_sessions: int, k: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Are real users more self-consistent across sessions than pseudo-users built from bot sessions?

    Units: real test users (>= ``min_sessions`` sessions), pseudo-users of each attacker (``k`` random bot sessions each, as many
    units as real users) and a CONTROL of pseudo-users built from random human sessions of different users (real marginal
    distribution, no user persistence). Reports per-unit session-to-session std, its ratio to the total std of all sessions,
    a one-sided Mann-Whitney test (real std < unit std) and the ICC(1) of every feature.
    Returns ``(summary, per_unit)``.
    """
    real = per_unit_std(test.assign(unit=test["user_id"]), "unit", cols, min_sessions)
    n_units = len(real)
    groups: dict[str, pd.DataFrame] = {"real users": test.assign(unit=test["user_id"])[test["user_id"].isin(real["unit"])]}
    rng = np.random.default_rng([seed, 41])
    groups["shuffled humans (control)"] = pseudo_users(test, n_units, k, rng)
    for a in attackers:
        groups[f"{a} pseudo-users"] = pseudo_users(bots_te[a], n_units, k, np.random.default_rng([seed, 42, ATTACKER_CODES[a]]))
    rows, per_unit = [], []
    real_std = None
    for name, df in groups.items():
        std = per_unit_std(df, "unit", cols, min_sessions)
        per_unit.append(std.assign(group=name))
        if name == "real users":
            real_std = std
        for c in cols:
            total = df[c].std(ddof=1)
            v, r = std[c].to_numpy(), real_std[c].to_numpy()
            mw = mannwhitneyu(r, v, alternative="less") if name != "real users" else None
            ks = ks_2samp(r, v) if name != "real users" else None
            rows.append({"group": name, "feature": c, "n_units": len(std), "sessions_per_unit_median": float(std["n_sessions"].median()),
                         "median_std": float(np.median(v)), "q25_std": float(np.quantile(v, 0.25)), "q75_std": float(np.quantile(v, 0.75)),
                         "median_std_over_total_std": float(np.median(v) / total), "ratio_median_std_vs_real": float(np.median(v) / np.median(r)),
                         "mannwhitney_real_less_p": float(mw.pvalue) if mw else np.nan, "ks_stat_vs_real": float(ks.statistic) if ks else np.nan,
                         "icc1": icc1(df[c].to_numpy(dtype="float64"), df["unit"].to_numpy())})
    return pd.DataFrame(rows), pd.concat(per_unit, ignore_index=True)
