"""Metrics, user-level bootstrap CIs, ROC and confusion matrices.

Terminology (used everywhere in the results):
  * FNR = fraction of HUMAN sessions rejected (score < tau)
  * FPR = fraction of BOT sessions accepted (score >= tau)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from joblib import Parallel, delayed  # noqa: E402

from .attacks.base import Attacker  # noqa: E402
from .features import FeatureParams, compute_features  # noqa: E402
from .model import fit_ocsvm, tau_for_fnr  # noqa: E402
from .selection import tau_key  # noqa: E402


def reject_rate(scores: np.ndarray, tau: float) -> float:
    """Fraction of ``scores`` strictly below ``tau`` (FNR for humans)."""
    return float(np.mean(np.asarray(scores) < tau))


def accept_rate(scores: np.ndarray, tau: float) -> float:
    """Fraction of ``scores`` at or above ``tau`` (FPR for bots)."""
    return float(np.mean(np.asarray(scores) >= tau))


def bootstrap_reject_rate_by_user(
    scores: np.ndarray, user_ids: np.ndarray, tau: float, reps: int, seed: int, ci_level: float = 0.95,
) -> dict[str, float]:
    """FNR with a percentile bootstrap CI, resampling USERS (all their sessions) with replacement.

    Returns ``{"fnr", "ci_lo", "ci_hi", "n_users", "n_sessions", "reps"}``.
    """
    scores = np.asarray(scores)
    users, inv = np.unique(np.asarray(user_ids), return_inverse=True)
    n_all = np.bincount(inv, minlength=len(users)).astype("float64")
    n_rej = np.bincount(inv, weights=(scores < tau).astype("float64"), minlength=len(users))
    idx = np.random.default_rng(seed).integers(0, len(users), size=(reps, len(users)))
    rates = n_rej[idx].sum(axis=1) / n_all[idx].sum(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(rates, [alpha, 1.0 - alpha])
    return {
        "fnr": float(n_rej.sum() / n_all.sum()), "ci_lo": float(lo), "ci_hi": float(hi),
        "n_users": int(len(users)), "n_sessions": int(len(scores)), "reps": int(reps),
    }


def roc_auc(human_scores: np.ndarray, bot_scores: np.ndarray) -> float:
    """ROC-AUC with humans as the positive class (higher score = more human)."""
    y = np.r_[np.ones(len(human_scores)), np.zeros(len(bot_scores))]
    return float(roc_auc_score(y, np.r_[human_scores, bot_scores]))


def confusion_matrix(human_scores: np.ndarray, bot_scores: np.ndarray, tau: float) -> dict[str, Any]:
    """Confusion matrix at ``tau``: rows = true class, columns = decision.

    ``matrix = [[human_accepted, human_rejected], [bot_accepted, bot_rejected]]``;
    FNR = human_rejected / n_human, FPR = bot_accepted / n_bot.
    """
    ha = int(np.sum(np.asarray(human_scores) >= tau))
    ba = int(np.sum(np.asarray(bot_scores) >= tau))
    nh, nb = len(human_scores), len(bot_scores)
    return {
        "tau": float(tau),
        "rows": ["human", "bot"], "cols": ["accepted (judged human)", "rejected (judged bot)"],
        "matrix": [[ha, nh - ha], [ba, nb - ba]],
        "FNR_humans_rejected": (nh - ha) / nh, "FPR_bots_accepted": ba / nb,
    }


def plot_score_hist(scores: np.ndarray, taus: dict[str, float], path: Path, title: str, bins: int = 80) -> None:
    """Histogram of human scores with vertical lines at each threshold."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(scores, bins=bins, color="#4c72b0", alpha=0.85)
    for (name, tau), style in zip(taus.items(), ("--", ":")):
        ax.axvline(tau, color="#c44e52", linestyle=style, label=f"{name} = {tau:.3f}")
    ax.set_xlabel("OCSVM decision score (higher = more human-like)")
    ax.set_ylabel("sessions")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(human_scores: np.ndarray, bot_scores: np.ndarray, path: Path, title: str) -> None:
    """ROC curve (positive = human)."""
    y = np.r_[np.ones(len(human_scores)), np.zeros(len(bot_scores))]
    fpr, tpr, _ = roc_curve(y, np.r_[human_scores, bot_scores])
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot(fpr, tpr, color="#4c72b0")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":")
    ax.set_xlabel("bots accepted (FPR)")
    ax.set_ylabel("humans accepted (1 - FNR)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_bot_features(sentences: pd.Series, attacker: Attacker, params: FeatureParams, seed: int) -> pd.DataFrame:
    """One bot session per ``(session_id -> sentence)`` entry, featurised with the human code path.

    The frame is tagged as attack data (``is_attack`` column and ``attrs['kind'] == 'attack'``) so the
    attack-free selection code refuses it.
    """
    rng = np.random.default_rng([seed, 3])
    rows = []
    for k, (sid, sentence) in enumerate(sentences.items()):
        bot = attacker.generate(str(sentence), rng)
        bot["PARTICIPANT_ID"], bot["TEST_SECTION_ID"] = -1, k
        rows.append({"session_id": f"bot_{sid}", **compute_features(bot, params)})
    frame = pd.DataFrame(rows)
    frame["is_attack"] = True
    frame.attrs["kind"] = "attack"
    return frame


SENSITIVITY_NOTE = "post-hoc; not used for selection"


def _sensitivity_one(
    nu: float, gamma: str | float, x_tr: np.ndarray, x_val: np.ndarray, x_h: np.ndarray, x_b: np.ndarray, fnrs: Sequence[float],
) -> dict[str, Any]:
    svm, g_eff = fit_ocsvm(x_tr, nu, gamma)
    val, h, b = (svm.decision_function(x) for x in (x_val, x_h, x_b))
    rec: dict[str, Any] = {"nu": float(nu), "gamma": str(gamma), "gamma_effective": float(g_eff)}
    for f in fnrs:
        key = tau_key(f)
        tau = tau_for_fnr(val, f)
        rec[key] = tau
        rec[f"test_fnr_{key}"] = reject_rate(h, tau)
        rec[f"fpr_{key}"] = accept_rate(b, tau)
    rec["auc"] = roc_auc(h, b)
    return rec


def grid_attack_sensitivity(
    x_train: np.ndarray, x_val: np.ndarray, x_test_humans: np.ndarray, x_bots: np.ndarray,
    nu_grid: Sequence[float], gamma_grid: Sequence[str | float], target_fnrs: Sequence[float], n_jobs: int = 1,
) -> pd.DataFrame:
    """POST-HOC A1 evaluation of EVERY grid configuration (never used for selection).

    Each configuration is refitted on the (already transformed) train matrix and gets its own tau5 / tau10
    from the validation humans; human test sessions and bots are then scored with it. Returns one row per
    configuration with ``fpr_tau*`` (bots accepted), ``test_fnr_tau*`` (humans rejected) and ``auc``.
    """
    cells = [(nu, g) for nu in nu_grid for g in gamma_grid]
    rows = Parallel(n_jobs=n_jobs)(delayed(_sensitivity_one)(nu, g, x_train, x_val, x_test_humans, x_bots, target_fnrs) for nu, g in cells)
    df = pd.DataFrame(rows)
    df["note"] = SENSITIVITY_NOTE
    return df
