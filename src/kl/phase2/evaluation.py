"""Cell metrics with user-level bootstrap CIs, tables and figures for the Phase 2 experiments.

Terminology (both conventions are reported):
    APCER (= FPR)  fraction of ATTACK sessions accepted (score >= tau)
    BPCER (= FNR)  fraction of HUMAN sessions rejected (score < tau)
The bootstrap resamples USERS with replacement (a user carries its human sessions AND the bots matched to them),
implemented as integer user weights so that every cell of a table is evaluated on the SAME resamples.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

METRICS = ("apcer_fpr_tau5", "apcer_fpr_tau10", "bpcer_fnr_tau5", "bpcer_fnr_tau10", "auc")


def user_weights(n_users: int, reps: int, seed: int) -> np.ndarray:
    """``(reps, n_users)`` multinomial counts: how often each user is drawn in each bootstrap replicate."""
    return np.random.default_rng(seed).multinomial(n_users, np.full(n_users, 1.0 / n_users), size=reps).astype("float64")


def _weighted_auc(inv_h: np.ndarray, inv_b: np.ndarray, wh: np.ndarray, wb: np.ndarray, n_groups: int) -> float:
    hg = np.bincount(inv_h, weights=wh, minlength=n_groups)
    bg = np.bincount(inv_b, weights=wb, minlength=n_groups)
    higher = hg.sum() - np.cumsum(hg)                     # humans scoring strictly higher than group g
    return float(np.dot(bg, higher + 0.5 * hg) / (hg.sum() * bg.sum()))


def cell_metrics(
    h_scores: np.ndarray, h_users: np.ndarray, b_scores: np.ndarray, b_users: np.ndarray,
    taus: Mapping[str, float], weights: np.ndarray, ci_level: float = 0.95,
) -> dict[str, float]:
    """APCER/FPR, BPCER/FNR at tau5 & tau10 and ROC-AUC with percentile bootstrap CIs (by user).

    Args:
        h_scores, b_scores: detector scores of the human and the attack sessions (higher = more human).
        h_users, b_users: integer user index (row of ``weights`` columns) of every session.
        taus: ``{"tau5": ..., "tau10": ...}``.
        weights: output of :func:`user_weights` for the same user index space.
    """
    _, inv = np.unique(np.r_[h_scores, b_scores], return_inverse=True)
    n_g = int(inv.max()) + 1
    inv_h, inv_b = inv[:len(h_scores)], inv[len(h_scores):]
    rej = {k: (h_scores < t).astype("float64") for k, t in taus.items()}
    acc = {k: (b_scores >= t).astype("float64") for k, t in taus.items()}
    out: dict[str, float] = {}
    point = {}
    for k in taus:
        point[f"bpcer_fnr_{k}"] = float(rej[k].mean())
        point[f"apcer_fpr_{k}"] = float(acc[k].mean())
    point["auc"] = _weighted_auc(inv_h, inv_b, np.ones(len(h_scores)), np.ones(len(b_scores)), n_g)
    boot = {m: np.empty(len(weights)) for m in point}
    for r in range(len(weights)):
        wh, wb = weights[r][h_users], weights[r][b_users]
        sh, sb = wh.sum(), wb.sum()
        for k in taus:
            boot[f"bpcer_fnr_{k}"][r] = wh @ rej[k] / sh
            boot[f"apcer_fpr_{k}"][r] = wb @ acc[k] / sb
        boot["auc"][r] = _weighted_auc(inv_h, inv_b, wh, wb, n_g)
    alpha = (1.0 - ci_level) / 2.0
    for m, v in point.items():
        lo, hi = np.quantile(boot[m], [alpha, 1.0 - alpha])
        out[m], out[f"{m}_ci_lo"], out[f"{m}_ci_hi"] = v, float(lo), float(hi)
    return out


def cell_row(
    detector: str, trained_on: str, tested_on: str, h_scores: np.ndarray, h_users: np.ndarray, b_scores: np.ndarray,
    b_users: np.ndarray, taus: Mapping[str, float], weights: np.ndarray, ci_level: float = 0.95,
) -> dict[str, Any]:
    """One table row: identifiers, sample sizes, thresholds and :func:`cell_metrics`."""
    row: dict[str, Any] = {"detector": detector, "trained_on": trained_on, "tested_on": tested_on,
                           "n_humans": len(h_scores), "n_attacks": len(b_scores), "tau5": taus["tau5"], "tau10": taus["tau10"]}
    row.update(cell_metrics(h_scores, h_users, b_scores, b_users, taus, weights, ci_level))
    return row


#: column headers with both naming conventions (used in the markdown tables)
HEADERS = {
    "apcer_fpr_tau5": "APCER/FPR @τ5", "apcer_fpr_tau10": "APCER/FPR @τ10", "bpcer_fnr_tau5": "BPCER/FNR @τ5",
    "bpcer_fnr_tau10": "BPCER/FNR @τ10", "auc": "ROC-AUC",
}


def fmt_cell(row: Mapping[str, float], m: str) -> str:
    """``value [lo, hi]`` as percentages (AUC as a fraction)."""
    v, lo, hi = row[m], row[f"{m}_ci_lo"], row[f"{m}_ci_hi"]
    return f"{v:.4f} [{lo:.4f}, {hi:.4f}]" if m == "auc" else f"{100 * v:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def plot_heatmap(matrix: pd.DataFrame, path, title: str, cbar_label: str, percent: bool, vmin: float, vmax: float, cmap: str) -> None:
    """Annotated heatmap of a (rows = detector / training attack) x (columns = tested attack) matrix."""
    fig, ax = plt.subplots(figsize=(1.2 + 1.15 * matrix.shape[1], 1.2 + 0.55 * matrix.shape[0]))
    im = ax.imshow(matrix.to_numpy(dtype="float64"), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns)
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    ax.set_xlabel("tested on (attack)")
    ax.set_ylabel("detector / trained on")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.iat[i, j]
            if np.isfinite(v):
                dark = (v - vmin) / (vmax - vmin) > 0.6  # sequential light -> dark colormaps only
                ax.text(j, i, f"{100 * v:.1f}" if percent else f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if dark else "black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_per_attacker(human_scores: np.ndarray, bot_scores: Mapping[str, np.ndarray], path, title: str) -> None:
    """One ROC curve per attacker (positive class = human, x = attacks accepted)."""
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for name, b in bot_scores.items():
        y = np.r_[np.ones(len(human_scores)), np.zeros(len(b))]
        fpr, tpr, _ = roc_curve(y, np.r_[human_scores, b])
        auc = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":")
    ax.set_xlabel("attacks accepted (APCER = FPR)")
    ax.set_ylabel("humans accepted (1 - BPCER)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
