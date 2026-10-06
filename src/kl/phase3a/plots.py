"""Figures of the Phase 3a diagnostics."""
from __future__ import annotations

from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_centrality(dists: Mapping[str, np.ndarray], path_hist, path_ecdf, title: str) -> None:
    """Overlaid histograms (density, log10 distance) and ECDFs of the Mahalanobis distance of test humans and every attacker."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    lo = min(np.log10(np.maximum(v, 1e-3)).min() for v in dists.values())
    hi = max(np.quantile(np.log10(np.maximum(v, 1e-3)), 0.999) for v in dists.values())
    bins = np.linspace(lo, hi, 90)
    for name, v in dists.items():
        ax.hist(np.log10(np.maximum(v, 1e-3)), bins=bins, density=True, histtype="step", linewidth=2.2 if name == "human" else 1.4,
                color="black" if name == "human" else None, label=name)
    ax.set_xlabel("log10 Mahalanobis distance to the train-human distribution")
    ax.set_ylabel("density")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_hist, dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for name, v in dists.items():
        x = np.sort(v)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), linewidth=2.2 if name == "human" else 1.4, color="black" if name == "human" else None, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Mahalanobis distance")
    ax.set_ylabel("ECDF")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_ecdf, dpi=150)
    plt.close(fig)


def plot_matrix(matrix: pd.DataFrame, path, title: str, cbar: str, cmap: str, vmin: float, vmax: float, fmt: str = "{:.2f}", fig_w: float | None = None) -> None:
    """Annotated features x attackers heatmap."""
    fig, ax = plt.subplots(figsize=(fig_w or 1.6 + 0.95 * matrix.shape[1], 1.0 + 0.32 * matrix.shape[0]))
    im = ax.imshow(matrix.to_numpy(dtype="float64"), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns)
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    span = vmax - vmin
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.iat[i, j]
            if np.isfinite(v):
                dark = abs((v - (vmin + vmax) / 2) / span) > 0.3 if cmap == "RdBu_r" else (v - vmin) / span > 0.6
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7, color="white" if dark else "black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label=cbar)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_consistency(per_unit: pd.DataFrame, features: list[str], path, title: str) -> None:
    """Per-feature ECDFs of the per-unit session-to-session std for real users, the shuffled-human control and the bot pseudo-users."""
    fig, axes = plt.subplots(1, len(features), figsize=(5.4 * len(features), 4.2), squeeze=False)
    for ax, c in zip(axes[0], features):
        for name, g in per_unit.groupby("group", sort=False):
            x = np.sort(g[c].to_numpy())
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), linewidth=2.4 if name == "real users" else 1.4, color="black" if name == "real users" else None, label=name)
        ax.set_xlabel(f"per-unit std of {c} across sessions (ms)")
        ax.set_ylabel("ECDF")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
