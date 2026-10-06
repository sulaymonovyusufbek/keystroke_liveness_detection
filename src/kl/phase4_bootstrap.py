"""Phase 4 (docs/PHASE4_SPEC.md, A1-A3): test-user cluster bootstrap with seed averaging.

Resampling unit = TEST USER. Replicate ``r`` draws the test users with replacement; the draw is stored as integer
user weights ``W[r, u]`` (how often user ``u`` was drawn), exactly as in Phases 2 and 3b
(:func:`kl.phase2.evaluation.user_weights`, same seed, same sorted user index, hence the SAME replicates). A session,
human or bot, carries the weight of the test user it belongs to; every bot is matched to one test human session (same
``session_id`` and user), so a drawn user brings all of their human sessions and every bot matched to them.

Sequence models (several seeds): in each replicate the metric is computed for every seed and averaged over seeds; the
CI is the percentile interval of these averages. The point estimate is the mean over seeds of the full-sample metric;
the across-seed sd of the point estimate (ddof=1, no bootstrap) is reported separately.

Conventions (as in Phases 2 and 3b): scores are "higher = more human"; a session is rejected iff ``score < tau``;
APCER = attacks accepted (``score >= tau``), BPCER = humans rejected; ROC-AUC with humans positive, ties counted 1/2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .model import tau_for_fnr
from .phase2.evaluation import user_weights

METRICS: tuple[str, ...] = ("apcer_fpr_tau5", "apcer_fpr_tau10", "bpcer_fnr_tau5", "bpcer_fnr_tau10", "auc")
RATE_METRICS: tuple[str, ...] = METRICS[:4]
FNRS: tuple[float, ...] = (0.05, 0.10)


# ------------------------------------------------------------------------------------------------ resampling weights
def test_weights(n_users: int, reps: int, seed: int) -> np.ndarray:
    """``(reps, n_users)`` test-user draw counts: identical to the Phase 2 / 3b bootstrap replicates."""
    return user_weights(n_users, reps, seed)


test_weights.__test__ = False  # not a pytest test


def val_weights(n_users: int, reps: int, seed: int) -> np.ndarray:
    """``(reps, n_users)`` validation-user draw counts for the threshold sensitivity check.

    Drawn from an independent stream derived from the global seed, ``SeedSequence(seed, spawn_key=(4,))``, so the
    validation draws are independent of the test draws of the same replicate.
    """
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(4,)))
    return rng.multinomial(n_users, np.full(n_users, 1.0 / n_users), size=reps).astype("float64")


def identity_weights(n_users: int) -> np.ndarray:
    """One replicate in which every user is drawn exactly once (the full sample)."""
    return np.ones((1, n_users))


def user_index(users: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Position of every id of ``ids`` in the sorted unique ``users``; raises if one is not a known user."""
    pos = np.searchsorted(users, ids)
    pos = np.clip(pos, 0, len(users) - 1)
    if not np.array_equal(users[pos], ids):
        raise ValueError("a session belongs to a user outside the resampled population")
    return pos.astype("int64")


# ------------------------------------------------------------------------------------------------ metrics
@dataclass
class SeedScores:
    """Scores of one cell for one training seed (higher = more human) and that model's thresholds."""

    h: np.ndarray
    b: np.ndarray
    taus: Mapping[str, float]


def point_rates(h: np.ndarray, b: np.ndarray, taus: Mapping[str, float]) -> dict[str, float]:
    """Full-sample APCER / BPCER at every tau."""
    out = {}
    for k, t in taus.items():
        out[f"bpcer_fnr_{k}"] = float(np.mean(h < t))
        out[f"apcer_fpr_{k}"] = float(np.mean(b >= t))
    return out


def _rank_groups(h: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    _, inv = np.unique(np.r_[h, b], return_inverse=True)
    return inv[:len(h)], inv[len(h):], int(inv.max()) + 1


def _auc_from_groups(inv_h: np.ndarray, inv_b: np.ndarray, wh: np.ndarray, wb: np.ndarray, n_g: int) -> float:
    hg = np.bincount(inv_h, weights=wh, minlength=n_g)
    bg = np.bincount(inv_b, weights=wb, minlength=n_g)
    den = hg.sum() * bg.sum()
    if den == 0:
        return float("nan")
    higher = hg.sum() - np.cumsum(hg)              # humans scoring strictly higher than each group
    return float(np.dot(bg, higher + 0.5 * hg) / den)


def point_auc(h: np.ndarray, b: np.ndarray) -> float:
    """ROC-AUC (humans positive, ties 1/2), never flipped."""
    inv_h, inv_b, n_g = _rank_groups(h, b)
    return _auc_from_groups(inv_h, inv_b, np.ones(len(h)), np.ones(len(b)), n_g)


def boot_rate(indicator: np.ndarray, users: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean of a 0/1 ``indicator`` per replicate (sessions weighted by their user's draw count)."""
    n = weights.shape[1]
    num = np.bincount(users, weights=indicator.astype("float64"), minlength=n)
    den = np.bincount(users, minlength=n).astype("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        return (weights @ num) / (weights @ den)


def boot_auc(h: np.ndarray, b: np.ndarray, h_u: np.ndarray, b_u: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted ROC-AUC per replicate."""
    inv_h, inv_b, n_g = _rank_groups(h, b)
    return np.array([_auc_from_groups(inv_h, inv_b, w[h_u], w[b_u], n_g) for w in weights])


@dataclass
class BootResult:
    """Seed-averaged point estimates, per-seed point estimates and seed-averaged bootstrap replicates of one cell."""

    point: dict[str, float]
    per_seed: dict[str, list[float]]
    boot: dict[str, np.ndarray]
    n_humans: int
    n_attacks: int
    seeds: list[int] = field(default_factory=list)

    def ci(self, m: str, level: float = 0.95) -> tuple[float, float]:
        """Percentile CI of metric ``m``."""
        a = (1.0 - level) / 2.0
        v = self.boot[m]
        v = v[np.isfinite(v)]
        lo, hi = np.quantile(v, [a, 1.0 - a])
        return float(lo), float(hi)

    def seed_sd(self, m: str) -> float:
        """Across-seed sd (ddof=1) of the point estimate; NaN for a single model."""
        v = self.per_seed[m]
        return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")

    def row(self, level: float = 0.95, metrics: Sequence[str] | None = None) -> dict[str, float]:
        """Flat columns ``<m>``, ``<m>_ci_lo``, ``<m>_ci_hi``, ``<m>_seed_sd``, ``<m>_seed<s>``."""
        out: dict[str, float] = {"n_humans": self.n_humans, "n_attacks": self.n_attacks, "n_seeds": len(self.per_seed[next(iter(self.per_seed))])}
        for m in metrics or self.point:
            out[m] = self.point[m]
            out[f"{m}_ci_lo"], out[f"{m}_ci_hi"] = self.ci(m, level)
            out[f"{m}_seed_sd"] = self.seed_sd(m)
            for s, v in zip(self.seeds, self.per_seed[m]):
                out[f"{m}_seed{s}"] = v
        return out


def evaluate_cell(
    per_seed: Sequence[SeedScores], h_u: np.ndarray, b_u: np.ndarray, weights: np.ndarray,
    metrics: Sequence[str] = METRICS, seeds: Sequence[int] | None = None,
) -> BootResult:
    """Point estimates (mean over seeds) and seed-averaged bootstrap replicates of one cell.

    ``h_u`` / ``b_u``: user index (column of ``weights``) of every human / attack session.
    """
    point_s: dict[str, list[float]] = {m: [] for m in metrics}
    boot_s: dict[str, list[np.ndarray]] = {m: [] for m in metrics}
    for sc in per_seed:
        pr = point_rates(sc.h, sc.b, sc.taus)
        for m in metrics:
            if m == "auc":
                point_s[m].append(point_auc(sc.h, sc.b))
                boot_s[m].append(boot_auc(sc.h, sc.b, h_u, b_u, weights))
                continue
            kind, tau = m.rsplit("_", 1)
            point_s[m].append(pr[m])
            ind = sc.h < sc.taus[tau] if kind == "bpcer_fnr" else sc.b >= sc.taus[tau]
            boot_s[m].append(boot_rate(ind, h_u if kind == "bpcer_fnr" else b_u, weights))
    return BootResult({m: float(np.mean(v)) for m, v in point_s.items()}, point_s,
                      {m: np.mean(np.vstack(v), axis=0) for m, v in boot_s.items()},
                      len(per_seed[0].h), len(per_seed[0].b), list(seeds) if seeds is not None else list(range(len(per_seed))))


def average_cells(cells: Sequence[BootResult], metric: str) -> BootResult:
    """Mean of ``metric`` over several cells, replicate by replicate (so its CI accounts for their correlation)."""
    per_seed = np.mean([c.per_seed[metric] for c in cells], axis=0).tolist()
    return BootResult({metric: float(np.mean([c.point[metric] for c in cells]))}, {metric: per_seed},
                      {metric: np.mean([c.boot[metric] for c in cells], axis=0)}, sum(c.n_humans for c in cells),
                      sum(c.n_attacks for c in cells), cells[0].seeds)


# ------------------------------------------------------------------------------------------------ thresholds
def thresholds(val_human_scores: np.ndarray, fnrs: Sequence[float] = FNRS) -> dict[str, float]:
    """tau5 / tau10 = quantiles of VALIDATION-human scores (the frozen rule, :func:`kl.model.tau_for_fnr`)."""
    return {f"tau{int(round(f * 100))}": tau_for_fnr(val_human_scores, f) for f in fnrs}


def quantile_counts(sorted_x: np.ndarray, counts: np.ndarray, q: float) -> float:
    """``np.quantile(np.repeat(sorted_x, counts), q)`` (linear interpolation) without building the repeated array."""
    cum = np.cumsum(counts)
    n = int(cum[-1])
    pos = q * (n - 1)
    k = int(np.floor(pos))
    frac = pos - k
    lo = sorted_x[np.searchsorted(cum, k, side="right")]
    if frac == 0.0:
        return float(lo)
    hi = sorted_x[np.searchsorted(cum, k + 1, side="right")]
    return float(lo + frac * (hi - lo))


@dataclass
class ValScores:
    """Validation-human scores of one seed (for thresholds recomputed inside each replicate)."""

    v: np.ndarray


def _weight_below(sorted_x: np.ndarray, cw: np.ndarray, t: float) -> float:
    k = int(np.searchsorted(sorted_x, t, side="left"))   # sessions with score < t
    return float(cw[k - 1]) if k > 0 else 0.0


def sensitivity_cell(
    per_seed: Sequence[SeedScores], val: Sequence[ValScores], val_u: np.ndarray, val_w: np.ndarray,
    h_u: np.ndarray, b_u: np.ndarray, weights: np.ndarray, fnrs: Sequence[float] = FNRS,
) -> dict[str, np.ndarray]:
    """Seed-averaged APCER / BPCER replicates with tau recomputed in every replicate from RESAMPLED validation users.

    Replicate ``r`` uses test-user weights ``weights[r]`` and validation-user weights ``val_w[r]`` (independent draws).
    """
    reps = weights.shape[0]
    out = {f"{k}_tau{int(round(f * 100))}": np.zeros(reps) for k in ("apcer_fpr", "bpcer_fnr") for f in fnrs}
    for sc, vs in zip(per_seed, val):
        vo = np.argsort(vs.v, kind="stable")
        vx, vu = vs.v[vo], val_u[vo]
        ho, bo = np.argsort(sc.h, kind="stable"), np.argsort(sc.b, kind="stable")
        hx, hu, bx, bu = sc.h[ho], h_u[ho], sc.b[bo], b_u[bo]
        for r in range(reps):
            c = val_w[r][vu]
            wh, wb = weights[r][hu], weights[r][bu]
            cwh, cwb = np.cumsum(wh), np.cumsum(wb)
            for f in fnrs:
                t = quantile_counts(vx, c, f)
                tag = f"tau{int(round(f * 100))}"
                out[f"bpcer_fnr_{tag}"][r] += _weight_below(hx, cwh, t) / cwh[-1]
                out[f"apcer_fpr_{tag}"][r] += 1.0 - _weight_below(bx, cwb, t) / cwb[-1]
    return {k: v / len(per_seed) for k, v in out.items()}
