"""Attack-free hyperparameter selection with a product-of-marginals reference set (Revision 1).

For every ``(nu, gamma)`` the One-Class SVM is fitted on TRANSFORMED TRAIN humans (signed log +
RobustScaler, see ``model.py``); thresholds tau5 / tau10 come from the VALIDATION humans.

Selection criterion: the fraction of *reference points* the model still accepts at tau5. The reference
set is drawn from the product of the marginals of the transformed train data: every feature is
resampled independently, with replacement, from its own train column (the artificial-outlier
construction of Hempstalk, Frank & Witten, 2008, "One-class classification by combining density and
class probability estimation", ECML PKDD). It keeps the marginals of the data but destroys the
dependence structure, so a good one-class model should reject it.

The original Tax & Duin (2001) uniform-box acceptance is still computed as a DIAGNOSTIC
(``uniform_accept_tau*``) because it proved degenerate (all configurations ~0%); it is never used to select.

Revision 2 (FINAL): before the criterion is applied, configurations must be *admissible*: the fraction of
VAL humans below the native boundary (``val_fnr_native``) must satisfy
``|val_fnr_native - nu| <= max(0.02, 0.5 * nu)``. Under the nu-property of the One-Class SVM (Scholkopf et al.,
2001, Neural Computation 13(7)) nu approximates the held-out outlier fraction, so a large excess signals an
overfitted boundary. The reference criterion is minimised over admissible configurations only.

If the spread (max - min) of reference acceptance across the grid is below ``min_spread`` the grid cannot
discriminate and :func:`select_best` raises :class:`DegenerateSelectionError`; if no configuration is admissible it
raises :class:`NoAdmissibleConfigError`. In both cases nothing is selected and the tolerance is never relaxed.

No attack data may enter this module: every public function calls :func:`guard_no_attack` and raises
:class:`AttackDataError` if it is handed attack-labelled data, an attacker, or an attacker_pool split.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.svm import OneClassSVM

from .attacks.base import Attacker
from .model import FeatureTransform, fit_ocsvm, fit_transform_on_train, tau_for_fnr

ATTACK_MARKER_COLUMNS: frozenset[str] = frozenset({"is_attack", "attack", "bot", "attack_label", "attacker", "attack_type"})
FORBIDDEN_SPLITS: frozenset[str] = frozenset({"attacker_pool", "attack", "bot", "a1"})


class AttackDataError(ValueError):
    """Raised when attack data reaches the attack-free selection code."""


def guard_no_attack(**inputs: Any) -> None:
    """Raise :class:`AttackDataError` if any named input is (or is labelled as) attack data.

    Detected: :class:`Attacker` instances or objects with ``is_attack=True``; DataFrames carrying an
    attack-marker column, ``attrs['kind'] == 'attack'``, or a ``split`` / ``source`` column that
    contains an attack value (``attacker_pool``, ``attack``, ``bot``, ``a1``).
    """
    for name, obj in inputs.items():
        if isinstance(obj, Attacker) or getattr(obj, "is_attack", False) is True:
            raise AttackDataError(f"'{name}' is an attacker; selection must be attack-free")
        if isinstance(obj, pd.DataFrame):
            marker = ATTACK_MARKER_COLUMNS & {str(c).lower() for c in obj.columns}
            if marker:
                raise AttackDataError(f"'{name}' has attack-marker column(s) {sorted(marker)}")
            if str(obj.attrs.get("kind", "")).lower() == "attack":
                raise AttackDataError(f"'{name}' is tagged as attack data")
            for col in ("split", "source"):
                if col in obj.columns:
                    bad = FORBIDDEN_SPLITS & {str(v).lower() for v in obj[col].unique()}
                    if bad:
                        raise AttackDataError(f"'{name}' column '{col}' contains attack value(s) {sorted(bad)}")


class DegenerateSelectionError(RuntimeError):
    """Raised when reference acceptance does not discriminate between grid configurations."""

    def __init__(self, spread: float, lo: float, hi: float, min_spread: float) -> None:
        super().__init__(
            f"degenerate selection: reference acceptance at tau5 spans only {100 * spread:.3f} pp across the grid "
            f"(min {100 * lo:.4f}%, max {100 * hi:.4f}%) < required {100 * min_spread:.2f} pp; refusing to select"
        )
        self.spread, self.lo, self.hi, self.min_spread = spread, lo, hi, min_spread


class NoAdmissibleConfigError(RuntimeError):
    """Raised when no grid configuration satisfies the admissibility condition."""

    def __init__(self, n_grid: int, abs_tol: float, rel_tol: float, worst: str) -> None:
        super().__init__(
            f"no admissible configuration among {n_grid}: need |val_fnr_native - nu| <= max({abs_tol}, {rel_tol} * nu); "
            f"closest: {worst}; refusing to relax the tolerance"
        )


def tau_key(fnr: float) -> str:
    """Column/threshold key for a target FNR (0.05 -> ``tau5``)."""
    return f"tau{int(round(fnr * 100))}"


@dataclass
class GridResult:
    """One grid cell: its record (a row of grid_results.csv) and the fitted SVM."""

    record: dict[str, Any]
    svm: OneClassSVM


def uniform_outliers(x_train_scaled: np.ndarray, n: int, seed: int) -> np.ndarray:
    """DIAGNOSTIC ONLY: ``n`` seeded points uniform in the per-feature [min, max] box of the transformed train data."""
    lo, hi = x_train_scaled.min(axis=0), x_train_scaled.max(axis=0)
    return np.random.default_rng([seed, 2]).uniform(lo, hi, size=(n, x_train_scaled.shape[1]))


def product_of_marginals(x_train_scaled: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Reference set: ``n`` points whose every coordinate is resampled independently (with replacement) from its train column."""
    n_train, d = x_train_scaled.shape
    idx = np.random.default_rng([seed, 5]).integers(0, n_train, size=(n, d))
    return x_train_scaled[idx, np.arange(d)]


def _fit_one(
    nu: float, gamma: str | float, x_tr: np.ndarray, x_val: np.ndarray, x_ref: np.ndarray, x_unif: np.ndarray,
    fnrs: Sequence[float],
) -> GridResult:
    t0 = time.perf_counter()
    svm, g_eff = fit_ocsvm(x_tr, nu, gamma)
    fit_time = time.perf_counter() - t0
    val_scores = svm.decision_function(x_val)
    ref_scores = svm.decision_function(x_ref)
    unif_scores = svm.decision_function(x_unif)
    rec: dict[str, Any] = {
        "nu": float(nu), "gamma": str(gamma), "fit_time": fit_time,
        "val_fnr_native": float(np.mean(val_scores < 0.0)),
    }
    taus = {tau_key(f): tau_for_fnr(val_scores, f) for f in fnrs}
    rec.update(taus)
    for key, tau in taus.items():
        rec[f"ref_accept_{key}"] = float(np.mean(ref_scores >= tau))       # selection criterion
    for key, tau in taus.items():
        rec[f"uniform_accept_{key}"] = float(np.mean(unif_scores >= tau))  # diagnostic only
    rec["n_support_vectors"] = int(svm.support_vectors_.shape[0])
    rec["gamma_effective"] = float(g_eff)
    return GridResult(record=rec, svm=svm)


def run_grid(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: Sequence[str],
    nu_grid: Sequence[float],
    gamma_grid: Sequence[str | float],
    target_fnrs: Sequence[float],
    n_reference: int,
    n_uniform: int,
    seed: int,
    n_jobs: int = 1,
) -> tuple[FeatureTransform, list[GridResult]]:
    """Fit the feature transform on TRAIN and evaluate every ``(nu, gamma)`` cell (attack-free).

    Args:
        train: TRAIN-split feature frame (``split`` column, if present, must be ``train``).
        val: VAL-split feature frame (``split`` column, if present, must be ``val``).
        feature_cols: model feature columns.
        nu_grid, gamma_grid: hyperparameter grid (gamma may be ``"scale"``/``"auto"``).
        target_fnrs: validation FNRs defining tau5/tau10.
        n_reference: size M of the product-of-marginals reference set (selection criterion).
        n_uniform: size of the uniform-box diagnostic set (never used for selection).
        seed: global seed.
        n_jobs: parallel workers over grid cells.

    Returns:
        ``(transform, results)`` with ``results`` in grid order (nu-major).
    """
    guard_no_attack(train=train, val=val)
    for name, frame, expected in (("train", train, "train"), ("val", val, "val")):
        if "split" in frame.columns and set(frame["split"].unique()) != {expected}:
            raise AttackDataError(f"'{name}' must contain only split == '{expected}', got {sorted(frame['split'].unique())}")
    x_train = train[list(feature_cols)].to_numpy(dtype="float64")
    x_val = val[list(feature_cols)].to_numpy(dtype="float64")
    transform = fit_transform_on_train(x_train, feature_cols)
    x_tr, x_va = transform.transform(x_train), transform.transform(x_val)
    x_ref = product_of_marginals(x_tr, n_reference, seed)
    x_unif = uniform_outliers(x_tr, n_uniform, seed)
    cells = [(nu, g) for nu in nu_grid for g in gamma_grid]
    results = Parallel(n_jobs=n_jobs)(delayed(_fit_one)(nu, g, x_tr, x_va, x_ref, x_unif, target_fnrs) for nu, g in cells)
    return transform, list(results)


def admissible_mask(grid: pd.DataFrame, abs_tol: float = 0.02, rel_tol: float = 0.5) -> pd.Series:
    """Boolean Series: ``|val_fnr_native - nu| <= max(abs_tol, rel_tol * nu)`` (attack-free overfitting check)."""
    tol = np.maximum(abs_tol, rel_tol * grid["nu"])
    return (grid["val_fnr_native"] - grid["nu"]).abs() <= tol + 1e-12


def select_best(
    grid: pd.DataFrame,
    tie_tolerance: float,
    min_spread: float = 0.01,
    primary: str = "tau5",
    abs_tol: float = 0.02,
    rel_tol: float = 0.5,
) -> int:
    """Row position of the selected config.

    Among ADMISSIBLE configs (see :func:`admissible_mask`), the lowest reference-set acceptance at the primary
    tau; configs within ``tie_tolerance`` (absolute, 0.001 = 0.1 percentage points) of that minimum are tied and
    the smaller ``nu`` wins, then the smaller effective gamma.

    Raises:
        DegenerateSelectionError: if ``max - min`` of the reference acceptance over the WHOLE grid is below
            ``min_spread`` (0.01 = 1 percentage point): the criterion cannot tell the configs apart.
        NoAdmissibleConfigError: if no configuration is admissible.
    """
    guard_no_attack(grid=grid)
    col = f"ref_accept_{primary}"
    lo, hi = float(grid[col].min()), float(grid[col].max())
    if hi - lo < min_spread:
        raise DegenerateSelectionError(hi - lo, lo, hi, min_spread)
    adm = admissible_mask(grid, abs_tol, rel_tol)
    if not adm.any():
        dev = (grid["val_fnr_native"] - grid["nu"]).abs() - np.maximum(abs_tol, rel_tol * grid["nu"])
        w = grid.loc[dev.idxmin()]
        raise NoAdmissibleConfigError(len(grid), abs_tol, rel_tol, f"nu={w['nu']}, val_fnr_native={w['val_fnr_native']:.4f}")
    cand = grid[adm]
    best = float(cand[col].min())
    tied = cand[cand[col] <= best + tie_tolerance]
    winner = tied.sort_values(["nu", "gamma_effective"], kind="mergesort").index[0]
    return int(grid.index.get_loc(winner))
