"""Feature transform (signed log + RobustScaler) + One-Class SVM wrapper with save/load.

Transform (Revision 1): bounded rates in [0, 1] are left as they are; every other feature goes through
the signed log ``y = sign(x) * log1p(|x|)``; then a RobustScaler (median / IQR) fitted on TRAIN only.

Score convention: ``score = OneClassSVM.decision_function(scaled X)``; higher = more human-like. A
session is *accepted* (judged human) iff ``score >= tau`` and *rejected* iff ``score < tau``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from .config import read_json, write_json
from .features import BOUNDED_RATE_FEATURES


def resolve_gamma(gamma: str | float, x_scaled: np.ndarray) -> float:  # x_scaled = transformed train matrix
    """Numeric value of a gamma setting (``"scale"`` / ``"auto"`` follow scikit-learn's definitions)."""
    if isinstance(gamma, str):
        if gamma == "scale":
            var = float(x_scaled.var())
            return 1.0 / (x_scaled.shape[1] * var) if var > 0 else 1.0
        if gamma == "auto":
            return 1.0 / x_scaled.shape[1]
        raise ValueError(f"unknown gamma string: {gamma!r}")
    return float(gamma)


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """``sign(x) * log1p(|x|)`` (odd, monotone, defined for negative values such as rollover flights)."""
    return np.sign(x) * np.log1p(np.abs(x))


class FeatureTransform:
    """Signed log on non-bounded features, then RobustScaler. Fit on TRAIN data only.

    Args:
        feature_names: column order of the matrices passed to :meth:`fit` / :meth:`transform`.
            Names in :data:`kl.features.BOUNDED_RATE_FEATURES` skip the log.
    """

    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names: list[str] = list(feature_names)
        self.log_mask: np.ndarray = np.array([n not in BOUNDED_RATE_FEATURES for n in self.feature_names])
        self.scaler: RobustScaler = RobustScaler()

    def _pre(self, x: np.ndarray) -> np.ndarray:
        out = np.array(x, dtype="float64", copy=True)
        out[:, self.log_mask] = signed_log1p(out[:, self.log_mask])
        return out

    def fit(self, x_train: np.ndarray) -> "FeatureTransform":
        """Fit the RobustScaler (median / IQR) on the log-transformed TRAIN matrix."""
        self.scaler.fit(self._pre(x_train))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the fitted transform to any matrix with the same columns."""
        return self.scaler.transform(self._pre(x))

    def fit_transform(self, x_train: np.ndarray) -> np.ndarray:
        """Fit on ``x_train`` and return its transform."""
        return self.fit(x_train).transform(x_train)


def fit_transform_on_train(x_train: np.ndarray, feature_names: Sequence[str]) -> FeatureTransform:
    """A :class:`FeatureTransform` fitted on TRAIN data only."""
    return FeatureTransform(feature_names).fit(x_train)


def fit_ocsvm(x_scaled: np.ndarray, nu: float, gamma: str | float) -> tuple[OneClassSVM, float]:
    """Fit an RBF One-Class SVM; returns ``(model, effective_numeric_gamma)``."""
    g = resolve_gamma(gamma, x_scaled)
    return OneClassSVM(kernel="rbf", nu=float(nu), gamma=g).fit(x_scaled), g


def tau_for_fnr(val_scores: np.ndarray, fnr: float) -> float:
    """Threshold such that a fraction ``fnr`` of validation humans has ``score < tau``."""
    return float(np.quantile(val_scores, fnr))


@dataclass
class LivenessModel:
    """Frozen detector: feature transform + SVM + feature order + thresholds."""

    transform: FeatureTransform
    svm: OneClassSVM
    feature_names: list[str]
    thresholds: dict[str, float] = field(default_factory=dict)  # {"tau5": ..., "tau10": ...}
    params: dict[str, Any] = field(default_factory=dict)        # nu, gamma, gamma_effective, ...
    config_hash: str = ""

    def _matrix(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            x = x[self.feature_names].to_numpy(dtype="float64")
        return np.asarray(x, dtype="float64")

    def score(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Decision scores (higher = more human-like)."""
        return self.svm.decision_function(self.transform.transform(self._matrix(x)))

    def accept(self, x: pd.DataFrame | np.ndarray, tau: str | float) -> np.ndarray:
        """Boolean array: judged human at threshold ``tau`` (a value or a key such as ``"tau5"``)."""
        t = self.thresholds[tau] if isinstance(tau, str) else float(tau)
        return self.score(x) >= t

    def save(self, model_dir: Path) -> None:
        """Write ``model.joblib`` (transform + SVM) and a human-readable ``thresholds.json``."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"transform": self.transform, "svm": self.svm}, model_dir / "model.joblib")
        write_json(model_dir / "thresholds.json", {
            "thresholds": self.thresholds, "feature_names": self.feature_names,
            "params": self.params, "config_hash": self.config_hash,
        })

    @classmethod
    def load(cls, model_dir: Path) -> "LivenessModel":
        """Load a model saved with :meth:`save`."""
        model_dir = Path(model_dir)
        blob = joblib.load(model_dir / "model.joblib")
        meta = read_json(model_dir / "thresholds.json")
        return cls(
            transform=blob["transform"], svm=blob["svm"], feature_names=list(meta["feature_names"]),
            thresholds={k: float(v) for k, v in meta["thresholds"].items()},
            params=meta["params"], config_hash=meta["config_hash"],
        )
