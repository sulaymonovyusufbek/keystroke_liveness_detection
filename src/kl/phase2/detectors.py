"""Detectors D1 (frozen OCSVM), D2 (Isolation Forest, pre-declared hyper-parameters), D3 (supervised Random Forest).

All detectors expose ``score(features_df)`` (higher = more human-like) and thresholds ``tau5`` / ``tau10`` such that
5% / 10% of VAL HUMANS have ``score < tau``. D2 / D3 use the frozen Phase 1 input transform; nothing is tuned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from ..model import FeatureTransform, LivenessModel, tau_for_fnr


@dataclass
class Detector:
    """A scoring function plus its validation-human thresholds."""

    name: str
    score_fn: Callable[[pd.DataFrame], np.ndarray]
    thresholds: dict[str, float] = field(default_factory=dict)
    description: str = ""

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Scores of the sessions in ``df`` (higher = more human-like)."""
        return self.score_fn(df)

    def set_thresholds_on_val_humans(self, val_humans: pd.DataFrame, fnrs: Sequence[float] = (0.05, 0.10)) -> "Detector":
        """tau5 / tau10 = empirical quantiles of the validation-human scores (reject iff score < tau)."""
        scores = self.score(val_humans)
        self.thresholds = {f"tau{int(round(f * 100))}": tau_for_fnr(scores, f) for f in fnrs}
        return self


def d1_from_model(model: LivenessModel) -> Detector:
    """D1: the FROZEN Phase 1 One-Class SVM with its frozen thresholds (never refitted)."""
    return Detector("D1", model.score, dict(model.thresholds), "frozen Phase 1 One-Class SVM (RBF)")


def _matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    return df[list(cols)].to_numpy(dtype="float64")


def fit_d2(
    train_humans: pd.DataFrame, transform: FeatureTransform, cols: Sequence[str], seed: int, hp: Mapping[str, int],
    val_humans: pd.DataFrame,
) -> Detector:
    """D2: Isolation Forest (pre-declared hyper-parameters) on transformed TRAIN humans; thresholds on VAL humans."""
    forest = IsolationForest(n_estimators=int(hp["n_estimators"]), max_samples=int(hp["max_samples"]), random_state=seed)
    forest.fit(transform.transform(_matrix(train_humans, cols)))
    det = Detector("D2", lambda df: forest.score_samples(transform.transform(_matrix(df, cols))), {},
                   f"Isolation Forest (n_estimators={hp['n_estimators']}, max_samples={hp['max_samples']}, random_state={seed})")
    return det.set_thresholds_on_val_humans(val_humans)


def fit_d3(
    train_humans: pd.DataFrame, train_bots: pd.DataFrame, transform: FeatureTransform, cols: Sequence[str], seed: int,
    hp: Mapping[str, object], val_humans: pd.DataFrame, n_jobs: int = 1, name: str = "D3",
) -> Detector:
    """D3: Random Forest on transformed TRAIN humans (label 1) + TRAIN bots (label 0); thresholds on VAL HUMANS only."""
    x = np.vstack([_matrix(train_humans, cols), _matrix(train_bots, cols)])
    y = np.r_[np.ones(len(train_humans)), np.zeros(len(train_bots))]
    rf = RandomForestClassifier(n_estimators=int(hp["n_estimators"]), class_weight=str(hp["class_weight"]), random_state=seed, n_jobs=n_jobs)
    rf.fit(transform.transform(x), y)
    human_col = int(np.flatnonzero(rf.classes_ == 1.0)[0])
    det = Detector(name, lambda df: rf.predict_proba(transform.transform(_matrix(df, cols)))[:, human_col], {},
                   f"Random Forest (n_estimators={hp['n_estimators']}, class_weight={hp['class_weight']}, random_state={seed})")
    return det.set_thresholds_on_val_humans(val_humans)
