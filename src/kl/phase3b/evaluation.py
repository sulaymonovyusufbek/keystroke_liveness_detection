"""Thresholds (VAL humans only), single test-split access, and seed-aggregated cell metrics for Phase 3b.

A neural cell is evaluated for every training seed. The table reports the mean and sample sd (ddof=1) over seeds, and
seed 42's value with the percentile bootstrap CI of :func:`kl.phase2.evaluation.cell_metrics` (users resampled with
the SAME weights as the Phase 2 tables).
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..model import tau_for_fnr
from ..phase2.evaluation import METRICS, cell_metrics

FNRS: tuple[float, ...] = (0.05, 0.10)


class ThresholdDataError(ValueError):
    """Thresholds were requested on something other than the validation humans."""


class TestReuseError(RuntimeError):
    """The test split was requested a second time for the same (detector, seed)."""

    __test__ = False  # not a pytest test class


@dataclass(frozen=True)
class Scored:
    """Human scores (higher = more human-like) of one set, with what the set is."""

    scores: np.ndarray
    split: str
    kind: str            # "human" / "bot"
    name: str = ""


def thresholds_from_val_humans(val: Scored, fnrs: Sequence[float] = FNRS) -> dict[str, float]:
    """tau5 / tau10 as quantiles of VALIDATION-HUMAN scores; anything else raises :class:`ThresholdDataError`."""
    if val.split != "val" or val.kind != "human":
        raise ThresholdDataError(f"thresholds must come from validation humans only, got {val.name!r} ({val.split}, {val.kind})")
    return {f"tau{int(round(f * 100))}": tau_for_fnr(val.scores, f) for f in fnrs}


class TestAccess:
    """Ledger of test-split uses: each key (detector variant + seed) may load the test split once.

    The ledger persists in ``test_access.json`` so a second RUN is refused as well, unless ``allow_rerun``.
    """

    __test__ = False

    def __init__(self, path: Path, allow_rerun: bool = False) -> None:
        self.path, self.allow_rerun = Path(path), allow_rerun
        self._session: set[str] = set()

    def _ledger(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def used(self, key: str) -> bool:
        """Has ``key`` already used the test split (in any run)?"""
        return key in self._ledger()

    def claim(self, key: str) -> None:
        """Record a test-split use by ``key``; raise :class:`TestReuseError` on a second use."""
        ledger = self._ledger()
        if key in self._session or (key in ledger and not self.allow_rerun):
            raise TestReuseError(f"the test split was already used by {key!r} ({ledger.get(key, 'this run')}); pass --allow-rerun to override")
        self._session.add(key)
        ledger.setdefault(key, []).append(_dt.datetime.now().isoformat(timespec="seconds"))
        self.path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


class TestSplit:
    """The TEST-split encoded sets (humans + bots per attacker); :meth:`for_detector` hands them out once per key."""

    __test__ = False

    def __init__(self, humans: Any, bots: Mapping[str, Any], access: TestAccess) -> None:
        if humans.split != "test" or any(b.split != "test" for b in bots.values()):
            raise ValueError("TestSplit holds test-split sets only")
        self._humans, self._bots, self.access = humans, dict(bots), access

    def users(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """User ids of the test humans and of every attacker's bots (for the bootstrap; no scores involved)."""
        return self._humans.user_ids, {a: b.user_ids for a, b in self._bots.items()}

    def for_detector(self, key: str) -> tuple[Any, dict[str, Any]]:
        """``(humans, {attacker: bots})`` for ``key`` (e.g. ``"D4/seed42"``); a second call raises."""
        self.access.claim(key)
        return self._humans, self._bots


def point_metrics(h_scores: np.ndarray, b_scores: np.ndarray, taus: Mapping[str, float]) -> dict[str, float]:
    """APCER/FPR, BPCER/FNR at every tau and ROC-AUC (human = positive, ties 1/2; never flipped)."""
    out = {}
    for k, t in taus.items():
        out[f"bpcer_fnr_{k}"] = float(np.mean(h_scores < t))
        out[f"apcer_fpr_{k}"] = float(np.mean(b_scores >= t))
    y = np.r_[np.ones(len(h_scores)), np.zeros(len(b_scores))]
    out["auc"] = float(roc_auc_score(y, np.r_[h_scores, b_scores]))
    return out


def seed_cell(
    detector: str, trained_on: str, tested_on: str, per_seed: Mapping[int, Mapping[str, Any]], h_users: np.ndarray,
    b_users: np.ndarray, weights: np.ndarray, ci_level: float, ci_seed: int = 42,
) -> dict[str, Any]:
    """One table row from ``{seed: {"h": human scores, "b": bot scores, "taus": {...}}}``.

    Columns per metric: ``<m>_mean``, ``<m>_sd`` (ddof=1 over seeds), ``<m>_ci_lo`` / ``<m>_ci_hi`` (bootstrap of seed
    ``ci_seed``), ``<m>_seed<s>``; plus ``tau5_seed<s>`` / ``tau10_seed<s>``.
    """
    seeds = sorted(per_seed)
    first = per_seed[seeds[0]]
    row: dict[str, Any] = {"detector": detector, "trained_on": trained_on, "tested_on": tested_on, "source": "phase3b",
                           "n_seeds": len(seeds), "n_humans": len(first["h"]), "n_attacks": len(first["b"])}
    vals: dict[str, list[float]] = {m: [] for m in METRICS}
    for s in seeds:
        d = per_seed[s]
        if s == ci_seed:
            pm = cell_metrics(d["h"], h_users, d["b"], b_users, d["taus"], weights, ci_level)
            for m in METRICS:
                row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = pm[f"{m}_ci_lo"], pm[f"{m}_ci_hi"]
        else:
            pm = point_metrics(d["h"], d["b"], d["taus"])
        for m in METRICS:
            vals[m].append(pm[m])
            row[f"{m}_seed{s}"] = pm[m]
        for k, t in d["taus"].items():
            row[f"{k}_seed{s}"] = t
    for m in METRICS:
        v = np.asarray(vals[m])
        row[f"{m}_mean"] = float(v.mean())
        row[f"{m}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    return row


def phase2_rows(table: pd.DataFrame, detectors: Sequence[str]) -> pd.DataFrame:
    """Phase 2 rows of ``detectors`` in the Phase 3b column layout (a deterministic single model: sd = NaN)."""
    sub = table[table["detector"].isin(detectors)].copy()
    rows = []
    for _, r in sub.iterrows():
        row = {"detector": r["detector"], "trained_on": r["trained_on"], "tested_on": r["tested_on"], "source": "phase2 (copied)",
               "n_seeds": 1, "n_humans": int(r["n_humans"]), "n_attacks": int(r["n_attacks"])}
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_sd"] = float(r[m]), float("nan")
            row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = float(r[f"{m}_ci_lo"]), float(r[f"{m}_ci_hi"])
        row["tau5_seed42"], row["tau10_seed42"] = float(r["tau5"]), float(r["tau10"])
        rows.append(row)
    return pd.DataFrame(rows)


def ordered_columns(df: pd.DataFrame, seeds: Sequence[int]) -> pd.DataFrame:
    """Stable column order for the Phase 3b tables."""
    head = ["detector", "trained_on", "tested_on", "source", "n_seeds", "n_humans", "n_attacks"]
    body = [c for m in METRICS for c in (f"{m}_mean", f"{m}_sd", f"{m}_ci_lo", f"{m}_ci_hi")]
    per_seed = [f"{m}_seed{s}" for m in METRICS for s in seeds] + [f"{t}_seed{s}" for t in ("tau5", "tau10") for s in seeds]
    cols = [c for c in head + body + per_seed if c in df.columns] + [c for c in df.columns if c not in head + body + per_seed]
    return df[cols]
