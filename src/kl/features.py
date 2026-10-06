"""Session -> feature dict. ONE code path for human sessions and bot sessions.

All times are in ms. For consecutive keys ``i, i+1`` (sorted by PRESS_TIME, KEYSTROKE_ID as the
tie-break only)::

    hold_i   = RELEASE_i - PRESS_i        pp_i     = PRESS_{i+1}   - PRESS_i
    rr_i     = RELEASE_{i+1} - RELEASE_i  flight_i = PRESS_{i+1}   - RELEASE_i   (< 0: rollover)

Keys are identified by KEYCODE only (LETTER is unreliable for Backspace / Shift in the corpus).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

STD_DDOF: int = 0  #: population standard deviation (ddof=0) everywhere; stated in METHODS.md


@dataclass(frozen=True)
class FeatureSpec:
    """Documentation record of one feature (used to generate FEATURES.md)."""

    name: str
    group: str
    formula: str
    unit: str


def _timing_specs() -> list[FeatureSpec]:
    """TIMING features. ``flight`` has no cv (Revision 1: undefined when the mean flight is ~0 because of rollover)."""
    defs = {
        "hold": "RELEASE_i - PRESS_i",
        "pp": "PRESS_{i+1} - PRESS_i",
        "rr": "RELEASE_{i+1} - RELEASE_i",
        "flight": "PRESS_{i+1} - RELEASE_i (negative = rollover)",
    }
    specs: list[FeatureSpec] = []
    for x, d in defs.items():
        specs += [
            FeatureSpec(f"{x}_mean", "TIMING", f"mean of {x}_i = {d}", "ms"),
            FeatureSpec(f"{x}_std", "TIMING", f"population std (ddof=0) of {x}_i", "ms"),
        ]
        if x == "flight":
            specs += [
                FeatureSpec("flight_median", "TIMING", "median of flight_i", "ms"),
                FeatureSpec("flight_iqr", "TIMING", "75th - 25th percentile of flight_i (linear interpolation, np.percentile)", "ms"),
            ]
        else:
            specs.append(FeatureSpec(f"{x}_cv", "TIMING", f"{x}_std / max(|{x}_mean|, 1e-6)", "ratio"))
    return specs


FEATURE_SPECS: list[FeatureSpec] = _timing_specs() + [
    FeatureSpec("pause_count", "RHYTHM", "number of flight_i > pause_threshold (500 ms)", "count"),
    FeatureSpec("pause_rate", "RHYTHM", "pause_count / (n_keys - 1)", "ratio"),
    FeatureSpec("rollover_rate", "RHYTHM", "(number of flight_i < 0) / (n_keys - 1)", "ratio"),
    FeatureSpec("corrections_count", "BEHAVIOUR", "number of keys with KEYCODE in {8, 46}", "count"),
    FeatureSpec("correction_rate", "BEHAVIOUR", "corrections_count / n_keys", "ratio"),
    FeatureSpec("lev_norm", "BEHAVIOUR", "Levenshtein(SENTENCE, USER_INPUT) / max(len(SENTENCE), len(USER_INPUT))", "ratio"),
    FeatureSpec("shift_overlap_count", "BEHAVIOUR",
                "per KEYCODE-16 hold: (non-Shift keys pressed while it is held) - 1, floored at 0; summed", "count"),
    FeatureSpec("n_keys", "GLOBAL", "number of keystroke rows in the session", "count"),
    FeatureSpec("total_duration_ms", "GLOBAL", "last PRESS - first PRESS", "ms"),
    FeatureSpec("wpm", "GLOBAL", "(len(USER_INPUT) / 5) / (total_duration_ms / 60000)", "words/min"),
]

#: Features bounded in [0, 1]; the model transform leaves them untouched (no signed log).
BOUNDED_RATE_FEATURES: frozenset[str] = frozenset({"pause_rate", "rollover_rate", "correction_rate", "lev_norm"})

FEATURE_NAMES: list[str] = [s.name for s in FEATURE_SPECS]
FEATURE_GROUPS: dict[str, list[str]] = {}
for _s in FEATURE_SPECS:
    FEATURE_GROUPS.setdefault(_s.group, []).append(_s.name)


def feature_columns(groups: Sequence[str]) -> list[str]:
    """Feature names belonging to ``groups`` (order of FEATURE_NAMES)."""
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")
    wanted = set(groups)
    return [s.name for s in FEATURE_SPECS if s.group in wanted]


@dataclass(frozen=True)
class FeatureParams:
    """Tunable constants of the feature definitions (defaults match the spec)."""

    pause_threshold_ms: float = 500.0
    correction_keycodes: tuple[int, ...] = (8, 46)
    shift_keycode: int = 16

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> "FeatureParams":
        """Build from the ``features`` section of config.yaml."""
        return cls(
            pause_threshold_ms=float(section["pause_threshold_ms"]),
            correction_keycodes=tuple(int(k) for k in section["correction_keycodes"]),
            shift_keycode=int(section["shift_keycode"]),
        )


def _stats(x: np.ndarray) -> tuple[float, float, float]:
    """(mean, std, cv) with cv = std / max(|mean|, 1e-6)."""
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=STD_DDOF))
    return mean, std, std / max(abs(mean), 1e-6)


def _features_sorted(
    press: np.ndarray,
    release: np.ndarray,
    keycode: np.ndarray,
    sentence: str,
    user_input: str,
    params: FeatureParams,
) -> dict[str, float]:
    """Core computation on arrays that are already sorted by (PRESS_TIME, KEYSTROKE_ID)."""
    n = len(press)
    if n < 2:
        raise ValueError("a session needs at least 2 keystrokes to compute features")
    hold = release - press
    pp = np.diff(press)
    rr = np.diff(release)
    flight = press[1:] - release[:-1]

    feats: dict[str, float] = {}
    for name, arr in (("hold", hold), ("pp", pp), ("rr", rr)):
        feats[f"{name}_mean"], feats[f"{name}_std"], feats[f"{name}_cv"] = _stats(arr)
    feats["flight_mean"] = float(np.mean(flight))
    feats["flight_std"] = float(np.std(flight, ddof=STD_DDOF))
    feats["flight_median"] = float(np.median(flight))
    q25, q75 = np.percentile(flight, [25, 75])
    feats["flight_iqr"] = float(q75 - q25)

    pause_count = int(np.count_nonzero(flight > params.pause_threshold_ms))
    feats["pause_count"] = float(pause_count)
    feats["pause_rate"] = pause_count / (n - 1)
    feats["rollover_rate"] = float(np.count_nonzero(flight < 0)) / (n - 1)

    corrections = int(np.count_nonzero(np.isin(keycode, params.correction_keycodes)))
    feats["corrections_count"] = float(corrections)
    feats["correction_rate"] = corrections / n
    denom = max(len(sentence), len(user_input))
    feats["lev_norm"] = Levenshtein.distance(sentence, user_input) / denom if denom else 0.0

    is_shift = keycode == params.shift_keycode
    overlap = 0
    for j in np.flatnonzero(is_shift):
        inside = (~is_shift) & (press > press[j]) & (press < release[j])
        overlap += max(0, int(np.count_nonzero(inside)) - 1)
    feats["shift_overlap_count"] = float(overlap)

    duration = float(np.max(press) - np.min(press))
    feats["n_keys"] = float(n)
    feats["total_duration_ms"] = duration
    feats["wpm"] = (len(user_input) / 5.0) / (duration / 60000.0) if duration > 0 else float("nan")
    return {name: feats[name] for name in FEATURE_NAMES}


def _sort_order(press: np.ndarray, kid: np.ndarray | None) -> np.ndarray:
    """Indices sorting by PRESS_TIME with KEYSTROKE_ID as tie-break only."""
    if kid is None:
        return np.argsort(press, kind="stable")
    return np.lexsort((kid, press))


def compute_features(session: pd.DataFrame, params: FeatureParams | None = None) -> dict[str, float]:
    """Features of ONE session (Aalto-schema keystroke rows) as an ordered dict.

    Args:
        session: rows of a single session; needs PRESS_TIME, RELEASE_TIME, KEYCODE, SENTENCE,
            USER_INPUT (KEYSTROKE_ID optional, used only as the sort tie-break).
        params: feature constants (defaults follow the spec).

    Returns:
        ``{feature_name: value}`` in :data:`FEATURE_NAMES` order.
    """
    params = params or FeatureParams()
    press = session["PRESS_TIME"].to_numpy(dtype="float64")
    release = session["RELEASE_TIME"].to_numpy(dtype="float64")
    keycode = session["KEYCODE"].to_numpy(dtype="float64", na_value=np.nan)
    kid = session["KEYSTROKE_ID"].to_numpy(dtype="float64", na_value=np.nan) if "KEYSTROKE_ID" in session else None
    order = _sort_order(press, kid)
    return _features_sorted(
        press[order], release[order], keycode[order],
        str(session["SENTENCE"].iloc[0]), str(session["USER_INPUT"].iloc[0]), params,
    )


def extract_features_frame(keystrokes: pd.DataFrame, params: FeatureParams | None = None) -> pd.DataFrame:
    """Features for many sessions at once (same core as :func:`compute_features`).

    Args:
        keystrokes: rows with a ``session_id`` column (plus optional ``LAYOUT``); sessions must be
            complete (not split across calls).

    Returns:
        One row per session: ``session_id, user_id, LAYOUT`` + the features, sorted by session_id.
    """
    params = params or FeatureParams()
    if keystrokes.empty:
        return pd.DataFrame(columns=["session_id", "user_id", "LAYOUT"] + FEATURE_NAMES)
    sid_all = keystrokes["session_id"].to_numpy(dtype=object)
    press_all = keystrokes["PRESS_TIME"].to_numpy(dtype="float64")
    release_all = keystrokes["RELEASE_TIME"].to_numpy(dtype="float64")
    key_all = keystrokes["KEYCODE"].to_numpy(dtype="float64", na_value=np.nan)
    kid_all = keystrokes["KEYSTROKE_ID"].to_numpy(dtype="float64", na_value=np.nan)
    sent_all = keystrokes["SENTENCE"].to_numpy(dtype=object)
    inp_all = keystrokes["USER_INPUT"].to_numpy(dtype=object)
    pid_all = keystrokes["PARTICIPANT_ID"].to_numpy()
    layout_all = keystrokes["LAYOUT"].to_numpy(dtype=object) if "LAYOUT" in keystrokes else np.full(len(keystrokes), "", dtype=object)

    codes, uniques = pd.factorize(sid_all, sort=True)
    order = np.lexsort((kid_all, press_all, codes))
    codes = codes[order]
    bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
    rows: list[dict[str, Any]] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        feats = _features_sorted(
            press_all[idx], release_all[idx], key_all[idx], str(sent_all[idx[0]]), str(inp_all[idx[0]]), params,
        )
        rows.append({
            "session_id": uniques[codes[a]], "user_id": int(pid_all[idx[0]]), "LAYOUT": str(layout_all[idx[0]]), **feats,
        })
    return pd.DataFrame(rows)


def worked_example() -> pd.DataFrame:
    """The 3-key session used in FEATURES.md and in the unit tests (hold=[100,80,110])."""
    return pd.DataFrame({
        "PARTICIPANT_ID": [1, 1, 1], "TEST_SECTION_ID": [1, 1, 1],
        "SENTENCE": ["abc"] * 3, "USER_INPUT": ["abd"] * 3,
        "KEYSTROKE_ID": [1, 2, 3],
        "PRESS_TIME": [0.0, 200.0, 450.0], "RELEASE_TIME": [100.0, 280.0, 560.0],
        "LETTER": ["a", "b", "c"], "KEYCODE": [65, 66, 67],
    })
