"""Session definition and cleaning filters.

A *session* is the set of rows sharing ``(PARTICIPANT_ID, TEST_SECTION_ID)``, i.e. one typed
sentence. A session is dropped when ANY rule fires; every rule is counted independently (a session
that violates two rules increments both counters) and ``dropped_total`` counts unique sessions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

#: Drop-rule names, in the order they are reported.
DROP_REASONS: tuple[str, ...] = (
    "too_few_keystrokes", "bad_times", "hold_out_of_range", "gap_too_long", "empty_text",
)


@dataclass(frozen=True)
class CleaningParams:
    """Thresholds of the cleaning rules (all in ms except ``min_keystrokes``)."""

    min_keystrokes: int = 10
    min_hold_ms: float = 0.0
    max_hold_ms: float = 2000.0
    max_press_gap_ms: float = 5000.0

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> "CleaningParams":
        """Build from the ``cleaning`` section of config.yaml."""
        return cls(
            min_keystrokes=int(section["min_keystrokes"]),
            min_hold_ms=float(section["min_hold_ms"]),
            max_hold_ms=float(section["max_hold_ms"]),
            max_press_gap_ms=float(section["max_press_gap_ms"]),
        )


def sort_keystrokes(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by session, then PRESS_TIME, with KEYSTROKE_ID as the tie-break only."""
    return df.sort_values(
        ["PARTICIPANT_ID", "TEST_SECTION_ID", "PRESS_TIME", "KEYSTROKE_ID"],
        kind="mergesort", na_position="last",
    ).reset_index(drop=True)


def empty_counts() -> dict[str, int]:
    """Zeroed counter dict for :func:`clean_sessions`."""
    out = {"rows_in": 0, "sessions_total": 0}
    out.update({f"drop_{r}": 0 for r in DROP_REASONS})
    out.update({"dropped_total": 0, "sessions_kept": 0, "rows_kept": 0})
    return out


def merge_counts(a: dict[str, int], b: Mapping[str, int]) -> dict[str, int]:
    """Element-wise sum of two counter dicts."""
    return {k: int(a.get(k, 0)) + int(b.get(k, 0)) for k in set(a) | set(b)}


def clean_sessions(df: pd.DataFrame, params: CleaningParams) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the cleaning rules to keystroke rows of whole participants.

    Args:
        df: keystroke rows (Aalto schema, ``LAYOUT`` optional). Must contain complete sessions.
        params: thresholds.

    Returns:
        ``(kept, counts)``: ``kept`` holds the rows of surviving sessions, sorted by session and
        PRESS_TIME, with an extra ``session_id`` column (``"<participant>_<test_section>"``).
    """
    counts = empty_counts()
    counts["rows_in"] = int(len(df))
    if df.empty:
        return df.assign(session_id=pd.Series(dtype="str")), counts

    df = sort_keystrokes(df)
    pid = df["PARTICIPANT_ID"].to_numpy()
    tsid = df["TEST_SECTION_ID"].to_numpy()
    new_group = np.r_[True, (pid[1:] != pid[:-1]) | (tsid[1:] != tsid[:-1])]
    sid = np.cumsum(new_group) - 1
    n_sess = int(sid[-1]) + 1
    starts = np.flatnonzero(new_group)

    press = df["PRESS_TIME"].to_numpy(dtype="float64")
    release = df["RELEASE_TIME"].to_numpy(dtype="float64")
    hold = release - press
    gap = np.r_[np.nan, np.diff(press)]
    gap[new_group] = np.nan  # no gap across session boundaries

    def any_per_session(mask: np.ndarray) -> np.ndarray:
        return np.bincount(sid, weights=mask.astype("float64"), minlength=n_sess) > 0

    with np.errstate(invalid="ignore"):  # NaN comparisons are False by design
        flags = {
            "too_few_keystrokes": np.bincount(sid, minlength=n_sess) < params.min_keystrokes,
            "bad_times": any_per_session(np.isnan(press) | np.isnan(release)),
            "hold_out_of_range": any_per_session((hold < params.min_hold_ms) | (hold > params.max_hold_ms)),
            "gap_too_long": any_per_session(gap > params.max_press_gap_ms),
            "empty_text": (
                (df["SENTENCE"].iloc[starts].str.strip() == "").to_numpy()
                | (df["USER_INPUT"].iloc[starts].str.strip() == "").to_numpy()
            ),
        }
    drop = np.zeros(n_sess, dtype=bool)
    for name, flag in flags.items():
        counts[f"drop_{name}"] = int(flag.sum())
        drop |= flag
    keep_row = ~drop[sid]
    kept = df.loc[keep_row].reset_index(drop=True)
    kept["session_id"] = (kept["PARTICIPANT_ID"].astype(str) + "_" + kept["TEST_SECTION_ID"].astype(str)).astype("str")
    counts["sessions_total"] = n_sess
    counts["dropped_total"] = int(drop.sum())
    counts["sessions_kept"] = int((~drop).sum())
    counts["rows_kept"] = int(len(kept))
    return kept, counts


def session_index(kept: pd.DataFrame) -> pd.DataFrame:
    """One row per kept session: ids, LAYOUT, sentence, typed text and keystroke count."""
    if kept.empty:
        return pd.DataFrame(columns=["session_id", "PARTICIPANT_ID", "TEST_SECTION_ID", "LAYOUT", "SENTENCE", "USER_INPUT", "n_keys"])
    grp = kept.groupby("session_id", sort=False)
    first = grp[["PARTICIPANT_ID", "TEST_SECTION_ID", "SENTENCE", "USER_INPUT"] + (["LAYOUT"] if "LAYOUT" in kept else [])].first()
    first["n_keys"] = grp.size()
    if "LAYOUT" not in first:
        first["LAYOUT"] = ""
    return first.reset_index()[["session_id", "PARTICIPANT_ID", "TEST_SECTION_ID", "LAYOUT", "SENTENCE", "USER_INPUT", "n_keys"]]
