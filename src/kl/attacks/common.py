"""Shared helpers of the Phase 2 attackers: keystroke assembly with Shift rows, QWERTY neighbours.

Conventions are those of A1 (``a1_gaussian``): Shift = keycode 16 with LETTER ``"SHIFT"``, Backspace = keycode 8 with
LETTER ``"BKSP"``, letters use the uppercase ASCII code, times are integer ms.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .a1_gaussian import SHIFT_KEYCODE, char_to_key
from .base import AALTO_COLUMNS
from .pool import PoolStats

#: first press of every bot session (ms, arbitrary epoch origin, leaves room for a leading Shift press)
BOT_START_MS: int = 1_500_000_000_000 + 10_000
BACKSPACE_KEYCODE = 8

_QWERTY_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
_ROW_OFFSET = (0.0, 0.25, 0.75)


def _build_neighbours() -> dict[str, tuple[str, ...]]:
    pos = {ch: (c + _ROW_OFFSET[r], float(r)) for r, row in enumerate(_QWERTY_ROWS) for c, ch in enumerate(row)}
    out: dict[str, tuple[str, ...]] = {}
    for ch, (x, y) in pos.items():
        near = sorted(o for o, (ox, oy) in pos.items() if o != ch and np.hypot(ox - x, oy - y) <= 1.3)
        out[ch] = tuple(near)
    return out


#: adjacent keys on a US QWERTY keyboard (key centres within 1.3 key widths; row stagger 0 / 0.25 / 0.75)
QWERTY_NEIGHBOURS: dict[str, tuple[str, ...]] = _build_neighbours()


def frame_from_events(sentence: str, user_input: str, events: Sequence[tuple[float, float, str, int]]) -> pd.DataFrame:
    """Aalto-schema frame from ``(press, release, letter, keycode)`` events (stable-sorted by press; ids 1..n)."""
    order = sorted(range(len(events)), key=lambda i: events[i][0])
    ev = [events[i] for i in order]
    n = len(ev)
    return pd.DataFrame({
        "PARTICIPANT_ID": np.full(n, -1, dtype="int64"), "TEST_SECTION_ID": np.full(n, -1, dtype="int64"),
        "SENTENCE": [sentence] * n, "USER_INPUT": [user_input] * n,
        "KEYSTROKE_ID": np.arange(1, n + 1, dtype="int64"),
        "PRESS_TIME": np.array([e[0] for e in ev], dtype="float64"), "RELEASE_TIME": np.array([e[1] for e in ev], dtype="float64"),
        "LETTER": [e[2] for e in ev], "KEYCODE": np.array([e[3] for e in ev], dtype="int64"),
    })[AALTO_COLUMNS]


def place_shift(
    key_press: int, key_release: int, prev_press: int | None, next_press: int | None,
    stats: PoolStats, rng: np.random.Generator, overlap: bool = False,
) -> tuple[int, int]:
    """Shift press / release around a shifted key (all draws from the pool's Shift statistics).

    * Shift is pressed ``lead`` ms before the key, never earlier than 1 ms after the previous key press.
    * Non-overlapping hold: released ``lag`` ms after the key release, strictly before the next key press.
    * Overlapping hold (``overlap=True`` and a next key exists): released ``tail`` ms after the next key press.
    """
    lead = stats.sample_shift_lead(rng)
    if prev_press is not None:
        lead = min(lead, max(1, key_press - prev_press - 1))
    shift_press = key_press - lead
    if overlap and next_press is not None:
        return shift_press, next_press + stats.sample_shift_tail(rng)
    release = max(key_press + 1, key_release + stats.sample_shift_lag(rng))
    if next_press is not None:
        release = max(key_press + 1, min(release, next_press - 1))
    return shift_press, release


def build_events(
    keys: Sequence[tuple[str, bool]], presses: Sequence[int], holds: Sequence[int], stats: PoolStats,
    rng: np.random.Generator, overlap_p: float = 0.0,
) -> list[tuple[float, float, str, int]]:
    """Events for typed ``keys`` (``(char, is_backspace)``) with given press times and holds, adding Shift rows.

    ``overlap_p`` is the per-session probability that a Shift hold outlasts the next key press (A4 only).
    """
    events: list[tuple[float, float, str, int]] = []
    n = len(keys)
    for i, (ch, is_bs) in enumerate(keys):
        p, rel = presses[i], presses[i] + holds[i]
        if is_bs:
            events.append((p, rel, "BKSP", BACKSPACE_KEYCODE))
            continue
        code, shifted = char_to_key(ch)
        if shifted:
            prev_p = presses[i - 1] if i > 0 else None
            next_p = presses[i + 1] if i + 1 < n else None
            ov = overlap_p > 0 and next_p is not None and rng.random() < overlap_p
            sp, sr = place_shift(p, rel, prev_p, next_p, stats, rng, ov)
            events.append((sp, sr, "SHIFT", SHIFT_KEYCODE))
        events.append((p, rel, ch, code))
    return events
