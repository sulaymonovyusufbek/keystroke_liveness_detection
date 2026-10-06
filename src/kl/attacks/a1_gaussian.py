"""A1: independent-Gaussian timing generator (reconstruction of the original bot).

Every keystroke gets an independent hold ``max(HOLD_MIN, N(116.24, 23.88))`` ms and consecutive
characters are separated by an independent flight ``max(FLIGHT_MIN, N(238.656, 111.6))`` ms:
``PRESS_{next} = RELEASE_{prev} + flight``. The sentence is typed exactly (USER_INPUT = SENTENCE).

ASSUMPTIONS (the original generator is lost; all are reported in METHODS.md):
  * HOLD_MIN = 30 ms and FLIGHT_MIN = 60 ms clamp the draws.
  * Times are rounded to integer ms (like the human logs) after clamping.
  * Shifted character = Shift(16) down -> key down/up -> Shift up, nested and never overlapping
    another character. The Shift-down -> key-down delay is a flight draw; Shift is released
    ``HOLD_MIN`` ms after the key is released; the next character starts a flight after the Shift
    release.
  * A character without a JavaScript keycode in the tables below is emitted with keycode 229
    (the "IME / unidentified" code) and no Shift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .base import AALTO_COLUMNS, Attacker

SHIFT_KEYCODE = 16
UNMAPPED_KEYCODE = 229
START_TIME_MS = 1_500_000_000_000  # arbitrary epoch origin for the first press

#: US-layout punctuation typed WITHOUT Shift -> JavaScript keyCode.
UNSHIFTED_KEYCODES: dict[str, int] = {
    " ": 32, ";": 186, "=": 187, ",": 188, "-": 189, ".": 190, "/": 191, "`": 192,
    "[": 219, "\\": 220, "]": 221, "'": 222,
}
#: US-layout characters typed WITH Shift -> keyCode of the physical key.
SHIFTED_KEYCODES: dict[str, int] = {
    ")": 48, "!": 49, "@": 50, "#": 51, "$": 52, "%": 53, "^": 54, "&": 55, "*": 56, "(": 57,
    ":": 186, "+": 187, "<": 188, "_": 189, ">": 190, "?": 191, "~": 192,
    "{": 219, "|": 220, "}": 221, '"': 222,
}


def char_to_key(ch: str) -> tuple[int, bool]:
    """JavaScript keycode and whether Shift is needed for ``ch``.

    Letters map to the uppercase ASCII code for both cases ('a'/'A' -> 65, Shift for 'A'); digits
    map to their ASCII code; space is 32; punctuation uses the tables above.
    """
    if len(ch) == 1 and ch.isascii():
        if ch.isalpha():
            return ord(ch.upper()), ch.isupper()
        if ch.isdigit():
            return ord(ch), False
        if ch in UNSHIFTED_KEYCODES:
            return UNSHIFTED_KEYCODES[ch], False
        if ch in SHIFTED_KEYCODES:
            return SHIFTED_KEYCODES[ch], True
    return UNMAPPED_KEYCODE, False


@dataclass(frozen=True)
class A1Params:
    """Timing distribution of A1 (ms)."""

    hold_mean_ms: float = 116.24
    hold_std_ms: float = 23.88
    hold_min_ms: float = 30.0
    flight_mean_ms: float = 238.656
    flight_std_ms: float = 111.6
    flight_min_ms: float = 60.0

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> "A1Params":
        """Build from the ``a1`` section of config.yaml."""
        return cls(**{k: float(v) for k, v in section.items()})


class A1Gaussian(Attacker):
    """Independent-Gaussian hold / flight generator."""

    name = "A1"

    def __init__(self, params: A1Params | None = None) -> None:
        self.params = params or A1Params()

    def _hold(self, rng: np.random.Generator) -> int:
        p = self.params
        return int(round(max(p.hold_min_ms, rng.normal(p.hold_mean_ms, p.hold_std_ms))))

    def _flight(self, rng: np.random.Generator) -> int:
        p = self.params
        return int(round(max(p.flight_min_ms, rng.normal(p.flight_mean_ms, p.flight_std_ms))))

    def generate(self, sentence: str, rng: np.random.Generator) -> pd.DataFrame:
        """One bot session typing ``sentence`` exactly, in the Aalto schema."""
        rows: list[tuple[int, int, str, int]] = []  # (press, release, letter, keycode) in press order
        t = START_TIME_MS
        for i, ch in enumerate(sentence):
            if i > 0:
                t += self._flight(rng)  # PRESS_next = RELEASE_prev + flight
            code, shifted = char_to_key(ch)
            if shifted:
                shift_press = t
                key_press = shift_press + self._flight(rng)
                key_release = key_press + self._hold(rng)
                shift_release = key_release + int(round(self.params.hold_min_ms))
                rows.append((shift_press, shift_release, "SHIFT", SHIFT_KEYCODE))
                rows.append((key_press, key_release, ch, code))
                t = shift_release
            else:
                release = t + self._hold(rng)
                rows.append((t, release, ch, code))
                t = release
        n = len(rows)
        return pd.DataFrame({
            "PARTICIPANT_ID": np.full(n, -1, dtype="int64"),
            "TEST_SECTION_ID": np.full(n, -1, dtype="int64"),
            "SENTENCE": [sentence] * n,
            "USER_INPUT": [sentence] * n,
            "KEYSTROKE_ID": np.arange(1, n + 1, dtype="int64"),
            "PRESS_TIME": np.array([r[0] for r in rows], dtype="float64"),
            "RELEASE_TIME": np.array([r[1] for r in rows], dtype="float64"),
            "LETTER": [r[2] for r in rows],
            "KEYCODE": np.array([r[3] for r in rows], dtype="int64"),
        })[AALTO_COLUMNS]
