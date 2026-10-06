"""A5: replay attacks built from real attacker_pool recordings.

* :class:`A5aVerbatim` replays a random pool session EXACTLY (its own sentence, keystrokes and timings). It is a real
  human recording, so acceptance by the detector is expected: it marks the boundary of session-level behavioural
  liveness (motivating challenge sentences / freshness checks). The ``sentence`` argument of ``generate`` is ignored.
* :class:`A5bTransplant` applies the hold / flight sequence of a random pool session, in order, to the target
  sentence's keystrokes (a Shift row precedes each shifted character and counts as one keystroke of the sequence). If
  the target needs more keystrokes than the source session has, the sequence is continued with contiguous chunks of
  other random pool sessions; if it needs fewer, the source sequence is truncated. USER_INPUT = target sentence.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .a1_gaussian import SHIFT_KEYCODE, char_to_key
from .base import Attacker
from .common import BOT_START_MS, frame_from_events
from .pool import SessionBank


class A5aVerbatim(Attacker):
    """Verbatim replay of a random attacker_pool session."""

    name = "A5a"

    def __init__(self, bank: SessionBank, source_idx: Sequence[int]) -> None:
        self.bank = bank
        self.source_idx = np.asarray(sorted(source_idx), dtype="int64")

    def generate(self, sentence: str, rng: np.random.Generator) -> pd.DataFrame:
        """A copy of one random pool session (``sentence`` is ignored)."""
        return self.bank.frame(int(self.source_idx[rng.integers(len(self.source_idx))]))


class A5bTransplant(Attacker):
    """Hold / flight sequence of a pool session applied to the target sentence."""

    name = "A5b"

    def __init__(self, bank: SessionBank, source_idx: Sequence[int], min_rows: int = 2) -> None:
        self.bank = bank
        self.source_idx = np.asarray([i for i in sorted(source_idx) if bank.n_rows(i) >= min_rows], dtype="int64")

    def sequence(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
        """``(holds[n], flights[n-1], source_session)`` for ``n`` keystrokes (see module docstring)."""
        src = int(self.source_idx[rng.integers(len(self.source_idx))])
        h, f = self.bank.hold_flight(src)
        holds, flights = list(h[:n]), list(f[:max(n - 1, 0)])
        while len(holds) < n:
            other = int(self.source_idx[rng.integers(len(self.source_idx))])
            oh, of = self.bank.hold_flight(other)
            start = int(rng.integers(1, len(oh)))          # holds[start:], joined by flight[start - 1]
            take = min(n - len(holds), len(oh) - start)
            holds += list(oh[start:start + take])
            flights += list(of[start - 1:start - 1 + take])
        return np.asarray(holds, dtype="float64"), np.asarray(flights, dtype="float64"), src

    def generate(self, sentence: str, rng: np.random.Generator) -> pd.DataFrame:
        """One bot session typing ``sentence`` with transplanted timings."""
        keystrokes: list[tuple[str, int]] = []
        for ch in sentence:
            code, shifted = char_to_key(ch)
            if shifted:
                keystrokes.append(("SHIFT", SHIFT_KEYCODE))
            keystrokes.append((ch, code))
        holds, flights, _ = self.sequence(len(keystrokes), rng)
        events = []
        press = float(BOT_START_MS)
        for k, (letter, code) in enumerate(keystrokes):
            release = press + holds[k]
            events.append((press, release, letter, code))
            if k + 1 < len(keystrokes):
                press = release + flights[k]
        return frame_from_events(sentence, sentence, events)
