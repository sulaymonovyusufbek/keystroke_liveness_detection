"""A2: per-digraph empirical timing.

Every press-to-press interval is drawn from the empirical pp distribution of the lower-cased digraph (previous
character, current character) measured on attacker_pool FIT sessions; every hold is drawn from the empirical hold
distribution of the key. Backoff when a digraph has fewer than ``min_samples`` (30) samples: the second character's pp
distribution, then the global one (holds: key -> global). All draws are independent, so negative flights arise
naturally whenever hold > pp. Shift rows follow the pool's Shift lead / lag statistics without overlap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Attacker
from .common import BOT_START_MS, build_events, frame_from_events
from .pool import PoolStats


class A2Digraph(Attacker):
    """Independent per-digraph pp and per-key hold draws from empirical pool distributions."""

    name = "A2"

    def __init__(self, stats: PoolStats) -> None:
        self.stats = stats

    def generate(self, sentence: str, rng: np.random.Generator) -> pd.DataFrame:
        """One bot session typing ``sentence`` exactly."""
        st = self.stats
        presses: list[int] = []
        holds: list[int] = []
        t = BOT_START_MS
        for i, ch in enumerate(sentence):
            if i > 0:
                t += max(1, st.sample_pp(sentence[i - 1], ch, rng))
            presses.append(t)
            holds.append(max(1, int(st.sample_hold(ch, rng))))
        events = build_events([(ch, False) for ch in sentence], presses, holds, st, rng)
        return frame_from_events(sentence, sentence, events)
