"""A3: conditional / temporal timing.

Per session: draw a speed offset ``s`` from the pool's per-session offsets, then an AR(1) process on log pp around the
digraph mean::

    log pp_i = mu_digraph(c_{i-1}, c_i) + s + phi * (log pp_{i-1} - mu_digraph(c_{i-2}, c_{i-1}) - s) + eps_i,
    eps_i ~ N(0, sigma^2)                       (residual state starts at 0)

``mu`` is the mean log pp of non-pause transitions (backoff digraph -> second char -> global), ``phi`` and ``sigma``
are fitted on attacker_pool residuals. pp is rounded and clipped to [1, max_pp]. Holds are drawn from the per-key
empirical distribution and scaled by ``exp(slope * (s - mean s))`` (the pool's hold-speed regression), clipped to
[1, max_hold]. No pauses are generated (the pool statistics exclude flights > pause threshold); see A4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Attacker
from .common import BOT_START_MS, build_events, frame_from_events
from .pool import PoolStats


class A3Temporal(Attacker):
    """AR(1) log-pp process around digraph means with a session speed factor."""

    name = "A3"

    def __init__(self, stats: PoolStats) -> None:
        self.stats = stats

    def generate(self, sentence: str, rng: np.random.Generator) -> pd.DataFrame:
        """One bot session typing ``sentence`` exactly."""
        st = self.stats
        s = st.sample_speed(rng)
        presses: list[int] = []
        holds: list[int] = []
        t = BOT_START_MS
        r = 0.0
        for i, ch in enumerate(sentence):
            if i > 0:
                r = st.phi * r + rng.normal(0.0, st.sigma)
                t += st.clip_pp(float(np.exp(st.mu(sentence[i - 1], ch) + s + r)))
            presses.append(t)
            holds.append(st.scaled_hold(ch, s, rng))
        events = build_events([(ch, False) for ch in sentence], presses, holds, st, rng)
        return frame_from_events(sentence, sentence, events)
