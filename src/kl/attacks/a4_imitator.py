"""A4: realistic imitator = A3 timing + pauses + substitution typos + Shift overlap.

All behaviour rates are drawn per session from attacker_pool empirical distributions:

* ``(c, u, p_boundary, p_within)`` jointly from one random pool session: ``c`` = Backspace presses per typed
  character, ``u`` = lev_norm of the session, ``p_*`` = share of char->char transitions that are pauses (flight >
  pause threshold) when the previous character is a space (word boundary) / otherwise (within word).
* ``p_so`` from a random pool session with Shift holds: share of Shift holds released after the next key press.

Errors are SUBSTITUTION typos with adjacent QWERTY keys (letters only; other characters never mistyped). The number of
corrected events is ``round_stochastic(c * L / mean_backspaces_per_event)`` and each is corrected with Backspace after
``d`` further characters (``d`` in 0..3 from the pool's Backspace-run lengths: run of b presses -> d = b - 1; runs
longer than 4 are excluded from the delay distribution, an ASSUMPTION documented in ATTACKS.md): typo, ``d`` intended
characters, ``d + 1`` Backspaces, then the intended characters again. The number of UNCORRECTED typos is
``round_stochastic(u * L)``; they stay in USER_INPUT. Timing: char->char transitions use the A3 process; a pause
replaces the transition by a flight drawn from the pool's pause flights; transitions involving Backspace draw pp from
the pool's char->bs / bs->bs / bs->char empirical distributions and reset the AR state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Attacker
from .common import BOT_START_MS, QWERTY_NEIGHBOURS, build_events, frame_from_events
from .pool import PoolStats


def stochastic_round(x: float, rng: np.random.Generator) -> int:
    """floor(x) plus one with probability frac(x): unbiased integer rounding."""
    base = int(np.floor(x))
    return base + int(rng.random() < x - base)


class A4Imitator(Attacker):
    """A3 timing plus pauses, corrected / uncorrected typos and Shift overlap."""

    name = "A4"

    def __init__(self, stats: PoolStats) -> None:
        self.stats = stats

    @staticmethod
    def _typo(ch: str, rng: np.random.Generator) -> str:
        near = QWERTY_NEIGHBOURS[ch.lower()]
        t = near[rng.integers(len(near))]
        return t.upper() if ch.isupper() else t

    def _plan(self, sentence: str, c: float, u: float, rng: np.random.Generator) -> tuple[list[tuple[str, bool]], str]:
        """Typed key sequence ``[(char, is_backspace)]`` and the resulting USER_INPUT."""
        st = self.stats
        L = len(sentence)
        n_events = stochastic_round(c * L / st.mean_bs_per_event, rng)
        n_uncorrected = stochastic_round(u * L, rng)
        eligible = np.array([i for i, ch in enumerate(sentence) if ch.lower() in QWERTY_NEIGHBOURS], dtype="int64")
        order = rng.permutation(eligible) if len(eligible) else eligible
        reserved = np.zeros(L, dtype=bool)
        events: dict[int, int] = {}
        for i in order:
            if len(events) >= n_events:
                break
            d = min(st.sample_delay(rng), L - 1 - int(i))
            if reserved[i:i + d + 1].any():
                continue
            reserved[i:i + d + 1] = True
            events[int(i)] = d
        uncorrected: set[int] = set()
        for i in order:
            if len(uncorrected) >= n_uncorrected:
                break
            if not reserved[i]:
                reserved[i] = True
                uncorrected.add(int(i))
        keys: list[tuple[str, bool]] = []
        typed = list(sentence)
        i = 0
        while i < L:
            ch = sentence[i]
            if i in events:
                d = events[i]
                keys.append((self._typo(ch, rng), False))
                keys += [(sentence[j], False) for j in range(i + 1, i + d + 1)]
                keys += [("", True)] * (d + 1)
                keys += [(sentence[j], False) for j in range(i, i + d + 1)]
                i += d + 1
            elif i in uncorrected:
                typo = self._typo(ch, rng)
                keys.append((typo, False))
                typed[i] = typo
                i += 1
            else:
                keys.append((ch, False))
                i += 1
        return keys, "".join(typed)

    def generate(self, sentence: str, rng: np.random.Generator) -> pd.DataFrame:
        """One bot session for ``sentence`` (USER_INPUT may differ from it by uncorrected typos)."""
        st = self.stats
        s = st.sample_speed(rng)
        c, u, p_boundary, p_within = st.sample_rates(rng)
        p_so = st.sample_pso(rng)
        keys, user_input = self._plan(sentence, c, u, rng)
        presses: list[int] = []
        holds: list[int] = []
        r = 0.0
        for k, (ch, is_bs) in enumerate(keys):
            if k == 0:
                press = BOT_START_MS
            else:
                pch, pbs = keys[k - 1]
                if not pbs and not is_bs:
                    if rng.random() < (p_boundary if pch == " " else p_within):
                        press = presses[k - 1] + holds[k - 1] + st.sample_pause_flight(rng)
                        r = 0.0
                    else:
                        r = st.phi * r + rng.normal(0.0, st.sigma)
                        press = presses[k - 1] + st.clip_pp(float(np.exp(st.mu(pch, ch) + s + r)))
                else:
                    kind = "c2b" if is_bs and not pbs else ("b2b" if is_bs else "b2c")
                    press = presses[k - 1] + st.sample_bs_pp(kind, rng)
                    r = 0.0
            presses.append(press)
            holds.append(st.sample_bs_hold(rng) if is_bs else st.scaled_hold(ch, s, rng))
        events = build_events(keys, presses, holds, st, rng, overlap_p=p_so)
        return frame_from_events(sentence, user_input, events)
