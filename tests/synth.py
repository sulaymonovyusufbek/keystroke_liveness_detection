"""Synthetic 'human' sessions with known typing behaviour, used to test the Phase 2 attackers without the dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd

from kl.attacks.a1_gaussian import char_to_key
from kl.attacks.common import QWERTY_NEIGHBOURS

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs!",
    "How vexingly quick daft zebras jump, said Tom.",
    "She sells sea shells by the sea shore.",
    "A journey of a thousand miles begins with one step.",
    "Waltz, bad nymph, for quick jigs vex Fred.",
]


def synth_session(rng: np.random.Generator, sentence: str, uid: int, tsid: int, p_err: float = 0.04, p_uncorr: float = 0.01) -> pd.DataFrame:
    """One human-like session: digraph-dependent pp, holds ~ N(100, 20), corrected typos (delay 0-2) and rare uncorrected ones."""
    keys: list[tuple[str, bool]] = []
    typed = list(sentence)
    i, L = 0, len(sentence)
    while i < L:
        ch = sentence[i]
        near = QWERTY_NEIGHBOURS.get(ch.lower())
        r = rng.random()
        if near and r < p_err:
            d = min(int(rng.integers(0, 3)), L - 1 - i)
            keys.append((near[0], False))
            keys += [(sentence[j], False) for j in range(i + 1, i + d + 1)]
            keys += [("", True)] * (d + 1)
            keys += [(sentence[j], False) for j in range(i, i + d + 1)]
            i += d + 1
        elif near and r < p_err + p_uncorr:
            keys.append((near[-1], False))
            typed[i] = near[-1]
            i += 1
        else:
            keys.append((ch, False))
            i += 1
    t, ev = 1_400_000_000_000, []
    prev_press, prev_release, prev_ch = None, None, None
    for ch, bs in keys:
        if prev_press is None:
            press = t
        elif bs or prev_ch == "":
            press = prev_press + int(rng.integers(120, 260))
        else:
            base = 140 + 12 * (ord(ch.lower()) % 7) + (500 if prev_ch == " " and rng.random() < 0.15 else 0)
            press = prev_press + max(20, int(rng.lognormal(np.log(base), 0.25)))
        hold = max(20, int(rng.normal(100, 20)))
        code, shifted = char_to_key(ch) if not bs else (8, False)
        if shifted:
            ev.append((press - 60, press + hold + 30, "SHIFT", 16))
        ev.append((press, press + hold, "BKSP" if bs else ch, code))
        prev_press, prev_ch = press, ("" if bs else ch)
    ev.sort(key=lambda e: e[0])
    n = len(ev)
    return pd.DataFrame({
        "PARTICIPANT_ID": uid, "TEST_SECTION_ID": tsid, "SENTENCE": sentence, "USER_INPUT": "".join(typed),
        "KEYSTROKE_ID": np.arange(1, n + 1), "PRESS_TIME": [float(e[0]) for e in ev], "RELEASE_TIME": [float(e[1]) for e in ev],
        "LETTER": [e[2] for e in ev], "KEYCODE": [e[3] for e in ev],
        "session_id": f"{uid}_{tsid}",
    })


def synth_pool(n_users: int = 40, per_user: int = 15, seed: int = 0, **kw) -> pd.DataFrame:
    """A pool of ``n_users * per_user`` synthetic sessions (Aalto schema + ``session_id``)."""
    rng = np.random.default_rng(seed)
    frames = [synth_session(rng, SENTENCES[(u + k) % len(SENTENCES)], 1000 + u, k, **kw) for u in range(n_users) for k in range(per_user)]
    return pd.concat(frames, ignore_index=True)
