"""Shared builders for the tests."""
from __future__ import annotations

import pandas as pd


def make_session(press, release, keycode, letters=None, sentence="hello", user_input="hello", kid=None) -> pd.DataFrame:
    """Aalto-schema session from parallel lists."""
    n = len(press)
    return pd.DataFrame({
        "PARTICIPANT_ID": [1] * n, "TEST_SECTION_ID": [1] * n,
        "SENTENCE": [sentence] * n, "USER_INPUT": [user_input] * n,
        "KEYSTROKE_ID": kid if kid is not None else list(range(1, n + 1)),
        "PRESS_TIME": [float(p) for p in press], "RELEASE_TIME": [float(r) for r in release],
        "LETTER": letters if letters is not None else ["x"] * n, "KEYCODE": list(keycode),
    })
