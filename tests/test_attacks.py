"""A1 generator: schema, keycodes, timing rules, and the shared features.py path."""
import numpy as np
import pytest

from kl.attacks.a1_gaussian import A1Gaussian, A1Params, UNMAPPED_KEYCODE, char_to_key
from kl.attacks.base import AALTO_COLUMNS, Attacker
from kl.features import FEATURE_NAMES, compute_features


def test_keycode_table():
    assert char_to_key("a") == (65, False) and char_to_key("A") == (65, True)
    assert char_to_key(" ") == (32, False) and char_to_key("7") == (55, False)
    assert char_to_key(".") == (190, False) and char_to_key("?") == (191, True) and char_to_key("!") == (49, True)
    assert char_to_key("é") == (UNMAPPED_KEYCODE, False)


def test_output_schema_and_exact_typing():
    bot = A1Gaussian().generate("Hello, World.", np.random.default_rng(0))
    assert list(bot.columns) == AALTO_COLUMNS
    assert (bot["SENTENCE"] == "Hello, World.").all() and (bot["USER_INPUT"] == "Hello, World.").all()
    assert bot["KEYSTROKE_ID"].is_unique
    typed = [l for l in bot["LETTER"] if l != "SHIFT"]
    assert "".join(typed) == "Hello, World."
    assert (bot["KEYCODE"] == 16).sum() == 2 and (bot.loc[bot["LETTER"] == "SHIFT", "KEYCODE"] == 16).all()
    assert bot.loc[bot["LETTER"] == "h", "KEYCODE"].iloc[0] == 72 if (bot["LETTER"] == "h").any() else True


def test_timing_constraints_and_no_shift_overlap():
    rng = np.random.default_rng(1)
    gen = A1Gaussian()
    for _ in range(20):
        bot = gen.generate("The Quick Brown Fox Jumps, Over: (the) Lazy Dog!", rng)
        assert ((bot["RELEASE_TIME"] - bot["PRESS_TIME"]) >= 30).all()
        chars = bot.reset_index(drop=True)
        for i, r in chars.iterrows():
            if r["LETTER"] == "SHIFT":
                key = chars.iloc[i + 1]
                assert r["PRESS_TIME"] < key["PRESS_TIME"] and key["RELEASE_TIME"] < r["RELEASE_TIME"]  # key nested inside Shift
        # a new character starts >= FLIGHT_MIN after the previous character ends
        ends = chars["RELEASE_TIME"].cummax().to_numpy()
        starts = [i for i in range(1, len(chars)) if chars.iloc[i - 1]["LETTER"] != "SHIFT"]
        for i in starts:
            assert chars.iloc[i]["PRESS_TIME"] - ends[i - 1] >= 60


def test_no_shift_overlap_feature_for_bots_and_shared_features_path():
    bot = A1Gaussian().generate("Hello World. It Is I.", np.random.default_rng(2))
    f = compute_features(bot)
    assert list(f) == FEATURE_NAMES and np.isfinite(list(f.values())).all()
    assert f["shift_overlap_count"] == 0 and f["corrections_count"] == 0 and f["lev_norm"] == 0


def test_determinism_and_parameter_moments():
    a = A1Gaussian().generate("abc def", np.random.default_rng(5))
    b = A1Gaussian().generate("abc def", np.random.default_rng(5))
    assert a.equals(b)
    gen = A1Gaussian(A1Params())
    holds = np.concatenate([(lambda d: (d["RELEASE_TIME"] - d["PRESS_TIME"]).to_numpy())(gen.generate("a" * 200, np.random.default_rng(s))) for s in range(5)])
    assert holds.mean() == pytest.approx(116.24, abs=2.0) and holds.std() == pytest.approx(23.88, abs=2.0)


def test_attacker_marker():
    assert isinstance(A1Gaussian(), Attacker) and A1Gaussian().is_attack
