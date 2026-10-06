"""Feature definitions: worked example, Shift overlap, KEYCODE-based corrections, sorting."""
import numpy as np
import pandas as pd
import pytest

from kl.features import FEATURE_NAMES, FEATURE_GROUPS, compute_features, extract_features_frame, feature_columns, worked_example
from helpers import make_session


def test_timing_worked_example():
    s = make_session([0, 200, 450], [100, 280, 560], [65, 66, 67], sentence="abc", user_input="abc")
    f = compute_features(s)
    hold, pp, rr, flight = [100, 80, 110], [200, 250], [180, 280], [100, 170]
    for name, arr in (("hold", hold), ("pp", pp), ("rr", rr), ("flight", flight)):
        assert f[f"{name}_mean"] == pytest.approx(np.mean(arr))
        assert f[f"{name}_std"] == pytest.approx(np.std(arr))  # population std
        if name != "flight":
            assert f[f"{name}_cv"] == pytest.approx(np.std(arr) / abs(np.mean(arr)))
    assert "flight_cv" not in f  # removed in Revision 1
    assert f["flight_median"] == pytest.approx(135) and f["flight_iqr"] == pytest.approx(35)  # percentiles 117.5 and 152.5
    assert f["hold_mean"] == pytest.approx(96.6667, abs=1e-3) and f["pp_mean"] == 225 and f["rr_std"] == 50 and f["flight_std"] == 35
    assert f["n_keys"] == 3 and f["total_duration_ms"] == 450
    assert f["wpm"] == pytest.approx((3 / 5) / (450 / 60000))
    assert f["lev_norm"] == 0.0 and f["corrections_count"] == 0 and f["shift_overlap_count"] == 0
    assert list(f) == FEATURE_NAMES


def test_cv_uses_epsilon_for_zero_mean():
    # holds = [-50, 50, -50, 50]: mean 0, std 50 -> cv = std / 1e-6 (only defined this way for near-zero means)
    s = make_session([0, 100, 200, 300], [-50, 150, 150, 350], [65, 66, 67, 68], sentence="abcd", user_input="abcd")
    f = compute_features(s)
    assert f["hold_mean"] == pytest.approx(0) and f["hold_cv"] == pytest.approx(f["hold_std"] / 1e-6)


def test_flight_median_and_iqr_are_robust_to_one_long_pause():
    # one 5270 ms pause among small flights: median / IQR barely move, std explodes
    press = [0, 110, 240, 390, 570, 6000]
    release = [100, 200, 330, 500, 730, 6100]
    flights = np.array(press[1:]) - np.array(release[:-1])
    f = compute_features(make_session(press, release, [65] * 6, sentence="abcdef", user_input="abcdef"))
    assert f["flight_median"] == pytest.approx(np.median(flights)) and f["flight_iqr"] == pytest.approx(np.subtract(*np.percentile(flights, [75, 25])))
    assert f["flight_iqr"] < f["flight_std"] / 10


def test_pause_and_rollover():
    # flights: 600 (pause), -20 (rollover), 100
    s = make_session([0, 700, 780, 900], [100, 800, 800, 1000], [65, 66, 67, 68], sentence="abcd", user_input="abcd")
    f = compute_features(s)
    assert f["pause_count"] == 1 and f["pause_rate"] == pytest.approx(1 / 3) and f["rollover_rate"] == pytest.approx(1 / 3)


def test_shift_overlap_counts_keys_beyond_first():
    # Shift held 0-300; keys pressed at 50, 150, 250 -> 3 keys inside, first free -> 2
    s = make_session([0, 50, 150, 250], [300, 90, 190, 290], [16, 65, 66, 67], letters=["SHIFT", "A", "B", "C"])
    assert compute_features(s)["shift_overlap_count"] == 2


def test_shift_with_single_key_has_no_overlap():
    s = make_session([0, 50, 400], [200, 120, 500], [16, 65, 66])
    assert compute_features(s)["shift_overlap_count"] == 0


def test_shift_detected_by_keycode_when_letter_garbled():
    s = make_session([0, 50, 150, 250], [300, 90, 190, 290], [16, 65, 66, 67], letters=["?", "A", "B", "C"])
    assert compute_features(s)["shift_overlap_count"] == 2


def test_garbled_letter_with_backspace_keycode_is_a_correction():
    s = make_session([0, 100, 200, 300], [50, 150, 250, 350], [65, 8, 46, 66], letters=["a", "=", "\\", "b"])
    f = compute_features(s)
    assert f["corrections_count"] == 2 and f["correction_rate"] == pytest.approx(2 / 4)


def test_letter_bksp_with_wrong_keycode_is_not_a_correction():
    s = make_session([0, 100, 200], [50, 150, 250], [65, 66, 67], letters=["a", "BKSP", "c"])
    assert compute_features(s)["corrections_count"] == 0


def test_lev_norm_and_wpm():
    s = make_session([0, 1000, 2000], [100, 1100, 2100], [65, 66, 67], sentence="kitten", user_input="sitting")
    f = compute_features(s)
    assert f["lev_norm"] == pytest.approx(3 / 7)
    assert f["wpm"] == pytest.approx((7 / 5) / (2000 / 60000))


def test_sorted_by_press_time_with_keystroke_id_tiebreak_only():
    a = make_session([200, 0, 450], [280, 100, 560], [66, 65, 67], kid=[9, 8, 7])
    b = make_session([0, 200, 450], [100, 280, 560], [65, 66, 67], kid=[1, 2, 3])
    assert compute_features(a) == compute_features(b)  # KEYSTROKE_ID order must not matter
    # equal presses: KEYSTROKE_ID breaks the tie
    t = make_session([0, 0, 100], [50, 80, 150], [65, 16, 66], kid=[2, 1, 3])
    assert np.isfinite(list(compute_features(t).values())).all()


def test_extract_frame_matches_single_session_path():
    ex = worked_example()
    other = make_session([0, 130, 300, 420], [90, 200, 380, 500], [65, 8, 66, 16], sentence="abab", user_input="abaa")
    other["PARTICIPANT_ID"] = 2
    frame = pd.concat([ex.assign(session_id="1_1", LAYOUT="qwerty"), other.assign(session_id="2_1", LAYOUT="azerty")], ignore_index=True)
    out = extract_features_frame(frame)
    assert list(out["session_id"]) == ["1_1", "2_1"] and list(out["LAYOUT"]) == ["qwerty", "azerty"] and list(out["user_id"]) == [1, 2]
    for sid, sess in (("1_1", ex), ("2_1", other)):
        row = out[out["session_id"] == sid].iloc[0]
        for k, v in compute_features(sess).items():
            assert row[k] == pytest.approx(v)


def test_feature_groups_and_columns():
    assert set(FEATURE_GROUPS) == {"TIMING", "RHYTHM", "BEHAVIOUR", "GLOBAL"}
    assert len(FEATURE_GROUPS["TIMING"]) == 13 and len(FEATURE_NAMES) == 23
    assert "flight_cv" not in FEATURE_NAMES and {"flight_median", "flight_iqr"} <= set(FEATURE_GROUPS["TIMING"])
    assert feature_columns(["GLOBAL"]) == ["n_keys", "total_duration_ms", "wpm"]
    with pytest.raises(ValueError):
        feature_columns(["NOPE"])
