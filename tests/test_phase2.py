"""Phase 2: attackers (schema, determinism, A2/A4/A5 properties), pool leakage, freeze guard, bootstrap metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from kl.attacks.a1_gaussian import A1Gaussian
from kl.attacks.a2_digraph import A2Digraph
from kl.attacks.a3_temporal import A3Temporal
from kl.attacks.a4_imitator import A4Imitator
from kl.attacks.a5_replay import A5aVerbatim, A5bTransplant
from kl.attacks.base import AALTO_COLUMNS
from kl.attacks.pool import SessionBank, cp, fit_pool_stats
from kl.features import FEATURE_NAMES, compute_features
from kl.phase2.evaluation import cell_metrics, user_weights
from kl.phase2.generate import PoolLeakageError, assert_pool_disjoint
from kl.phase2.guard import FrozenViolation, sha256_file, verify_files
from synth import SENTENCES, synth_pool

PARAMS = dict(min_samples=30, min_session_pairs=10, min_rate_transitions=3, pause_threshold_ms=500, max_pp_ms=5000, max_hold_ms=2000)
TARGET = "The Quick brown fox, said Tom."


@pytest.fixture(scope="module")
def pool():
    df = synth_pool(n_users=60, per_user=20, seed=0)
    bank = SessionBank.from_frame(df)
    stats = fit_pool_stats(bank, range(bank.n_sessions), PARAMS)
    return df, bank, stats


@pytest.fixture(scope="module")
def attackers(pool):
    _, bank, stats = pool
    idx = list(range(bank.n_sessions))
    return {"A1": A1Gaussian(), "A2": A2Digraph(stats), "A3": A3Temporal(stats), "A4": A4Imitator(stats),
            "A5a": A5aVerbatim(bank, idx), "A5b": A5bTransplant(bank, idx)}


# ---------------------------------------------------------------- schema + determinism (every attacker)
@pytest.mark.parametrize("name", ["A1", "A2", "A3", "A4", "A5a", "A5b"])
def test_schema_validity_and_determinism(attackers, name):
    a = attackers[name]
    f = a.generate(TARGET, np.random.default_rng(3))
    assert list(f.columns) == AALTO_COLUMNS and len(f) >= len(TARGET)
    assert f["KEYSTROKE_ID"].is_unique and (f["RELEASE_TIME"] >= f["PRESS_TIME"]).all()
    assert (f["PRESS_TIME"].diff().dropna() >= 0).all()                       # rows are in press order
    assert f["PRESS_TIME"].dtype == "float64" and f["KEYCODE"].dtype == "int64"
    assert (f["PRESS_TIME"] == np.round(f["PRESS_TIME"])).all()                # integer ms
    if name != "A5a":
        assert (f["SENTENCE"] == TARGET).all()
    feats = compute_features(f)                                                # all bots go through the unchanged features.py
    assert list(feats) == FEATURE_NAMES and np.isfinite(list(feats.values())).all()
    g = a.generate(TARGET, np.random.default_rng(3))
    pd.testing.assert_frame_equal(f, g)                                        # same seed -> identical
    other = a.generate(TARGET, np.random.default_rng(4))
    assert not (len(other) == len(f) and np.array_equal(other["PRESS_TIME"], f["PRESS_TIME"]))


@pytest.mark.parametrize("name", ["A1", "A2", "A3", "A5b"])
def test_typed_text_is_the_target_and_keycodes_follow_a1_convention(attackers, name):
    f = attackers[name].generate(TARGET, np.random.default_rng(0))
    assert (f["USER_INPUT"] == TARGET).all()
    typed = "".join(l for l in f["LETTER"] if l != "SHIFT")
    assert typed == TARGET
    assert (f.loc[f["LETTER"] == "SHIFT", "KEYCODE"] == 16).all() and (f["LETTER"] == "SHIFT").sum() == 3  # 'T', 'Q', 'T'
    assert f.loc[f["LETTER"] == "Q", "KEYCODE"].iloc[0] == 81 and f.loc[f["LETTER"] == " ", "KEYCODE"].iloc[0] == 32


# ---------------------------------------------------------------- A2
def test_a2_reproduces_digraph_means_within_5_percent(pool):
    _, _, stats = pool
    letters = {cp(c) for c in "abcdefghijklmnopqrstuvwxyz "}
    cands = [(k, len(v)) for k, v in stats.dg_pp.items() if (k >> 21) in letters and (k & ((1 << 21) - 1)) in letters and (k >> 21) != (k & ((1 << 21) - 1))]
    key = max(cands, key=lambda t: t[1])[0]
    x, y = chr(key >> 21), chr(key & ((1 << 21) - 1))
    emp = stats.dg_pp[key].mean()
    sentence = (x + y) * 10_000
    f = A2Digraph(stats).generate(sentence, np.random.default_rng(0))
    assert len(f) == len(sentence)
    pp = np.diff(f["PRESS_TIME"].to_numpy())
    xy = pp[0::2]                                   # transitions x -> y
    assert xy.mean() == pytest.approx(emp, rel=0.05)
    assert abs(np.mean(f["RELEASE_TIME"] - f["PRESS_TIME"]) - stats.hold_samples(x).mean() / 2 - stats.hold_samples(y).mean() / 2) < 0.05 * 100


def test_a2_backoff_to_second_char_then_global(pool):
    _, _, stats = pool
    assert stats.pp_samples("é", "è") is stats.glob_pp                       # unseen second char -> global
    seen = next(k & ((1 << 21) - 1) for k in stats.sec_pp)
    assert stats.pp_samples("é", chr(seen)) is stats.sec_pp[seen]                # unseen digraph, seen second char
    top = max(stats.dg_pp, key=lambda k: len(stats.dg_pp[k]))
    assert stats.pp_samples(chr(top >> 21), chr(top & ((1 << 21) - 1))) is stats.dg_pp[top]
    assert all(len(v) >= 30 for v in stats.dg_pp.values())


# ---------------------------------------------------------------- A3
def test_a3_fitted_parameters_are_sane(pool):
    _, _, stats = pool
    assert -1 < stats.phi < 1 and 0.1 < stats.sigma < 0.5 and len(stats.speed) > 100      # synthetic pool has sigma = 0.25, phi = 0


# ---------------------------------------------------------------- A4
def _pool_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = [compute_features(g) for _, g in df.groupby("session_id", sort=False)]
    return pd.DataFrame(rows)


def test_a4_corrections_and_lev_norm_match_the_pool(pool):
    df, bank, stats = pool
    human = _pool_features(df)
    a4 = A4Imitator(stats)
    sents = [bank.sentence(i) for i in range(bank.n_sessions)]
    bots = pd.DataFrame([compute_features(a4.generate(s, np.random.default_rng([1, i]))) for i, s in enumerate(sents)])
    for col, rel in (("corrections_count", 0.20), ("correction_rate", 0.20), ("lev_norm", 0.30)):
        assert bots[col].mean() == pytest.approx(human[col].mean(), rel=rel), col
    for col in ("corrections_count", "lev_norm"):                                          # distribution shape, not just the mean
        h, b = human[col].to_numpy(), bots[col].to_numpy()
        assert abs((h == 0).mean() - (b == 0).mean()) < 0.12, col
        assert abs(np.quantile(h, 0.9) - np.quantile(b, 0.9)) <= max(1.0, 0.5 * np.quantile(h, 0.9)), col


def test_a4_errors_are_adjacent_qwerty_substitutions_corrected_with_backspace(pool):
    _, _, stats = pool
    from kl.attacks.common import QWERTY_NEIGHBOURS
    a4 = A4Imitator(stats)
    rng = np.random.default_rng(11)
    seen_bs, seen_uncorrected = 0, 0
    for _ in range(300):
        keys, typed = a4._plan("the quick brown fox jumps over the lazy dog", 0.2, 0.05, rng)
        seen_bs += sum(b for _, b in keys)
        sent = "the quick brown fox jumps over the lazy dog"
        assert len(typed) == len(sent)
        for a_, b_ in zip(sent, typed):
            if a_ != b_:
                seen_uncorrected += 1
                assert b_ in QWERTY_NEIGHBOURS[a_]
        # replaying the key sequence with Backspace semantics yields exactly USER_INPUT
        buf: list[str] = []
        for ch, bs in keys:
            buf.pop() if bs else buf.append(ch)
        assert "".join(buf) == typed
    assert seen_bs > 0 and seen_uncorrected > 0


def test_a4_shift_overlap_and_pauses_are_produced_when_rates_say_so(pool):
    _, _, stats = pool
    import copy
    st = copy.deepcopy(stats)
    st.pso = np.array([1.0])                                   # every Shift hold outlasts the next key press
    st.rates = np.array([[0.0, 0.0, 1.0, 1.0]])                # every transition is a pause
    f = A4Imitator(st).generate("The Quick Brown Fox", np.random.default_rng(0))
    ft = compute_features(f)
    assert ft["shift_overlap_count"] >= 1 and ft["pause_count"] >= 15
    st.pso = np.array([0.0])
    ft0 = compute_features(A4Imitator(st).generate("The Quick Brown Fox", np.random.default_rng(0)))
    assert ft0["shift_overlap_count"] == 0


# ---------------------------------------------------------------- A5
def test_a5a_is_byte_identical_to_a_pool_session(pool, attackers):
    _, bank, _ = pool
    f = attackers["A5a"].generate("ignored sentence", np.random.default_rng(9))
    matches = [i for i in range(bank.n_sessions) if bank.n_rows(i) == len(f) and bank.sentence(i) == f["SENTENCE"].iloc[0]
               and np.array_equal(bank.press[bank.offsets[i]:bank.offsets[i + 1]], f["PRESS_TIME"].to_numpy())]
    assert matches
    i = matches[0]
    sl = slice(bank.offsets[i], bank.offsets[i + 1])
    assert f["PRESS_TIME"].to_numpy().tobytes() == bank.press[sl].tobytes()
    assert f["RELEASE_TIME"].to_numpy().tobytes() == bank.release[sl].tobytes()
    assert (f["SENTENCE"] == bank.sentence(i)).all() and (f["USER_INPUT"] == bank.user_input(i)).all()


def test_a5b_preserves_order_and_handles_both_length_cases(pool, attackers):
    _, bank, _ = pool
    a5b = attackers["A5b"]
    for n in (12, 400):                                           # shorter and (much) longer than typical pool sessions
        rng = np.random.default_rng(21)
        holds, flights, src = a5b.sequence(n, rng)
        h, f = bank.hold_flight(src)
        m = len(h)
        assert len(holds) == n and len(flights) == n - 1
        k = min(n, m)
        assert np.array_equal(holds[:k], h[:k]) and np.array_equal(flights[:k - 1], f[:k - 1])   # source order preserved / truncated
        if n > m:                                                  # continuation = contiguous chunks of pool sessions (hold, joining flight)
            tail_h = holds[m:]
            found = False
            for j in range(bank.n_sessions):
                oh, of = bank.hold_flight(j)
                if len(oh) > 1 and len(tail_h) <= len(oh) - 1 and any(np.array_equal(oh[s:s + len(tail_h)], tail_h) for s in range(1, len(oh) - len(tail_h) + 1)):
                    found = True
                    break
            assert found or n > 2 * m                              # a long tail spans several chunks; the short-tail case must be a single chunk
    sentence = "Hello World, again!"
    rng = np.random.default_rng(5)
    holds, flights, _ = a5b.sequence(len(sentence) + 3, rng)       # 'H', 'W' and '!' are shifted -> 3 extra Shift rows
    frame = a5b.generate(sentence, np.random.default_rng(5))
    assert len(frame) == len(holds)
    assert np.allclose(frame["RELEASE_TIME"] - frame["PRESS_TIME"], holds)
    assert np.allclose(frame["PRESS_TIME"].to_numpy()[1:] - frame["RELEASE_TIME"].to_numpy()[:-1], flights)
    assert (frame["USER_INPUT"] == sentence).all()


# ---------------------------------------------------------------- fitting uses only the given sessions
def test_stats_depend_only_on_the_fit_sessions(pool):
    df, bank, _ = pool
    ids = df["session_id"].unique()
    subset = set(ids[:150])
    idx = [i for i, s in enumerate(bank.session_ids) if s in subset]
    on_full_bank = fit_pool_stats(bank, idx, PARAMS)
    only_subset = fit_pool_stats(SessionBank.from_frame(df[df["session_id"].isin(subset)]), range(len(subset)), PARAMS)
    assert on_full_bank.n_fit_sessions == only_subset.n_fit_sessions == 150
    assert on_full_bank.phi == pytest.approx(only_subset.phi) and on_full_bank.sigma == pytest.approx(only_subset.sigma)
    assert np.array_equal(np.sort(on_full_bank.glob_pp), np.sort(only_subset.glob_pp))
    assert set(on_full_bank.dg_pp) == set(only_subset.dg_pp)
    assert np.array_equal(on_full_bank.delay_p, only_subset.delay_p)


# ---------------------------------------------------------------- leakage + freeze guards
def test_attacker_pool_users_never_in_generation_inputs():
    assert_pool_disjoint([1, 2, 3], [4, 5, 6])
    with pytest.raises(PoolLeakageError):
        assert_pool_disjoint([1, 2, 3], [3, 4])
    manifest = {"users": {"train": [1, 2], "val": [3], "test": [4, 5], "attacker_pool": [6, 7]}}
    for split in ("train", "val", "test"):
        assert_pool_disjoint(manifest["users"]["attacker_pool"], manifest["users"][split], split)


def test_real_manifest_pool_is_disjoint_when_available():
    p = Path("outputs/quick/results/split_manifest.json")
    if not p.exists():
        pytest.skip("no quick-mode outputs")
    users = json.loads(p.read_text())["users"]
    for split in ("train", "val", "test"):
        assert_pool_disjoint(users["attacker_pool"], users[split], split)


def test_frozen_files_guard(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("a = 1")
    good = {"x.py": sha256_file(f)}
    verify_files(good, tmp_path)
    f.write_text("a = 2")
    with pytest.raises(FrozenViolation):
        verify_files(good, tmp_path)


def test_phase1_source_files_still_match_the_freeze_record():
    rec = json.loads(Path("docs/phase1_frozen_hashes.json").read_text())
    verify_files(rec["source_files"], Path("."))


# ---------------------------------------------------------------- metrics
def test_cell_metrics_match_sklearn_and_bootstrap_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    users_h, users_b = np.repeat(np.arange(50), 10), np.repeat(np.arange(50), 10)
    h = np.round(rng.normal(1.0, 1.0, 500), 1)                      # rounded -> ties
    b = np.round(rng.normal(-0.5, 1.0, 500), 1)
    taus = {"tau5": float(np.quantile(h, 0.05)), "tau10": float(np.quantile(h, 0.10))}
    w = user_weights(50, 300, seed=1)
    m = cell_metrics(h, users_h, b, users_b, taus, w)
    assert m["auc"] == pytest.approx(roc_auc_score(np.r_[np.ones(500), np.zeros(500)], np.r_[h, b]))
    assert m["bpcer_fnr_tau5"] == pytest.approx(np.mean(h < taus["tau5"])) and m["apcer_fpr_tau10"] == pytest.approx(np.mean(b >= taus["tau10"]))
    for k in ("auc", "apcer_fpr_tau5", "bpcer_fnr_tau10"):
        assert m[f"{k}_ci_lo"] <= m[k] <= m[f"{k}_ci_hi"]
    assert m == cell_metrics(h, users_h, b, users_b, taus, w)       # deterministic given the weights
    assert (w.sum(axis=1) == 50).all()
