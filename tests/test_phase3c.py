"""Phase 3c (post-hoc artefact check): swap construction, letters-only masking, keycode realism helpers."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from kl.phase3b import nets
from kl.phase3b.sequences import Encoder, SeqSet
from kl.phase3c.realism import enriched, jsd, shift_stats
from kl.phase3c.swap import d4_letters_scores, identical_keys, letter_ids, letters_step_mask, pair_index, swap_sets


def _set(name, kind, ids, seqs, attacker=None):
    """SeqSet from ``[(keycodes, holds, pps_without_last)]``."""
    lens = [len(k) for k, _, _ in seqs]
    return SeqSet(name, "test", kind, attacker, np.array(ids, dtype="U"), np.arange(len(ids), dtype="int64"),
                  np.r_[0, np.cumsum(lens)].astype("int64"),
                  np.concatenate([np.asarray(k, dtype="int32") for k, _, _ in seqs]),
                  np.concatenate([np.asarray(h, dtype="float64") for _, h, _ in seqs]),
                  np.concatenate([np.r_[np.asarray(p, dtype="float64"), np.nan] for _, _, p in seqs]))


@pytest.fixture
def pair():
    humans = _set("h", "human", ["s1", "s2", "s3"], [
        ([72, 73, 32, 8, 65], [101, 102, 103, 104, 105], [201, 202, 203, 204]),        # 5 keys
        ([66, 67], [111, 112], [211]),                                                  # 2 keys
        ([68, 69, 70], [121, 122, 123], [221, 222]),                                    # 3 keys (no bot)
    ])
    bots = _set("b", "bot", ["s2", "s1"], [                                              # other order, s3 missing
        ([66, 67, 190], [11, 12, 13], [21, 22]),                                        # 3 keys (human has 2)
        ([72, 73, 32], [1, 2, 3], [4, 5]),                                              # 3 keys (human has 5)
    ], attacker="AX")
    return humans, bots


def test_pairs_follow_session_id(pair):
    humans, bots = pair
    hi, bi = pair_index(humans, bots)
    assert list(humans.session_ids[hi]) == list(bots.session_ids[bi]) == ["s2", "s1"]


def test_swap_takes_keys_and_timing_from_the_right_side_and_truncates(pair):
    humans, bots = pair
    sets, summ = swap_sets(humans, bots)
    s3, s4 = sets["S3"], sets["S4"]
    assert list(s3.lengths) == list(s4.lengths) == [2, 3]
    # S3 = human keys + bot timing
    assert list(s3.keycode) == [66, 67, 72, 73, 32]
    assert list(s3.hold) == [11, 12, 1, 2, 3]
    np.testing.assert_array_equal(s3.pp, [21, np.nan, 4, 5, np.nan])
    # S4 = bot keys + human timing
    assert list(s4.keycode) == [66, 67, 72, 73, 32]
    assert list(s4.hold) == [111, 112, 101, 102, 103]
    np.testing.assert_array_equal(s4.pp, [211, np.nan, 201, 202, np.nan])     # human pp 203 dropped: last kept key masked
    # controls: truncated originals
    assert list(sets["S1t"].keycode) == [66, 67, 72, 73, 32] and list(sets["S2t"].hold) == [11, 12, 1, 2, 3]
    assert list(s3.session_ids) == ["s2", "s1"]
    assert summ["n_pairs"] == 2 and summ["n_unpaired_bots"] == 0 and summ["n_mismatch"] == 2
    assert summ["n_bot_longer"] == 1 and summ["n_bot_shorter"] == 1 and summ["abs_diff_max"] == 2
    # sources untouched
    assert np.isnan(bots.pp[2]) and humans.pp[0] == 201


def test_swap_of_a_set_with_itself_reproduces_it(pair):
    humans, _ = pair
    sets, summ = swap_sets(humans, humans)
    for k in ("S3", "S4", "S1t", "S2t"):
        np.testing.assert_array_equal(sets[k].keycode, humans.keycode)
        np.testing.assert_array_equal(sets[k].hold, humans.hold)
        np.testing.assert_array_equal(sets[k].pp, humans.pp)
    assert summ["n_mismatch"] == 0


def test_letters_step_mask():
    allowed = torch.tensor([5, 6])
    keys = torch.tensor([[5, 6, 9, 5, 5, 6, 0], [6, 5, 0, 0, 0, 0, 0]])
    lens = torch.tensor([6, 2])
    m = letters_step_mask(keys, lens, allowed)
    # row 0: (5,6) ok, (6,9) no, (9,5) no, (5,5) ok, (5,6) ok, last step (6) has no next key -> no
    assert m[0].tolist() == [True, False, False, True, True, False, False]
    assert m[1].tolist() == [True, False, False, False, False, False, False]


def _tiny_encoder():
    return Encoder(np.array([8, 16, 32, 65, 66, 67, 190], dtype="int64"), 4.6, 0.4, 5.2, 0.7, 12)


def test_letter_ids_exclude_unk_and_non_letters():
    enc = _tiny_encoder()
    ids = letter_ids(enc.key_ids)
    assert set(ids) == set(enc.key_ids(np.array([32, 65, 66, 67])))


def test_letters_only_score_is_the_mean_nll_over_letter_pairs():
    torch.manual_seed(0)
    enc = _tiny_encoder()
    s = _set("h", "human", ["a", "b"], [
        ([65, 66, 16, 67, 32, 65], [90, 100, 150, 80, 95, 85], [120, 60, 30, 140, 110]),
        ([8, 190, 16], [70, 80, 90], [100, 110]),                                      # no letter pair at all
    ])
    data = enc.encode(s)
    model = nets.D4Model(enc.vocab_size, 8, 16, 1, 0.0, 2).eval()
    score, n = d4_letters_scores(model, data, letter_ids(enc.key_ids), 8, torch.device("cpu"))
    with torch.no_grad():
        nll, _ = model.step_nll(nets.to_device(data.batch(np.arange(2)), torch.device("cpu")))
    # session a: steps (A,B)=0 ok, (B,Shift)=1 no, (Shift,C)=2 no, (C,Space)=3 ok, (Space,A)=4 ok, 5 = last
    assert list(n) == [3, 0]
    assert score[0] == pytest.approx(float(nll[0, [0, 3, 4]].double().mean()), abs=1e-6)
    assert np.isnan(score[1])
    # extra padding / a different batch composition does not change the score
    score1, _ = d4_letters_scores(model, data, letter_ids(enc.key_ids), 1, torch.device("cpu"))
    assert score1[0] == pytest.approx(score[0], abs=1e-6)


def test_jsd_bounds():
    assert jsd({1: 5, 2: 5}, {1: 1, 2: 1}) == pytest.approx(0.0)
    assert jsd({1: 3}, {2: 7}) == pytest.approx(1.0)


def test_enriched_lists_tokens_in_both_directions_with_prevalence_filter():
    h_tok = np.array([65] * 100 + [8] * 50)
    h_sess = np.arange(150) % 50
    b_tok = np.array([65] * 100 + [229] * 20 + [8] * 1)
    b_sess = np.arange(121) % 50
    df = enriched(h_tok, h_sess, 50, b_tok, b_sess, 50, str)
    got = {(int(t), m) for t, m in zip(df["token"], df["more_in"])}
    assert got == {(229, "bots"), (8, "humans")}
    assert not enriched(h_tok, h_sess, 50, b_tok, b_sess, 50, str, min_prev=0.9).query("token == 229").shape[0]


def test_shift_stats_counts_capitals_and_release_classes():
    # "Ab": Shift down 0, A down 50 (hold 60 -> up 110), Shift up at 100 (during A), B at 200
    s = _set("x", "bot", ["x"], [([16, 65, 66], [100, 60, 50], [50, 150])])
    st = shift_stats(s, ["Ab"])
    assert st["n_capitals"] == 1 and st["shift_per_capital"] == 1.0
    assert st["next_is_letter"] == 1.0 and st["released_during_next_key"] == 1.0
    assert st["lead_ms_median"] == 50 and st["hold_ms_median"] == 100


def test_identical_key_sequences_and_aligned_share(pair):
    humans, bots = pair
    hi, bi = pair_index(humans, bots)                   # pairs (s2: [66,67] vs [66,67,190]), (s1: [72,73,32,8,65] vs [72,73,32])
    assert list(identical_keys(humans, hi, bots, bi)) == [False, False]
    assert list(identical_keys(humans, np.array([0, 1]), humans, np.array([0, 1]))) == [True, True]
    _, summ = swap_sets(humans, bots)
    assert summ["frac_aligned_positions_same_key"] == 1.0 and summ["n_identical_key_sequences"] == 0
