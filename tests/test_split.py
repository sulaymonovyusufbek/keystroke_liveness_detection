"""User-level split: disjointness, proportions, cap, determinism."""
import pandas as pd
import pytest

from kl.split import SPLITS, assert_disjoint, assign_users, build_manifest, cap_sessions

FR = {"train": 0.6, "val": 0.15, "test": 0.15, "attacker_pool": 0.1}


def _index(n_users=200, per_user=7) -> pd.DataFrame:
    rows = [(f"{u}_{k}", u) for u in range(1000, 1000 + n_users) for k in range(per_user)]
    return pd.DataFrame(rows, columns=["session_id", "PARTICIPANT_ID"])


def test_user_sets_pairwise_disjoint_and_complete():
    users = assign_users(range(1000), FR, seed=42)
    assert_disjoint(users)
    assert sum(len(v) for v in users.values()) == 1000
    assert [len(users[s]) for s in SPLITS] == [600, 150, 150, 100]


def test_assert_disjoint_raises_on_overlap():
    with pytest.raises(AssertionError):
        assert_disjoint({"train": [1, 2], "val": [2, 3], "test": [4], "attacker_pool": [5]})


def test_manifest_sessions_follow_users_and_no_leak():
    m = build_manifest(_index(), FR, None, seed=42)
    owner = {u: s for s, us in m["users"].items() for u in us}
    for split, sids in m["sessions"].items():
        assert all(owner[int(sid.split("_")[0])] == split for sid in sids)
    assert m["counts"]["sessions"]["train"] == m["counts"]["train_sessions_before_cap"]


def test_train_cap_is_seeded_uniform_subset():
    idx = _index()
    m = build_manifest(idx, FR, 100, seed=42)
    full = build_manifest(idx, FR, None, seed=42)
    assert len(m["sessions"]["train"]) == 100 and set(m["sessions"]["train"]) <= set(full["sessions"]["train"])
    assert m["sessions"]["val"] == full["sessions"]["val"]  # cap touches train only
    assert m["counts"]["train_sessions_before_cap"] == len(full["sessions"]["train"])
    assert cap_sessions([f"s{i}" for i in range(50)], 10, 1) == cap_sessions([f"s{i}" for i in range(50)], 10, 1)
    assert cap_sessions([f"s{i}" for i in range(50)], 10, 1) != cap_sessions([f"s{i}" for i in range(50)], 10, 2)


def test_manifest_deterministic_and_seed_sensitive():
    a, b = build_manifest(_index(), FR, 300, 42), build_manifest(_index(), FR, 300, 42)
    assert a == b
    assert build_manifest(_index(), FR, 300, 43)["users"] != a["users"]


def test_bad_fractions_rejected():
    with pytest.raises(ValueError):
        assign_users(range(10), {"train": 0.5, "val": 0.5, "test": 0.5, "attacker_pool": 0.5}, 0)
    with pytest.raises(ValueError):
        assign_users(range(10), {"train": 1.0}, 0)
