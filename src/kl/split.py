"""User-level train / val / test / attacker_pool split and its manifest."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

SPLITS: tuple[str, ...] = ("train", "val", "test", "attacker_pool")


def assign_users(user_ids: Sequence[int], fractions: Mapping[str, float], seed: int) -> dict[str, list[int]]:
    """Randomly partition users (seeded permutation of the sorted ids) according to ``fractions``.

    The first three splits get ``round(cumulative fraction * n)`` users; attacker_pool takes the rest.
    """
    if set(fractions) != set(SPLITS):
        raise ValueError(f"fractions must have exactly the keys {SPLITS}, got {sorted(fractions)}")
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError(f"split fractions must sum to 1, got {sum(fractions.values())}")
    users = np.array(sorted(set(int(u) for u in user_ids)), dtype="int64")
    perm = np.random.default_rng(seed).permutation(users)
    cum = np.cumsum([fractions[s] for s in SPLITS[:-1]])
    cuts = [int(round(c * len(users))) for c in cum]
    parts = np.split(perm, cuts)
    return {s: sorted(int(u) for u in part) for s, part in zip(SPLITS, parts)}


def cap_sessions(session_ids: Sequence[str], max_n: int | None, seed: int) -> list[str]:
    """Seeded uniform sample (without replacement) of at most ``max_n`` sessions; ``None`` = no cap."""
    ordered = sorted(session_ids)
    if max_n is None or len(ordered) <= max_n:
        return ordered
    rng = np.random.default_rng([seed, 1])
    idx = rng.choice(len(ordered), size=int(max_n), replace=False)
    return [ordered[i] for i in sorted(idx)]


def assert_disjoint(users: Mapping[str, Sequence[int]]) -> None:
    """Raise AssertionError unless the four user sets are pairwise disjoint."""
    names = list(users)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(users[a]) & set(users[b])
            assert not overlap, f"user sets '{a}' and '{b}' overlap on {len(overlap)} users"


def build_manifest(
    index: pd.DataFrame,
    fractions: Mapping[str, float],
    max_train_sessions: int | None,
    seed: int,
) -> dict[str, Any]:
    """Split the users that own at least one clean session and list session ids per split.

    Args:
        index: session index with columns ``session_id`` and ``PARTICIPANT_ID``.
        fractions: user fractions per split.
        max_train_sessions: cap on the number of train sessions (seeded uniform sampling).
        seed: global seed.
    """
    users = assign_users(index["PARTICIPANT_ID"].unique().tolist(), fractions, seed)
    assert_disjoint(users)
    owner = {u: s for s, us in users.items() for u in us}
    split_of = index["PARTICIPANT_ID"].map(owner)
    sessions = {s: sorted(index.loc[split_of == s, "session_id"].tolist()) for s in SPLITS}
    n_train_before = len(sessions["train"])
    sessions["train"] = cap_sessions(sessions["train"], max_train_sessions, seed)
    return {
        "seed": seed,
        "fractions": dict(fractions),
        "max_train_sessions": max_train_sessions,
        "counts": {
            "users": {s: len(users[s]) for s in SPLITS},
            "sessions": {s: len(sessions[s]) for s in SPLITS},
            "train_sessions_before_cap": n_train_before,
        },
        "users": users,
        "sessions": sessions,
    }
