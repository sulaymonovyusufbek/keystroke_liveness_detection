"""Bot-session generation and featurisation (one bot per matched human session), fully seeded and parallel-safe."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..attacks.a1_gaussian import A1Gaussian, A1Params
from ..attacks.a2_digraph import A2Digraph
from ..attacks.a3_temporal import A3Temporal
from ..attacks.a4_imitator import A4Imitator
from ..attacks.a5_replay import A5aVerbatim, A5bTransplant
from ..attacks.base import Attacker
from ..attacks.pool import PoolStats, SessionBank
from ..features import FEATURE_NAMES, FeatureParams, compute_features

ATTACKER_CODES: dict[str, int] = {"A1": 1, "A2": 2, "A3": 3, "A4": 4, "A5a": 5, "A5b": 6}
SPLIT_CODES: dict[str, int] = {"train": 1, "val": 2, "test": 3, "pool_heldout": 4}


class PoolLeakageError(AssertionError):
    """Raised when attacker_pool users appear among the users a bot is generated for."""


def assert_pool_disjoint(pool_users: Sequence[int], target_users: Sequence[int], where: str = "bot generation") -> None:
    """attacker_pool users may only be the STATISTICS source; they must not be among the matched human users."""
    overlap = set(int(u) for u in pool_users) & set(int(u) for u in target_users)
    if overlap:
        raise PoolLeakageError(f"{len(overlap)} attacker_pool user(s) appear in the {where} input, e.g. {sorted(overlap)[:5]}")


def build_attackers(
    stats: PoolStats, bank: SessionBank, fit_idx: Sequence[int], a1_cfg: Mapping[str, Any], names: Sequence[str],
) -> dict[str, Attacker]:
    """Attacker objects by name; every pool-dependent parameter comes from ``stats`` / the FIT sessions ``fit_idx``."""
    all_ = {
        "A1": A1Gaussian(A1Params.from_config(a1_cfg)),
        "A2": A2Digraph(stats), "A3": A3Temporal(stats), "A4": A4Imitator(stats),
        "A5a": A5aVerbatim(bank, fit_idx), "A5b": A5bTransplant(bank, fit_idx),
    }
    return {n: all_[n] for n in names}


def _chunk(
    attacker: Attacker, items: list[tuple[int, str, int, str]], seed: int, a_code: int, s_code: int, params: FeatureParams,
) -> pd.DataFrame:
    rows = []
    for gi, sid, uid, sentence in items:
        rng = np.random.default_rng([seed, a_code, s_code, gi])
        bot = attacker.generate(sentence, rng)
        bot["PARTICIPANT_ID"], bot["TEST_SECTION_ID"] = -1, gi
        rows.append({"session_id": sid, "user_id": int(uid), **compute_features(bot, params)})
    return pd.DataFrame(rows, columns=["session_id", "user_id"] + FEATURE_NAMES)


def generate_bot_features(
    attacker: Attacker, matched: pd.DataFrame, split: str, seed: int, params: FeatureParams,
    n_jobs: int = 1, chunk_size: int = 2000,
) -> pd.DataFrame:
    """One bot per row of ``matched`` (columns ``session_id, user_id, sentence``), featurised by features.py.

    The RNG of bot ``i`` is ``default_rng([seed, attacker_code, split_code, i])`` so the result does not depend on the
    number of workers. Rows keep the matched human's ``session_id`` / ``user_id``; sessions with non-finite features
    are dropped (counted by the caller via ``attrs['n_dropped']``).
    """
    items = [(i, r.session_id, r.user_id, r.sentence) for i, r in enumerate(matched.itertuples(index=False))]
    chunks = [items[k:k + chunk_size] for k in range(0, len(items), chunk_size)]
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_chunk)(attacker, c, seed, ATTACKER_CODES[attacker.name], SPLIT_CODES[split], params) for c in chunks
    )
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["session_id", "user_id"] + FEATURE_NAMES)
    finite = np.isfinite(out[FEATURE_NAMES].to_numpy(dtype="float64")).all(axis=1)
    kept = out.loc[finite].reset_index(drop=True)
    kept.attrs["n_dropped"] = int((~finite).sum())
    return kept
