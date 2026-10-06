"""Deterministic regeneration of the Phase 2 bot sessions WITH their keystrokes, and the check against Phase 2.

Phase 2 (``kl.phase2.generate``) stored only features. Bot ``i`` of split ``s`` for attacker ``a`` is regenerated with the
FROZEN attacker object and the SAME generator ``np.random.default_rng([seed, ATTACKER_CODES[a], SPLIT_CODES[s], i])``
for the SAME sentence, then featurised by the unchanged features.py; :func:`verify_against_phase2` asserts the result
equals the stored Phase 2 features.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..attacks.base import Attacker
from ..features import FEATURE_NAMES, FeatureParams, compute_features
from ..phase2.generate import ATTACKER_CODES, SPLIT_CODES
from .sequences import SeqSet, session_arrays


class BotMismatchError(AssertionError):
    """Regenerated bots do not reproduce the stored Phase 2 features."""


def _chunk(attacker: Attacker, items: list[tuple[int, str, int, str]], seed: int, a_code: int, s_code: int,
           params: FeatureParams) -> tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    rows, seqs = [], []
    for gi, sid, uid, sentence in items:
        rng = np.random.default_rng([seed, a_code, s_code, gi])          # identical to kl.phase2.generate._chunk
        bot = attacker.generate(sentence, rng)
        bot["PARTICIPANT_ID"], bot["TEST_SECTION_ID"] = -1, gi
        rows.append({"session_id": sid, "user_id": int(uid), **compute_features(bot, params)})
        seqs.append(session_arrays(bot["PRESS_TIME"], bot["RELEASE_TIME"], bot["KEYCODE"], bot["KEYSTROKE_ID"]))
    return pd.DataFrame(rows, columns=["session_id", "user_id"] + FEATURE_NAMES), seqs


def regenerate_bots(
    attacker: Attacker, matched: pd.DataFrame, split: str, seed: int, params: FeatureParams, n_jobs: int = 1, chunk_size: int = 2000,
) -> tuple[pd.DataFrame, SeqSet]:
    """Features and keystroke sequences of one bot per row of ``matched`` (``session_id, user_id, sentence``), in row order.

    Sessions with non-finite features are dropped from both outputs (as in Phase 2; count in ``feats.attrs['n_dropped']``).
    """
    items = [(i, r.session_id, r.user_id, r.sentence) for i, r in enumerate(matched.itertuples(index=False))]
    chunks = [items[k:k + chunk_size] for k in range(0, len(items), chunk_size)]
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_chunk)(attacker, c, seed, ATTACKER_CODES[attacker.name], SPLIT_CODES[split], params) for c in chunks
    )
    feats = pd.concat([p[0] for p in parts], ignore_index=True) if parts else pd.DataFrame(columns=["session_id", "user_id"] + FEATURE_NAMES)
    seqs = [s for p in parts for s in p[1]]
    finite = np.isfinite(feats[FEATURE_NAMES].to_numpy(dtype="float64")).all(axis=1)
    keep = np.flatnonzero(finite)
    kept = feats.loc[finite].reset_index(drop=True)
    kept.attrs["n_dropped"] = int((~finite).sum())
    lens = np.array([len(seqs[i][0]) for i in keep], dtype="int64")
    seqset = SeqSet(
        f"bot_{attacker.name}_{split}", split, "bot", attacker.name, kept["session_id"].to_numpy(dtype="U"),
        kept["user_id"].to_numpy(dtype="int64"), np.r_[0, np.cumsum(lens)].astype("int64"),
        np.concatenate([seqs[i][0] for i in keep]) if len(keep) else np.zeros(0, "int32"),
        np.concatenate([seqs[i][1] for i in keep]) if len(keep) else np.zeros(0),
        np.concatenate([seqs[i][2] for i in keep]) if len(keep) else np.zeros(0),
    )
    return kept, seqset


def verify_against_phase2(regen: pd.DataFrame, stored: pd.DataFrame, tol: float = 1e-9) -> dict[str, Any]:
    """Compare regenerated bot features with the stored Phase 2 file: rows, ids, order and every feature within ``tol``.

    Returns a report dict with ``ok``; never raises (the caller decides to stop).
    """
    rep: dict[str, Any] = {"n_regenerated": int(len(regen)), "n_stored": int(len(stored)), "tolerance": tol}
    if len(regen) != len(stored):
        return {**rep, "ok": False, "reason": "row count differs"}
    same_ids = bool((regen["session_id"].astype(str).to_numpy() == stored["session_id"].astype(str).to_numpy()).all()
                    and (regen["user_id"].to_numpy(dtype="int64") == stored["user_id"].to_numpy(dtype="int64")).all())
    a = regen[FEATURE_NAMES].to_numpy(dtype="float64")
    b = stored[FEATURE_NAMES].to_numpy(dtype="float64")
    diff = np.abs(a - b)
    per_feature = {f: float(np.max(diff[:, j])) if len(diff) else 0.0 for j, f in enumerate(FEATURE_NAMES)}
    nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
    max_diff = float(np.nanmax(diff)) if len(diff) else 0.0
    ok = same_ids and nan_mismatch == 0 and not np.isnan(diff).any() and max_diff <= tol
    n_bad = int((~(diff <= tol)).any(axis=1).sum()) if len(diff) else 0
    return {**rep, "ok": bool(ok), "same_ids_and_order": same_ids, "max_abs_diff": max_diff, "n_sessions_over_tol": n_bad,
            "nan_mismatches": nan_mismatch, "max_abs_diff_per_feature": per_feature}
