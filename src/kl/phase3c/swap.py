"""Swap test and letters-only masking (docs/PHASE3C_SPEC.md, items 2 and 3).

Swap: each test bot is paired with the test human whose session it was generated for (same ``session_id``, hence the
same target sentence). The two raw sequences are aligned by position and truncated to the shorter length ``n``:

    S3 = human keycodes[:n] + bot hold[:n] / bot pp[:n]        S4 = bot keycodes[:n] + human hold[:n] / human pp[:n]
    S1t / S2t = human / bot truncated to n (length controls)

``pp`` of the last kept key is undefined (NaN -> masked by the encoder), exactly as for any session end.

Letters-only D4 score: mean per-step NLL over the steps ``i`` where key ``i`` AND key ``i + 1`` are letters a-z or
space (so the last step, which has no next key, never counts); every other step is masked.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch

from ..phase3b.nets import D4Model, _chunks, to_device
from ..phase3b.sequences import EncodedSet, SeqSet, _ranges

LETTER_KEYCODES: tuple[int, ...] = (32, *range(65, 91))     # space + A-Z (letters carry the uppercase code)


def pair_index(humans: SeqSet, bots: SeqSet) -> tuple[np.ndarray, np.ndarray]:
    """``(hi, bi)``: row indices of the matched human / bot sessions (same ``session_id``), in bot order."""
    pos = pd.Index(humans.session_ids.astype(str))
    if not pos.is_unique or not pd.Index(bots.session_ids.astype(str)).is_unique:
        raise ValueError("session ids must be unique within each set")
    hi = pos.get_indexer(bots.session_ids.astype(str))
    bi = np.flatnonzero(hi >= 0)
    return hi[bi].astype("int64"), bi.astype("int64")


def _assemble(name: str, src_keys: SeqSet, ki: np.ndarray, src_time: SeqSet, ti: np.ndarray, n: np.ndarray,
              ids: np.ndarray, users: np.ndarray, kind: str, attacker: str | None) -> SeqSet:
    rows_k = _ranges(src_keys.offsets[:-1][ki], n)
    rows_t = _ranges(src_time.offsets[:-1][ti], n)
    offsets = np.r_[0, np.cumsum(n)].astype("int64")
    pp = src_time.pp[rows_t].copy()
    pp[offsets[1:] - 1] = np.nan                                   # last kept key: p undefined (masked)
    return SeqSet(name, src_keys.split, kind, attacker, ids, users, offsets, src_keys.keycode[rows_k].copy(),
                  src_time.hold[rows_t].copy(), pp)


def swap_sets(humans: SeqSet, bots: SeqSet) -> tuple[dict[str, SeqSet], dict[str, float]]:
    """``{"S3", "S4", "S1t", "S2t"}`` for every matched pair, plus the length-mismatch summary."""
    hi, bi = pair_index(humans, bots)
    lh, lb = humans.lengths[hi], bots.lengths[bi]
    n = np.minimum(lh, lb)
    if (n < 1).any():
        raise ValueError("empty session in a pair")
    ids, users, att = humans.session_ids[hi], humans.user_ids[hi], bots.attacker
    sets = {
        "S3": _assemble(f"S3_{att}", humans, hi, bots, bi, n, ids, users, "swap", att),
        "S4": _assemble(f"S4_{att}", bots, bi, humans, hi, n, ids, users, "swap", att),
        "S1t": _assemble(f"S1t_{att}", humans, hi, humans, hi, n, ids, users, "human", att),
        "S2t": _assemble(f"S2t_{att}", bots, bi, bots, bi, n, ids, users, "bot", att),
    }
    d = lb - lh
    mism = d != 0
    same = key_agreement(humans, hi, bots, bi, n)
    summary = {"n_bots": int(bots.n_sessions), "n_pairs": int(len(bi)), "n_unpaired_bots": int(bots.n_sessions - len(bi)),
               "n_mismatch": int(mism.sum()), "frac_mismatch": float(mism.mean()) if len(d) else float("nan"),
               "n_bot_longer": int((d > 0).sum()), "n_bot_shorter": int((d < 0).sum()),
               "abs_diff_mean_mismatched": float(np.abs(d[mism]).mean()) if mism.any() else 0.0,
               "abs_diff_median_mismatched": float(np.median(np.abs(d[mism]))) if mism.any() else 0.0,
               "abs_diff_p95_mismatched": float(np.percentile(np.abs(d[mism]), 95)) if mism.any() else 0.0,
               "abs_diff_max": int(np.abs(d).max()) if len(d) else 0,
               "frac_keys_dropped_human": float(1 - n.sum() / lh.sum()), "frac_keys_dropped_bot": float(1 - n.sum() / lb.sum()),
               "frac_aligned_positions_same_key": float(same.sum() / n.sum()),
               "n_identical_key_sequences": int(identical_keys(humans, hi, bots, bi).sum())}
    return sets, summary


def key_agreement(humans: SeqSet, hi: np.ndarray, bots: SeqSet, bi: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Per aligned position (flat, pairs in order): does the human's keycode equal the bot's?"""
    return humans.keycode[_ranges(humans.offsets[:-1][hi], n)] == bots.keycode[_ranges(bots.offsets[:-1][bi], n)]


def identical_keys(humans: SeqSet, hi: np.ndarray, bots: SeqSet, bi: np.ndarray) -> np.ndarray:
    """Per pair: the human typed exactly the bot's keycode sequence (same length, same key at every position)."""
    lh, lb = humans.lengths[hi], bots.lengths[bi]
    out = lh == lb
    n = np.where(out, lh, 0)
    agree = key_agreement(humans, hi, bots, bi, n)
    sess = np.repeat(np.arange(len(hi)), n)
    bad = np.zeros(len(hi), dtype=bool)
    np.logical_or.at(bad, sess, ~agree)
    return out & ~bad


def letter_ids(enc_key_ids, codes: Iterable[int] = LETTER_KEYCODES) -> np.ndarray:
    """Vocabulary ids of ``codes`` under an encoder's ``key_ids`` (UNK never counts as a letter)."""
    ids = np.asarray(enc_key_ids(np.asarray(list(codes), dtype="int64")))
    return np.unique(ids[ids > 1])


def letters_step_mask(keys: torch.Tensor, lens: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """``(B, L)`` bool: step ``i`` is valid, ``i < len - 1``, key ``i`` and key ``i + 1`` are both in ``allowed``."""
    B, L = keys.shape
    ar = torch.arange(L, device=keys.device)[None, :]
    has_next = ar < (lens.to(keys.device) - 1)[:, None]
    cur = torch.isin(keys, allowed)
    nxt = torch.cat([keys[:, 1:], torch.zeros_like(keys[:, :1])], dim=1)
    return has_next & cur & torch.isin(nxt, allowed)


@torch.no_grad()
def d4_letters_scores(model: D4Model, data: EncodedSet, allowed_ids: np.ndarray, batch_size: int,
                      device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """``(score, n_steps)``: mean per-step NLL over the letters-only steps (NaN where a session has none)."""
    allowed = torch.as_tensor(np.asarray(allowed_ids, dtype="int64"), device=device)
    scores, counts = [], []
    for idx in _chunks(np.arange(len(data)), batch_size):
        b = to_device(data.batch(idx), device)
        nll, valid = model.step_nll(b)
        m = letters_step_mask(b["keys"], b["lens"], allowed) & valid
        c = m.sum(1)
        s = (nll * m).sum(1).double() / c.double()
        scores.append(torch.where(c > 0, s, torch.full_like(s, float("nan"))).cpu().numpy())
        counts.append(c.cpu().numpy())
    if not scores:
        return np.zeros(0), np.zeros(0, dtype="int64")
    return np.concatenate(scores), np.concatenate(counts).astype("int64")
