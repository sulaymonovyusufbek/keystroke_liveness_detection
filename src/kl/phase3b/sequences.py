"""Keystroke sequences for the Phase 3b detectors: ragged storage, the encoder (vocabulary, standardisation, max_len)
and padded batches.

Per keystroke ``i`` of a session sorted by PRESS_TIME (KEYSTROKE_ID as the tie-break only, the order of features.py)::

    key_i = KEYCODE_i      hold_i = RELEASE_i - PRESS_i      pp_i = PRESS_{i+1} - PRESS_i   (last key: undefined)

The encoder maps ``key -> vocabulary index`` (0 = PAD, 1 = UNK), ``h = (log1p(hold) - mu_h) / sd_h`` and
``p = (log1p(pp) - mu_p) / sd_p`` with statistics of TRAIN humans, and truncates to ``max_len``. The last KEPT key of a
session is its last key (``p`` masked, stored as 0) whether or not the session was truncated.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed

PAD, UNK = 0, 1
READ_COLS = ["session_id", "PARTICIPANT_ID", "PRESS_TIME", "RELEASE_TIME", "KEYCODE", "KEYSTROKE_ID"]


class NegativeHoldError(ValueError):
    """A keystroke with RELEASE < PRESS reached the encoder (pre-registered: stop, never clip)."""


def session_arrays(press: Sequence[float], release: Sequence[float], keycode: Sequence[int],
                   kid: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(keycode, hold, pp)`` of ONE session in keystroke order; ``pp`` has NaN at the last key (masked)."""
    press, release = np.asarray(press, dtype="float64"), np.asarray(release, dtype="float64")
    order = np.argsort(press, kind="stable") if kid is None else np.lexsort((np.asarray(kid, dtype="float64"), press))
    p, r = press[order], release[order]
    return np.asarray(keycode, dtype="int64")[order].astype("int32"), r - p, np.r_[np.diff(p), np.nan]


def _ranges(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Concatenation of ``arange(s, s + n)`` for every ``(s, n)``."""
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, dtype="int64")
    shift = np.repeat(starts - np.r_[0, np.cumsum(lens)[:-1]], lens)
    return shift + np.arange(total)


@dataclass
class SeqSet:
    """Ragged keystroke sequences of many sessions plus what they are (split, human / bot, attacker)."""

    name: str
    split: str
    kind: str                      # "human" or "bot"
    attacker: str | None
    session_ids: np.ndarray        # str
    user_ids: np.ndarray           # int64
    offsets: np.ndarray            # int64, n_sessions + 1
    keycode: np.ndarray            # int32 (flat)
    hold: np.ndarray               # float64 ms (flat)
    pp: np.ndarray                 # float64 ms (flat), NaN at the last key of every session

    @property
    def n_sessions(self) -> int:
        """Number of sessions."""
        return len(self.session_ids)

    @property
    def lengths(self) -> np.ndarray:
        """Keystrokes per session."""
        return np.diff(self.offsets)

    def take(self, idx: np.ndarray, name: str | None = None) -> "SeqSet":
        """Sessions ``idx`` (in that order)."""
        idx = np.asarray(idx, dtype="int64")
        starts, lens = self.offsets[:-1][idx], self.lengths[idx]
        rows = _ranges(starts, lens)
        return SeqSet(name or self.name, self.split, self.kind, self.attacker, self.session_ids[idx], self.user_ids[idx],
                      np.r_[0, np.cumsum(lens)].astype("int64"), self.keycode[rows], self.hold[rows], self.pp[rows])

    def sorted_by_session(self) -> "SeqSet":
        """Sessions ordered by ``session_id`` (string order, as the Phase 1 feature files)."""
        return self.take(np.argsort(self.session_ids, kind="stable"))

    def select_ids(self, ids: Iterable[str], name: str) -> "SeqSet":
        """The sessions whose id is in ``ids`` (kept in this set's order)."""
        want = set(ids)
        return self.take(np.flatnonzero([s in want for s in self.session_ids]), name)

    def save(self, path: Path) -> None:
        """Write a compressed ``.npz``."""
        np.savez_compressed(path, meta=np.array(json.dumps({"name": self.name, "split": self.split, "kind": self.kind, "attacker": self.attacker})),
                            session_ids=self.session_ids.astype("U"), user_ids=self.user_ids, offsets=self.offsets,
                            keycode=self.keycode, hold=self.hold, pp=self.pp)

    @classmethod
    def load(cls, path: Path) -> "SeqSet":
        """Read a file written by :meth:`save`."""
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(meta["name"], meta["split"], meta["kind"], meta["attacker"], z["session_ids"], z["user_ids"], z["offsets"],
                   z["keycode"], z["hold"], z["pp"])

    @classmethod
    def concat(cls, parts: Sequence["SeqSet"], name: str, split: str, kind: str, attacker: str | None) -> "SeqSet":
        """Sessions of all ``parts`` one after another."""
        parts = [p for p in parts if p.n_sessions]
        if not parts:
            return cls(name, split, kind, attacker, np.zeros(0, dtype="U1"), np.zeros(0, dtype="int64"), np.zeros(1, dtype="int64"),
                       np.zeros(0, dtype="int32"), np.zeros(0), np.zeros(0))
        lens = np.concatenate([p.lengths for p in parts])
        return cls(name, split, kind, attacker, np.concatenate([p.session_ids.astype("U") for p in parts]),
                   np.concatenate([p.user_ids for p in parts]).astype("int64"), np.r_[0, np.cumsum(lens)].astype("int64"),
                   np.concatenate([p.keycode for p in parts]), np.concatenate([p.hold for p in parts]), np.concatenate([p.pp for p in parts]))


def seqset_from_frame(df: pd.DataFrame, name: str, split: str, kind: str, attacker: str | None,
                      user_col: str = "PARTICIPANT_ID") -> SeqSet:
    """Sequences of all sessions in ``df`` (Aalto columns + ``session_id``), sorted by session_id, then keystroke order."""
    if df.empty:
        return SeqSet.concat([], name, split, kind, attacker)
    codes, uniques = pd.factorize(df["session_id"].to_numpy(dtype=object), sort=True)
    press = df["PRESS_TIME"].to_numpy(dtype="float64")
    kid = df["KEYSTROKE_ID"].to_numpy(dtype="float64", na_value=np.nan)
    order = np.lexsort((kid, press, codes))
    codes, press = codes[order], press[order]
    release = df["RELEASE_TIME"].to_numpy(dtype="float64")[order]
    first = np.r_[True, codes[1:] != codes[:-1]]
    last = np.r_[codes[1:] != codes[:-1], True]
    pp = np.r_[np.diff(press), np.nan]
    pp[last] = np.nan
    starts = np.flatnonzero(first)
    users = df[user_col].to_numpy()[order][starts].astype("int64")
    return SeqSet(name, split, kind, attacker, np.asarray(uniques, dtype="U"), users, np.r_[starts, len(codes)].astype("int64"),
                  df["KEYCODE"].to_numpy(dtype="int64")[order].astype("int32"), release - press, pp)


def _read_group(path: str, group: int, split_of_user: Mapping[int, str], split_of_session: Mapping[str, str]) -> dict[str, SeqSet]:
    df = pq.ParquetFile(path).read_row_group(group, columns=READ_COLS).to_pandas()
    by_user = df["PARTICIPANT_ID"].map(split_of_user)
    by_session = df["session_id"].map(split_of_session)
    out = {}
    for name, mask in (("train_all", by_user == "train"), ("val", by_session == "val"), ("test", by_session == "test")):
        out[name] = seqset_from_frame(df.loc[mask.to_numpy()], name, "train" if name == "train_all" else name, "human", None)
    return out


def build_human_sets(sessions_path: Path, manifest: Mapping[str, Any], n_jobs: int = 1) -> dict[str, SeqSet]:
    """``train_all`` (every clean session of a TRAIN user), ``train_20k`` (the manifest's capped train sessions = D1's
    training set), ``val`` and ``test`` (the manifest's sessions), each sorted by session_id."""
    split_of_user = {int(u): "train" for u in manifest["users"]["train"]}
    split_of_session = {sid: s for s in ("val", "test") for sid in manifest["sessions"][s]}
    n_groups = pq.ParquetFile(sessions_path).num_row_groups
    parts = Parallel(n_jobs=n_jobs)(delayed(_read_group)(str(sessions_path), g, split_of_user, split_of_session) for g in range(n_groups))
    sets = {name: SeqSet.concat([p[name] for p in parts], name, "train" if name == "train_all" else name, "human", None).sorted_by_session()
            for name in ("train_all", "val", "test")}
    sets["train_20k"] = sets["train_all"].select_ids(manifest["sessions"]["train"], "train_20k")
    for name, want in (("train_20k", manifest["sessions"]["train"]), ("val", manifest["sessions"]["val"]), ("test", manifest["sessions"]["test"])):
        if list(sets[name].session_ids) != sorted(want):
            raise AssertionError(f"human set {name}: sessions differ from the Phase 1 manifest")
    return sets


def check_against_features(seqs: SeqSet, feats: pd.DataFrame, tol: float = 1e-9) -> dict[str, float]:
    """Assert that ``n_keys``, ``hold_mean`` and ``pp_mean`` recomputed from ``seqs`` equal the Phase 1/2 feature rows."""
    if list(seqs.session_ids) != list(feats["session_id"].astype(str)) or not np.array_equal(seqs.user_ids, feats["user_id"].to_numpy(dtype="int64")):
        raise AssertionError(f"{seqs.name}: session / user order differs from the feature file")
    lens = seqs.lengths
    starts = seqs.offsets[:-1]
    hold_mean = np.add.reduceat(seqs.hold, starts) / lens
    pp0 = np.nan_to_num(seqs.pp, nan=0.0)
    pp_mean = np.add.reduceat(pp0, starts) / (lens - 1)
    diffs = {"n_keys": float(np.max(np.abs(lens - feats["n_keys"].to_numpy()))),
             "hold_mean": float(np.max(np.abs(hold_mean - feats["hold_mean"].to_numpy()))),
             "pp_mean": float(np.max(np.abs(pp_mean - feats["pp_mean"].to_numpy())))}
    bad = {k: v for k, v in diffs.items() if not v <= tol}
    if bad:
        raise AssertionError(f"{seqs.name}: sequences disagree with the feature file: {bad}")
    return diffs


# ------------------------------------------------------------------------------------------------------ encoder
@dataclass
class Encoder:
    """Vocabulary, standardisation and max_len, all fitted on TRAIN humans (``train_all``)."""

    keycodes: np.ndarray           # sorted kept keycodes; index i -> vocabulary id i + 2
    mu_h: float
    sd_h: float
    mu_p: float
    sd_p: float
    max_len: int
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def vocab_size(self) -> int:
        """Including PAD and UNK."""
        return len(self.keycodes) + 2

    @classmethod
    def fit(cls, train: SeqSet, min_count: int, percentile: float) -> "Encoder":
        """Fit on TRAIN human sequences."""
        if train.kind != "human" or train.split != "train":
            raise ValueError(f"the encoder is fitted on TRAIN humans only, got {train.name} ({train.split}, {train.kind})")
        _check_holds(train)
        codes, counts = np.unique(train.keycode, return_counts=True)
        kept = codes[counts >= min_count]
        lh = np.log1p(train.hold)
        lp = np.log1p(train.pp[np.isfinite(train.pp)])
        max_len = int(math.ceil(float(np.percentile(train.lengths, percentile))))
        info = {"n_keycodes_seen": int(len(codes)), "n_keycodes_kept": int(len(kept)), "min_count": int(min_count),
                "unk_keycodes": [int(c) for c in codes[counts < min_count]], "max_len_percentile": float(percentile),
                "n_sessions": int(train.n_sessions), "n_keystrokes": int(len(train.hold))}
        return cls(kept.astype("int64"), float(lh.mean()), float(lh.std()), float(lp.mean()), float(lp.std()), max_len, info)

    def to_json(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {"keycodes": [int(k) for k in self.keycodes], "mu_h": self.mu_h, "sd_h": self.sd_h, "mu_p": self.mu_p, "sd_p": self.sd_p,
                "max_len": self.max_len, "vocab_size": self.vocab_size, "info": self.info}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "Encoder":
        """Inverse of :meth:`to_json`."""
        return cls(np.array(d["keycodes"], dtype="int64"), float(d["mu_h"]), float(d["sd_h"]), float(d["mu_p"]), float(d["sd_p"]),
                   int(d["max_len"]), dict(d.get("info", {})))

    def key_ids(self, keycode: np.ndarray) -> np.ndarray:
        """Vocabulary ids (UNK for anything not kept)."""
        keycode = np.asarray(keycode, dtype="int64")
        if not len(self.keycodes):
            return np.full(len(keycode), UNK, dtype="int64")
        pos = np.clip(np.searchsorted(self.keycodes, keycode), 0, len(self.keycodes) - 1)
        return np.where(self.keycodes[pos] == keycode, pos + 2, UNK).astype("int64")

    def encode(self, s: SeqSet) -> "EncodedSet":
        """Padded ``(n, max_len)`` arrays; sessions longer than ``max_len`` keep their first ``max_len`` keys."""
        _check_holds(s)
        n, L = s.n_sessions, self.max_len
        lens = np.minimum(s.lengths, L)
        ar = np.arange(L)[None, :]
        valid = ar < lens[:, None]
        rows = np.where(valid, s.offsets[:-1][:, None] + ar, 0)
        pmask = ar < (lens - 1)[:, None]
        keys = np.where(valid, self.key_ids(s.keycode)[rows] if len(s.keycode) else 0, PAD).astype("int16")
        h = np.where(valid, (np.log1p(s.hold[rows]) - self.mu_h) / self.sd_h, 0.0).astype("float32") if len(s.hold) else np.zeros((n, L), "float32")
        with np.errstate(invalid="ignore"):
            p = np.where(pmask, (np.log1p(s.pp[rows]) - self.mu_p) / self.sd_p, 0.0).astype("float32") if len(s.pp) else np.zeros((n, L), "float32")
        if not np.isfinite(h).all() or not np.isfinite(p).all():
            raise ValueError(f"{s.name}: non-finite encoded value")
        return EncodedSet(s.name, s.split, s.kind, s.attacker, s.session_ids, s.user_ids, keys, h, p, pmask, lens.astype("int64"),
                          int((s.lengths > L).sum()), int(s.lengths.max()) if n else 0)


def _check_holds(s: SeqSet) -> None:
    if len(s.hold) and float(np.min(s.hold)) < 0:
        raise NegativeHoldError(f"{s.name}: {int((s.hold < 0).sum())} keystrokes with RELEASE < PRESS")


@dataclass
class EncodedSet:
    """Model-ready padded arrays of one set (``keys`` int16, ``h`` / ``p`` float32, ``pmask`` bool, ``lens``)."""

    name: str
    split: str
    kind: str
    attacker: str | None
    session_ids: np.ndarray
    user_ids: np.ndarray
    keys: np.ndarray
    h: np.ndarray
    p: np.ndarray
    pmask: np.ndarray
    lens: np.ndarray
    n_truncated: int
    max_n_keys: int

    def __len__(self) -> int:
        return len(self.lens)

    def batch(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        """Rows ``idx``, trimmed to the longest sequence among them."""
        idx = np.asarray(idx, dtype="int64")
        lens = self.lens[idx]
        w = int(lens.max())
        return {"keys": self.keys[idx, :w].astype("int64"), "h": self.h[idx, :w], "p": self.p[idx, :w], "pmask": self.pmask[idx, :w], "lens": lens}

    def truncation_row(self, max_len: int) -> dict[str, Any]:
        """One row of the truncation report."""
        return {"set": self.name, "split": self.split, "kind": self.kind, "attacker": self.attacker or "", "n_sessions": len(self),
                "n_truncated": self.n_truncated, "frac_truncated": self.n_truncated / len(self) if len(self) else float("nan"),
                "max_len": max_len, "max_n_keys": self.max_n_keys}
