"""Glue shared by the Phase 3b training scripts: encoded-set loading, the test split, score files, training curves,
the runtime log and the seed-aggregated tables."""
from __future__ import annotations

import datetime as _dt
import fcntl
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..config import read_json, write_json
from ..phase2.evaluation import user_weights
from .evaluation import TestAccess, TestSplit, seed_cell
from .sequences import EncodedSet, Encoder, SeqSet


class LockBusyError(RuntimeError):
    """Another Phase 3b launcher / training process holds the lock."""


def acquire_lock(path: Path, what: str):
    """Exclusive, non-blocking ``flock`` on ``path`` held until the returned handle is closed or the process dies.

    A process paused with SIGSTOP keeps its lock, so nothing can start next to it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.seek(0)
        holder = fh.read().strip() or "unknown"
        fh.close()
        raise LockBusyError(f"{what}: {path.name} is held by another process ({holder}); refusing to start") from None
    fh.seek(0)
    fh.truncate()
    fh.write(f"pid {os.getpid()} since {_dt.datetime.now().isoformat(timespec='seconds')}\n")
    fh.flush()
    return fh


def gpu_lock(out: Path):
    """The single-training lock: only one Phase 3b training process per run mode at a time."""
    return acquire_lock(out / ".training.lock", "training")


def run_state(out: Path, key: str, model_file: Path, required: Sequence[str]) -> str:
    """``"complete"`` (weights, a loadable score file with every ``required`` array, test-ledger entry and runtime entry),
    ``"interrupted-after-test-access"`` (ledger entry but no complete result) or ``"not-done"``."""
    ledger = json.loads((out / "test_access.json").read_text(encoding="utf-8")) if (out / "test_access.json").exists() else {}
    runs = (read_json(out / "runtime_log.json", {}) or {}).get("runs", {})
    sp = score_path(out, key)
    complete = False
    if sp.exists() and model_file.exists() and key in ledger and key in runs:
        try:
            z = load_scores(sp)
            complete = all(k in z and np.isfinite(z[k]).all() for k in required)
        except Exception:  # noqa: BLE001 - an unreadable file is simply not complete
            complete = False
    if complete:
        return "complete"
    return "interrupted-after-test-access" if key in ledger else "not-done"


def load_encoder(out: Path) -> Encoder:
    """The encoder fitted by step 30."""
    return Encoder.from_json(read_json(out / "interim" / "encoder.json"))


def load_encoded(out: Path, enc: Encoder, file_stem: str) -> EncodedSet:
    """``interim/<file_stem>.npz`` encoded with ``enc``."""
    return enc.encode(SeqSet.load(out / "interim" / f"{file_stem}.npz"))


def load_test_split(out: Path, enc: Encoder, attackers: Sequence[str], allow_rerun: bool) -> TestSplit:
    """Test humans + test bots of ``attackers`` behind the single-use ledger ``test_access.json``."""
    humans = load_encoded(out, enc, "human_test")
    bots = {a: load_encoded(out, enc, f"bot_{a}_test") for a in attackers}
    for a, b in bots.items():                       # bots are matched one-to-one to test humans (same users)
        if not set(np.unique(b.user_ids)) <= set(np.unique(humans.user_ids)):
            raise AssertionError(f"test bots of {a} are matched to non-test users")
    return TestSplit(humans, bots, TestAccess(out / "test_access.json", allow_rerun))


def score_path(out: Path, key: str) -> Path:
    """``scores/<key with '/' -> '_'>.npz``."""
    return out / "scores" / (key.replace("/", "_").replace("<-", "_from_") + ".npz")


def save_scores(path: Path, **arrays: Any) -> None:
    """Compressed npz of score arrays / scalars."""
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})


def load_scores(path: Path) -> dict[str, np.ndarray]:
    """Inverse of :func:`save_scores`."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def append_curves(out: Path, meta: Mapping[str, Any], curve: Sequence[Mapping[str, float]]) -> None:
    """Append one run's per-epoch train/val loss to ``training_curves.csv`` (replacing an older copy of the same run)."""
    path = out / "training_curves.csv"
    new = pd.DataFrame([{**meta, **row} for row in curve])
    if path.exists():
        old = pd.read_csv(path)
        same = np.ones(len(old), dtype=bool)
        for k, v in meta.items():
            same &= old[k].astype(str).to_numpy() == str(v)
        new = pd.concat([old.loc[~same], new], ignore_index=True)
    new.to_csv(path, index=False)


def update_runtime(out: Path, section: str, key: str, value: Any) -> None:
    """``runtime_log.json[section][key] = value`` (read-modify-write)."""
    path = out / "runtime_log.json"
    data = read_json(path, {}) or {}
    data.setdefault(section, {})[key] = value
    data["updated_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    write_json(path, data)


def build_table(
    specs: Sequence[Mapping[str, Any]], test_humans_users: np.ndarray, bot_users: Mapping[str, np.ndarray],
    reps: int, seed: int, ci_level: float, n_jobs: int,
) -> pd.DataFrame:
    """Seed-aggregated rows. Each spec: ``detector, trained_on, tested_on, per_seed={seed: {h, b, taus}}``."""
    users = np.sort(np.unique(test_humans_users))
    idx_of = pd.Index(users).get_indexer
    h_users = idx_of(test_humans_users)
    b_idx = {a: idx_of(u) for a, u in bot_users.items()}
    if (h_users < 0).any() or any((v < 0).any() for v in b_idx.values()):
        raise AssertionError("bot matched to a non-test user")
    weights = user_weights(len(users), reps, seed)          # the Phase 2 resamples (same users, same seed)
    rows = Parallel(n_jobs=n_jobs)(
        delayed(seed_cell)(s["detector"], s["trained_on"], s["tested_on"], s["per_seed"], h_users, b_idx[s["tested_on"]], weights, ci_level, seed)
        for s in specs
    )
    return pd.DataFrame(rows)
