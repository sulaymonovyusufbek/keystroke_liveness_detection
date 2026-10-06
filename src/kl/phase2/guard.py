"""Phase 1 freeze guard: Phase 2 must never run against a modified Phase 1.

``docs/phase1_frozen_hashes.json`` records SHA-256 of features.py / model.py / selection.py / config.yaml and, per
run mode, the protocol config hash and the SHA-256 of the frozen model files. :func:`assert_frozen` verifies all of it
and additionally checks that the config hash stored INSIDE the loaded model equals the current config hash.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from ..config import ROOT, Settings
from ..model import LivenessModel

RECORD_PATH: Path = ROOT / "docs" / "phase1_frozen_hashes.json"


class FrozenViolation(RuntimeError):
    """Raised when something Phase 1 froze has changed."""


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_files(expected: Mapping[str, str], root: Path = ROOT) -> None:
    """Raise :class:`FrozenViolation` if any ``{relative_path: sha256}`` entry differs."""
    bad = [p for p, h in expected.items() if sha256_file(root / p) != h]
    if bad:
        raise FrozenViolation(f"frozen Phase 1 files changed: {bad}")


def assert_frozen(st: Settings, record_path: Path = RECORD_PATH) -> LivenessModel:
    """Verify the Phase 1 freeze for this run mode and return the frozen model (never refit it).

    Checks: frozen source files unchanged; protocol config hash equals the recorded one AND the one stored in the
    model; model / threshold files byte-identical to the record.
    """
    rec = json.loads(Path(record_path).read_text(encoding="utf-8"))
    verify_files(rec["source_files"])
    mode = "quick" if st.quick else "full"
    m = rec["modes"][mode]
    if st.config_hash != m["config_hash"]:
        raise FrozenViolation(f"config hash {st.config_hash} != frozen {m['config_hash']} ({mode})")
    for fname, key in (("model.joblib", "model_joblib_sha256"), ("thresholds.json", "thresholds_json_sha256")):
        if sha256_file(st.model_dir / fname) != m[key]:
            raise FrozenViolation(f"frozen model file {st.model_dir / fname} changed")
    model = LivenessModel.load(st.model_dir)
    if model.config_hash != st.config_hash:
        raise FrozenViolation(f"model config hash {model.config_hash} != current {st.config_hash}")
    return model
