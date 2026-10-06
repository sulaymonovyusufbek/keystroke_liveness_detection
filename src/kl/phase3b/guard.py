"""Freeze guard for Phase 3b: Phases 1, 2 and 3a unchanged, and the pre-registration unchanged.

* Phases 1 + 2: :func:`kl.phase3a.guard.assert_frozen_p12` (sources, ``config.yaml``, config hash, frozen model, Phase 2
  sources and outputs).
* Phase 3a: ``docs/phase3a_frozen_hashes.json`` (sources / config and every ``outputs/<mode>/phase3a`` file).
* Pre-registration: ``docs/phase3b_prereg_hash.txt`` (SHA-256 of ``docs/PHASE3B_PREREGISTRATION.md`` and
  ``config_phase3b.yaml``).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import ROOT, Settings
from ..model import LivenessModel
from ..phase2.guard import FrozenViolation, sha256_file, verify_files
from ..phase3a.guard import assert_frozen_p12

PHASE3A_RECORD: Path = ROOT / "docs" / "phase3a_frozen_hashes.json"
PREREG_RECORD: Path = ROOT / "docs" / "phase3b_prereg_hash.txt"


def read_prereg_record(path: Path = PREREG_RECORD) -> dict[str, str]:
    """``key: value`` lines of the pre-registration record (``#`` lines are comments)."""
    out: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def verify_prereg(path: Path = PREREG_RECORD, root: Path = ROOT) -> dict[str, str]:
    """Raise :class:`FrozenViolation` if the pre-registration or ``config_phase3b.yaml`` changed since it was recorded."""
    rec = read_prereg_record(path)
    for f, h in (("prereg_file", "prereg_sha256"), ("config_file", "config_sha256")):
        if sha256_file(root / rec[f]) != rec[h]:
            raise FrozenViolation(f"pre-registered file {rec[f]} changed since {rec['recorded_at']}")
    return rec


def verify_phase3a(st: Settings, record_path: Path = PHASE3A_RECORD, root: Path = ROOT) -> None:
    """Raise :class:`FrozenViolation` if a Phase 3a source or output file changed."""
    rec = json.loads(Path(record_path).read_text(encoding="utf-8"))
    verify_files(rec["source_files"], root)
    mode = "quick" if st.quick else "full"
    bad = [p for p, h in rec["modes"][mode]["output_files"].items() if sha256_file(root / p) != h]
    if bad:
        raise FrozenViolation(f"frozen Phase 3a outputs changed: {bad}")


def assert_frozen_p123(st: Settings) -> LivenessModel:
    """Verify Phases 1, 2, 3a and the Phase 3b pre-registration; return the frozen D1 model (never refit)."""
    model = assert_frozen_p12(st)
    verify_phase3a(st)
    verify_prereg()
    return model
