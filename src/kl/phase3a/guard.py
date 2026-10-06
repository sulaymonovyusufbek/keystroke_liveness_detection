"""Freeze guard for Phase 3a: Phase 1 AND Phase 2 must be unchanged (source files and the outputs Phase 3a reads)."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import ROOT, Settings
from ..model import LivenessModel
from ..phase2.guard import FrozenViolation, assert_frozen, sha256_file, verify_files

PHASE2_RECORD: Path = ROOT / "docs" / "phase2_frozen_hashes.json"


def assert_frozen_p12(st: Settings, record_path: Path = PHASE2_RECORD) -> LivenessModel:
    """Verify Phase 1 (via :func:`kl.phase2.guard.assert_frozen`) and Phase 2 (sources + output files) and return the frozen model."""
    model = assert_frozen(st)
    rec = json.loads(Path(record_path).read_text(encoding="utf-8"))
    verify_files(rec["source_files"])
    mode = "quick" if st.quick else "full"
    bad = [p for p, h in rec["modes"][mode]["output_files"].items() if sha256_file(ROOT / p) != h]
    if bad:
        raise FrozenViolation(f"frozen Phase 2 outputs changed: {bad}")
    return model
