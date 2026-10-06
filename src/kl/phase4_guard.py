"""Phase 4 freeze guard: Phases 1-3c must be unchanged and docs/PHASE4_SPEC.md must match its recorded hash.

Checks (raises on any mismatch):
* Phases 1, 2, 3a, 3b pre-registration and 3b outputs: :func:`kl.phase3c.guard.assert_frozen_p1234`;
* Phase 3c sources and the outputs of the current mode: ``docs/phase3c_frozen_hashes.json``;
* the Phase 4 spec: ``docs/phase4_prereg_hash.txt``.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import ROOT, Settings
from .model import LivenessModel
from .phase2.guard import FrozenViolation, sha256_file, verify_files
from .phase3c.guard import assert_frozen_p1234

PHASE3C_RECORD = ROOT / "docs" / "phase3c_frozen_hashes.json"
PHASE4_RECORD = ROOT / "docs" / "phase4_prereg_hash.txt"


def read_record(path: Path = PHASE4_RECORD) -> dict[str, str]:
    """``key: value`` lines of the Phase 4 hash record (comments skipped)."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def verify_spec(path: Path = PHASE4_RECORD, root: Path = ROOT) -> dict[str, str]:
    """The spec's SHA-256 equals the recorded one."""
    rec = read_record(path)
    got = sha256_file(root / rec["spec_file"])
    if got != rec["spec_sha256"]:
        raise FrozenViolation(f"{rec['spec_file']} changed since it was recorded ({got} != {rec['spec_sha256']})")
    return rec


def verify_phase3c(st: Settings, record_path: Path = PHASE3C_RECORD, root: Path = ROOT) -> None:
    """Phase 3c sources and this mode's Phase 3c outputs equal the freeze record."""
    rec = json.loads(record_path.read_text(encoding="utf-8"))
    verify_files(rec["source_files"], root)
    mode = "quick" if st.quick else "full"
    bad = [p for p, h in rec["modes"][mode]["output_files"].items() if sha256_file(root / p) != h]
    if bad:
        raise FrozenViolation(f"Phase 3c outputs changed ({mode}): {bad}")


def assert_frozen_p4(st: Settings) -> LivenessModel:
    """All earlier phases frozen + Phase 4 spec unchanged; returns the frozen D1 model."""
    model = assert_frozen_p1234(st)
    verify_phase3c(st)
    verify_spec()
    return model
