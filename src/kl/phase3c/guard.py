"""Freeze guard for Phase 3c: Phases 1, 2, 3a (+ the Phase 3b pre-registration) via ``kl.phase3b.guard``, plus the
Phase 3b record ``docs/phase3b_frozen_hashes.json`` (Phase 3b sources / config / pre-registration and every
``outputs/<mode>/phase3b`` result file)."""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from ..config import ROOT, Settings
from ..model import LivenessModel
from ..phase2.guard import FrozenViolation, sha256_file, verify_files
from ..phase3b.guard import assert_frozen_p123

PHASE3B_RECORD: Path = ROOT / "docs" / "phase3b_frozen_hashes.json"
#: launcher bookkeeping / archives in outputs/<mode>/phase3b that are not results
_NOT_RESULTS = {"launcher.out", "launcher_status.json", ".launcher.lock", ".training.lock", "thermal_stop.json"}


def phase3b_sources(root: Path = ROOT) -> list[str]:
    """Phase 3b source, script, config and pre-registration files (relative paths)."""
    files = sorted((root / "src" / "kl" / "phase3b").glob("*.py"))
    files += [root / "scripts" / n for n in ("30_build_sequences.py", "31_train_d4.py", "32_train_d5.py", "33_report_phase3b.py", "run_phase3b.py")]
    files += [root / "config_phase3b.yaml", root / "docs" / "PHASE3B_PREREGISTRATION.md", root / "docs" / "phase3b_prereg_hash.txt"]
    return [str(f.relative_to(root)) for f in files]


def phase3b_outputs(mode: str, root: Path = ROOT) -> list[str]:
    """Every Phase 3b result file of ``mode`` (models, scores, interim sequences, tables, report, figures, logs)."""
    base = root / "outputs" / mode / "phase3b"
    out = [p for p in base.rglob("*") if p.is_file() and p.name not in _NOT_RESULTS and "log_archive" not in p.relative_to(base).parts]
    return sorted(str(p.relative_to(root)) for p in out)


def record_phase3b(path: Path = PHASE3B_RECORD, root: Path = ROOT, overwrite: bool = False) -> dict:
    """Write the Phase 3b freeze record (refuses to overwrite an existing one unless ``overwrite``)."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; the Phase 3b freeze is recorded once")
    rec = {"description": "Phase 3b freeze record (Phase 3c): SHA-256 of the Phase 3b sources/config/pre-registration and of every "
                          "outputs/<mode>/phase3b result file. Verified by kl.phase3c.guard.",
           "recorded_at": _dt.datetime.now().isoformat(timespec="seconds"),
           "source_files": {p: sha256_file(root / p) for p in phase3b_sources(root)},
           "modes": {m: {"output_files": {p: sha256_file(root / p) for p in phase3b_outputs(m, root)}}
                     for m in ("full", "quick") if (root / "outputs" / m / "phase3b").exists()}}
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def verify_phase3b(st: Settings, record_path: Path = PHASE3B_RECORD, root: Path = ROOT) -> None:
    """Raise :class:`FrozenViolation` if a Phase 3b source or result file changed (or a result file disappeared)."""
    rec = json.loads(Path(record_path).read_text(encoding="utf-8"))
    verify_files(rec["source_files"], root)
    mode = "quick" if st.quick else "full"
    bad = [p for p, h in rec["modes"][mode]["output_files"].items() if not (root / p).exists() or sha256_file(root / p) != h]
    if bad:
        raise FrozenViolation(f"frozen Phase 3b outputs changed: {bad[:10]}{' ...' if len(bad) > 10 else ''}")


def assert_frozen_p1234(st: Settings) -> LivenessModel:
    """Verify Phases 1, 2, 3a, the Phase 3b pre-registration and the Phase 3b results; return the frozen D1 model."""
    model = assert_frozen_p123(st)
    verify_phase3b(st)
    return model
