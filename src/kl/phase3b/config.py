"""Phase 3b configuration (``config_phase3b.yaml``, pre-registered) and output layout."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..config import ROOT, Settings

PHASE3B_CONFIG: Path = ROOT / "config_phase3b.yaml"


def load_phase3b_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config_phase3b.yaml``."""
    with open(Path(path) if path else PHASE3B_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def phase3b_dir(st: Settings) -> Path:
    """``outputs/<mode>/phase3b`` (created, with its ``interim``, ``models`` and ``scores`` sub-directories)."""
    d = st.out_dir / "phase3b"
    for sub in ("interim", "models", "scores"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d
