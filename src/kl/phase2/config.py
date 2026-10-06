"""Phase 2 configuration (``config_phase2.yaml``) and output layout."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..config import ROOT, Settings

PHASE2_CONFIG: Path = ROOT / "config_phase2.yaml"


def load_phase2_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config_phase2.yaml``."""
    with open(Path(path) if path else PHASE2_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def phase2_dir(st: Settings) -> Path:
    """``outputs/<mode>/phase2`` (created)."""
    d = st.out_dir / "phase2"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pool_params(st: Settings, p2: dict[str, Any]) -> dict[str, Any]:
    """Parameters of :func:`kl.attacks.pool.fit_pool_stats`: Phase 2 pool section + shared Phase 1 limits."""
    return {
        **{k: v for k, v in p2["pool"].items() if k != "holdout_user_fraction"},
        "pause_threshold_ms": st.cfg["features"]["pause_threshold_ms"],
        "max_pp_ms": st.cfg["cleaning"]["max_press_gap_ms"],
        "max_hold_ms": st.cfg["cleaning"]["max_hold_ms"],
    }
