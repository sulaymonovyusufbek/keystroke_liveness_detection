"""Configuration loading, output-directory layout, JSON helpers and logging.

Every script builds a :class:`Settings` from ``config.yaml`` (plus ``--quick`` overrides).
The settings expose the global seed and the effective-config hash that is stored next to the model.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG: Path = ROOT / "config.yaml"
#: config sections that do not influence results (paths, parallelism, the separate legacy replica)
HASH_EXCLUDED_KEYS: frozenset[str] = frozenset({"n_jobs", "paths", "legacy", "quick_overrides"})


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``override`` merged in recursively (neither input is mutated)."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for numpy scalars/arrays and paths."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


@dataclass(frozen=True)
class Settings:
    """Effective configuration for one run (``full`` or ``quick``)."""

    cfg: dict[str, Any]
    quick: bool
    root: Path = ROOT

    def __getitem__(self, key: str) -> Any:
        return self.cfg[key]

    @property
    def seed(self) -> int:
        """Global random seed."""
        return int(self.cfg["seed"])

    @property
    def n_jobs(self) -> int:
        """Number of joblib workers."""
        return int(self.cfg["n_jobs"])

    @property
    def raw_dir(self) -> Path:
        """Directory holding the participant files and the metadata file."""
        return self.root / self.cfg["paths"]["raw_dir"]

    @property
    def metadata_path(self) -> Path:
        """Path of ``metadata_participants.txt``."""
        return self.raw_dir / self.cfg["paths"]["metadata_file"]

    @property
    def out_dir(self) -> Path:
        """``outputs/full`` or ``outputs/quick``."""
        return self.root / self.cfg["paths"]["outputs_dir"] / ("quick" if self.quick else "full")

    @property
    def interim_dir(self) -> Path:
        """Bulky intermediate parquet files."""
        return self.out_dir / "interim"

    @property
    def model_dir(self) -> Path:
        """Frozen model, scaler, thresholds."""
        return self.out_dir / "model"

    @property
    def results_dir(self) -> Path:
        """Small JSON / CSV / MD / PNG results."""
        return self.out_dir / "results"

    def ensure_dirs(self) -> None:
        """Create all output directories."""
        for d in (self.interim_dir, self.model_dir, self.results_dir, self.out_dir / "logs"):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def config_hash(self) -> str:
        """SHA-256 (first 16 hex chars) of the canonical JSON of the protocol-relevant effective config."""
        protocol = {k: v for k, v in self.cfg.items() if k not in HASH_EXCLUDED_KEYS}
        blob = json.dumps({"cfg": protocol, "quick": self.quick}, sort_keys=True, default=_json_default)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_settings(config_path: str | Path | None = None, quick: bool = False) -> Settings:
    """Load ``config.yaml`` and apply ``quick_overrides`` when ``quick`` is set."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if quick:
        cfg = _deep_merge(cfg, cfg.get("quick_overrides") or {})
    settings = Settings(cfg=cfg, quick=quick)
    settings.ensure_dirs()
    return settings


def write_json(path: str | Path, obj: Any) -> None:
    """Write ``obj`` as pretty JSON (numpy types handled)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, default=_json_default, ensure_ascii=False)


def read_json(path: str | Path, default: Any = None) -> Any:
    """Read JSON, returning ``default`` if the file does not exist."""
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def update_json_section(path: str | Path, section: str, value: Any) -> None:
    """Set ``section`` in the JSON object stored at ``path`` (creating the file if needed)."""
    data = read_json(path, default={}) or {}
    data[section] = value
    write_json(path, data)


def setup_logging(out_dir: Path, name: str) -> logging.Logger:
    """Console + file logger writing to ``<out_dir>/logs/<name>.log``."""
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / f"{name}.log", mode="w")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger
