"""Aalto 136M Keystrokes: file discovery, robust parsing and metadata loading.

Parsing never raises for a bad file: problems are returned in :class:`ParseStats` so the caller can
log and count them. Text is decoded as UTF-8 first and falls back to latin-1 (the real encoding of
the corpus; latin-1 cannot fail, so ``errors="replace"`` is only a last-resort safety net).
"""
from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import pyarrow as pa

KEYSTROKE_COLUMNS: list[str] = [
    "PARTICIPANT_ID", "TEST_SECTION_ID", "SENTENCE", "USER_INPUT", "KEYSTROKE_ID",
    "PRESS_TIME", "RELEASE_TIME", "LETTER", "KEYCODE",
]

#: Arrow schema of ``keystrokes.parquet`` (raw columns + the LAYOUT metadata column).
KEYSTROKE_SCHEMA: pa.Schema = pa.schema([
    ("PARTICIPANT_ID", pa.int64()),
    ("TEST_SECTION_ID", pa.int64()),
    ("SENTENCE", pa.string()),
    ("USER_INPUT", pa.string()),
    ("KEYSTROKE_ID", pa.int64()),
    ("PRESS_TIME", pa.float64()),
    ("RELEASE_TIME", pa.float64()),
    ("LETTER", pa.string()),
    ("KEYCODE", pa.int64()),
    ("LAYOUT", pa.string()),
])

MISSING_LABEL = "<missing>"


@dataclass
class ParseStats:
    """Per-file parsing outcome (all counts are exact except ``lines_not_parsed``, an approximation)."""

    participant_id: int
    file: str
    encoding: str = ""
    rows_read: int = 0
    lines_not_parsed: int = 0
    rows_bad_ids: int = 0
    error: str = ""


def list_participant_files(raw_dir: Path, pattern: str = "*_keystrokes.txt") -> dict[int, Path]:
    """Map PARTICIPANT_ID -> path for every participant file (metadata/readme are not matched)."""
    files: dict[int, Path] = {}
    for path in sorted(Path(raw_dir).glob(pattern)):
        stem = path.name.split("_", 1)[0]
        if stem.isdigit():
            files[int(stem)] = path
    return files


def read_text_robust(path: Path) -> tuple[str, str]:
    """Decode a file as UTF-8, falling back to latin-1. Returns ``(text, encoding_used)``."""
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace"), "latin-1"


def _read_tsv(text: str) -> pd.DataFrame:
    """Parse tab-separated text without quoting/NA magic; every cell stays a string."""
    return pd.read_csv(
        io.StringIO(text), sep="\t", quoting=csv.QUOTE_NONE, on_bad_lines="skip",
        dtype=str, keep_default_na=False, na_filter=False,
    )


def parse_keystroke_file(path: Path, participant_id: int) -> tuple[pd.DataFrame | None, ParseStats]:
    """Parse one participant file into the Aalto schema; never raises.

    Rows whose PARTICIPANT_ID does not equal ``participant_id`` (or whose ids are non-numeric) are
    dropped and counted. Non-numeric times/keycodes become missing values (handled in cleaning).
    Returns ``(None, stats)`` when the file cannot be used at all (``stats.error`` explains why).
    """
    stats = ParseStats(participant_id=participant_id, file=Path(path).name)
    try:
        text, stats.encoding = read_text_robust(path)
        raw = _read_tsv(text)
        raw.columns = [str(c).strip() for c in raw.columns]
        missing = [c for c in KEYSTROKE_COLUMNS if c not in raw.columns]
        if missing:
            stats.error = f"missing columns: {missing}"
            return None, stats
        n_lines = text.count("\n") + (0 if text.endswith("\n") else 1) - 1
        stats.lines_not_parsed = max(0, n_lines - len(raw))
        stats.rows_read = len(raw)
        pid = pd.to_numeric(raw["PARTICIPANT_ID"], errors="coerce")
        tsid = pd.to_numeric(raw["TEST_SECTION_ID"], errors="coerce")
        ok = (pid == participant_id) & tsid.notna()
        stats.rows_bad_ids = int((~ok).sum())
        raw = raw.loc[ok]
        df = pd.DataFrame({
            "PARTICIPANT_ID": pd.Series(participant_id, index=raw.index, dtype="int64"),
            "TEST_SECTION_ID": tsid.loc[ok].astype("int64"),
            "SENTENCE": raw["SENTENCE"],
            "USER_INPUT": raw["USER_INPUT"],
            "KEYSTROKE_ID": pd.to_numeric(raw["KEYSTROKE_ID"], errors="coerce").astype("Int64"),
            "PRESS_TIME": pd.to_numeric(raw["PRESS_TIME"], errors="coerce").astype("float64"),
            "RELEASE_TIME": pd.to_numeric(raw["RELEASE_TIME"], errors="coerce").astype("float64"),
            "LETTER": raw["LETTER"],
            "KEYCODE": pd.to_numeric(raw["KEYCODE"], errors="coerce").astype("Int64"),
        }).reset_index(drop=True)
        return df, stats
    except Exception as exc:  # noqa: BLE001 - a bad file must never crash the pipeline
        stats.error = f"{type(exc).__name__}: {exc}"
        return None, stats


def parse_chunk(items: Sequence[tuple[int, str, str]]) -> tuple[pa.Table | None, list[dict[str, Any]]]:
    """Parse several files into one Arrow table (one parquet row group). Picklable for joblib.

    ``items`` are ``(participant_id, path, layout)`` triples; ``layout`` is attached as ``LAYOUT``.
    """
    frames: list[pd.DataFrame] = []
    stats: list[dict[str, Any]] = []
    for pid, path, layout in items:
        df, st = parse_keystroke_file(Path(path), pid)
        stats.append(asdict(st))
        if df is not None and len(df):
            df["LAYOUT"] = layout
            frames.append(df)
    if not frames:
        return None, stats
    table = pa.Table.from_pandas(pd.concat(frames, ignore_index=True), schema=KEYSTROKE_SCHEMA, preserve_index=False)
    return table, stats


def load_metadata(path: Path) -> pd.DataFrame:
    """Load ``metadata_participants.txt`` (tab-separated) with a numeric PARTICIPANT_ID.

    KEYBOARD_TYPE / LAYOUT are stripped strings ("" when missing). Rows without a numeric id are
    dropped. Duplicate ids keep the first row; the drop counts are stored in ``df.attrs``.
    """
    text, _ = read_text_robust(path)
    meta = _read_tsv(text)
    meta.columns = [str(c).strip() for c in meta.columns]
    n_raw = len(meta)
    meta["PARTICIPANT_ID"] = pd.to_numeric(meta["PARTICIPANT_ID"], errors="coerce")
    n_bad_id = int(meta["PARTICIPANT_ID"].isna().sum())
    meta = meta.dropna(subset=["PARTICIPANT_ID"]).copy()
    meta["PARTICIPANT_ID"] = meta["PARTICIPANT_ID"].astype("int64")
    n_dup = int(meta["PARTICIPANT_ID"].duplicated().sum())
    meta = meta.drop_duplicates("PARTICIPANT_ID", keep="first").reset_index(drop=True)
    for col in ("KEYBOARD_TYPE", "LAYOUT"):
        meta[col] = meta[col].astype(str).str.strip()
    meta.attrs.update({"rows_raw": n_raw, "rows_bad_id": n_bad_id, "rows_duplicate_id": n_dup})
    return meta


def eligibility(
    meta: pd.DataFrame,
    file_ids: Iterable[int],
    include: Sequence[str],
    exclude: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the keyboard-type filter and join with the available keystroke files.

    Only exact ``include`` strings are eligible. Values in neither list (including empty/missing)
    are excluded and listed under ``unmapped_values`` so nothing is dropped silently.

    Returns ``(eligible_with_file, report)`` where the frame has one row per eligible participant
    that also has a keystroke file (sorted by PARTICIPANT_ID).
    """
    file_set = set(int(i) for i in file_ids)
    types = meta["KEYBOARD_TYPE"].where(meta["KEYBOARD_TYPE"] != "", MISSING_LABEL)
    counts = {str(k): int(v) for k, v in types.value_counts().items()}
    known = set(include) | set(exclude)
    unmapped = {k: v for k, v in counts.items() if k not in known}
    eligible = meta[meta["KEYBOARD_TYPE"].isin(list(include))]
    with_file = eligible[eligible["PARTICIPANT_ID"].isin(file_set)].sort_values("PARTICIPANT_ID").reset_index(drop=True)
    report = {
        "metadata_rows_raw": meta.attrs.get("rows_raw"),
        "metadata_rows_bad_id": meta.attrs.get("rows_bad_id"),
        "metadata_rows_duplicate_id": meta.attrs.get("rows_duplicate_id"),
        "metadata_participants": int(len(meta)),
        "keyboard_type_counts": counts,
        "keyboard_types_include": list(include),
        "keyboard_types_exclude": list(exclude),
        "unmapped_values": unmapped,
        "eligible": int(len(eligible)),
        "keystroke_files_found": len(file_set),
        "eligible_with_file": int(len(with_file)),
    }
    return with_file, report
