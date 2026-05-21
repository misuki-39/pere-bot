"""Parquet capture loader.

The spread_monitor writes Hive-partitioned hourly Parquet under
`logs/spread_<base>_<ts>/date=YYYY-MM-DD/HH.parquet`. The active (currently
being written) hour file has no valid footer and raises `ArrowInvalid` when
read — we skip such files with a warning so backtests can run while a capture
is still in progress.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..strategy.spread_monitor import SPREAD_PARQUET_SCHEMA

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BBORow:
    """One captured row, with all numeric columns coerced to Decimal."""
    ts_ms: int
    left_venue: str
    right_venue: str
    left_bid: Decimal
    left_bid_size: Decimal
    left_ask: Decimal
    left_ask_size: Decimal
    right_bid: Decimal
    right_bid_size: Decimal
    right_ask: Decimal
    right_ask_size: Decimal
    mid_left: Decimal
    mid_right: Decimal
    raw_spread: Decimal
    bias_ewma: Decimal
    # VWAP columns are None when capture-time depth gate failed
    vwap_left_sell: Decimal | None
    vwap_left_buy: Decimal | None
    vwap_right_sell: Decimal | None
    vwap_right_buy: Decimal | None
    edge_A_bps: Decimal | None
    edge_B_bps: Decimal | None
    gates_passed: bool
    left_ts_ms: int
    right_ts_ms: int
    gap_ms: int


# Decimal-valued columns derived from the capture schema (excluding the
# string-typed venue-name columns). Partitioned into "always-present" vs
# "may-be-None when capture-time depth gate failed" so the loader can't drift
# from the writer.
_NON_DECIMAL_STRING_COLS = {"left_venue", "right_venue"}
_DECIMAL_COL_NAMES = {
    f.name for f in SPREAD_PARQUET_SCHEMA
    if f.type == pa.string() and f.name not in _NON_DECIMAL_STRING_COLS
}
_OPT_DECIMAL_COLS = (
    "vwap_left_sell", "vwap_left_buy", "vwap_right_sell", "vwap_right_buy",
    "edge_A_bps", "edge_B_bps",
)
_STR_COLS = tuple(sorted(_DECIMAL_COL_NAMES - set(_OPT_DECIMAL_COLS)))


def _discover_files(root: Path) -> list[Path]:
    """Find every Parquet file under root (Hive-partitioned or flat)."""
    return sorted(root.rglob("*.parquet"))


def _read_one(path: Path) -> list[BBORow] | None:
    """Read one Parquet file. Returns None (with WARN) if the file is corrupt
    or partially written (active capture hour)."""
    try:
        table = pq.read_table(path)
    except Exception as e:  # noqa: BLE001
        _log.warning("skipping unreadable parquet %s: %s", path, e)
        return None

    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    n = table.num_rows
    rows: list[BBORow] = []
    for i in range(n):
        kwargs: dict[str, object] = {
            "ts_ms": int(cols["ts_ms"][i]),
            "left_venue": str(cols["left_venue"][i]),
            "right_venue": str(cols["right_venue"][i]),
            "gates_passed": bool(cols["gates_passed"][i]),
            "left_ts_ms": int(cols["left_ts_ms"][i]),
            "right_ts_ms": int(cols["right_ts_ms"][i]),
            "gap_ms": int(cols["gap_ms"][i]),
        }
        for c in _STR_COLS:
            kwargs[c] = Decimal(cols[c][i])
        for c in _OPT_DECIMAL_COLS:
            v = cols[c][i]
            kwargs[c] = Decimal(v) if v is not None else None
        rows.append(BBORow(**kwargs))  # type: ignore[arg-type]
    return rows


def load_capture(root: Path) -> list[BBORow]:
    """Load every readable Parquet file under root, sorted by ts_ms.

    Returns a list (not an iterator) because the engine needs random access
    via BookIndex anyway and 19h of WTI data (~420k rows) is comfortably
    in-memory.
    """
    files = _discover_files(root)
    if not files:
        raise FileNotFoundError(f"no .parquet files under {root}")
    all_rows: list[BBORow] = []
    skipped = 0
    for f in files:
        rows = _read_one(f)
        if rows is None:
            skipped += 1
            continue
        all_rows.extend(rows)
    if not all_rows:
        raise RuntimeError(f"every parquet file under {root} was unreadable")
    all_rows.sort(key=lambda r: r.ts_ms)
    _log.info(
        "loaded %d rows from %d files (skipped %d unreadable) under %s",
        len(all_rows), len(files) - skipped, skipped, root,
    )
    return all_rows


