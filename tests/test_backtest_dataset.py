"""Dataset loader: Hive discovery + skip-corrupt + decimal coercion."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from perp_arb.backtest.dataset import load_capture
from perp_arb.strategy.spread_monitor import SPREAD_PARQUET_SCHEMA


def _row(ts_ms: int, left_ts: int, right_ts: int, *, gates: bool = True,
         left_bid: str = "100.00", right_bid: str = "100.05") -> dict:
    return {
        "ts_ms": ts_ms,
        "left_venue": "lighter", "right_venue": "aster",
        "left_bid": left_bid, "left_bid_size": "10",
        "left_ask": str(Decimal(left_bid) + Decimal("0.01")), "left_ask_size": "10",
        "right_bid": right_bid, "right_bid_size": "10",
        "right_ask": str(Decimal(right_bid) + Decimal("0.01")), "right_ask_size": "10",
        "mid_left": str((Decimal(left_bid) * 2 + Decimal("0.01")) / 2),
        "mid_right": str((Decimal(right_bid) * 2 + Decimal("0.01")) / 2),
        "raw_spread": "0.05", "bias_ewma": "0.05",
        "vwap_left_sell": left_bid, "vwap_left_buy": str(Decimal(left_bid) + Decimal("0.01")),
        "vwap_right_sell": right_bid, "vwap_right_buy": str(Decimal(right_bid) + Decimal("0.01")),
        "edge_A_bps": "1.0", "edge_B_bps": "-1.0",
        "gates_passed": gates,
        "left_ts_ms": left_ts, "right_ts_ms": right_ts,
        "gap_ms": 0,
    }


def _write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = []
    # SPREAD_PARQUET_SCHEMA has the dictionary `date` column at the end; tests
    # write to date= partitions, so we just drop that field from the schema.
    schema = pa.schema([f for f in SPREAD_PARQUET_SCHEMA if f.name != "date"])
    for field in schema:
        arrays.append(pa.array([r[field.name] for r in rows], type=field.type))
    table = pa.table(arrays, schema=schema)
    pq.write_table(table, path)


def test_load_capture_skips_corrupt_files(tmp_path) -> None:
    rows_a = [_row(1000, 1000, 950), _row(1100, 1100, 1100)]
    rows_b = [_row(1200, 1200, 1150), _row(1300, 1300, 1300)]
    _write_parquet(rows_a, tmp_path / "date=2026-05-20" / "00.parquet")
    _write_parquet(rows_b, tmp_path / "date=2026-05-20" / "01.parquet")
    # mimic an active-writer file: empty bytes → ArrowInvalid on read
    (tmp_path / "date=2026-05-20" / "02.parquet").write_bytes(b"")

    rows = load_capture(tmp_path)
    assert len(rows) == 4
    assert [r.ts_ms for r in rows] == [1000, 1100, 1200, 1300]
    # decimal coercion
    assert isinstance(rows[0].left_bid, Decimal)
    assert rows[0].left_bid == Decimal("100.00")
    # passthrough fields
    assert rows[0].left_venue == "lighter" and rows[0].right_venue == "aster"
    assert rows[0].left_ts_ms == 1000 and rows[0].right_ts_ms == 950


def test_load_capture_sorts_across_files(tmp_path) -> None:
    # file B is written first chronologically by name but contains LATER ts_ms;
    # loader must still emit by ts_ms order
    rows_b = [_row(1200, 1200, 1150), _row(1300, 1300, 1300)]
    rows_a = [_row(1000, 1000, 950), _row(1100, 1100, 1100)]
    _write_parquet(rows_b, tmp_path / "date=2026-05-20" / "00.parquet")
    _write_parquet(rows_a, tmp_path / "date=2026-05-20" / "01.parquet")
    rows = load_capture(tmp_path)
    assert [r.ts_ms for r in rows] == [1000, 1100, 1200, 1300]
