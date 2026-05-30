"""Backtest CSV telemetry sink.

Persistence is normalised into two joinable files (`decisions_*.csv`,
`legs_*.csv`); the strategy never sees that split. CSV headers are derived from
the `Decision` / `LegOutcome` dataclass fields so header and row cannot drift.
The live path uses `core/sqlite_recorder.py::SqliteRecorder` instead — both
implement the `Recorder` contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from ..logging import CsvWriter
from ..types import LegOutcome
from .decision import Decision, Timeline
from .recorder import Recorder

# These fields are live-SQLite-only telemetry (or non-serialisable); keep them
# out of the CSV projection so the backtest decisions CSV header/rows are
# byte-for-byte stable.
_DECISION_SKIP = {"timeline", "thr_throttle_bps", "failure_reason"}


def _decision_header() -> list[str]:
    cols = [f.name for f in fields(Decision) if f.name not in _DECISION_SKIP]
    return cols + list(Timeline().latencies())


def _leg_header() -> list[str]:
    return ["decision_id", "ts_ms", *LegOutcome.csv_header()]


class CsvRecorder(Recorder):
    """Backtest CSV sink. `emit(decision)` writes the decisions row; `emit_legs`
    writes one legs row per leg. The only place the strategy's telemetry reaches
    disk in backtest."""

    def __init__(
        self,
        log_dir: Path,
        run_ts: str | None = None,
        *,
        strategy_id: str = "taker_taker",
    ) -> None:
        ts = run_ts or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self._dec = CsvWriter(log_dir / f"decisions_{strategy_id}_{ts}.csv", _decision_header())
        self._legs = CsvWriter(log_dir / f"legs_{strategy_id}_{ts}.csv", _leg_header())

    def emit(self, d: Decision) -> None:
        row = {f.name: getattr(d, f.name) for f in fields(d) if f.name not in _DECISION_SKIP}
        row |= d.timeline.latencies()
        # CSV-only precision shaping: keep the in-memory Decimals
        # full-precision so analytics math doesn't lose digits, but the
        # human-facing file only carries what's meaningful.
        #   - edge_bps: bps scale, 1 dp ≈ 0.1 bps is well below noise.
        #   - bias: price-scale quantity; dp tracks the price magnitude
        #     so an asset at ~3000 shows ~mille-px resolution and one
        #     at ~1 shows micro-px resolution (4-ish sigfigs in either).
        row["edge_bps"] = d.edge_bps.quantize(Decimal("0.1"))
        row["bias"] = _quantize_to_price_scale(d.bias, d.mid_left)
        self._dec.write([row[h] for h in self._dec.header])

    def emit_legs(self, decision_id: str, ts_ms: int, legs: Sequence[LegOutcome]) -> None:
        for leg in legs:
            self._legs.write([decision_id, ts_ms, *leg.to_csv_row()])

    def close(self) -> None:
        self._dec.close()
        self._legs.close()


def _quantize_to_price_scale(value: Decimal, price: Decimal) -> Decimal:
    """Round `value` to a decimal place appropriate for a price-scale
    quantity at the given price level — yields ~4 sigfigs of resolution
    relative to the underlying ratio. Falls back to 4 dp if price is
    non-positive (only seen pre-warmup / aborted tick)."""
    if price <= 0:
        return value.quantize(Decimal("0.0001"))
    magnitude = len(str(int(price))) - 1  # ⌊log10(price)⌋
    dp = max(2, 6 - magnitude)
    return value.quantize(Decimal(10) ** -dp)
