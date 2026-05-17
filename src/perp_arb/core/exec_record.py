"""Execution telemetry: one Decision per evaluated opportunity, one emit point.

The strategy builds exactly one `Decision` per tick it would act on, mutates
it as evaluation/firing proceeds, and a single `ExecutionRecorder.emit` call
persists it — so a missed write (the old partial-fill gap) is structurally
impossible and the domain code never touches CSV.

Persistence is normalised into two joinable files (`decisions_*.csv`,
`legs_*.csv`); the strategy never sees that split. CSV headers are derived
from the dataclass fields so header and row cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from ..utils.time import mono_ms
from .logging import CsvWriter


class Timeline:
    """Named monotonic checkpoints. Latencies are derived, never hand-subtracted
    at call sites."""

    def __init__(self) -> None:
        self._marks: dict[str, int] = {}

    def mark(self, phase: str) -> None:
        self._marks[phase] = mono_ms()

    def span(self, a: str, b: str) -> int | None:
        if a in self._marks and b in self._marks:
            return self._marks[b] - self._marks[a]
        return None


@dataclass
class LegReport:
    """One venue leg of a fired decision — always produced, success or not."""

    exchange: str
    side: str
    requested_qty: Decimal
    filled_qty: Decimal | None
    expected_price: Decimal       # decision-time VWAP for this leg/side
    realized_price: Decimal | None
    status: str
    success: bool
    error: str | None = None
    order_id: str | None = None
    client_id: str | None = None
    fee: Decimal | None = None
    latency_ms: int | None = None   # send → THIS leg's result (inter-leg skew)
    kind: str = "entry"             # "entry" | "unwind"


@dataclass
class Decision:
    """One evaluated opportunity. `outcome` is terminal; `legs` is populated
    only when it FIRED."""

    decision_id: str
    ts_ms: int
    mid_a: Decimal
    mid_l: Decimal
    a_quote_ts_ms: int              # for decision-time staleness analysis
    l_quote_ts_ms: int
    # below are unknown at an early (pre-edge) abort, hence defaulted
    bias: Decimal = Decimal(0)
    vwap_a_sell: Decimal = Decimal(0)
    vwap_a_buy: Decimal = Decimal(0)
    vwap_l_sell: Decimal = Decimal(0)
    vwap_l_buy: Decimal = Decimal(0)
    edge_bps: Decimal = Decimal(0)  # chosen direction's net edge, bps
    direction: str = ""             # "A" | "B"
    outcome: str = "PENDING"        # FIRED|ABORT_STALE|ABORT_SLIPPAGE|
                                    # ABORT_NO_DEPTH|BLOCKED_RISK
    abort_reason: str | None = None
    timeline: Timeline = field(default_factory=Timeline)
    legs: list[LegReport] = field(default_factory=list)


_DECISION_SKIP = {"timeline", "legs"}
_LAT_COLS = ["lat_decision_send_ms", "lat_send_result_ms", "lat_total_ms"]


def _decision_header() -> list[str]:
    return [f.name for f in fields(Decision) if f.name not in _DECISION_SKIP] + _LAT_COLS


def _leg_header() -> list[str]:
    return ["decision_id", "ts_ms"] + [f.name for f in fields(LegReport)]


class ExecutionRecorder:
    """Single sink. `emit(decision)` writes one decisions row + one legs row
    per leg. The only place the strategy's telemetry reaches disk."""

    def __init__(self, log_dir: Path, run_ts: str | None = None) -> None:
        ts = run_ts or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self._dec_header = _decision_header()
        self._leg_header = _leg_header()
        self._dec = CsvWriter(log_dir / f"decisions_taker_taker_{ts}.csv", self._dec_header)
        self._legs = CsvWriter(log_dir / f"legs_taker_taker_{ts}.csv", self._leg_header)

    def emit(self, d: Decision) -> None:
        row = {f.name: getattr(d, f.name) for f in fields(d) if f.name not in _DECISION_SKIP}
        row["lat_decision_send_ms"] = d.timeline.span("decision", "send")
        row["lat_send_result_ms"] = d.timeline.span("send", "result")
        row["lat_total_ms"] = d.timeline.span("decision", "result")
        self._dec.write([row[h] for h in self._dec_header])
        for leg in d.legs:
            lrow = {"decision_id": d.decision_id, "ts_ms": d.ts_ms}
            lrow |= {f.name: getattr(leg, f.name) for f in fields(leg)}
            self._legs.write([lrow[h] for h in self._leg_header])

    def close(self) -> None:
        self._dec.close()
        self._legs.close()
