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
from enum import StrEnum
from pathlib import Path

from ..utils.time import mono_ms
from .logging import CsvWriter
from .types import LegOutcome


class Outcome(StrEnum):
    PENDING = "PENDING"
    FIRED = "FIRED"
    ABORT_STALE = "ABORT_STALE"
    ABORT_NO_DEPTH = "ABORT_NO_DEPTH"
    BLOCKED_RISK = "BLOCKED_RISK"


class Direction(StrEnum):
    A = "A"   # sell leg_a, buy leg_b
    B = "B"   # reverse


class Phase(StrEnum):
    """Canonical Timeline checkpoints `latencies()` derives columns from. The
    recorder depends only on these — the strategy owns any other marks.

    No RESULT phase: end-to-end latency is derived at analysis time from
    `LegOutcome.fill_ts_ms - LegOutcome.send_ts_ms` (mixed clock — local
    epoch vs venue matching-engine clock; NTP skew is ms-scale)."""

    DECISION = "decision"
    SEND = "send"


class Timeline:
    """Named monotonic checkpoints. Latencies are derived, never hand-subtracted
    at call sites."""

    def __init__(self) -> None:
        self._marks: dict[str, int] = {}

    def mark(self, phase: str) -> None:
        self._marks[phase] = mono_ms()

    def mark_at(self, phase: str, ts_ms: int) -> None:
        """Stamp a mark with an explicit timestamp instead of mono_ms().
        Backtest uses this to record sim-time spans; live never calls it."""
        self._marks[phase] = ts_ms

    def get(self, phase: str) -> int | None:
        return self._marks.get(phase)

    def span(self, a: str, b: str) -> int | None:
        if a in self._marks and b in self._marks:
            return self._marks[b] - self._marks[a]
        return None

    def latencies(self) -> dict[str, int | None]:
        """One derived latency column: decision-compute → fire. Pure
        local-clock CPU/wait. End-to-end execution latency is derived at
        analysis time from LegOutcome.fill_ts_ms - LegOutcome.send_ts_ms."""
        return {
            "lat_decision_send_ms": self.span(Phase.DECISION, Phase.SEND),
        }


@dataclass
class Decision:
    """One evaluated opportunity. `outcome` is terminal; `legs` is populated
    only when it FIRED."""

    decision_id: str
    ts_ms: int
    mid_left: Decimal
    mid_right: Decimal
    left_quote_ts_ms: int           # for decision-time staleness analysis
    right_quote_ts_ms: int
    # below are unknown at an early (pre-edge) abort, hence defaulted
    bias: Decimal = Decimal(0)
    vwap_left_sell: Decimal = Decimal(0)
    vwap_left_buy: Decimal = Decimal(0)
    vwap_right_sell: Decimal = Decimal(0)
    vwap_right_buy: Decimal = Decimal(0)
    edge_bps: Decimal = Decimal(0)  # chosen direction's net edge, bps
    direction: Direction | None = None
    outcome: Outcome = Outcome.PENDING
    abort_reason: str | None = None
    # Our local clock at SEND (epoch ms). Pair with `LegOutcome.fill_ts_ms`
    # (exchange clock) for end-to-end submit→fill latency, modulo NTP skew.
    send_ts_ms: int | None = None
    # Cash-flow realized PnL for this two-leg trade, net of fees. None on
    # non-FIRED outcomes or partial-failure unwinds (success=False).
    realised_pnl: Decimal | None = None
    timeline: Timeline = field(default_factory=Timeline)
    legs: list[LegOutcome] = field(default_factory=list)


_DECISION_SKIP = {"timeline", "legs"}


def _decision_header() -> list[str]:
    cols = [f.name for f in fields(Decision) if f.name not in _DECISION_SKIP]
    return cols + list(Timeline().latencies())


def _leg_header() -> list[str]:
    return ["decision_id", "ts_ms", *LegOutcome.csv_header()]


class ExecutionRecorder:
    """Single sink. `emit(decision)` writes one decisions row + one legs row
    per leg. The only place the strategy's telemetry reaches disk."""

    def __init__(
        self,
        log_dir: Path,
        run_ts: str | None = None,
        *,
        strategy_id: str = "taker_taker",
    ) -> None:
        ts = run_ts or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self._dec = CsvWriter(log_dir / f"decisions_{strategy_id}_{ts}.csv",
                              _decision_header())
        self._legs = CsvWriter(log_dir / f"legs_{strategy_id}_{ts}.csv",
                               _leg_header())

    def emit(self, d: Decision) -> None:
        row = {f.name: getattr(d, f.name)
               for f in fields(d) if f.name not in _DECISION_SKIP}
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
        for leg in d.legs:
            self._legs.write([d.decision_id, d.ts_ms, *leg.to_csv_row()])

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
    magnitude = len(str(int(price))) - 1   # ⌊log10(price)⌋
    dp = max(2, 6 - magnitude)
    return value.quantize(Decimal(10) ** -dp)
