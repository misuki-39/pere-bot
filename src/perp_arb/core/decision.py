"""Execution-decision domain model.

The pre-trade decision and its supporting value types — no I/O, no CSV/SQLite
knowledge. The strategy builds exactly one `Decision` per tick it would act on
and mutates it as evaluation/firing proceeds; persistence is a separate concern
owned by a `Recorder` (see `core/recorder.py` and the CSV / SQLite backends).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from ..utils.time import mono_ms


class Verdict(StrEnum):
    """The disposition of an evaluated opportunity — a *pre-execution* verdict
    (fire / abort / blocked), NOT an execution result. The actual fill / PnL
    result lives on `LegOutcome` and `Decision.realised_pnl`. String values are
    persisted (SQLite/CSV `outcome` column); do not change them."""

    PENDING = "PENDING"
    FIRED = "FIRED"
    ABORT_STALE = "ABORT_STALE"
    ABORT_NO_DEPTH = "ABORT_NO_DEPTH"
    BLOCKED_RISK = "BLOCKED_RISK"


class Direction(StrEnum):
    A = "A"  # sell leg_a, buy leg_b
    B = "B"  # reverse

    @property
    def sign(self) -> int:
        """Signed unit change to the (left-leg) position when firing this
        direction: A sells the left leg (−1), B buys it (+1)."""
        return -1 if self is Direction.A else 1


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
    """One evaluated opportunity — the pre-trade decision record. `outcome` is
    terminal. It carries no `LegOutcome`: execution legs are a post-trade result
    (`ExecutionResult.legs`) recorded separately via the recorder's `emit_legs`;
    only scalar result summaries (`realised_pnl`, `failure_reason`, `send_ts_ms`)
    are stamped back onto the decision."""

    decision_id: str
    ts_ms: int
    mid_left: Decimal
    mid_right: Decimal
    left_quote_ts_ms: int  # for decision-time staleness analysis
    right_quote_ts_ms: int
    # below are unknown at an early (pre-edge) abort, hence defaulted
    bias: Decimal = Decimal(0)
    vwap_left_sell: Decimal = Decimal(0)
    vwap_left_buy: Decimal = Decimal(0)
    vwap_right_sell: Decimal = Decimal(0)
    vwap_right_buy: Decimal = Decimal(0)
    edge_bps: Decimal = Decimal(0)  # chosen direction's net edge, bps
    direction: Direction | None = None
    outcome: Verdict = Verdict.PENDING
    abort_reason: str | None = None
    # Our local clock at SEND (epoch ms). Pair with `LegOutcome.fill_ts_ms`
    # (exchange clock) for end-to-end submit→fill latency, modulo NTP skew.
    send_ts_ms: int | None = None
    # Cash-flow realized PnL for this two-leg trade, net of fees. None on
    # non-FIRED outcomes or partial-failure unwinds (success=False).
    realised_pnl: Decimal | None = None
    # Executor's failure narrative for a fired trade (None = all legs filled).
    # The live SQLite `trades` row's success/failure_reason derive from this —
    # the result summary, sourced from `ExecutionResult.failure_reason`, not the
    # legs. Like `thr_throttle_bps`: live-SQLite-only, kept out of the CSV.
    failure_reason: str | None = None
    # Same-direction throttle bump applied to the chosen direction's threshold
    # (bps). The only threshold component that is path-dependent and so cannot
    # be reconstructed post-hoc — the live SQLite recorder persists it on the
    # `trades` row. Excluded from the CSV projection (backtest stays unchanged).
    thr_throttle_bps: Decimal = Decimal(0)
    timeline: Timeline = field(default_factory=Timeline)
