"""Event-driven backtest engine.

Replays rows in `ts_ms` order. At each tick: (1) drains all pending orders
whose `arrival_ts ≤ row.ts_ms`, filling them against the venue's book at
arrival (via `BookIndex`); (2) builds a `MarketSnapshot` and asks the
strategy for new intents; (3) schedules each intent's arrival.

Atomic-pair semantics: a `Decision` is emitted to the recorder only once
ALL its legs have resolved. Position is updated only when EVERY leg
succeeded — any single-leg reject leaves the position untouched and the
Decision still emits with `LegReport.success=False` legs (Outcome stays
FIRED — the strategy did fire; the venue just didn't fill).
"""

from __future__ import annotations

import json
import logging
from bisect import insort
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from ..core.exec_record import (
    Decision,
    ExecutionRecorder,
    LegKind,
    LegReport,
    Phase,
)
from ..core.types import OrderResult, OrderStatus, Side
from .base import BacktestStrategy, EngineView, StrategyContext
from .dataset import BBORow
from .fills import FillModelKind, VenueSide, fill_model_for
from .intents import FillEvent, OrderIntent, PendingOrder
from .latency import BookIndex, LatencyModel
from .pnl import SyntheticPositions
from .snapshot import build_snapshot

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """All knobs the engine needs to run. Built by `runner.build_config`."""
    data_root: Path
    out_dir: Path
    capture_qty: Decimal
    latency: LatencyModel
    fill_model: FillModelKind                  # applied to both legs in v1
    fee_bps_per_leg: Decimal
    strategy_id: str


@dataclass(slots=True)
class EngineSummary:
    rows_processed: int = 0
    intents_emitted: int = 0
    fills_succeeded: int = 0
    fills_rejected: int = 0
    decisions_emitted: int = 0
    realised_pnl: Decimal = Decimal("0")
    final_positions: dict[str, Decimal] = field(default_factory=dict)
    outage_count: int = 0
    outage_max_ms: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "rows_processed": self.rows_processed,
            "intents_emitted": self.intents_emitted,
            "fills_succeeded": self.fills_succeeded,
            "fills_rejected": self.fills_rejected,
            "decisions_emitted": self.decisions_emitted,
            "realised_pnl": str(self.realised_pnl),
            "final_positions": {k: str(v) for k, v in self.final_positions.items()},
            "outage_count": self.outage_count,
            "outage_max_ms": self.outage_max_ms,
            "reject_reasons": self.reject_reasons,
        }


_OUTAGE_THRESHOLD_MS = 5_000


class Engine:
    """Single-shot backtest. Construct with strategy + config, call `run()`."""

    def __init__(
        self,
        rows: list[BBORow],
        strategy: BacktestStrategy,
        cfg: EngineConfig,
        ctx: StrategyContext,
    ) -> None:
        if not rows:
            raise ValueError("engine: empty rows")
        self.rows = rows
        self.strategy = strategy
        self.cfg = cfg
        self.ctx = ctx
        self.positions = SyntheticPositions()
        self.pending: dict[str, list[PendingOrder]] = {
            ctx.left_venue: [],
            ctx.right_venue: [],
        }
        self._venue_side: dict[str, VenueSide] = {ctx.left_venue: "left", ctx.right_venue: "right"}
        self.book_index = {
            ctx.left_venue: BookIndex.build(rows, "left"),
            ctx.right_venue: BookIndex.build(rows, "right"),
        }
        self.fill_models = {
            FillModelKind.BBO: fill_model_for(FillModelKind.BBO),
            FillModelKind.VWAP: fill_model_for(FillModelKind.VWAP),
        }
        self.in_flight: dict[str, Decision] = {}
        self.remaining_legs: dict[str, int] = {}
        # signed in-flight qty per venue (scheduled but not yet resolved).
        # Combined with `positions.sizes` by `EngineView.position()` so
        # `max_qty` caps total commitment, not just settled exposure.
        self.in_flight_qty: dict[str, Decimal] = {ctx.left_venue: Decimal("0"),
                                                  ctx.right_venue: Decimal("0")}
        self.summary = EngineSummary()
        self.view = EngineView(_positions=self.positions.sizes,
                               _in_flight=self.in_flight_qty,
                               _sim_now_ms=0, _pending_count=0)

    def _pop_due(self, venue: str, up_to: int) -> list[PendingOrder]:
        bucket = self.pending[venue]
        due: list[PendingOrder] = []
        while bucket and bucket[0].arrival_ts_ms <= up_to:
            due.append(bucket.pop(0))
        return due

    def _apply_fill(self, fill: FillEvent, intent: OrderIntent) -> None:
        # Decrement in-flight regardless of success — the intent has now
        # resolved one way or the other and no longer represents pending
        # exposure. Settled positions only move via `apply_pair` (atomic).
        self.in_flight_qty[intent.venue] = (
            self.in_flight_qty.get(intent.venue, Decimal("0"))
            - intent.qty * Decimal(intent.side.sign)
        )
        d = self.in_flight[fill.decision_id]
        r = OrderResult(
            success=fill.success,
            order_id=f"bt-{fill.decision_id[:6]}-{fill.venue[:3]}",
            side=fill.side,
            requested_qty=fill.requested_qty,
            filled_qty=fill.filled_qty,
            avg_price=fill.realized_price,
            status=OrderStatus.FILLED if fill.success else OrderStatus.REJECTED,
            error_message=fill.error,
        )
        # latency = simulated submit delay (= the per-venue network leg latency
        # the user dialled in). Matches the live semantic where latency_ms is
        # the span between SEND and "result_<venue>".
        latency_ms = fill.arrival_ts_ms - intent.sim_ts_ms
        d.legs.append(LegReport.build(
            venue=fill.venue, side=fill.side, qty=fill.requested_qty,
            expected=intent.expected_price, ack=r,
            latency_ms=latency_ms, kind=LegKind.ENTRY,
        ))
        if fill.success:
            self.summary.fills_succeeded += 1
        else:
            self.summary.fills_rejected += 1
            reason = fill.error or "unknown"
            self.summary.reject_reasons[reason] = self.summary.reject_reasons.get(reason, 0) + 1
        self.remaining_legs[fill.decision_id] -= 1

    def _maybe_emit(self, decision_id: str, recorder: ExecutionRecorder) -> None:
        if self.remaining_legs[decision_id] > 0:
            return
        d = self.in_flight.pop(decision_id)
        del self.remaining_legs[decision_id]
        all_success = bool(d.legs) and all(leg.success for leg in d.legs)
        if all_success and len(d.legs) == 2:
            left_leg = next((lg for lg in d.legs if lg.exchange == self.ctx.left_venue), None)
            right_leg = next((lg for lg in d.legs if lg.exchange == self.ctx.right_venue), None)
            assert left_leg is not None and right_leg is not None
            self.positions.apply_pair(
                self.ctx.left_venue, Side(left_leg.side), left_leg.realized_price or Decimal(0),
                self.ctx.right_venue, Side(right_leg.side), right_leg.realized_price or Decimal(0),
                left_leg.filled_qty or Decimal(0),
                self.cfg.fee_bps_per_leg,
            )
        latest_latency = max((leg.latency_ms or 0) for leg in d.legs) if d.legs else 0
        send_mark = d.timeline.get(Phase.SEND) or 0
        d.timeline.mark_at(Phase.RESULT, send_mark + latest_latency)
        recorder.emit(d)
        self.summary.decisions_emitted += 1

    def _schedule(self, intent: OrderIntent) -> None:
        intent.decision.timeline.mark_at(Phase.SEND, intent.sim_ts_ms)
        self.in_flight[intent.decision_id] = intent.decision
        self.remaining_legs[intent.decision_id] = self.remaining_legs.get(intent.decision_id, 0) + 1
        arrival = self.cfg.latency.arrival_ts(intent.venue, intent.sim_ts_ms)
        insort(self.pending[intent.venue], PendingOrder(intent, arrival),
               key=lambda p: p.arrival_ts_ms)
        self.in_flight_qty[intent.venue] = (
            self.in_flight_qty.get(intent.venue, Decimal("0"))
            + intent.qty * Decimal(intent.side.sign)
        )
        self.summary.intents_emitted += 1

    def _refresh_view(self, now_ms: int) -> None:
        self.view._sim_now_ms = now_ms
        self.view._pending_count = sum(len(v) for v in self.pending.values())

    def _drain(self, up_to: int, recorder: ExecutionRecorder) -> None:
        """Drain every pending arrival whose `arrival_ts ≤ up_to` across both
        venues. Called twice per row (before + after the strategy step) so a
        zero-latency intent scheduled this tick lands the same tick."""
        for venue in (self.ctx.left_venue, self.ctx.right_venue):
            for p in self._pop_due(venue, up_to):
                venue_row = self.book_index[venue].book_at(p.arrival_ts_ms)
                side_label = self._venue_side[venue]
                fill = self.fill_models[p.intent.fill_model].try_fill(
                    p.intent, p.arrival_ts_ms, venue_row, side_label, self.cfg.capture_qty,
                )
                self._apply_fill(fill, p.intent)
                self._refresh_view(up_to)
                self.strategy.on_fill(fill, self.view)
                self._maybe_emit(p.intent.decision_id, recorder)

    def run(self, recorder: ExecutionRecorder) -> EngineSummary:
        for row in self.rows:
            self.summary.rows_processed += 1
            if row.gap_ms > _OUTAGE_THRESHOLD_MS:
                self.summary.outage_count += 1
                self.summary.outage_max_ms = max(self.summary.outage_max_ms, row.gap_ms)
                _log.info("outage: gap %dms at ts=%d", row.gap_ms, row.ts_ms)

            self._drain(row.ts_ms, recorder)

            self._refresh_view(row.ts_ms)
            snap = build_snapshot(row)
            for intent in self.strategy.on_tick(snap, self.view):
                self._schedule(intent)

            # Drain again so zero-delay intents scheduled this tick land the
            # same tick (their arrival_ts == row.ts_ms).
            self._drain(row.ts_ms, recorder)

        # EOD: reject every still-pending order
        last_ts = self.rows[-1].ts_ms
        for venue, bucket in self.pending.items():
            for p in bucket:
                fill = FillEvent(
                    decision_id=p.intent.decision_id, venue=venue,
                    side=p.intent.side, requested_qty=p.intent.qty,
                    filled_qty=None, realized_price=None,
                    arrival_ts_ms=p.arrival_ts_ms, fill_ts_ms=last_ts,
                    success=False, error="eod_unfilled",
                    fill_model=p.intent.fill_model,
                )
                self._apply_fill(fill, p.intent)
                self.strategy.on_fill(fill, self.view)
                self._maybe_emit(p.intent.decision_id, recorder)
            bucket.clear()

        self.strategy.on_end(self.view)
        self.summary.realised_pnl = self.positions.realised_pnl
        self.summary.final_positions = dict(self.positions.sizes)
        return self.summary


def write_summary(summary: EngineSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(summary.to_dict(), f, indent=2)
