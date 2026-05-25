"""Event-driven backtest engine.

Replays rows in `ts_ms` order. At each tick: (1) drains all pending orders
whose `arrival_ts ≤ row.ts_ms`, filling them against the venue's book at
arrival (via `BookIndex`); (2) builds a `MarketSnapshot` and asks the
strategy for new intents; (3) schedules each intent's arrival.

Atomic-pair semantics: a `Decision` is emitted to the recorder only once
ALL its legs have resolved. Position is updated only when EVERY leg
succeeded — any single-leg reject leaves the position untouched and the
Decision still emits with `LegOutcome.success=False` legs (Outcome stays
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
    Direction,
    ExecutionRecorder,
    Phase,
)
from ..core.types import LegKind, LegOutcome, OrderStatus
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
    # Direction-resolved fire counts (only successful 2-leg pairs).
    fires_dir_a: int = 0
    fires_dir_b: int = 0
    # Per-venue tick count where |position| ≥ max_qty (combined settled+in_flight,
    # sampled once per row at the moment the strategy decides). Diagnoses the
    # position-cap blocking new same-direction fires.
    ticks_pinned: dict[str, int] = field(default_factory=dict)
    duration_ms: int = 0
    max_qty: Decimal = Decimal("0")

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
            "fires_dir_a": self.fires_dir_a,
            "fires_dir_b": self.fires_dir_b,
            "ticks_pinned": dict(self.ticks_pinned),
            "duration_ms": self.duration_ms,
            "max_qty": str(self.max_qty),
        }

    def pretty(self) -> str:
        """Human-rendered multi-line summary. Same source of truth as `to_dict`."""
        dur_h, rem = divmod(self.duration_ms // 1000, 3600)
        dur_m = rem // 60
        per_day = (
            self.realised_pnl * Decimal(86_400_000) / Decimal(self.duration_ms)
            if self.duration_ms > 0 else Decimal(0)
        )
        # PnL comes from filled pairs only; dividing by `decisions_emitted`
        # would deflate the figure with BLOCKED/aborted decisions that
        # never contributed PnL. Use the fired-pair count instead.
        filled_pairs = self.fires_dir_a + self.fires_dir_b
        per_filled = (
            self.realised_pnl / Decimal(filled_pairs)
            if filled_pairs > 0 else Decimal(0)
        )
        pin_lines = []
        for venue, n in self.ticks_pinned.items():
            pct = (n / self.rows_processed * 100) if self.rows_processed > 0 else 0.0
            pin_lines.append(f"{venue}={n}/{self.rows_processed} ({pct:.2f}%)")
        final_pos = ", ".join(f"{k}={v}" for k, v in self.final_positions.items()) or "(none)"
        return (
            "backtest done\n"
            f"  duration:       {dur_h}h {dur_m}m  ({self.duration_ms} ms, {self.rows_processed} rows)\n"
            f"  intents:        {self.intents_emitted}  (filled {self.fills_succeeded}, rejected {self.fills_rejected})\n"
            f"  pairs:          {self.decisions_emitted}  (A={self.fires_dir_a} B={self.fires_dir_b})\n"
            f"  pnl:            {self.realised_pnl:.4f}  (${per_day:.3f}/day, ${per_filled:.4f}/filled-pair)\n"
            f"  final pos:      {final_pos}  (cap=±{self.max_qty})\n"
            f"  pinned ticks:   {'  '.join(pin_lines) if pin_lines else '(none)'}\n"
            f"  outages:        {self.outage_count}  (max {self.outage_max_ms} ms)"
        )


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
        self.summary.ticks_pinned = {ctx.left_venue: 0, ctx.right_venue: 0}
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
        # Mirror live: the outcome carries the venue-side fill instant as
        # `exchange_ts_ms`. In backtest, that's the simulated arrival ts
        # (= submit ts + venue latency). Recorder derives latency from
        # `fill_ts_ms - send_ts_ms` — both stamped here on the outcome
        # so the backtest CSV is shape-identical to live.
        # Atomic: filled_qty + _weighted_price_sum must be committed together
        # via set_fill(). A fill with qty>0 but no realized_price would
        # otherwise leave avg_price returning Decimal('0') (fabricated $0
        # fill) instead of None, blowing up downstream PnL math.
        out = LegOutcome(
            success=fill.success,
            side=fill.side,
            requested_qty=fill.requested_qty,
            status=OrderStatus.FILLED if fill.success else OrderStatus.REJECTED,
            error_message=fill.error,
            exchange_ts_ms=fill.arrival_ts_ms,
            venue=fill.venue,
            expected_price=intent.expected_price,
            send_ts_ms=intent.sim_ts_ms,
            kind=LegKind.ENTRY,
        )
        if (
            fill.success and fill.filled_qty is not None
            and fill.realized_price is not None and fill.filled_qty > 0
        ):
            out.set_fill(fill.filled_qty, fill.realized_price)
        d.legs.append(out)
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
            left_leg = next((lg for lg in d.legs if lg.venue == self.ctx.left_venue), None)
            right_leg = next((lg for lg in d.legs if lg.venue == self.ctx.right_venue), None)
            assert left_leg is not None and right_leg is not None
            assert left_leg.side is not None and right_leg.side is not None
            self.positions.apply_pair(
                self.ctx.left_venue, left_leg.side, left_leg.avg_price or Decimal(0),
                self.ctx.right_venue, right_leg.side, right_leg.avg_price or Decimal(0),
                left_leg.filled_qty or Decimal(0),
                self.cfg.fee_bps_per_leg,
            )
            if d.direction is Direction.A:
                self.summary.fires_dir_a += 1
            elif d.direction is Direction.B:
                self.summary.fires_dir_b += 1
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
        if not self.rows:
            return self.summary
        for row in self.rows:
            self.summary.rows_processed += 1
            if row.gap_ms > _OUTAGE_THRESHOLD_MS:
                self.summary.outage_count += 1
                self.summary.outage_max_ms = max(self.summary.outage_max_ms, row.gap_ms)
                _log.info("outage: gap %dms at ts=%d", row.gap_ms, row.ts_ms)

            self._drain(row.ts_ms, recorder)

            self._refresh_view(row.ts_ms)
            # Position-pin sample: at the moment the strategy decides, is
            # |settled+in_flight| ≥ max_qty for either venue? Combined exposure
            # is what `assess_reversion` actually gates on.
            for venue in (self.ctx.left_venue, self.ctx.right_venue):
                if abs(self.view.position(venue)) >= self.ctx.max_qty:
                    self.summary.ticks_pinned[venue] += 1
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
        self.summary.duration_ms = self.rows[-1].ts_ms - self.rows[0].ts_ms
        self.summary.max_qty = self.ctx.max_qty
        return self.summary


def write_summary(summary: EngineSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(summary.to_dict(), f, indent=2)
