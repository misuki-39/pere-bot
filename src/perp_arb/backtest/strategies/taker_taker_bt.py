"""Backtest port of the taker-taker arbitrage strategy.

Composes the shared pure decision function `assess_reversion` (fed by
`compute_taker_fills`) with a fresh `SpreadModel` (EWMA bias / scale) and a
per-venue synthetic position. When `assess_reversion` returns `FIRED`, this
strategy issues two `OrderIntent`s
(left + right legs) sharing the same `Decision`; for any other terminal
outcome it emits the Decision directly via `recorder.emit`. None outcomes
(warmup or no-edge ticks) are silently dropped — that's the spread monitor's
job, not ours.
"""

from __future__ import annotations

from ...core.recording.decision import Direction, Phase, Verdict
from ...strategy.base import SpreadModel
from ...strategy.persistence_gate import PersistenceGate
from ...strategy.reversion_signal import (
    AssessInputs,
    AssessParams,
    assess_reversion,
    leg_sides,
)
from ...strategy.taker_fill_model import TakerFillParams, compute_taker_fills
from ..base import BacktestStrategy, EngineView
from ..intents import FillEvent, OrderIntent
from ..snapshot import MarketSnapshot


class TakerTakerBT(BacktestStrategy):
    name = "taker_taker"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._spread = SpreadModel(
            center_half_life_s=ctx.bias_halflife_s,
            scale_half_life_s=ctx.scale_halflife_s,
            warmup_s=ctx.warmup_seconds,
        )
        self._fill_params = TakerFillParams(
            qty=ctx.capture_qty,
            max_levels=1,                      # BBO snapshot → one level
        )
        self._params = AssessParams(
            qty=ctx.capture_qty,
            fees_bps=ctx.fees_bps,             # round-trip, matches live convention
            min_profit_bps=ctx.min_profit_bps,
            max_stale_ms=ctx.max_stale_ms,
            max_qty=ctx.max_qty,
            inventory_skew_bps=ctx.inventory_skew_bps,
            inventory_skew_close_bps=ctx.inventory_skew_close_bps,
        )
        # Per-direction in-flight cap. K=0 disables. K=1 = at most one outstanding
        # entry of that direction at a time; new fires are recorded as
        # BLOCKED_RISK (no intents emitted). State here, not in the engine,
        # because Direction is a strategy concept.
        self._inflight_cap = int(ctx.in_flight_cap_per_direction)
        self._inflight_dir: dict[str, Direction] = {}     # decision_id -> direction
        self._inflight_legs: dict[str, int] = {}          # decision_id -> legs remaining

        # Edge-persistence confirmation gate — a temporal filter applied to the
        # `assess_reversion` output, identical to the live wiring in
        # `strategy/taker_taker.py`. Off (identity pass-through) by default.
        self._gate = PersistenceGate(ctx.persistence)
        self._prev_left_ts: int | None = None
        self._prev_right_ts: int | None = None

    def on_tick(self, snap: MarketSnapshot, view: EngineView) -> list[OrderIntent]:
        # Top-of-book mids; the snapshot is always built one-level, so present.
        mid_l, mid_r = snap.left_book.mid, snap.right_book.mid
        assert mid_l is not None and mid_r is not None
        bias = self._spread.update(mid_l - mid_r, snap.ts_ms).center

        fills = compute_taker_fills(
            self._fill_params, snap.left_book, snap.right_book)
        d = assess_reversion(self._params, AssessInputs(
            now_ms=snap.ts_ms,
            mid_left=mid_l, mid_right=mid_r,
            left_ts_ms=snap.left_ts_ms, right_ts_ms=snap.right_ts_ms,
            fills=fills,
            bias=bias,
            is_warm=self._spread.is_warm,
            position=view.position(self.ctx.left_venue),
        ))

        # Persistence-confirm gate: suppress a FIRED decision until its edge
        # has survived the confirmation window. No-op when disabled.
        left_ticked = snap.left_ts_ms != self._prev_left_ts
        right_ticked = snap.right_ts_ms != self._prev_right_ts
        self._prev_left_ts = snap.left_ts_ms
        self._prev_right_ts = snap.right_ts_ms
        d = self._gate.admit(d, left_ticked=left_ticked, right_ticked=right_ticked)

        if d is None:
            return []
        if d.outcome is not Verdict.FIRED:
            # abort/blocked decisions get recorded immediately — no legs.
            self.ctx.recorder.emit(d)
            return []

        # FIRED → two legs sharing this Decision.
        assert d.direction is not None

        # Per-direction in-flight cap: block if already K same-direction
        # entries pending. Records as BLOCKED_RISK (no intents emitted).
        if self._inflight_cap > 0:
            same_dir_inflight = sum(
                1 for dir_ in self._inflight_dir.values() if dir_ is d.direction
            )
            if same_dir_inflight >= self._inflight_cap:
                d.outcome = Verdict.BLOCKED_RISK
                d.abort_reason = (
                    f"in-flight cap {self._inflight_cap} reached "
                    f"for direction {d.direction.value}"
                )
                self.ctx.recorder.emit(d)
                return []

        d.timeline.mark_at(Phase.DECISION, snap.ts_ms)

        # Track this decision for the in-flight cap.
        if self._inflight_cap > 0:
            self._inflight_dir[d.decision_id] = d.direction
            self._inflight_legs[d.decision_id] = 2
        l_side, r_side = leg_sides(d.direction)
        l_exp = d.vwap_left_sell  if l_side.value == "sell" else d.vwap_left_buy
        r_exp = d.vwap_right_sell if r_side.value == "sell" else d.vwap_right_buy
        intent_left = OrderIntent(
            decision_id=d.decision_id, decision=d,
            venue=self.ctx.left_venue, side=l_side, qty=self.ctx.capture_qty,
            expected_price=l_exp, fill_model=self.ctx.fill_model,
            sim_ts_ms=snap.ts_ms,
        )
        intent_right = OrderIntent(
            decision_id=d.decision_id, decision=d,
            venue=self.ctx.right_venue, side=r_side, qty=self.ctx.capture_qty,
            expected_price=r_exp, fill_model=self.ctx.fill_model,
            sim_ts_ms=snap.ts_ms,
        )
        return [intent_left, intent_right]

    def on_fill(self, fill: FillEvent, view: EngineView) -> None:
        # Engine handles position bookkeeping; strategy only tracks per-direction
        # in-flight count (needed when in_flight_cap_per_direction > 0).
        if self._inflight_cap <= 0:
            return
        did = fill.decision_id
        if did not in self._inflight_legs:
            return
        self._inflight_legs[did] -= 1
        if self._inflight_legs[did] <= 0:
            del self._inflight_legs[did]
            del self._inflight_dir[did]
