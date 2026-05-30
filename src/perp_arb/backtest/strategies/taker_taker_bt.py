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

from decimal import Decimal

from ...core.decision import Direction, Phase, Verdict
from ...strategy.base import SpreadModel, TimeEwma
from ...strategy.persistence_gate import PersistenceGate
from ...strategy.reversion_signal import (
    AssessInputs,
    AssessParams,
    assess_reversion,
    left_side,
    right_side,
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
        # Same-direction throttle: each FIRED on direction X bumps that
        # direction's threshold by `throttle_bump_bps`; the bump decays back
        # toward 0 with `throttle_halflife_s` half-life. Δ=0 = throttle off.
        self._throttle_enabled = ctx.throttle_bump_bps > 0
        self._bump_a = TimeEwma(half_life_s=ctx.throttle_halflife_s)
        self._bump_b = TimeEwma(half_life_s=ctx.throttle_halflife_s)
        self._throttle_bump_bps = ctx.throttle_bump_bps
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
        bias = self._spread.update(snap.left_quote.mid - snap.right_quote.mid, snap.ts_ms).center

        # Decay current bumps to "now" before reading them (TimeEwma.update with
        # the current value acts as a pure time-decay step — alpha applies to
        # the new sample which equals the current value, so output = current
        # * (1 - alpha) + current * alpha = current; the side-effect is the
        # advancement of _last_ts_ms so the *next* update sees the right dt).
        bump_a = self._bump_a.value if self._bump_a.value is not None else Decimal(0)
        bump_b = self._bump_b.value if self._bump_b.value is not None else Decimal(0)
        if self._throttle_enabled and self._bump_a.value is not None:
            bump_a = self._bump_a.update(Decimal(0), snap.ts_ms)  # decay toward 0
            bump_b = self._bump_b.update(Decimal(0), snap.ts_ms)

        fills = compute_taker_fills(
            self._fill_params, snap.left_book, snap.right_book)
        d = assess_reversion(self._params, AssessInputs(
            now_ms=snap.ts_ms,
            left_quote=snap.left_quote,
            right_quote=snap.right_quote,
            fills=fills,
            bias=bias,
            is_warm=self._spread.is_warm,
            position=view.position(self.ctx.left_venue),
            bump_a_bps=bump_a if self._throttle_enabled else Decimal(0),
            bump_b_bps=bump_b if self._throttle_enabled else Decimal(0),
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

        # Same-direction throttle: bump this direction's threshold; let it
        # decay over `throttle_halflife_s` (handled at top-of-tick).
        if self._throttle_enabled:
            target = self._bump_a if d.direction is Direction.A else self._bump_b
            target.bump(self._throttle_bump_bps, snap.ts_ms)

        # Track this decision for the in-flight cap.
        if self._inflight_cap > 0:
            self._inflight_dir[d.decision_id] = d.direction
            self._inflight_legs[d.decision_id] = 2
        l_side = left_side(d.direction)
        r_side = right_side(d.direction)
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
