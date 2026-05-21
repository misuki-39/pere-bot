"""Backtest port of the taker-taker arbitrage strategy.

Composes the shared pure decision function `assess_taker_taker` with a fresh
`SpreadModel` (EWMA bias / scale) and a per-venue synthetic position. When
`assess_taker_taker` returns `FIRED`, this strategy issues two `OrderIntent`s
(left + right legs) sharing the same `Decision`; for any other terminal
outcome it emits the Decision directly via `recorder.emit`. None outcomes
(warmup or no-edge ticks) are silently dropped — that's the spread monitor's
job, not ours.
"""

from __future__ import annotations

from ...core.exec_record import Outcome, Phase
from ...strategy.base import SpreadModel
from ...strategy.taker_taker_core import (
    AssessInputs,
    AssessParams,
    assess_taker_taker,
    left_side,
    right_side,
)
from ..intents import FillEvent, OrderIntent
from ..snapshot import MarketSnapshot
from ..strategy import BacktestStrategy, EngineView


class TakerTakerBT(BacktestStrategy):
    name = "taker_taker"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._spread = SpreadModel(
            center_half_life_s=ctx.bias_halflife_s,
            scale_half_life_s=ctx.scale_halflife_s,
            warmup_s=ctx.warmup_seconds,
        )
        self._params = AssessParams(
            qty=ctx.capture_qty,
            max_levels=1,                      # BBO snapshot → one level
            fees_bps=ctx.fees_bps,             # round-trip, matches live convention
            min_profit_bps=ctx.min_profit_bps,
            max_slippage_bps=ctx.max_slippage_bps,
            max_stale_ms=ctx.max_stale_ms,
            max_qty=ctx.max_qty,
        )

    def on_tick(self, snap: MarketSnapshot, view: EngineView) -> list[OrderIntent]:
        bias = self._spread.update(snap.left_quote.mid - snap.right_quote.mid, snap.ts_ms).center
        d = assess_taker_taker(self._params, AssessInputs(
            now_ms=snap.ts_ms,
            left_book=snap.left_book,
            right_book=snap.right_book,
            left_quote=snap.left_quote,
            right_quote=snap.right_quote,
            bias=bias,
            is_warm=self._spread.is_warm,
            position_left=view.position(self.ctx.left_venue),
            position_right=view.position(self.ctx.right_venue),
        ))
        if d is None:
            return []
        if d.outcome is not Outcome.FIRED:
            # abort/blocked decisions get recorded immediately — no legs.
            self.ctx.recorder.emit(d)
            return []

        # FIRED → two legs sharing this Decision.
        assert d.direction is not None
        d.timeline.mark_at(Phase.DECISION, snap.ts_ms)
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
        # Engine handles position bookkeeping; nothing strategy-specific to do.
        return
