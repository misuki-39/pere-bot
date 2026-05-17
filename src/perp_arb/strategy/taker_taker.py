"""TakerTakerArbitrage — depth-aware VWAP edge vs EWMA bias on mid-mid.

Modes:
  * `paper` — full decision math, orders are no-ops that log synthetic fills.
  * `live`  — real `place_market_order` calls on both venues concurrently.

What it bets on:
  An EWMA of (mid_aster - mid_lighter) tracks the long-run inter-venue
  *bias*. The bias itself can be structurally non-zero (different oracles,
  funding rates, USDT-vs-USDC quote currencies). What matters is that the
  current spread oscillates around the bias. The strategy fires when the
  current depth-aware spread deviates from the EWMA bias by more than
  `fees_bps + min_profit_bps`, betting on reversion to bias.

Exit policy is implicit: a reverse-direction entry IS the exit. Direction B
fires with the same `qty` exactly cancel a direction-A position (and vice
versa). The risk manager caps absolute exposure via `max_qty`, so positions
stack up to the cap and unwind naturally when the spread reverts toward bias.

Failure modes (when max-qty position gets stuck):
  * Spread *trends* away from bias without reverting (regime change faster
    than the EWMA can chase) → one-direction entries keep stacking.
  * Spread variance is too low — never exceeds threshold → no signal.
  * `bias_halflife_s` mistuned: too short eats the ~2 s reversion signal
    (center chases its own residual to zero); too long lags the slow
    intraday center wander.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import NamedTuple

from ..core.config import RunMode
from ..core.exec_record import (
    Decision,
    Direction,
    ExecutionRecorder,
    LegKind,
    LegReport,
    Outcome,
    Phase,
)
from ..core.types import OrderResult, Side
from ..risk.manager import RiskManager
from ..utils.precision import BPS, vwap_fill
from ..utils.time import now_ms
from .base import BaseStrategy, SpreadModel

_log = logging.getLogger(__name__)


class _Vwaps(NamedTuple):
    """The four depth-aware fill prices the edge math uses, passed as one
    typed value so Decision construction stays statically checkable."""

    a_sell: Decimal
    a_buy: Decimal
    l_sell: Decimal
    l_buy: Decimal


@dataclass
class SyntheticPosition:
    """Bot-side position tracker (signed). For paper + live both."""

    aster: Decimal = Decimal("0")
    lighter: Decimal = Decimal("0")
    realised_pnl: Decimal = Decimal("0")


class TakerTakerArbitrage(BaseStrategy):
    name = "taker_taker"

    def __init__(self, cfg, exchanges, markets) -> None:
        super().__init__(cfg, exchanges, markets)
        s = cfg.strategy
        self._spread = SpreadModel(
            center_half_life_s=s.bias_halflife_s,
            scale_half_life_s=s.scale_halflife_s,
            warmup_s=s.warmup_seconds,
        )
        self._recorder: ExecutionRecorder | None = None
        self._evaluating = False
        self._position = SyntheticPosition()
        self._risk = RiskManager(s.risk, max_qty=s.max_qty)
        self._last_heartbeat_ms = 0
        self._heartbeat_interval_ms = 60_000  # liveness only; trades go to CSV
        self._risk_blocked = False
        # hoist per-tick constants into Decimals once
        self._slip_cap = s.max_slippage_bps / BPS
        self._fee_frac = s.fees_bps / BPS
        self._min_profit_frac = s.min_profit_bps / BPS

    async def run(self) -> None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self._recorder = ExecutionRecorder(self.cfg.runtime.log_dir, ts)
        _log.info("taker_taker mode=%s recording to %s",
                  self.cfg.strategy.mode, self.cfg.runtime.log_dir)

        self._aster().subscribe_book(self._aster_market(), lambda _b: self._schedule_eval())
        self._lighter().subscribe_book(self._lighter_market(), lambda _b: self._schedule_eval())

        try:
            await self._stop.wait()
        finally:
            if self._recorder:
                self._recorder.close()

    # ---- evaluation ----

    def _schedule_eval(self) -> None:
        if self._evaluating:
            return
        self._evaluating = True
        asyncio.create_task(self._evaluate_once(), name="taker_taker-eval")

    async def _evaluate_once(self) -> None:
        try:
            await self._evaluate()
        finally:
            self._evaluating = False

    async def _evaluate(self) -> None:
        a_book = self._aster().order_book(self._aster_market())
        l_book = self._lighter().order_book(self._lighter_market())
        a_q = self._aster().best_quote(self._aster_market())
        l_q = self._lighter().best_quote(self._lighter_market())
        if a_book is None or l_book is None or a_q is None or l_q is None:
            return

        d = self._assess(a_book, l_book, a_q, l_q)
        if d is None:
            return
        try:
            if d.outcome is Outcome.FIRED:
                await self._fire(d)
        finally:
            if self._recorder:
                self._recorder.emit(d)

    def _assess(self, a_book, l_book, a_q, l_q) -> Decision | None:
        """Pure decision logic: no order placement, no persistence. Returns the
        Decision to record (outcome already terminal), or None for ticks not
        worth recording (not warm, or no tradeable edge — that is the spread
        monitor's job)."""
        s = self.cfg.strategy
        now = now_ms()
        mid_a = a_q.mid
        mid_l = l_q.mid

        def new(
            outcome: Outcome, reason: str | None = None, *,
            bias: Decimal = Decimal(0), edge_bps: Decimal = Decimal(0),
            direction: Direction | None = None, vwaps: _Vwaps | None = None,
        ) -> Decision:
            v = vwaps or _Vwaps(Decimal(0), Decimal(0), Decimal(0), Decimal(0))
            return Decision(
                decision_id=f"d-{uuid.uuid4().hex[:10]}",
                ts_ms=now, mid_a=mid_a, mid_l=mid_l,
                a_quote_ts_ms=a_q.ts_ms, l_quote_ts_ms=l_q.ts_ms,
                bias=bias, vwap_a_sell=v.a_sell, vwap_a_buy=v.a_buy,
                vwap_l_sell=v.l_sell, vwap_l_buy=v.l_buy,
                edge_bps=edge_bps, direction=direction,
                outcome=outcome, abort_reason=reason,
            )

        if (now - max(a_q.ts_ms, l_q.ts_ms)) > s.max_stale_ms:
            return new(Outcome.ABORT_STALE, "quote older than max_stale_ms")

        bias = self._spread.update(mid_a - mid_l, now).center
        if not self._spread.is_warm:
            return None  # warmup: not interesting telemetry

        qty = s.qty
        vwap_a_sell, _ = vwap_fill(a_book.bids, qty, max_levels=s.max_levels)
        vwap_a_buy,  _ = vwap_fill(a_book.asks, qty, max_levels=s.max_levels)
        vwap_l_sell, _ = vwap_fill(l_book.bids, qty, max_levels=s.max_levels)
        vwap_l_buy,  _ = vwap_fill(l_book.asks, qty, max_levels=s.max_levels)
        if None in (vwap_a_sell, vwap_a_buy, vwap_l_sell, vwap_l_buy):
            return new(Outcome.ABORT_NO_DEPTH,
                       "qty does not fill within max_levels", bias=bias)

        vw = _Vwaps(vwap_a_sell, vwap_a_buy, vwap_l_sell, vwap_l_buy)

        slip = self._slip_cap
        if (abs((vwap_a_sell - mid_a) / mid_a) > slip
                or abs((vwap_a_buy - mid_a) / mid_a) > slip
                or abs((vwap_l_sell - mid_l) / mid_l) > slip
                or abs((vwap_l_buy - mid_l) / mid_l) > slip):
            return new(Outcome.ABORT_SLIPPAGE, "vwap-mid exceeds max_slippage_bps",
                       bias=bias, vwaps=vw)

        ref_mid = (mid_a + mid_l) / Decimal(2)
        threshold = ref_mid * (self._fee_frac + self._min_profit_frac)
        edge_A = (vwap_a_sell - vwap_l_buy) - bias - threshold   # sell A, buy L
        edge_B = (vwap_l_sell - vwap_a_buy) + bias - threshold   # reverse

        self._maybe_heartbeat(now, mid_a, mid_l, bias, edge_A, edge_B, ref_mid)

        if edge_A <= 0 and edge_B <= 0:
            return None  # nothing we'd act on

        direction = Direction.A if edge_A >= edge_B else Direction.B
        edge_bps = max(edge_A, edge_B) / ref_mid * BPS

        post_a = self._position.aster + qty * Decimal(self._a_side(direction).sign)
        post_l = self._position.lighter + qty * Decimal(self._l_side(direction).sign)
        ok, reason = self._risk.can_trade(
            post_trade_abs_position=max(abs(post_a), abs(post_l)),
        )
        if not ok:
            if not self._risk_blocked:
                self._risk_blocked = True
                _log.info("entry blocked by risk: %s (suppressing until cleared)", reason)
            return new(Outcome.BLOCKED_RISK, reason, bias=bias,
                       edge_bps=edge_bps, direction=direction, vwaps=vw)

        if self._risk_blocked:
            self._risk_blocked = False
            _log.info("risk block cleared — resuming entries")

        d = new(Outcome.FIRED, bias=bias, edge_bps=edge_bps,
                direction=direction, vwaps=vw)
        d.timeline.mark(Phase.DECISION)  # latency clock starts at decision
        return d

    @staticmethod
    def _a_side(direction: Direction) -> Side:
        return Side.SELL if direction is Direction.A else Side.BUY

    @staticmethod
    def _l_side(direction: Direction) -> Side:
        return Side.BUY if direction is Direction.A else Side.SELL

    def _maybe_heartbeat(
        self,
        now: int,
        mid_a: Decimal, mid_l: Decimal, bias: Decimal,
        edge_A: Decimal, edge_B: Decimal, ref_mid: Decimal,
    ) -> None:
        """Throttled INFO log so paper runs are observable when nothing fires."""
        if now - self._last_heartbeat_ms < self._heartbeat_interval_ms:
            return
        self._last_heartbeat_ms = now
        edge_A_bps = edge_A / ref_mid * BPS
        edge_B_bps = edge_B / ref_mid * BPS
        _log.info(
            "edge: mid_a=%.2f mid_l=%.2f bias=%.4f edge_A=%+.2fbps edge_B=%+.2fbps pos_a=%s pos_l=%s",
            mid_a, mid_l, bias, edge_A_bps, edge_B_bps,
            self._position.aster, self._position.lighter,
        )

    # ---- order firing ----

    async def _fire(self, d: Decision) -> None:
        qty = self.cfg.strategy.qty
        a_side, l_side = self._a_side(d.direction), self._l_side(d.direction)
        if d.direction is Direction.A:
            exp_a, exp_l = d.vwap_a_sell, d.vwap_l_buy
        else:
            exp_a, exp_l = d.vwap_a_buy, d.vwap_l_sell

        _log.info("[%s] FIRE qty=%s aster=%s lighter=%s mode=%s",
                  d.decision_id, qty, a_side, l_side, self.cfg.strategy.mode)

        d.timeline.mark(Phase.SEND)
        if self.cfg.strategy.mode is RunMode.PAPER:
            ra, rl = self._paper_fill(a_side, l_side, qty)
            d.timeline.mark("result_aster")
            d.timeline.mark("result_lighter")
        else:
            async def _leg(coro, venue: str) -> OrderResult:
                r = await coro
                d.timeline.mark(f"result_{venue}")  # this leg's own fill instant
                return r
            ra, rl = await asyncio.gather(
                _leg(self._aster().place_market_order(
                    self._aster_market(), a_side, qty, client_id=f"{d.decision_id}-a"),
                    "aster"),
                _leg(self._lighter().place_market_order(
                    self._lighter_market(), l_side, qty, client_id=f"{d.decision_id}-l"),
                    "lighter"),
            )
        d.timeline.mark(Phase.RESULT)

        lat_a = d.timeline.span(Phase.SEND, "result_aster")
        lat_l = d.timeline.span(Phase.SEND, "result_lighter")
        d.legs = [
            LegReport.from_result("aster", a_side, qty, exp_a, ra, lat_a),
            LegReport.from_result("lighter", l_side, qty, exp_l, rl, lat_l),
        ]
        latency = d.timeline.span(Phase.SEND, Phase.RESULT)

        if ra.success and rl.success:
            self._position.aster += qty * Decimal(a_side.sign)
            self._position.lighter += qty * Decimal(l_side.sign)
            self._risk.record_success(leg_latency_ms=latency or 0)
            _log.info(
                "[%s] FILLED aster=%s lighter=%s latency=%sms pos_a=%s pos_l=%s",
                d.decision_id, ra.order_id, rl.order_id, latency,
                self._position.aster, self._position.lighter,
            )
        else:
            await self._handle_partial_failure(d, a_side, l_side, qty, ra, rl)

    async def _handle_partial_failure(
        self,
        d: Decision,
        a_side: Side,
        l_side: Side,
        qty: Decimal,
        ra: OrderResult,
        rl: OrderResult,
    ) -> None:
        if ra.success and not rl.success:
            _log.error(
                "[%s] PARTIAL: aster filled, lighter failed (%s) — unwinding aster",
                d.decision_id, rl.error_message,
            )
            await self._unwind_leg(d, "aster", a_side.opposite, qty, ra.avg_price)
            self._risk.record_failure(f"lighter leg failed: {rl.error_message}")
        elif rl.success and not ra.success:
            _log.error(
                "[%s] PARTIAL: lighter filled, aster failed (%s) — unwinding lighter",
                d.decision_id, ra.error_message,
            )
            await self._unwind_leg(d, "lighter", l_side.opposite, qty, rl.avg_price)
            self._risk.record_failure(f"aster leg failed: {ra.error_message}")
        else:
            _log.warning(
                "[%s] BOTH FAILED: aster=%s lighter=%s",
                d.decision_id, ra.error_message, rl.error_message,
            )
            self._risk.record_failure("both legs failed")

    async def _unwind_leg(
        self, d: Decision, venue: str, side: Side, qty: Decimal,
        cost_basis: Decimal | None,
    ) -> None:
        """Flatten the stranded leg and record the unwind as a first-class leg
        (kind=unwind). expected_price is the stranded fill we are reversing, so
        the round-trip cost of a partial is directly computable offline."""
        if self.cfg.strategy.mode is RunMode.PAPER:
            return
        ex = self.exchanges[venue]
        mkt = self.markets[venue]
        d.timeline.mark("unwind_send")
        try:
            r = await ex.place_market_order(mkt, side, qty, reduce_only=True)
            if not r.success:
                _log.error("unwind on %s FAILED: %s", venue, r.error_message)
        except Exception as e:  # noqa: BLE001
            _log.exception("unwind on %s raised: %s", venue, e)
            r = OrderResult(success=False, side=side, requested_size=qty,
                             error_message=f"unwind raised: {e}")
        d.timeline.mark("unwind_result")
        d.legs.append(LegReport.from_result(
            venue, side, qty, cost_basis, r,
            d.timeline.span("unwind_send", "unwind_result"), kind=LegKind.UNWIND,
        ))

    def _paper_fill(
        self,
        aster_side: Side,
        lighter_side: Side,
        qty: Decimal,
    ) -> tuple[OrderResult, OrderResult]:
        # use current book VWAP as the synthetic fill price
        a_book = self._aster().order_book(self._aster_market())
        l_book = self._lighter().order_book(self._lighter_market())
        assert a_book is not None and l_book is not None

        a_levels = a_book.bids if aster_side is Side.SELL else a_book.asks
        l_levels = l_book.bids if lighter_side is Side.SELL else l_book.asks
        vwap_a, _ = vwap_fill(a_levels, qty, max_levels=self.cfg.strategy.max_levels)
        vwap_l, _ = vwap_fill(l_levels, qty, max_levels=self.cfg.strategy.max_levels)
        assert vwap_a is not None and vwap_l is not None

        return (
            OrderResult(
                success=True,
                order_id="paper-" + uuid.uuid4().hex[:8],
                side=aster_side,
                requested_size=qty,
                filled_size=qty,
                avg_price=vwap_a,
            ),
            OrderResult(
                success=True,
                order_id="paper-" + uuid.uuid4().hex[:8],
                side=lighter_side,
                requested_size=qty,
                filled_size=qty,
                avg_price=vwap_l,
            ),
        )
