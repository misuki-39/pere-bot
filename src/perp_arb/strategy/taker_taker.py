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
from ..core.types import FillDelta, OrderResult, OrderSnapshot, OrderStatus, Side
from ..risk.manager import RiskManager
from ..utils.precision import vwap_fill
from ..utils.time import now_ms
from .base import BaseStrategy, SpreadModel, TimeEwma
from .markout import MarkoutTable
from .taker_taker_core import (
    AssessInputs,
    AssessParams,
    assess_taker_taker,
    left_side,
    right_side,
)

_log = logging.getLogger(__name__)


@dataclass
class SyntheticPosition:
    """Bot-side position tracker (signed). For paper + live both."""

    aster: Decimal = Decimal("0")
    lighter: Decimal = Decimal("0")
    realised_pnl: Decimal = Decimal("0")


@dataclass(slots=True)
class _FillAccumulator:
    """Per-`client_id` aggregate of two event types:

      * `OrderSnapshot` — cumulative running totals (lighter
        `account_market.orders`); accumulator OVERWRITES filled_qty + price.
      * `FillDelta` — per-fill event (aster `l`/`L`); accumulator
        ACCUMULATES qty * price.

    `last_status` is taken from any terminal-status signal so `is_complete`
    can short-circuit the wait the moment the venue confirms the parent
    order is settled (FILLED / CANCELED / REJECTED / EXPIRED)."""

    filled_qty: Decimal = Decimal("0")
    weighted_price_sum: Decimal = Decimal("0")
    last_ts_ms: int = 0
    last_status: OrderStatus | None = None

    def add(self, event: OrderSnapshot | FillDelta) -> None:
        if event.ts_ms:
            self.last_ts_ms = max(self.last_ts_ms, event.ts_ms)
        match event:
            case OrderSnapshot():
                if event.status.terminal:
                    self.last_status = event.status
                if event.filled_size > 0:
                    self.filled_qty = event.filled_size
                    if event.avg_fill_price is not None:
                        self.weighted_price_sum = event.filled_size * event.avg_fill_price
            case FillDelta():
                if event.terminal_status is not None:
                    self.last_status = event.terminal_status
                # FillDelta adapter invariant: qty > 0 (non-fills dropped at source).
                self.filled_qty += event.qty
                self.weighted_price_sum += event.qty * event.price

    def is_complete(self, requested_qty: Decimal) -> bool:
        # Terminal status = no more fills coming (filled / canceled /
        # rejected / expired) → stop waiting. Otherwise fall back to qty
        # comparison for the trade-only path (account_orders unsubscribed
        # or lagging).
        if self.last_status is not None and self.last_status.terminal:
            return True
        return self.filled_qty >= requested_qty


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
        # parameters consumed by the pure decision function.
        # Wave-1 optimisations (markout, throttle, cap) come from
        # `s.optimisations` — defaults are all "off" so legacy configs that
        # omit the block keep their pre-Wave-1 behaviour.
        opt = s.optimisations
        markout = (
            MarkoutTable.from_json(opt.markout_table_path)
            if opt.markout_table_path is not None
            else MarkoutTable.disabled()
        )
        self._params = AssessParams(
            qty=Decimal(str(s.qty)),
            max_levels=s.max_levels,
            fees_bps=Decimal(str(s.fees_bps)),
            min_profit_bps=Decimal(str(s.min_profit_bps)),
            max_slippage_bps=Decimal(str(s.max_slippage_bps)),
            max_stale_ms=s.max_stale_ms,
            max_qty=Decimal(str(s.max_qty)),
            markout=markout,
            # inventory_skew_bps intentionally left at default 0; not
            # exposed in OptimisationsCfg until rolling markout calibration
            # is stable.
        )

        # Same-direction throttle: each FILLED on direction X bumps that
        # direction's threshold by `throttle_bump_bps`; the bump decays
        # back toward 0 with `throttle_halflife_s` half-life. Δ=0 disables.
        self._throttle_enabled = opt.throttle_bump_bps > 0
        self._throttle_bump_bps = Decimal(str(opt.throttle_bump_bps))
        self._bump_a = TimeEwma(half_life_s=opt.throttle_halflife_s)
        self._bump_b = TimeEwma(half_life_s=opt.throttle_halflife_s)

        # Per-direction in-flight cap. K=0 disables. K=1 = at most one
        # outstanding entry of that direction at a time.
        # NOTE: live's `_evaluating` gate in `_schedule_eval` already
        # serializes evaluation, so `_inflight_dir` is in practice always
        # empty by the time `_assess` runs. The cap is wired for parity
        # with the backtest path and as a forward-safety belt if the gate
        # is ever relaxed.
        self._inflight_cap = int(opt.in_flight_cap_per_direction)
        self._inflight_dir: dict[str, Direction] = {}

        # Per-`client_id` fill-event slots. Pre-registered in `_fire` BEFORE
        # the REST submit, so a fast aster fill can't race past the
        # registration and be dropped by `_on_fill`'s unknown-cid guard.
        # Lighter's REST returns submit-ack only, so the WS fill event is
        # the ONLY source of realized_price + fill_ts_ms for that leg.
        self._fill_events: dict[str, asyncio.Event] = {}
        self._fill_acc: dict[str, _FillAccumulator] = {}
        # 5 s comfortably absorbs lighter sequencer's long tail (typical <1 s).
        self._fill_wait_timeout_s = 5.0

        if markout.direction_A or markout.direction_B:
            _log.info(
                "markout enabled: %s",
                markout.latency_label or "(unlabelled)",
            )
        if self._throttle_enabled:
            _log.info(
                "same-side throttle enabled: bump=%s bps halflife=%ss",
                self._throttle_bump_bps, opt.throttle_halflife_s,
            )
        if self._inflight_cap > 0:
            _log.info(
                "in-flight cap enabled K=%d (note: structural no-op in "
                "live due to _evaluating serialization)",
                self._inflight_cap,
            )

    async def run(self) -> None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self._recorder = ExecutionRecorder(self.cfg.runtime.log_dir, ts)
        _log.info("taker_taker mode=%s recording to %s",
                  self.cfg.strategy.mode, self.cfg.runtime.log_dir)

        self._aster().subscribe_book(self._aster_market(), lambda _b: self._schedule_eval())
        self._lighter().subscribe_book(self._lighter_market(), lambda _b: self._schedule_eval())

        if self.cfg.strategy.mode is RunMode.LIVE:
            self._aster().subscribe_fills(self._aster_market(), self._on_fill)
            self._lighter().subscribe_fills(self._lighter_market(), self._on_fill)

        try:
            await self._stop.wait()
        finally:
            if self._recorder:
                self._recorder.close()

    # ---- fill event handling ----

    def _on_fill(self, event: OrderSnapshot | FillDelta) -> None:
        # Drop fills for cids we didn't pre-register (stale orders, other
        # sessions on the same account).
        cid = event.client_id
        if not cid or cid not in self._fill_events:
            return
        self._fill_acc.setdefault(cid, _FillAccumulator()).add(event)
        self._fill_events[cid].set()

    async def _await_fill(
        self, client_id: str, requested_qty: Decimal,
    ) -> _FillAccumulator | None:
        """Wait up to `_fill_wait_timeout_s` for accumulated fills to reach
        `requested_qty`. Returns whatever landed (None if no events at all)
        and unconditionally releases the slot."""
        ev = self._fill_events.get(client_id)
        if ev is None:
            return None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._fill_wait_timeout_s
        try:
            while (remaining := deadline - loop.time()) > 0:
                try:
                    await asyncio.wait_for(ev.wait(), timeout=remaining)
                except TimeoutError:
                    break
                acc = self._fill_acc.get(client_id)
                if acc and acc.is_complete(requested_qty):
                    return acc
                ev.clear()
            return self._fill_acc.get(client_id)
        finally:
            self._fill_events.pop(client_id, None)
            self._fill_acc.pop(client_id, None)

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
        """Thin wrapper around the pure `assess_taker_taker` decision math.

        Live-only responsibilities kept here: update the EWMA, run the
        operational `RiskManager` gates (halted / consecutive-failures / daily
        PnL — the pure function only checks the structural `max_qty` cap), and
        emit the throttled heartbeat log."""
        now = now_ms()
        bias = self._spread.update(a_q.mid - l_q.mid, now).center

        # Same-side throttle bumps: decay to "now" before reading. Calling
        # `update` with the current value advances the internal timestamp
        # without altering the value, so a follow-up `update(0, now+dt)` in
        # a later tick decays correctly. We pass 0 here as the new sample
        # to push the EWMA toward 0 over the halflife.
        bump_a = Decimal(0)
        bump_b = Decimal(0)
        if self._throttle_enabled and self._bump_a.value is not None:
            bump_a = self._bump_a.update(Decimal(0), now)
            bump_b = self._bump_b.update(Decimal(0), now)

        d = assess_taker_taker(self._params, AssessInputs(
            now_ms=now,
            left_book=a_book, right_book=l_book,
            left_quote=a_q, right_quote=l_q,
            bias=bias, is_warm=self._spread.is_warm,
            position_left=self._position.aster,
            position_right=self._position.lighter,
            bump_a_bps=bump_a,
            bump_b_bps=bump_b,
        ))

        if d is not None and d.outcome is Outcome.FIRED:
            assert d.direction is not None

            # In-flight cap: block if K same-direction entries are already
            # pending. Live's `_evaluating` gate already serializes
            # evaluation so this is in practice unreachable, but it
            # mirrors the backtest contract and protects against future
            # async relaxation. Records BLOCKED_RISK with a distinct
            # `abort_reason` so the cap can be distinguished from
            # RiskManager blocks in logs.
            if self._inflight_cap > 0:
                same_dir = sum(
                    1 for x in self._inflight_dir.values() if x is d.direction
                )
                if same_dir >= self._inflight_cap:
                    d.outcome = Outcome.BLOCKED_RISK
                    d.abort_reason = (
                        f"in-flight cap {self._inflight_cap} reached for "
                        f"direction {d.direction.value}"
                    )

        # The RiskManager gate runs on the (possibly already-blocked-by-cap)
        # decision. If the cap already rewrote to BLOCKED_RISK we still let
        # RiskManager observe the FIRED-intent state via the position math —
        # but we must not overwrite the cap's reason. Guard with the outcome.
        if d is not None and d.outcome is Outcome.FIRED:
            d.timeline.mark(Phase.DECISION)
            qty = self.cfg.strategy.qty
            post_a = self._position.aster + qty * Decimal(left_side(d.direction).sign)
            post_l = self._position.lighter + qty * Decimal(right_side(d.direction).sign)
            ok, reason = self._risk.can_trade(
                post_trade_abs_position=max(abs(post_a), abs(post_l)),
            )
            if not ok:
                if not self._risk_blocked:
                    self._risk_blocked = True
                    _log.info("entry blocked by risk: %s (suppressing until cleared)", reason)
                d.outcome = Outcome.BLOCKED_RISK
                d.abort_reason = reason
            elif self._risk_blocked:
                self._risk_blocked = False
                _log.info("risk block cleared — resuming entries")

        self._maybe_heartbeat(now, a_q.mid, l_q.mid, bias, d)
        return d

    def _maybe_heartbeat(
        self,
        now: int,
        mid_left: Decimal, mid_right: Decimal, bias: Decimal,
        d: Decision | None,
    ) -> None:
        """Throttled INFO log so paper runs are observable when nothing fires.

        When `d` has VWAPs (any post-warmup non-stale tick), log both raw
        edge directions in bps — operators rely on seeing whether the *other*
        direction is approaching threshold even when this one isn't firing.
        """
        if now - self._last_heartbeat_ms < self._heartbeat_interval_ms:
            return
        self._last_heartbeat_ms = now
        if d is not None and d.vwap_left_sell > 0:
            ref_mid = (mid_left + mid_right) / Decimal(2)
            edge_A = (d.vwap_left_sell  - d.vwap_right_buy) - bias
            edge_B = (d.vwap_right_sell - d.vwap_left_buy)  + bias
            edge_str = (f"edge_A={edge_A / ref_mid * Decimal(10_000):+.2f}bps "
                        f"edge_B={edge_B / ref_mid * Decimal(10_000):+.2f}bps")
        else:
            edge_str = "no-edge"
        _log.info(
            "tick: mid_left=%.2f mid_right=%.2f bias=%.4f %s pos_a=%s pos_l=%s",
            mid_left, mid_right, bias, edge_str,
            self._position.aster, self._position.lighter,
        )

    # ---- order firing ----

    async def _fire(self, d: Decision) -> None:
        assert d.direction is not None
        qty = self.cfg.strategy.qty
        a_side, l_side = left_side(d.direction), right_side(d.direction)
        if d.direction is Direction.A:
            exp_a, exp_l = d.vwap_left_sell, d.vwap_right_buy
        else:
            exp_a, exp_l = d.vwap_left_buy, d.vwap_right_sell

        _log.info("[%s] FIRE qty=%s aster=%s lighter=%s mode=%s",
                  d.decision_id, qty, a_side, l_side, self.cfg.strategy.mode)

        if self._inflight_cap > 0:
            self._inflight_dir[d.decision_id] = d.direction

        aster_cid = f"{d.decision_id}-a"
        lighter_cid = f"{d.decision_id}-l"
        is_live = self.cfg.strategy.mode is RunMode.LIVE

        try:
            if is_live:
                # Register fill slots BEFORE submit so a fast aster fill
                # can't race past `_on_fill`'s unknown-cid guard.
                self._fill_events[aster_cid] = asyncio.Event()
                self._fill_events[lighter_cid] = asyncio.Event()

            d.timeline.mark(Phase.SEND)
            d.send_ts_ms = now_ms()
            if self.cfg.strategy.mode is RunMode.PAPER:
                ra, rl = self._paper_fill(a_side, l_side, qty)
                d.timeline.mark("result_aster")
                d.timeline.mark("result_lighter")
                aster_fill: _FillAccumulator | None = None
                lighter_fill: _FillAccumulator | None = None
            else:
                async def _submit_and_await(
                    place_coro, venue: str, cid: str,
                ) -> tuple[OrderResult, _FillAccumulator | None]:
                    r = await place_coro
                    d.timeline.mark(f"result_{venue}")
                    fill = await self._await_fill(cid, qty) if r.success else None
                    return r, fill

                (ra, aster_fill), (rl, lighter_fill) = await asyncio.gather(
                    _submit_and_await(self._aster().place_market_order(
                        self._aster_market(), a_side, qty, client_id=aster_cid),
                        "aster", aster_cid),
                    _submit_and_await(self._lighter().place_market_order(
                        self._lighter_market(), l_side, qty, client_id=lighter_cid),
                        "lighter", lighter_cid),
                )
            d.timeline.mark(Phase.RESULT)

            lat_a = d.timeline.span(Phase.SEND, "result_aster")
            lat_l = d.timeline.span(Phase.SEND, "result_lighter")
            d.legs = [
                LegReport.build(venue="aster", side=a_side, qty=qty, expected=exp_a,
                                rest=ra, fill=aster_fill, latency_ms=lat_a),
                LegReport.build(venue="lighter", side=l_side, qty=qty, expected=exp_l,
                                rest=rl, fill=lighter_fill, latency_ms=lat_l),
            ]
            latency = d.timeline.span(Phase.SEND, Phase.RESULT)

            if ra.success and rl.success:
                self._position.aster += qty * Decimal(a_side.sign)
                self._position.lighter += qty * Decimal(l_side.sign)
                self._risk.record_success(leg_latency_ms=latency or 0)

                # Seed throttle bump only on confirmed FILLED — a partial /
                # both-fail consumed no edge.
                if self._throttle_enabled:
                    target = self._bump_a if d.direction is Direction.A else self._bump_b
                    target.bump(self._throttle_bump_bps, now_ms())

                _log.info(
                    "[%s] FILLED aster=%s lighter=%s latency=%sms pos_a=%s pos_l=%s",
                    d.decision_id, ra.order_id, rl.order_id, latency,
                    self._position.aster, self._position.lighter,
                )
            else:
                await self._handle_partial_failure(d, a_side, l_side, qty, ra, rl)
        finally:
            if self._inflight_cap > 0:
                self._inflight_dir.pop(d.decision_id, None)
            # `_await_fill` cleans up on the happy path; this covers the
            # short window where Event() registered but REST never reached
            # `_await_fill` (e.g. failed REST + exception before gather).
            self._fill_events.pop(aster_cid, None)
            self._fill_acc.pop(aster_cid, None)
            self._fill_events.pop(lighter_cid, None)
            self._fill_acc.pop(lighter_cid, None)

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
        d.legs.append(LegReport.build(
            venue=venue, side=side, qty=qty, expected=cost_basis,
            rest=r, latency_ms=d.timeline.span("unwind_send", "unwind_result"),
            kind=LegKind.UNWIND,
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
