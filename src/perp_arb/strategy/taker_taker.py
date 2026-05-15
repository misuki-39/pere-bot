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
from ..core.logging import TRADES_CSV_HEADER, CsvWriter
from ..core.types import OrderResult, Side
from ..risk.manager import RiskManager
from ..utils.precision import vwap_fill
from ..utils.time import mono_ms, now_ms
from .base import BaseStrategy, SpreadModel

_log = logging.getLogger(__name__)


@dataclass
class LegPair:
    leg_pair_id: str
    ts_ms: int
    aster_side: Side
    lighter_side: Side
    qty: Decimal


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
        self._csv: CsvWriter | None = None
        self._evaluating = False
        self._position = SyntheticPosition()
        self._risk = RiskManager(s.risk, max_qty=s.max_qty)
        self._last_heartbeat_ms = 0
        self._heartbeat_interval_ms = 5000
        # hoist per-tick constants into Decimals once
        self._slip_cap = s.max_slippage_bps / Decimal(10_000)
        self._fee_frac = s.fees_bps / Decimal(10_000)
        self._min_profit_frac = s.min_profit_bps / Decimal(10_000)

    async def run(self) -> None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        csv_path = self.cfg.runtime.log_dir / f"trades_taker_taker_{ts}.csv"
        self._csv = CsvWriter(csv_path, TRADES_CSV_HEADER)
        _log.info("taker_taker mode=%s writing %s", self.cfg.strategy.mode, csv_path)

        self._aster().subscribe_book(self._aster_market(), lambda _b: self._schedule_eval())
        self._lighter().subscribe_book(self._lighter_market(), lambda _b: self._schedule_eval())

        try:
            await self._stop.wait()
        finally:
            if self._csv:
                self._csv.close()

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
        s = self.cfg.strategy
        a_book = self._aster().order_book(self._aster_market())
        l_book = self._lighter().order_book(self._lighter_market())
        a_q = self._aster().best_quote(self._aster_market())
        l_q = self._lighter().best_quote(self._lighter_market())
        if a_book is None or l_book is None or a_q is None or l_q is None:
            return

        now = now_ms()
        if (now - max(a_q.ts_ms, l_q.ts_ms)) > s.max_stale_ms:
            return

        mid_a = a_q.mid
        mid_l = l_q.mid
        st = self._spread.update(mid_a - mid_l, now)
        bias = st.center

        if not self._spread.is_warm:
            return

        qty = s.qty
        vwap_a_sell, _ = vwap_fill(a_book.bids, qty, max_levels=s.max_levels)
        vwap_a_buy,  _ = vwap_fill(a_book.asks, qty, max_levels=s.max_levels)
        vwap_l_sell, _ = vwap_fill(l_book.bids, qty, max_levels=s.max_levels)
        vwap_l_buy,  _ = vwap_fill(l_book.asks, qty, max_levels=s.max_levels)
        if None in (vwap_a_sell, vwap_a_buy, vwap_l_sell, vwap_l_buy):
            return

        slip = self._slip_cap
        if (abs((vwap_a_sell - mid_a) / mid_a) > slip
                or abs((vwap_a_buy - mid_a) / mid_a) > slip
                or abs((vwap_l_sell - mid_l) / mid_l) > slip
                or abs((vwap_l_buy - mid_l) / mid_l) > slip):
            return

        ref_mid = (mid_a + mid_l) / Decimal(2)
        threshold = ref_mid * (self._fee_frac + self._min_profit_frac)

        # direction A: sell aster, buy lighter; direction B: the reverse
        edge_A = (vwap_a_sell - vwap_l_buy) - bias - threshold
        edge_B = (vwap_l_sell - vwap_a_buy) + bias - threshold

        self._maybe_heartbeat(now, mid_a, mid_l, bias, edge_A, edge_B, ref_mid)

        if edge_A <= 0 and edge_B <= 0:
            return

        if edge_A >= edge_B:
            aster_side, lighter_side = Side.SELL, Side.BUY
        else:
            aster_side, lighter_side = Side.BUY, Side.SELL

        post_aster = self._position.aster + qty * Decimal(aster_side.sign)
        post_lighter = self._position.lighter + qty * Decimal(lighter_side.sign)
        ok, reason = self._risk.can_trade(
            post_trade_abs_position=max(abs(post_aster), abs(post_lighter)),
        )
        if not ok:
            _log.info("entry blocked by risk: %s", reason)
            return

        await self._fire(aster_side, lighter_side, qty)

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
        edge_A_bps = edge_A / ref_mid * Decimal(10_000)
        edge_B_bps = edge_B / ref_mid * Decimal(10_000)
        _log.info(
            "edge: mid_a=%.2f mid_l=%.2f bias=%.4f edge_A=%+.2fbps edge_B=%+.2fbps pos_a=%s pos_l=%s",
            mid_a, mid_l, bias, edge_A_bps, edge_B_bps,
            self._position.aster, self._position.lighter,
        )

    # ---- order firing ----

    async def _fire(self, aster_side: Side, lighter_side: Side, qty: Decimal) -> None:
        leg_pair_id = f"lp-{uuid.uuid4().hex[:10]}"
        t0 = mono_ms()
        _log.info(
            "[%s] FIRE qty=%s aster=%s lighter=%s mode=%s",
            leg_pair_id, qty, aster_side, lighter_side, self.cfg.strategy.mode,
        )

        if self.cfg.strategy.mode is RunMode.PAPER:
            ra, rl = self._paper_fill(aster_side, lighter_side, qty)
        else:
            ra_task = asyncio.create_task(self._aster().place_market_order(
                self._aster_market(), aster_side, qty,
                client_id=f"{leg_pair_id}-a",
            ))
            rl_task = asyncio.create_task(self._lighter().place_market_order(
                self._lighter_market(), lighter_side, qty,
                client_id=f"{leg_pair_id}-l",
            ))
            ra, rl = await asyncio.gather(ra_task, rl_task)

        leg_latency = mono_ms() - t0
        pair = LegPair(
            leg_pair_id=leg_pair_id,
            ts_ms=now_ms(),
            aster_side=aster_side,
            lighter_side=lighter_side,
            qty=qty,
        )

        if ra.success and rl.success:
            self._position.aster += qty * Decimal(aster_side.sign)
            self._position.lighter += qty * Decimal(lighter_side.sign)
            self._risk.record_success(leg_latency_ms=leg_latency)
            self._log_fill(pair, "aster", ra)
            self._log_fill(pair, "lighter", rl)
            _log.info(
                "[%s] FILLED aster=%s lighter=%s latency=%dms pos_a=%s pos_l=%s",
                leg_pair_id, ra.order_id, rl.order_id, leg_latency,
                self._position.aster, self._position.lighter,
            )
        else:
            await self._handle_partial_failure(pair, ra, rl, leg_latency)

    async def _handle_partial_failure(
        self,
        pair: LegPair,
        ra: OrderResult,
        rl: OrderResult,
        leg_latency: int,
    ) -> None:
        if ra.success and not rl.success:
            _log.error(
                "[%s] PARTIAL: aster filled, lighter failed (%s) — unwinding aster",
                pair.leg_pair_id, rl.error_message,
            )
            await self._emergency_unwind("aster", pair.aster_side.opposite, pair.qty)
            self._risk.record_failure(f"lighter leg failed: {rl.error_message}")
        elif rl.success and not ra.success:
            _log.error(
                "[%s] PARTIAL: lighter filled, aster failed (%s) — unwinding lighter",
                pair.leg_pair_id, ra.error_message,
            )
            await self._emergency_unwind("lighter", pair.lighter_side.opposite, pair.qty)
            self._risk.record_failure(f"aster leg failed: {ra.error_message}")
        else:
            _log.warning(
                "[%s] BOTH FAILED: aster=%s lighter=%s",
                pair.leg_pair_id, ra.error_message, rl.error_message,
            )
            self._risk.record_failure("both legs failed")

    async def _emergency_unwind(self, venue: str, side: Side, qty: Decimal) -> None:
        if self.cfg.strategy.mode is RunMode.PAPER:
            return
        ex = self.exchanges[venue]
        mkt = self.markets[venue]
        try:
            r = await ex.place_market_order(mkt, side, qty, reduce_only=True)
            if not r.success:
                _log.error("unwind on %s FAILED: %s", venue, r.error_message)
        except Exception as e:  # noqa: BLE001
            _log.exception("unwind on %s raised: %s", venue, e)

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

    def _log_fill(self, pair: LegPair, venue: str, r: OrderResult) -> None:
        if self._csv is None:
            return
        market = self.markets[venue]
        self._csv.write([
            pair.ts_ms,
            pair.leg_pair_id,
            venue,
            market.symbol.raw,
            r.side.value if r.side else "",
            r.filled_size or pair.qty,
            r.avg_price,
            None,           # fee — fill in once we have a fee accounting source
            r.status.value if r.status else "",
            r.order_id,
            r.client_id,
            None,           # realised_pnl — populated on close
        ])
