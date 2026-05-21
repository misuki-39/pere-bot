"""Two-leg execution: gathers REST submits + WS fills on both venues,
unwinds the stranded leg on partial failure, returns a `TradeReport`.

Layering: the executor sits between **strategy** (signal — produces
decisions and resolves them into venue-side intents) and **driver**
(per-venue REST + WS + cid-keyed fill tracking via `BaseExchange`). It
is strategy-agnostic — it never sees `Direction`, `vwap_*`, or any
strategy-internal field. The caller hands it concrete `LegIntent`s
(venue, side, expected_price, client_id) and a `Timeline` writeback
target; the executor handles `gather`, partial-failure unwind,
paper-fill synth, and `LegReport` assembly.

Paper mode is a single branch inside `execute()` rather than a separate
`PaperExecutor` because paper and live share 100% of the orchestration
scaffold and differ only in *where the leaf fill price comes from*.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from decimal import Decimal

from ..utils.precision import vwap_fill
from ..utils.time import now_ms
from .exchange import BaseExchange
from .exec_record import LegKind, LegReport, Phase, Timeline
from .types import MarketInfo, OrderResult, Side, TerminalFill

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LegIntent:
    """One venue leg of an execution intent. The caller has already
    resolved direction → side and decided the cid namespace; the
    executor never needs to know how either was derived."""

    venue: str               # e.g. "aster" / "lighter" — must key into the executor's `exchanges` dict
    side: Side
    expected_price: Decimal  # decision-time VWAP / cost basis — informational, passed through to LegReport
    client_id: str           # caller-owned namespace; executor pre-registers and releases the tracker slot


@dataclass
class TradeReport:
    """Result the executor returns to the caller.

    `success=True` iff both entry legs' REST submits returned success.
    Per-leg WS fill data lives on `legs[*]` via `LegReport.build`; legs
    without a WS event fall back to REST values.

    A partial-failure unwind appears as an additional leg with
    `kind=UNWIND`. Both-fail leaves `success` False with a populated
    `failure_reason`. `send_ts_ms` is the wall-clock moment the SEND
    timeline mark was stamped — caller copies it into its own record."""

    legs: list[LegReport] = field(default_factory=list)
    success: bool = False
    failure_reason: str | None = None
    latency_ms: int | None = None
    send_ts_ms: int = 0


class TwoLegExecutor:
    """Maps `(trade_id, legs, qty, timeline)` → `TradeReport`.

    Stateless across calls. Holds the venue handles and paper-mode flag."""

    def __init__(
        self,
        exchanges: dict[str, BaseExchange],
        markets: dict[str, MarketInfo],
        *,
        is_paper: bool,
        max_levels: int,
        fill_wait_timeout_s: float = 5.0,
    ) -> None:
        self.exchanges = exchanges
        self.markets = markets
        self.is_paper = is_paper
        self.max_levels = max_levels
        self.fill_wait_timeout_s = fill_wait_timeout_s

    async def execute(
        self,
        *,
        trade_id: str,
        legs: tuple[LegIntent, LegIntent],
        qty: Decimal,
        timeline: Timeline,
    ) -> TradeReport:
        a_intent, b_intent = legs

        timeline.mark(Phase.SEND)
        send_ts_ms = now_ms()

        if self.is_paper:
            ra = self._paper_fill_one(a_intent, qty)
            rb = self._paper_fill_one(b_intent, qty)
            timeline.mark(f"result_{a_intent.venue}")
            timeline.mark(f"result_{b_intent.venue}")
            a_fill: TerminalFill | None = None
            b_fill: TerminalFill | None = None
        else:
            (ra, a_fill), (rb, b_fill) = await asyncio.gather(
                self._submit_leg(timeline, a_intent, qty),
                self._submit_leg(timeline, b_intent, qty),
            )
        timeline.mark(Phase.RESULT)

        legs_out = [
            LegReport.build(
                venue=a_intent.venue, side=a_intent.side, qty=qty,
                expected=a_intent.expected_price, rest=ra, fill=a_fill,
                latency_ms=timeline.span(Phase.SEND, f"result_{a_intent.venue}"),
            ),
            LegReport.build(
                venue=b_intent.venue, side=b_intent.side, qty=qty,
                expected=b_intent.expected_price, rest=rb, fill=b_fill,
                latency_ms=timeline.span(Phase.SEND, f"result_{b_intent.venue}"),
            ),
        ]
        latency = timeline.span(Phase.SEND, Phase.RESULT)
        report = TradeReport(
            legs=legs_out, latency_ms=latency, send_ts_ms=send_ts_ms,
        )

        if ra.success and rb.success:
            report.success = True
            _log.info("[%s] FILLED %s=%s %s=%s latency=%sms",
                      trade_id,
                      a_intent.venue, ra.order_id,
                      b_intent.venue, rb.order_id, latency)
            return report

        await self._handle_partial_failure(
            trade_id, timeline, a_intent, b_intent, qty, ra, rb, report,
        )
        return report

    # ----- one-leg submit (live only) -----

    async def _submit_leg(
        self, timeline: Timeline, leg: LegIntent, qty: Decimal,
    ) -> tuple[OrderResult, TerminalFill | None]:
        """REST submit + WS fill await for one leg.

        Per-leg timeline mark `result_{venue}` lands the moment REST
        returns — independent of when (or whether) the WS terminal fill
        resolves. That preserves per-leg inter-venue latency semantics."""
        ex = self.exchanges[leg.venue]
        mkt = self.markets[leg.venue]
        ex.register_fill_slot(leg.client_id)
        try:
            rest = await ex.place_market_order(
                mkt, leg.side, qty, client_id=leg.client_id,
            )
            timeline.mark(f"result_{leg.venue}")
            if not rest.success:
                return rest, None
            fill = await ex.await_fill(
                leg.client_id, qty, self.fill_wait_timeout_s,
            )
            return rest, fill
        finally:
            ex.release_fill_slot(leg.client_id)

    # ----- partial-failure recovery -----

    async def _handle_partial_failure(
        self,
        trade_id: str,
        timeline: Timeline,
        a: LegIntent,
        b: LegIntent,
        qty: Decimal,
        ra: OrderResult,
        rb: OrderResult,
        report: TradeReport,
    ) -> None:
        if ra.success and not rb.success:
            _log.error(
                "[%s] PARTIAL: %s filled, %s failed (%s) — unwinding %s",
                trade_id, a.venue, b.venue, rb.error_message, a.venue,
            )
            unwind = await self._unwind_leg(
                timeline, a.venue, a.side.opposite, qty, ra.avg_price,
            )
            if unwind is not None:
                report.legs.append(unwind)
            report.failure_reason = f"{b.venue} leg failed: {rb.error_message}"
        elif rb.success and not ra.success:
            _log.error(
                "[%s] PARTIAL: %s filled, %s failed (%s) — unwinding %s",
                trade_id, b.venue, a.venue, ra.error_message, b.venue,
            )
            unwind = await self._unwind_leg(
                timeline, b.venue, b.side.opposite, qty, rb.avg_price,
            )
            if unwind is not None:
                report.legs.append(unwind)
            report.failure_reason = f"{a.venue} leg failed: {ra.error_message}"
        else:
            _log.warning(
                "[%s] BOTH FAILED: %s=%s %s=%s",
                trade_id, a.venue, ra.error_message, b.venue, rb.error_message,
            )
            report.failure_reason = "both legs failed"

    async def _unwind_leg(
        self,
        timeline: Timeline,
        venue: str,
        side: Side,
        qty: Decimal,
        cost_basis: Decimal | None,
    ) -> LegReport | None:
        """Flatten the stranded leg. `expected_price` is the stranded
        fill we are reversing, so the round-trip cost of a partial is
        directly computable offline. Returns `None` in paper mode
        (paper has no real position to flatten)."""
        if self.is_paper:
            return None
        ex = self.exchanges[venue]
        mkt = self.markets[venue]
        timeline.mark("unwind_send")
        try:
            r = await ex.place_market_order(mkt, side, qty, reduce_only=True)
            if not r.success:
                _log.error("unwind on %s FAILED: %s", venue, r.error_message)
        except Exception as e:  # noqa: BLE001
            _log.exception("unwind on %s raised: %s", venue, e)
            r = OrderResult(
                success=False, side=side, requested_qty=qty,
                error_message=f"unwind raised: {e}",
            )
        timeline.mark("unwind_result")
        return LegReport.build(
            venue=venue, side=side, qty=qty, expected=cost_basis, rest=r,
            latency_ms=timeline.span("unwind_send", "unwind_result"),
            kind=LegKind.UNWIND,
        )

    # ----- paper-mode synth -----

    def _paper_fill_one(self, leg: LegIntent, qty: Decimal) -> OrderResult:
        """Synthesize a single-leg paper fill from the current book VWAP
        on the opposite side (a BUY fills against asks, SELL against
        bids)."""
        ex = self.exchanges[leg.venue]
        mkt = self.markets[leg.venue]
        book = ex.order_book(mkt)
        assert book is not None
        levels = book.bids if leg.side is Side.SELL else book.asks
        vwap, _ = vwap_fill(levels, qty, max_levels=self.max_levels)
        assert vwap is not None
        return OrderResult(
            success=True,
            order_id="paper-" + uuid.uuid4().hex[:8],
            side=leg.side,
            requested_qty=qty,
            filled_qty=qty,
            avg_price=vwap,
        )
