"""Two-leg execution: gathers place-acks + WS fills on both venues,
unwinds the stranded leg on partial failure, returns a `TradeReport`.

Layering: the executor sits between **strategy** (signal — produces
decisions and resolves them into venue-side intents) and **driver**
(per-venue place_market_order + WS + cid-keyed fill tracking via
`BaseExchange`). It is strategy-agnostic — it never sees `Direction`,
`vwap_*`, or any strategy-internal field. The caller hands it concrete
`LegIntent`s (venue, side, expected_price) and a `Timeline` writeback
target; the executor owns cid generation, handles `gather`, partial-
failure unwind, paper-fill synth, and `LegReport` assembly.

Paper mode is a single branch inside `execute()` rather than a separate
`PaperExecutor` because paper and live share 100% of the orchestration
scaffold and differ only in *where the leaf fill price comes from*.
"""

from __future__ import annotations

import asyncio
import logging
import time
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
    resolved direction → side; cid lives inside the executor (venue-
    protocol detail), not on the intent."""

    venue: str               # leg-label key into the executor's `exchanges` dict (e.g. "leg_a")
    side: Side
    expected_price: Decimal  # decision-time VWAP / cost basis — informational, passed through to LegReport


@dataclass
class TradeReport:
    """Result the executor returns to the caller.

    `success=True` iff both entry legs' place acks returned success.
    Per-leg WS fill data lives on `legs[*]` via `LegReport.build`; legs
    without a WS event fall back to the place ack's values.

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
        cid_seed: int | None = None,
    ) -> None:
        self.exchanges = exchanges
        self.markets = markets
        self.is_paper = is_paper
        self.max_levels = max_levels
        self.fill_wait_timeout_s = fill_wait_timeout_s
        # cid is a venue-protocol detail; the executor owns the counter.
        # Both legs of a trade share a cid — each driver's `_fill_tracker`
        # is per-driver, so the same string keys disjoint event spaces.
        # Seed from epoch seconds (Lighter SDK's canonical pattern); cap
        # is Lighter's `2^48 - 10`, so an epoch-seeded int is comfortably
        # within range and monotonic across sessions.
        self._cid_counter = cid_seed if cid_seed is not None else int(time.time())

    def _next_cid(self) -> str:
        cid = str(self._cid_counter)
        self._cid_counter += 1
        return cid

    async def execute(
        self,
        *,
        trade_id: str,
        legs: tuple[LegIntent, LegIntent],
        qty: Decimal,
        timeline: Timeline,
    ) -> TradeReport:
        a_intent, b_intent = legs
        cid = self._next_cid()

        timeline.mark(Phase.SEND)
        send_ts_ms = now_ms()

        if self.is_paper:
            ack_a = self._paper_fill_one(a_intent, qty, cid)
            ack_b = self._paper_fill_one(b_intent, qty, cid)
            timeline.mark(f"result_{a_intent.venue}")
            timeline.mark(f"result_{b_intent.venue}")
            a_fill: TerminalFill | None = None
            b_fill: TerminalFill | None = None
        else:
            (ack_a, a_fill), (ack_b, b_fill) = await asyncio.gather(
                self._submit_leg(timeline, a_intent, qty, cid),
                self._submit_leg(timeline, b_intent, qty, cid),
            )
        # Phase.RESULT = the moment both place acks had returned, NOT the
        # moment gather() unblocked (which also waits on the WS fill timeout
        # inside _submit_leg). Without this, a missed WS fill inflates
        # `lat_send_result_ms` and TradeReport.latency_ms by the full
        # fill_wait_timeout_s — turning the per-leg "leg latency" budget into
        # a fill-wait check.
        a_res = timeline.get(f"result_{a_intent.venue}")
        b_res = timeline.get(f"result_{b_intent.venue}")
        if a_res is not None and b_res is not None:
            timeline.mark_at(Phase.RESULT, max(a_res, b_res))
        else:
            timeline.mark(Phase.RESULT)

        # CSV `exchange` column = actual venue ("aster"/"lighter") via
        # BaseExchange.name, not the leg label key ("leg_a") — keeps audit
        # traces joinable to venue-side records.
        legs_out = [
            LegReport.build(
                venue=self.exchanges[a_intent.venue].name,
                side=a_intent.side, qty=qty,
                expected=a_intent.expected_price, ack=ack_a, fill=a_fill,
                latency_ms=timeline.span(Phase.SEND, f"result_{a_intent.venue}"),
            ),
            LegReport.build(
                venue=self.exchanges[b_intent.venue].name,
                side=b_intent.side, qty=qty,
                expected=b_intent.expected_price, ack=ack_b, fill=b_fill,
                latency_ms=timeline.span(Phase.SEND, f"result_{b_intent.venue}"),
            ),
        ]
        latency = timeline.span(Phase.SEND, Phase.RESULT)
        report = TradeReport(
            legs=legs_out, latency_ms=latency, send_ts_ms=send_ts_ms,
        )

        if ack_a.success and ack_b.success:
            report.success = True
            _log.info("[%s] FILLED cid=%s latency=%sms",
                      trade_id, cid, latency)
            return report

        await self._handle_partial_failure(
            trade_id, timeline, a_intent, b_intent, qty, ack_a, ack_b, report,
        )
        return report

    # ----- one-leg submit (live only) -----

    async def _submit_leg(
        self, timeline: Timeline, leg: LegIntent, qty: Decimal, cid: str,
    ) -> tuple[OrderResult, TerminalFill | None]:
        """Place + WS fill await for one leg.

        Per-leg timeline mark `result_{venue}` lands the moment the
        place ack returns — independent of when (or whether) the WS
        terminal fill resolves. That preserves per-leg inter-venue
        latency semantics."""
        ex = self.exchanges[leg.venue]
        mkt = self.markets[leg.venue]
        ex.register_fill_slot(cid)
        try:
            ack = await ex.place_market_order(
                mkt, leg.side, qty, client_id=cid,
            )
            timeline.mark(f"result_{leg.venue}")
            if not ack.success:
                return ack, None
            fill = await ex.await_fill(
                cid, qty, self.fill_wait_timeout_s,
            )
            return ack, fill
        finally:
            ex.release_fill_slot(cid)

    # ----- partial-failure recovery -----

    async def _handle_partial_failure(
        self,
        trade_id: str,
        timeline: Timeline,
        a: LegIntent,
        b: LegIntent,
        qty: Decimal,
        ack_a: OrderResult,
        ack_b: OrderResult,
        report: TradeReport,
    ) -> None:
        # Log + failure_reason use the actual venue names ("aster"/"lighter")
        # — these strings are surfaced to operators, not to the strategy's
        # internal leg-label routing.
        a_name = self.exchanges[a.venue].name
        b_name = self.exchanges[b.venue].name
        if ack_a.success and not ack_b.success:
            _log.error(
                "[%s] PARTIAL: %s filled, %s failed (%s) — unwinding %s",
                trade_id, a_name, b_name, ack_b.error_message, a_name,
            )
            unwind = await self._unwind_leg(
                timeline, a.venue, a.side.opposite, qty, ack_a.avg_price,
            )
            if unwind is not None:
                report.legs.append(unwind)
            report.failure_reason = f"{b_name} leg failed: {ack_b.error_message}"
        elif ack_b.success and not ack_a.success:
            _log.error(
                "[%s] PARTIAL: %s filled, %s failed (%s) — unwinding %s",
                trade_id, b_name, a_name, ack_a.error_message, b_name,
            )
            unwind = await self._unwind_leg(
                timeline, b.venue, b.side.opposite, qty, ack_b.avg_price,
            )
            if unwind is not None:
                report.legs.append(unwind)
            report.failure_reason = f"{a_name} leg failed: {ack_a.error_message}"
        else:
            _log.warning(
                "[%s] BOTH FAILED: %s=%s %s=%s",
                trade_id, a_name, ack_a.error_message, b_name, ack_b.error_message,
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
            ack = await ex.place_market_order(mkt, side, qty, reduce_only=True)
            if not ack.success:
                _log.error("unwind on %s FAILED: %s", venue, ack.error_message)
        except Exception as e:  # noqa: BLE001
            _log.exception("unwind on %s raised: %s", venue, e)
            ack = OrderResult(
                success=False, side=side, requested_qty=qty,
                error_message=f"unwind raised: {e}",
            )
        timeline.mark("unwind_result")
        return LegReport.build(
            venue=ex.name, side=side, qty=qty, expected=cost_basis, ack=ack,
            latency_ms=timeline.span("unwind_send", "unwind_result"),
            kind=LegKind.UNWIND,
        )

    # ----- paper-mode synth -----

    def _paper_fill_one(self, leg: LegIntent, qty: Decimal, cid: str) -> OrderResult:
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
            client_id=cid,
            side=leg.side,
            requested_qty=qty,
            filled_qty=qty,
            avg_price=vwap,
        )
