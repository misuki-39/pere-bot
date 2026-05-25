"""Two-leg execution: gathers `submit_and_await` outcomes on both venues,
unwinds the stranded leg on partial failure, returns a `TradeReport`.

Layering: the executor sits between **strategy** (signal — produces
decisions and resolves them into venue-side intents) and **driver**
(per-venue `submit_and_await` + WS + cid-keyed fill tracking via
`BaseExchange`). It is strategy-agnostic — it never sees `Direction`,
`vwap_*`, or any strategy-internal field. The caller hands it concrete
`LegIntent`s (venue, side, expected_price) and a `Timeline` writeback
target; the executor owns cid generation, handles `gather`, partial-
failure unwind, paper-fill synth, and stamps presentation fields onto
each `LegOutcome` so the recorder can serialize them directly.

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
from .exec_record import Phase, Timeline
from .pnl import pair_pnl_from_legs
from .types import LegKind, LegOutcome, MarketInfo, OrderStatus, Side

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LegIntent:
    """One venue leg of an execution intent. The caller has already
    resolved direction → side; cid lives inside the executor (venue-
    protocol detail), not on the intent."""

    venue: str               # leg-label key into the executor's `exchanges` dict (e.g. "leg_a")
    side: Side
    expected_price: Decimal  # decision-time VWAP / cost basis — stamped onto LegOutcome


@dataclass
class TradeReport:
    """Result the executor returns to the caller.

    `success=True` iff both entry legs' place acks returned success.
    Per-leg ack + WS fill data live on `legs[*]` as `LegOutcome`s with
    presentation fields (venue, expected_price, send_ts_ms, kind) already
    stamped by the executor — the recorder serializes them directly.

    A partial-failure unwind appears as an additional leg with
    `kind=UNWIND`. Both-fail leaves `success` False with a populated
    `failure_reason`. `send_ts_ms` is the wall-clock moment the SEND
    timeline mark was stamped — caller copies it into its own record."""

    legs: list[LegOutcome] = field(default_factory=list)
    success: bool = False
    failure_reason: str | None = None
    send_ts_ms: int = 0
    # Cash-flow PnL of the two entry legs net of fees. Set only when
    # `success=True` and both legs carry realized price + filled qty.
    realised_pnl: Decimal | None = None


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
        self._is_paper = is_paper
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
        # Per-leg cids. cids are per-driver scoped (each driver has its
        # own _fill_tracker), so they don't collide cross-venue today —
        # but a same-venue two-leg strategy WOULD collide on a shared
        # cid because the first finishing leg's release_fill_slot would
        # wipe the second leg's accumulator mid-flight. Per-leg cids
        # cost one int and close that door.
        cid_a = self._next_cid()
        cid_b = self._next_cid()

        timeline.mark(Phase.SEND)
        send_ts_ms = now_ms()

        if self._is_paper:
            a_out = self._paper_outcome(a_intent, qty, cid_a)
            b_out = self._paper_outcome(b_intent, qty, cid_b)
        else:
            # `return_exceptions=True`: an unexpected raise out of one
            # submit_and_await (CancelledError aside) must NOT cancel
            # the sibling — that sibling may have already placed a real
            # venue order whose fill we still need to track. Coerce any
            # captured exception into a `LegOutcome(success=False)` so
            # the partial-failure / unwind path runs on both sides.
            results = await asyncio.gather(
                self.exchanges[a_intent.venue].submit_and_await(
                    self.markets[a_intent.venue], a_intent.side, qty,
                    client_id=cid_a, timeout_s=self.fill_wait_timeout_s,
                ),
                self.exchanges[b_intent.venue].submit_and_await(
                    self.markets[b_intent.venue], b_intent.side, qty,
                    client_id=cid_b, timeout_s=self.fill_wait_timeout_s,
                ),
                return_exceptions=True,
            )
            a_out = _coerce_outcome(results[0], a_intent.side, qty)
            b_out = _coerce_outcome(results[1], b_intent.side, qty)

        # Stamp presentation fields onto each outcome so they're recorder-
        # ready. `venue` = actual driver name ("aster"/"lighter") via
        # BaseExchange.name, not the leg label key ("leg_a") — keeps audit
        # traces joinable to venue-side records.
        self._stamp(a_out, a_intent, send_ts_ms, LegKind.ENTRY)
        self._stamp(b_out, b_intent, send_ts_ms, LegKind.ENTRY)

        report = TradeReport(legs=[a_out, b_out], send_ts_ms=send_ts_ms)

        if a_out.success and b_out.success:
            report.success = True
            report.realised_pnl = pair_pnl_from_legs(a_out, b_out)
            _log.info(
                "[%s] FILLED %s pnl=%s",
                trade_id,
                _spread_log(a_intent, b_intent, a_out, b_out),
                report.realised_pnl,
            )
            return report

        await self._handle_partial_failure(
            trade_id, a_intent, b_intent, qty, a_out, b_out, report,
        )
        return report

    # ----- partial-failure recovery -----

    async def _handle_partial_failure(
        self,
        trade_id: str,
        a: LegIntent,
        b: LegIntent,
        qty: Decimal,
        a_out: LegOutcome,
        b_out: LegOutcome,
        report: TradeReport,
    ) -> None:
        # Log + failure_reason use the actual venue names ("aster"/"lighter")
        # — these strings are surfaced to operators, not to the strategy's
        # internal leg-label routing.
        a_name = self.exchanges[a.venue].name
        b_name = self.exchanges[b.venue].name
        if a_out.success and not b_out.success:
            _log.error(
                "[%s] PARTIAL: %s filled, %s failed (%s) — unwinding %s",
                trade_id, a_name, b_name, b_out.error_message, a_name,
            )
            unwind = await self._unwind_leg(
                a.venue, a.side.opposite, qty, a_out.avg_price,
            )
            if unwind is not None:
                report.legs.append(unwind)
            report.failure_reason = f"{b_name} leg failed: {b_out.error_message}"
        elif b_out.success and not a_out.success:
            _log.error(
                "[%s] PARTIAL: %s filled, %s failed (%s) — unwinding %s",
                trade_id, b_name, a_name, a_out.error_message, b_name,
            )
            unwind = await self._unwind_leg(
                b.venue, b.side.opposite, qty, b_out.avg_price,
            )
            if unwind is not None:
                report.legs.append(unwind)
            report.failure_reason = f"{a_name} leg failed: {a_out.error_message}"
        else:
            _log.warning(
                "[%s] BOTH FAILED: %s=%s %s=%s",
                trade_id, a_name, a_out.error_message, b_name, b_out.error_message,
            )
            report.failure_reason = "both legs failed"

    async def _unwind_leg(
        self,
        venue: str,
        side: Side,
        qty: Decimal,
        cost_basis: Decimal | None,
    ) -> LegOutcome | None:
        """Flatten the stranded leg. `cost_basis` is the stranded fill
        we are reversing — stamped as `expected_price` so the round-trip
        cost of a partial is directly computable offline. Returns `None`
        in paper mode (paper has no real position to flatten).

        Routes through `submit_and_await` so the WS fill data populates
        the unwind outcome — `place_market_order` alone leaves lighter
        unwinds with no realized_price (lighter's REST carries no fill
        data; only WS does)."""
        if self._is_paper:
            return None
        ex = self.exchanges[venue]
        mkt = self.markets[venue]
        unwind_cid = self._next_cid()
        unwind_send_ts = now_ms()
        try:
            out = await ex.submit_and_await(
                mkt, side, qty, client_id=unwind_cid,
                timeout_s=self.fill_wait_timeout_s, reduce_only=True,
            )
            if not out.success:
                _log.error("unwind on %s FAILED: %s", venue, out.error_message)
        except Exception as e:  # noqa: BLE001
            _log.exception("unwind on %s raised: %s", venue, e)
            out = LegOutcome(
                success=False, side=side, requested_qty=qty,
                error_message=f"unwind raised: {e}",
            )
        out.venue = ex.name
        out.expected_price = cost_basis
        out.send_ts_ms = unwind_send_ts
        out.kind = LegKind.UNWIND
        return out

    # ----- presentation field stamping -----

    def _stamp(
        self,
        outcome: LegOutcome,
        intent: LegIntent,
        send_ts_ms: int,
        kind: LegKind,
    ) -> None:
        """Fill in the four presentation fields the recorder + analysis
        need. Drivers only know ack + fill data; venue label / expected
        price / send timestamp / leg kind are executor concerns."""
        outcome.venue = self.exchanges[intent.venue].name
        outcome.expected_price = intent.expected_price
        outcome.send_ts_ms = send_ts_ms
        outcome.kind = kind

    # ----- paper-mode synth -----

    def _paper_outcome(self, leg: LegIntent, qty: Decimal, cid: str) -> LegOutcome:
        """Synthesize a single-leg paper outcome from the current book VWAP
        on the opposite side (a BUY fills against asks, SELL against
        bids). Bypasses both the REST place ack and the WS tracker."""
        ex = self.exchanges[leg.venue]
        mkt = self.markets[leg.venue]
        book = ex.order_book(mkt)
        # Explicit raises (not asserts) — asserts are stripped under
        # `python -O` / PYTHONOPTIMIZE, and the next attribute access
        # would surface as a cryptic AttributeError.
        if book is None:
            raise RuntimeError(f"paper mode: no order book cached for {leg.venue}")
        levels = book.bids if leg.side is Side.SELL else book.asks
        vwap, _ = vwap_fill(levels, qty, max_levels=self.max_levels)
        if vwap is None:
            raise RuntimeError(
                f"paper mode: empty {('bids' if leg.side is Side.SELL else 'asks')} "
                f"on {leg.venue}",
            )
        return LegOutcome(
            success=True,
            client_id=cid,
            side=leg.side,
            requested_qty=qty,
            status=OrderStatus.FILLED,
            filled_qty=qty,
            weighted_price_sum=vwap * qty,
        )


def _coerce_outcome(
    result: LegOutcome | BaseException,
    side: Side,
    qty: Decimal,
) -> LegOutcome:
    """Convert a `gather(return_exceptions=True)` slot into a LegOutcome.
    Re-raise CancelledError (cooperative cancellation must propagate);
    convert any other exception into a `success=False` outcome so the
    partial-failure / unwind path can run."""
    if isinstance(result, asyncio.CancelledError):
        raise result
    if isinstance(result, BaseException):
        return LegOutcome(
            success=False, side=side, requested_qty=qty,
            error_message=f"submit_and_await raised: {type(result).__name__}: {result}",
        )
    return result


def _spread_log(
    a_intent: LegIntent, b_intent: LegIntent,
    a_out: LegOutcome, b_out: LegOutcome,
) -> str:
    """Format the per-leg sign-encoded prices + net spread for the
    FILLED log line. Sign convention: sell → +price (cash in), buy →
    -price (cash out). Sum = per-unit net cash captured pre-fees."""
    ap = a_out.avg_price
    bp = b_out.avg_price
    if ap is None or bp is None:
        # Keep key=value structure so log parsers don't misinterpret the
        # middle field as a price/spread report.
        return f"a={ap} b={bp}"
    ap_s = ap if a_intent.side is Side.SELL else -ap
    bp_s = bp if b_intent.side is Side.SELL else -bp
    return f"a={ap_s} b={bp_s} spread={ap_s + bp_s}"
