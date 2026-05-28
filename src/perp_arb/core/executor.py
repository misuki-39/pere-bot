"""Two-leg execution: gathers `submit_and_await` outcomes on both venues,
records the per-leg outcome, returns an `ExecutionResult`.

Layering: the executor sits between **strategy** (signal — produces
decisions and resolves them into venue-side intents) and **driver**
(per-venue `submit_and_await` + WS + cid-keyed fill tracking via
`BaseExchange`). It is strategy-agnostic — it never sees `Direction`,
`vwap_*`, or any strategy-internal field. The caller hands it concrete
`LegIntent`s (venue, side, expected_price) and a `Timeline` writeback
target; the executor owns cid generation, handles `gather`, paper-fill
synth, and stamps presentation fields onto each `LegOutcome` so the
recorder can serialize them directly.

On a partial leg failure the executor records the failure_reason and
returns; it does **not** auto-unwind. The strategy layer owns
reconcile-and-rebalance via `_reconcile_after_failure` (REST snapshot →
reduce-only flatten), which is safer than blind unwind in cases where
the "failed" leg actually executed (e.g. aster 400/timeout). See
docs/agile-waddling-beacon plan.

Paper mode is a single branch inside `execute()` rather than a separate
`PaperExecutor` because paper and live share 100% of the orchestration
scaffold and differ only in *where the leaf fill price comes from*.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """What `TwoLegExecutor.execute()` returns. `legs` carries entries
    + any UNWIND leg (kind-tagged). Strategy derives success / PnL /
    send_ts from the legs themselves — only the failure narrative
    needs executor-side context (venue names, error text)."""

    legs: list[LegOutcome]
    failure_reason: str | None = None


class TwoLegExecutor:
    """Maps `(trade_id, legs, qty, timeline)` → `ExecutionResult`.

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
        # cid policy lives on each `BaseExchange.client_id_generator`. Default
        # is a monotonic, epoch-seeded counter (preserves the historical cid
        # shape). Venues that pre-stage cids (lighter's pre-signed pool) swap
        # in their own generator; the executor stays venue-agnostic.
        if cid_seed is not None:
            from .client_id import CounterClientIdGenerator
            for ex in exchanges.values():
                ex.client_id_generator = CounterClientIdGenerator(seed=cid_seed)

    async def execute(
        self,
        *,
        trade_id: str,
        legs: tuple[LegIntent, LegIntent],
        qty: Decimal,
        timeline: Timeline,
    ) -> ExecutionResult:
        a_intent, b_intent = legs
        # Per-leg cids. cids are per-driver scoped (each driver has its
        # own _fill_tracker), so they don't collide cross-venue today —
        # but a same-venue two-leg strategy WOULD collide on a shared
        # cid because the first finishing leg's release_fill_slot would
        # wipe the second leg's accumulator mid-flight. Per-leg cids
        # cost one int and close that door.
        cid_a = self.exchanges[a_intent.venue].client_id_generator.next(side=a_intent.side)
        cid_b = self.exchanges[b_intent.venue].client_id_generator.next(side=b_intent.side)

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

        legs_out: list[LegOutcome] = [a_out, b_out]
        if a_out.success and b_out.success:
            _log.info(
                "[%s] FILLED %s pnl=%s",
                trade_id,
                _spread_log(a_intent, b_intent, a_out, b_out),
                pair_pnl_from_legs(a_out, b_out),
            )
            return ExecutionResult(legs=legs_out)

        failure_reason = await self._handle_partial_failure(
            trade_id, a_intent, b_intent, qty, a_out, b_out, legs_out,
        )
        return ExecutionResult(legs=legs_out, failure_reason=failure_reason)

    # ----- partial-failure narrative -----

    async def _handle_partial_failure(
        self,
        trade_id: str,
        a: LegIntent,
        b: LegIntent,
        qty: Decimal,
        a_out: LegOutcome,
        b_out: LegOutcome,
        legs: list[LegOutcome],
    ) -> str:
        """Build the failure_reason string for the strategy's reconcile
        path. We do NOT auto-unwind here anymore — strategy owns the
        reconcile-and-rebalance flow (REST snapshot + targeted reduce-only)
        so the recovery is based on venue truth, not the executor's
        own success/failure verdict.

        `legs` is left at exactly two entries. `qty` and `a_out`/`b_out`
        are intentionally retained in the signature so future telemetry
        (e.g. per-leg latency-on-failure) can be added without churn.
        """
        a_name = self.exchanges[a.venue].name
        b_name = self.exchanges[b.venue].name
        if a_out.success and not b_out.success:
            _log.error(
                "[%s] PARTIAL: %s filled, %s failed (%s) — strategy will reconcile",
                trade_id, a_name, b_name, b_out.error_message,
            )
            return f"{b_name} leg failed: {b_out.error_message}"
        if b_out.success and not a_out.success:
            _log.error(
                "[%s] PARTIAL: %s filled, %s failed (%s) — strategy will reconcile",
                trade_id, b_name, a_name, a_out.error_message,
            )
            return f"{a_name} leg failed: {a_out.error_message}"
        _log.warning(
            "[%s] BOTH FAILED: %s=%s %s=%s",
            trade_id, a_name, a_out.error_message, b_name, b_out.error_message,
        )
        return "both legs failed"

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
        out = LegOutcome(
            success=True,
            client_id=cid,
            side=leg.side,
            requested_qty=qty,
            status=OrderStatus.FILLED,
        )
        out.set_fill(qty, vwap)
        return out


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
