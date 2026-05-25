"""Unit tests for the live fill-enrichment plumbing:

  - `LegOutcome.add` aggregates `FillDelta` + `OrderSnapshot` events;
  - `BaseExchange.submit_and_await` merges the synchronous REST/WS-tx
    place ack with the WS fill aggregate, preferring WS for
    filled_qty / weighted_price_sum / last_ts_ms / total_fee;
  - The recorder writes the new `send_ts_ms` + `fill_ts_ms` columns.

These tests do not exercise the asyncio await loop directly — the loop
just wraps the accumulator + Event primitives covered here and in stdlib.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from perp_arb.core.exchange import BaseExchange
from perp_arb.core.exec_record import (
    Decision,
    Direction,
    ExecutionRecorder,
    LegReport,
    Outcome,
    _decision_header,
    _leg_header,
)
from perp_arb.core.types import (
    FillDelta,
    LegOutcome,
    MarketInfo,
    OrderBook,
    OrderSnapshot,
    OrderStatus,
    Position,
    Quote,
    Side,
    Symbol,
)
from perp_arb.core.types import (
    LegOutcome as _FillAccumulator,
)

_SYM = Symbol(exchange="aster", raw="CLUSDT", base="WTI", quote="USDT")
_MARKET = MarketInfo(
    symbol=_SYM, tick_size=Decimal("0.01"), lot_size=Decimal("1"),
    contract_id="CLUSDT",
)


# ---- _FillAccumulator (LegOutcome accumulator semantics) --------------

def _delta(qty: str, price: str, ts: int, *, client_id: str = "x",
           terminal: OrderStatus | None = None, fee: str = "0") -> FillDelta:
    return FillDelta(
        qty=Decimal(qty), price=Decimal(price), ts_ms=ts,
        side=Side.BUY, client_id=client_id,
        terminal_status=terminal, fee=Decimal(fee),
    )


def _snapshot(*, filled: str, avg: str, status: OrderStatus, ts: int = 0,
              client_id: str = "x") -> OrderSnapshot:
    return OrderSnapshot(
        client_id=client_id, symbol=_SYM, side=Side.BUY,
        size=Decimal("1.0"), price=Decimal("100"),
        status=status,
        filled_qty=Decimal(filled),
        realized_price=Decimal(avg) if avg else None,
        ts_ms=ts,
    )


def test_accumulator_single_delta() -> None:
    acc = _FillAccumulator()
    acc.add(_delta("1.0", "100.00", ts=1000))
    assert acc.filled_qty == Decimal("1.0")
    assert acc.weighted_price_sum / acc.filled_qty == Decimal("100.00")
    assert acc.last_ts_ms == 1000


def test_accumulator_aggregates_partial_deltas() -> None:
    """Two partial fills at different prices → size-weighted avg, latest ts."""
    acc = _FillAccumulator()
    acc.add(_delta("0.4", "100.00", ts=1000))
    acc.add(_delta("0.6", "100.10", ts=1050))
    assert acc.filled_qty == Decimal("1.0")
    # (0.4 * 100.00 + 0.6 * 100.10) / 1.0 = 100.06
    assert acc.weighted_price_sum / acc.filled_qty == Decimal("100.06")
    assert acc.last_ts_ms == 1050


def test_accumulator_complete_via_terminal_status() -> None:
    """FILLED status from account_orders short-circuits the qty check."""
    acc = _FillAccumulator()
    acc.add(_snapshot(filled="0.4", avg="100.00", status=OrderStatus.OPEN))
    assert not acc.is_complete(Decimal("1.0"))
    acc.add(_snapshot(filled="1.0", avg="100.05", status=OrderStatus.FILLED))
    assert acc.is_complete(Decimal("1.0"))


def test_accumulator_complete_on_cancel_with_partial_fill() -> None:
    """Cancel after a partial fill — terminal status short-circuits the wait."""
    acc = _FillAccumulator()
    acc.add(_snapshot(filled="0.3", avg="100.00", status=OrderStatus.CANCELED))
    assert acc.is_complete(Decimal("1.0"))
    assert acc.filled_qty == Decimal("0.3")


def test_accumulator_qty_fallback_when_status_absent() -> None:
    """Trade-delta only path (no snapshot stream): exact qty comparison."""
    acc = _FillAccumulator()
    acc.add(_delta("0.9995", "100", ts=1))
    assert not acc.is_complete(Decimal("1.0"))
    acc.add(_delta("0.0005", "100", ts=2))
    assert acc.is_complete(Decimal("1.0"))


def test_accumulator_snapshot_overwrites_delta() -> None:
    """Trade delta arrives first; snapshot arrives second and wins on qty/price.
    ts is taken from whichever event carries it (delta does, snapshot may not)."""
    acc = _FillAccumulator()
    acc.add(_delta("0.4", "100.00", ts=1000))
    acc.add(_snapshot(filled="1.0", avg="100.06", status=OrderStatus.FILLED))
    assert acc.filled_qty == Decimal("1.0")
    assert acc.weighted_price_sum / acc.filled_qty == Decimal("100.06")
    assert acc.last_ts_ms == 1000


def test_accumulator_delta_terminal_status_propagates() -> None:
    """Aster's final FILLED delta carries terminal_status → is_complete True."""
    acc = _FillAccumulator()
    acc.add(_delta("1.0", "100.00", ts=1, terminal=OrderStatus.FILLED))
    assert acc.is_complete(Decimal("2.0"))   # filled < requested but status terminal


# ---- BaseExchange.submit_and_await merge ------------------------------

class _FakeExchange(BaseExchange):
    """Concrete BaseExchange that returns a canned place-ack outcome and
    exposes its internal `_fill_tracker` so tests can inject WS events."""

    name = "fake"

    def __init__(self, ack: LegOutcome) -> None:
        super().__init__()
        self._ack = ack

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def load_market(self, raw_symbol: str) -> MarketInfo:
        raise NotImplementedError

    async def place_market_order(
        self, market: MarketInfo, side: Side, qty: Decimal,
        *, client_id: str, reduce_only: bool = False,
    ) -> LegOutcome:
        return self._ack

    async def get_position(self, market: MarketInfo) -> Position:
        raise NotImplementedError

    def subscribe_quotes(self, market: MarketInfo, cb: Callable) -> None: ...
    def subscribe_book(self, market: MarketInfo, cb: Callable) -> None: ...
    def subscribe_fills(self, market: MarketInfo, cb: Callable) -> None: ...
    def subscribe_positions(self, market: MarketInfo, cb: Callable) -> None: ...
    def best_quote(self, market: MarketInfo) -> Quote | None: return None
    def order_book(self, market: MarketInfo) -> OrderBook | None: return None
    def live_position(self, market: MarketInfo) -> Position | None: return None


def _ack_ok(*, avg_price: str = "100.00", exchange_ts: int | None = None,
            filled: str = "1.0") -> LegOutcome:
    """The shape aster's REST returns: ack fields + executedQty/avgPrice
    folded into the fill-side fields as a fallback before WS arrives."""
    f = Decimal(filled)
    return LegOutcome(
        success=True,
        client_id="x",
        side=Side.BUY,
        requested_qty=Decimal("1.0"),
        status=OrderStatus.FILLED,
        latency_ms=50,
        exchange_ts_ms=exchange_ts,
        filled_qty=f,
        weighted_price_sum=f * Decimal(avg_price),
    )


def _ack_lighter_open() -> LegOutcome:
    """Lighter's signed-tx ack: status=OPEN, no fill data — the WS account
    stream is the only realized-data source."""
    return LegOutcome(
        success=True, client_id="x", side=Side.SELL,
        requested_qty=Decimal("1.0"), status=OrderStatus.OPEN, latency_ms=200,
    )


@pytest.mark.asyncio
async def test_submit_and_await_ws_fill_overrides_rest() -> None:
    """WS fill is the matching-engine's authoritative view — its
    filled_qty / weighted_price_sum / last_ts_ms / total_fee overwrite the
    REST-reported placeholders. Ack-side fields (status, client_id,
    error_message, exchange_ts_ms) are not touched."""
    ex = _FakeExchange(_ack_ok(avg_price="100.00", exchange_ts=1_700_000_000_000))

    async def feed_ws() -> None:
        # Let submit_and_await register the slot first.
        await asyncio.sleep(0)
        ex._fill_tracker.on_event(_delta(
            "1.0", "100.05", ts=1_700_000_000_500,
            client_id="x", terminal=OrderStatus.FILLED, fee="0.02",
        ))

    feed = asyncio.create_task(feed_ws())
    outcome = await ex.submit_and_await(
        _MARKET, Side.BUY, Decimal("1.0"), client_id="x", timeout_s=1.0,
    )
    await feed
    assert outcome.avg_price == Decimal("100.05")
    assert outcome.filled_qty == Decimal("1.0")
    assert outcome.last_ts_ms == 1_700_000_000_500
    assert outcome.total_fee == Decimal("0.02")
    assert outcome.exchange_ts_ms == 1_700_000_000_000  # ack still owns this


@pytest.mark.asyncio
async def test_submit_and_await_falls_back_to_rest_on_ws_timeout() -> None:
    """REST ack carries filled_qty/avg_price; WS times out → outcome keeps
    the REST-reported values (no zero-out)."""
    ex = _FakeExchange(_ack_ok(avg_price="100.00", exchange_ts=1_700_000_000_000))
    outcome = await ex.submit_and_await(
        _MARKET, Side.BUY, Decimal("1.0"), client_id="x", timeout_s=0.05,
    )
    assert outcome.avg_price == Decimal("100.00")
    assert outcome.filled_qty == Decimal("1.0")
    assert outcome.fill_ts_ms == 1_700_000_000_000   # exchange_ts_ms fallback


@pytest.mark.asyncio
async def test_submit_and_await_lighter_ws_is_only_price_source() -> None:
    """Lighter's signed-tx ack carries no avg_price / filled_qty. WS fill
    is the only realized-data source for that leg."""
    ex = _FakeExchange(_ack_lighter_open())

    async def feed_ws() -> None:
        await asyncio.sleep(0)
        ex._fill_tracker.on_event(_delta(
            "1.0", "100.20", ts=1_700_000_001_000,
            client_id="x", terminal=OrderStatus.FILLED,
        ))

    feed = asyncio.create_task(feed_ws())
    outcome = await ex.submit_and_await(
        _MARKET, Side.SELL, Decimal("1.0"), client_id="x", timeout_s=1.0,
    )
    await feed
    assert outcome.avg_price == Decimal("100.20")
    assert outcome.filled_qty == Decimal("1.0")
    assert outcome.last_ts_ms == 1_700_000_001_000


@pytest.mark.asyncio
async def test_submit_and_await_failed_ack_skips_ws_await() -> None:
    """Place-ack failure short-circuits — no point waiting for WS events
    that will never come. Outcome is the ack as-is."""
    failed = LegOutcome(
        success=False, client_id="x", side=Side.BUY,
        requested_qty=Decimal("1.0"), error_message="oops",
    )
    ex = _FakeExchange(failed)
    outcome = await ex.submit_and_await(
        _MARKET, Side.BUY, Decimal("1.0"), client_id="x", timeout_s=0.5,
    )
    assert outcome.success is False
    assert outcome.error_message == "oops"
    assert outcome.filled_qty == Decimal("0")


@pytest.mark.asyncio
async def test_submit_and_await_protects_rest_full_fill_from_ws_partial() -> None:
    """Aster REST returns the complete sync fill (executedQty=1.0); a
    trailing partial WS delta with terminal_status arrives first. The
    overlay rule (`ws.filled_qty >= outcome.filled_qty`) must NOT let the
    WS partial downgrade the complete REST view."""
    ex = _FakeExchange(_ack_ok(avg_price="100.00", filled="1.0",
                                exchange_ts=1_700_000_000_000))

    async def feed_ws() -> None:
        await asyncio.sleep(0)
        # Single partial-with-terminal delta — would otherwise downgrade
        # REST's 1.0 → 0.4 under a naive overlay.
        ex._fill_tracker.on_event(_delta(
            "0.4", "99.95", ts=1_700_000_000_500,
            client_id="x", terminal=OrderStatus.FILLED,
        ))

    feed = asyncio.create_task(feed_ws())
    outcome = await ex.submit_and_await(
        _MARKET, Side.BUY, Decimal("1.0"), client_id="x", timeout_s=1.0,
    )
    await feed
    # REST's complete fill preserved; status promoted from WS terminal
    assert outcome.filled_qty == Decimal("1.0")
    assert outcome.avg_price == Decimal("100.00")
    assert outcome.status is OrderStatus.FILLED   # WS terminal propagated


@pytest.mark.asyncio
async def test_submit_and_await_promotes_ws_terminal_status() -> None:
    """Lighter ack returns OPEN; WS later confirms FILLED. Without the
    status reconciliation, legs CSV would forever show 'open' on
    successful lighter fills."""
    ex = _FakeExchange(_ack_lighter_open())

    async def feed_ws() -> None:
        await asyncio.sleep(0)
        ex._fill_tracker.on_event(_delta(
            "1.0", "100.20", ts=1_700_000_001_000,
            client_id="x", terminal=OrderStatus.FILLED,
        ))

    feed = asyncio.create_task(feed_ws())
    outcome = await ex.submit_and_await(
        _MARKET, Side.SELL, Decimal("1.0"), client_id="x", timeout_s=1.0,
    )
    await feed
    assert outcome.status is OrderStatus.FILLED
    assert outcome.last_status is OrderStatus.FILLED


def test_legoutcome_add_rejects_unknown_event_type() -> None:
    """Match statement now has a `case _:` default — silently dropping
    a future event subtype would mask fills as WS timeouts."""
    class _FutureEvent:   # not OrderSnapshot, not FillDelta
        ts_ms = 0          # satisfies the pre-match guard

    acc = _FillAccumulator()
    with pytest.raises(TypeError, match="unhandled event type"):
        acc.add(_FutureEvent())  # type: ignore[arg-type]


def test_legoutcome_add_snapshot_without_price_no_op() -> None:
    """OrderSnapshot with filled_qty>0 but realized_price=None is
    intermediate state — must NOT commit filled_qty alone (would leave
    avg_price returning fabricated 0)."""
    acc = _FillAccumulator()
    acc.add(_snapshot(filled="1.0", avg="", status=OrderStatus.FILLED, ts=1000))
    assert acc.filled_qty == Decimal("0")
    assert acc.avg_price is None
    assert acc.last_status is OrderStatus.FILLED  # terminal status still tracked


# ---- LegReport.from_outcome ------------------------------------------

_SEND_TS = 1_699_999_999_900   # reference SEND for from_outcome tests


def test_from_outcome_copies_fill_data() -> None:
    """Outcome already merged — `from_outcome` is a pure field copy."""
    out = LegOutcome(
        success=True, client_id="x", side=Side.BUY,
        requested_qty=Decimal("1.0"), status=OrderStatus.FILLED, latency_ms=50,
        exchange_ts_ms=1_700_000_000_000,
        filled_qty=Decimal("1.0"),
        weighted_price_sum=Decimal("100.05"),
        last_ts_ms=1_700_000_000_500,
        total_fee=Decimal("0.02"),
    )
    leg = LegReport.from_outcome(
        venue="aster", outcome=out,
        expected_price=Decimal("99.95"), send_ts_ms=_SEND_TS,
    )
    assert leg.realized_price == Decimal("100.05")
    assert leg.filled_qty == Decimal("1.0")
    assert leg.fill_ts_ms == 1_700_000_000_500          # WS ts wins via fill_ts_ms property
    assert leg.latency_ms == 600
    assert leg.fee == Decimal("0.02")
    assert leg.client_id == "x"


def test_from_outcome_latency_none_when_no_ts() -> None:
    """No WS ts AND no exchange_ts_ms (e.g. lighter pre-fill): fill_ts_ms
    and latency_ms both None — we don't fabricate a fill latency."""
    out = LegOutcome(
        success=True, client_id="x", side=Side.SELL,
        requested_qty=Decimal("1.0"), status=OrderStatus.OPEN, latency_ms=200,
    )
    leg = LegReport.from_outcome(
        venue="lighter", outcome=out,
        expected_price=Decimal("100.25"), send_ts_ms=_SEND_TS,
    )
    assert leg.fill_ts_ms is None
    assert leg.latency_ms is None


# ---- recorder CSV header contains the new columns ---------------------

def test_decision_header_contains_send_ts_ms() -> None:
    assert "send_ts_ms" in _decision_header()


def test_leg_header_contains_fill_ts_ms() -> None:
    assert "fill_ts_ms" in _leg_header()


def test_recorder_writes_send_ts_ms_to_csv(tmp_path: Path) -> None:
    """End-to-end: a FIRED Decision with send_ts_ms + a LegReport with
    fill_ts_ms should round-trip through the CSV writer."""
    rec = ExecutionRecorder(tmp_path, run_ts="TEST", strategy_id="taker_taker")
    d = Decision(
        decision_id="d-test",
        ts_ms=1_700_000_000_000,
        mid_left=Decimal("100"), mid_right=Decimal("100"),
        left_quote_ts_ms=1_700_000_000_000,
        right_quote_ts_ms=1_700_000_000_000,
        direction=Direction.A,
        outcome=Outcome.FIRED,
        send_ts_ms=1_700_000_000_010,
    )
    d.legs.append(LegReport(
        exchange="aster", side="buy",
        requested_qty=Decimal("1.0"), filled_qty=Decimal("1.0"),
        expected_price=Decimal("100"), realized_price=Decimal("100.05"),
        status="filled", success=True,
        fill_ts_ms=1_700_000_000_050,
    ))
    rec.emit(d)
    rec.close()

    decisions_csv = next(tmp_path.glob("decisions_*.csv")).read_text().splitlines()
    legs_csv = next(tmp_path.glob("legs_*.csv")).read_text().splitlines()
    assert "send_ts_ms" in decisions_csv[0]
    assert "1700000000010" in decisions_csv[1]
    assert "fill_ts_ms" in legs_csv[0]
    assert "1700000000050" in legs_csv[1]
