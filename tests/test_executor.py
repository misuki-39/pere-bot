"""Tests for `TwoLegExecutor` — the order-execution layer between
strategy decisions and venue drivers.

Tests pin the four partial-failure quadrants, the paper-mode synth path,
the WS-timeout fallback, and a few representative latency/leg-report
invariants. Driver behaviour is stubbed: these tests never hit the real
`aster` / `lighter` clients.

The executor is strategy-agnostic: it takes `LegIntent`s with already-
resolved sides + expected prices, plus a `Timeline` writeback target.
These tests never construct a `Decision` — that's the strategy's job."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pytest

from perp_arb.core.exec_record import LegKind, Phase, Timeline
from perp_arb.core.executor import LegIntent, TwoLegExecutor
from perp_arb.core.types import (
    BookLevel,
    MarketInfo,
    OrderBook,
    OrderResult,
    OrderStatus,
    Side,
    Symbol,
    TerminalFill,
)

_SYM_A = Symbol(exchange="aster", raw="ETHUSDT", base="ETH", quote="USDT")
_SYM_L = Symbol(exchange="lighter", raw="ETH", base="ETH", quote="USD")
_MARKETS = {
    "aster": MarketInfo(
        symbol=_SYM_A, tick_size=Decimal("0.01"), lot_size=Decimal("0.001"),
        contract_id="ETHUSDT",
    ),
    "lighter": MarketInfo(
        symbol=_SYM_L, tick_size=Decimal("0.01"), lot_size=Decimal("0.001"),
        contract_id=0,
    ),
}


def _book(symbol: Symbol, *, bid: str = "100.00", ask: str = "100.02") -> OrderBook:
    return OrderBook(
        symbol=symbol,
        bids=[BookLevel(price=Decimal(bid), size=Decimal("100"))],
        asks=[BookLevel(price=Decimal(ask), size=Decimal("100"))],
        ts_ms=0,
    )


@dataclass
class _StubExchange:
    """Stub `BaseExchange` for executor tests — only the methods the
    executor actually calls."""

    name: str
    rest_result: OrderResult | None = None
    rest_raises: Exception | None = None
    fill_result: TerminalFill | None = None
    book_a: OrderBook = field(default_factory=lambda: _book(_SYM_A))
    book_l: OrderBook = field(default_factory=lambda: _book(_SYM_L))
    rest_calls: list[dict[str, Any]] = field(default_factory=list)

    async def place_market_order(
        self, market: MarketInfo, side: Side, qty: Decimal,
        *, reduce_only: bool = False, client_id: str | None = None,
    ) -> OrderResult:
        self.rest_calls.append({
            "side": side, "qty": qty, "reduce_only": reduce_only,
            "client_id": client_id,
        })
        if self.rest_raises is not None:
            raise self.rest_raises
        assert self.rest_result is not None
        return self.rest_result

    def register_fill_slot(self, client_id: str) -> None:
        pass

    async def await_fill(
        self, client_id: str, requested_qty: Decimal, timeout_s: float,
    ) -> TerminalFill | None:
        return self.fill_result

    def release_fill_slot(self, client_id: str) -> None:
        pass

    def order_book(self, market: MarketInfo) -> OrderBook:
        return self.book_a if market.symbol.exchange == "aster" else self.book_l


def _ok(*, venue: str, side: Side, qty: Decimal, avg: str | None = "100.00") -> OrderResult:
    return OrderResult(
        success=True,
        order_id=f"{venue}-ok-1",
        client_id="cid",
        side=side,
        requested_qty=qty,
        filled_qty=qty if avg else None,
        avg_price=Decimal(avg) if avg else None,
        status=OrderStatus.FILLED if avg else OrderStatus.OPEN,
        latency_ms=10,
    )


def _bad(*, side: Side, qty: Decimal, msg: str) -> OrderResult:
    return OrderResult(
        success=False, side=side, requested_qty=qty, error_message=msg,
    )


def _legs(
    *,
    aster_side: Side = Side.SELL, aster_exp: str = "100.00",
    lighter_side: Side = Side.BUY, lighter_exp: str = "100.02",
    trade_id: str = "t-1",
) -> tuple[LegIntent, LegIntent]:
    return (
        LegIntent(
            venue="aster", side=aster_side,
            expected_price=Decimal(aster_exp),
            client_id=f"{trade_id}-a",
        ),
        LegIntent(
            venue="lighter", side=lighter_side,
            expected_price=Decimal(lighter_exp),
            client_id=f"{trade_id}-l",
        ),
    )


def _executor(
    aster: _StubExchange, lighter: _StubExchange, *, is_paper: bool = False,
) -> TwoLegExecutor:
    return TwoLegExecutor(
        {"aster": aster, "lighter": lighter}, _MARKETS,
        is_paper=is_paper, max_levels=3,
    )


# ---- live happy path -----------------------------------------------------

@pytest.mark.asyncio
async def test_both_legs_succeed_no_unwind() -> None:
    qty = Decimal("1.0")
    aster = _StubExchange(
        name="aster",
        rest_result=_ok(venue="aster", side=Side.SELL, qty=qty, avg="100.05"),
    )
    lighter = _StubExchange(
        name="lighter",
        rest_result=_ok(venue="lighter", side=Side.BUY, qty=qty, avg="100.07"),
    )
    ex = _executor(aster, lighter)
    timeline = Timeline()
    report = await ex.execute(
        trade_id="t-1", legs=_legs(), qty=qty, timeline=timeline,
    )

    assert report.success is True
    assert report.failure_reason is None
    assert [leg.kind for leg in report.legs] == [LegKind.ENTRY, LegKind.ENTRY]
    assert [leg.exchange for leg in report.legs] == ["aster", "lighter"]
    # cid pass-through: whatever the caller put in LegIntent
    assert aster.rest_calls[0]["client_id"] == "t-1-a"
    assert lighter.rest_calls[0]["client_id"] == "t-1-l"
    # send_ts_ms populated on report so the caller can stamp its own record
    assert report.send_ts_ms > 0


# ---- live partial-failure quadrants --------------------------------------

@pytest.mark.asyncio
async def test_aster_ok_lighter_fail_unwinds_aster() -> None:
    qty = Decimal("1.0")
    aster = _StubExchange(
        name="aster",
        rest_result=_ok(venue="aster", side=Side.SELL, qty=qty, avg="100.05"),
    )
    lighter = _StubExchange(
        name="lighter",
        rest_result=_bad(side=Side.BUY, qty=qty, msg="sequencer rejected"),
    )
    ex = _executor(aster, lighter)
    report = await ex.execute(
        trade_id="t-1", legs=_legs(), qty=qty, timeline=Timeline(),
    )

    assert report.success is False
    assert "lighter leg failed" in (report.failure_reason or "")
    assert "sequencer rejected" in (report.failure_reason or "")
    # unwind leg appended: aster reduce-only on the OPPOSITE side
    assert len(report.legs) == 3
    assert report.legs[2].kind is LegKind.UNWIND
    assert report.legs[2].exchange == "aster"
    # Second aster call is the unwind: opposite side + reduce_only
    assert len(aster.rest_calls) == 2
    unwind_call = aster.rest_calls[1]
    assert unwind_call["reduce_only"] is True
    assert unwind_call["side"] is Side.BUY  # opposite of SELL entry


@pytest.mark.asyncio
async def test_lighter_ok_aster_fail_unwinds_lighter() -> None:
    qty = Decimal("1.0")
    aster = _StubExchange(
        name="aster",
        rest_result=_bad(side=Side.SELL, qty=qty, msg="aster timeout"),
    )
    lighter = _StubExchange(
        name="lighter",
        rest_result=_ok(venue="lighter", side=Side.BUY, qty=qty, avg="100.07"),
    )
    ex = _executor(aster, lighter)
    report = await ex.execute(
        trade_id="t-1", legs=_legs(), qty=qty, timeline=Timeline(),
    )

    assert report.success is False
    assert "aster leg failed" in (report.failure_reason or "")
    assert len(report.legs) == 3
    assert report.legs[2].kind is LegKind.UNWIND
    assert report.legs[2].exchange == "lighter"
    unwind_call = lighter.rest_calls[1]
    assert unwind_call["reduce_only"] is True
    assert unwind_call["side"] is Side.SELL  # opposite of BUY entry


@pytest.mark.asyncio
async def test_both_legs_fail_no_unwind() -> None:
    qty = Decimal("1.0")
    aster = _StubExchange(
        name="aster", rest_result=_bad(side=Side.SELL, qty=qty, msg="A down"),
    )
    lighter = _StubExchange(
        name="lighter", rest_result=_bad(side=Side.BUY, qty=qty, msg="L down"),
    )
    ex = _executor(aster, lighter)
    report = await ex.execute(
        trade_id="t-1", legs=_legs(), qty=qty, timeline=Timeline(),
    )

    assert report.success is False
    assert report.failure_reason == "both legs failed"
    # No unwind leg — nothing to flatten
    assert all(leg.kind is LegKind.ENTRY for leg in report.legs)
    assert len(report.legs) == 2


# ---- WS fill paths -------------------------------------------------------

@pytest.mark.asyncio
async def test_ws_fill_overrides_rest_price() -> None:
    """When WS fill carries a real price, LegReport prefers it over REST."""
    qty = Decimal("1.0")
    aster = _StubExchange(
        name="aster",
        rest_result=_ok(venue="aster", side=Side.SELL, qty=qty, avg="100.00"),
        fill_result=TerminalFill(
            filled_qty=Decimal("1.0"),
            weighted_price_sum=Decimal("100.10"),
            last_ts_ms=1_700_000_000_500,
            last_status=OrderStatus.FILLED,
        ),
    )
    lighter = _StubExchange(
        name="lighter",
        rest_result=_ok(venue="lighter", side=Side.BUY, qty=qty, avg="100.00"),
    )
    ex = _executor(aster, lighter)
    report = await ex.execute(
        trade_id="t-1", legs=_legs(), qty=qty, timeline=Timeline(),
    )

    aster_leg = report.legs[0]
    assert aster_leg.realized_price == Decimal("100.10")
    assert aster_leg.fill_ts_ms == 1_700_000_000_500


@pytest.mark.asyncio
async def test_ws_fill_timeout_falls_back_to_rest() -> None:
    """REST ok but WS never delivered: report stays success=True (REST is
    the success oracle); LegReport falls back to REST avg_price."""
    qty = Decimal("1.0")
    aster = _StubExchange(
        name="aster",
        rest_result=_ok(venue="aster", side=Side.SELL, qty=qty, avg="100.05"),
        fill_result=None,
    )
    lighter = _StubExchange(
        name="lighter",
        rest_result=_ok(venue="lighter", side=Side.BUY, qty=qty, avg="100.07"),
        fill_result=TerminalFill(),  # empty: registered, no events
    )
    ex = _executor(aster, lighter)
    report = await ex.execute(
        trade_id="t-1", legs=_legs(), qty=qty, timeline=Timeline(),
    )

    assert report.success is True
    assert report.legs[0].realized_price == Decimal("100.05")
    assert report.legs[1].realized_price == Decimal("100.07")


# ---- paper mode ----------------------------------------------------------

@pytest.mark.asyncio
async def test_paper_synth_uses_book_vwap() -> None:
    """Paper mode synthesizes prices from current book VWAP — SELL fills
    against the bid, BUY against the ask."""
    qty = Decimal("1.0")
    aster = _StubExchange(name="aster", book_a=_book(_SYM_A, bid="100.00", ask="100.02"))
    lighter = _StubExchange(name="lighter", book_l=_book(_SYM_L, bid="100.04", ask="100.06"))
    ex = _executor(aster, lighter, is_paper=True)
    report = await ex.execute(
        trade_id="t-1", legs=_legs(), qty=qty, timeline=Timeline(),
    )

    assert report.success is True
    assert report.legs[0].realized_price == Decimal("100.00")  # aster bid (SELL)
    assert report.legs[1].realized_price == Decimal("100.06")  # lighter ask (BUY)
    # No REST calls — paper short-circuits
    assert aster.rest_calls == [] and lighter.rest_calls == []


@pytest.mark.asyncio
async def test_paper_does_not_unwind() -> None:
    qty = Decimal("1.0")
    aster = _StubExchange(name="aster", book_a=_book(_SYM_A))
    lighter = _StubExchange(name="lighter", book_l=_book(_SYM_L))
    ex = _executor(aster, lighter, is_paper=True)
    report = await ex.execute(
        trade_id="t-1", legs=_legs(), qty=qty, timeline=Timeline(),
    )

    assert len(report.legs) == 2
    assert all(leg.kind is LegKind.ENTRY for leg in report.legs)


# ---- side flip works through LegIntent ----------------------------------

@pytest.mark.asyncio
async def test_legs_with_opposite_sides() -> None:
    """The executor doesn't care which direction the strategy is in —
    LegIntent's sides are passed through verbatim."""
    qty = Decimal("1.0")
    aster = _StubExchange(
        name="aster",
        rest_result=_ok(venue="aster", side=Side.BUY, qty=qty, avg="100.02"),
    )
    lighter = _StubExchange(
        name="lighter",
        rest_result=_ok(venue="lighter", side=Side.SELL, qty=qty, avg="100.04"),
    )
    ex = _executor(aster, lighter)
    report = await ex.execute(
        trade_id="t-1",
        legs=_legs(aster_side=Side.BUY, lighter_side=Side.SELL),
        qty=qty, timeline=Timeline(),
    )

    assert report.success is True
    assert aster.rest_calls[0]["side"] is Side.BUY
    assert lighter.rest_calls[0]["side"] is Side.SELL


# ---- timeline marks survive ---------------------------------------------

@pytest.mark.asyncio
async def test_timeline_marks_set_send_and_per_leg_result() -> None:
    qty = Decimal("1.0")

    class _SlowLighter(_StubExchange):
        async def place_market_order(self, *args, **kwargs):  # type: ignore[override]
            await asyncio.sleep(0.01)
            return await super().place_market_order(*args, **kwargs)

    aster = _StubExchange(
        name="aster",
        rest_result=_ok(venue="aster", side=Side.SELL, qty=qty, avg="100.05"),
    )
    lighter = _SlowLighter(
        name="lighter",
        rest_result=_ok(venue="lighter", side=Side.BUY, qty=qty, avg="100.07"),
    )
    ex = _executor(aster, lighter)
    timeline = Timeline()
    await ex.execute(
        trade_id="t-1", legs=_legs(), qty=qty, timeline=timeline,
    )

    assert timeline.get(Phase.SEND) is not None
    assert timeline.get("result_aster") is not None
    assert timeline.get("result_lighter") is not None
    assert timeline.get(Phase.RESULT) is not None
    # Per-leg marks reflect REST completion order, not gather completion
    assert timeline.get("result_lighter") > timeline.get("result_aster")  # type: ignore[operator]
