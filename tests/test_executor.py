"""Tests for `TwoLegExecutor` — the order-execution layer between
strategy decisions and venue drivers.

Tests pin the four partial-failure quadrants, the paper-mode synth path,
PnL computation, and a few representative latency/leg-report invariants.
Driver behaviour is stubbed at the `submit_and_await` boundary: these
tests never hit the real `aster` / `lighter` clients. The ack/WS merge
itself is tested separately in test_fill_enrichment.py.

The executor is strategy-agnostic: it takes `LegIntent`s with already-
resolved sides + expected prices, plus a `Timeline` writeback target.
These tests never construct a `Decision` — that's the strategy's job."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pytest

from perp_arb.core.decision import Phase, Timeline
from perp_arb.core.executor import LegIntent, TwoLegExecutor
from perp_arb.core.pnl import pair_pnl_from_legs
from perp_arb.core.types import (
    BookLevel,
    LegKind,
    LegOutcome,
    MarketInfo,
    OrderBook,
    OrderStatus,
    Side,
    Symbol,
)

_SYM_A = Symbol(exchange="aster", raw="ETHUSDT", base="ETH", quote="USDT")
_SYM_L = Symbol(exchange="lighter", raw="ETH", base="ETH", quote="USD")
# Stub fixture binds leg_a -> aster, leg_b -> lighter. Strategy-layer code
# only sees the leg labels; the actual venue identity ("aster"/"lighter")
# is what each stub instance reports via its `.name` field and what the
# executor stamps into LegOutcome.venue.
_MARKETS = {
    "leg_a": MarketInfo(
        symbol=_SYM_A, tick_size=Decimal("0.01"), lot_size=Decimal("0.001"),
        contract_id="ETHUSDT",
    ),
    "leg_b": MarketInfo(
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
    """Stub `BaseExchange` for executor tests. The stub overrides
    `submit_and_await` (the merged-outcome entry point the executor uses)
    and `place_market_order` (the unwind path). The real driver-side
    ack/WS merge logic is exercised separately in test_fill_enrichment.py
    against `BaseExchange.submit_and_await`."""

    name: str
    submit_outcome: LegOutcome | None = None
    submit_raises: Exception | None = None
    book_a: OrderBook = field(default_factory=lambda: _book(_SYM_A))
    book_l: OrderBook = field(default_factory=lambda: _book(_SYM_L))
    submit_calls: list[dict[str, Any]] = field(default_factory=list)

    async def submit_and_await(
        self, market: MarketInfo, side: Side, qty: Decimal,
        *, client_id: str, timeout_s: float = 5.0, reduce_only: bool = False,
    ) -> LegOutcome:
        self.submit_calls.append({
            "side": side, "qty": qty, "reduce_only": reduce_only,
            "client_id": client_id,
        })
        if self.submit_raises is not None:
            raise self.submit_raises
        assert self.submit_outcome is not None
        return self.submit_outcome

    async def place_market_order(
        self, market: MarketInfo, side: Side, qty: Decimal,
        *, client_id: str, reduce_only: bool = False,
    ) -> LegOutcome:
        # Used only by the unwind path. Reuse submit_outcome's shape but
        # always return a success-shaped outcome (the unwind doesn't need
        # to fail in any test today; if it does, override per-instance).
        self.submit_calls.append({
            "side": side, "qty": qty, "reduce_only": reduce_only,
            "client_id": client_id,
        })
        assert self.submit_outcome is not None
        return self.submit_outcome

    def order_book(self, market: MarketInfo) -> OrderBook:
        return self.book_a if market.symbol.exchange == "aster" else self.book_l


def _ok(*, side: Side, qty: Decimal, avg: str = "100.00",
        fee: str = "0", exchange_ts: int | None = None) -> LegOutcome:
    """Build a successful merged outcome (ack + WS already reconciled)."""
    out = LegOutcome(
        success=True,
        client_id="cid",
        side=side,
        requested_qty=qty,
        status=OrderStatus.FILLED,
        exchange_ts_ms=exchange_ts,
        total_fee=Decimal(fee),
    )
    out.set_fill(qty, Decimal(avg))
    return out


def _bad(*, side: Side, qty: Decimal, msg: str) -> LegOutcome:
    return LegOutcome(
        success=False, side=side, requested_qty=qty, error_message=msg,
    )


def _legs(
    *,
    a_side: Side = Side.SELL, a_exp: str = "100.00",
    b_side: Side = Side.BUY, b_exp: str = "100.02",
) -> tuple[LegIntent, LegIntent]:
    return (
        LegIntent(venue="leg_a", side=a_side, expected_price=Decimal(a_exp)),
        LegIntent(venue="leg_b", side=b_side, expected_price=Decimal(b_exp)),
    )


_TEST_CID_SEED = 1_000  # deterministic so tests can assert on the cid string


def _executor(
    aster: _StubExchange, lighter: _StubExchange, *, is_paper: bool = False,
) -> TwoLegExecutor:
    # Stub fixture binds leg_a -> aster, leg_b -> lighter.
    return TwoLegExecutor(
        {"leg_a": aster, "leg_b": lighter}, _MARKETS,
        is_paper=is_paper, max_levels=3, cid_seed=_TEST_CID_SEED,
    )


# ---- live happy path -----------------------------------------------------

@pytest.mark.asyncio
async def test_both_legs_succeed_no_unwind() -> None:
    qty = Decimal("1.0")
    aster = _StubExchange(
        name="aster",
        submit_outcome=_ok(side=Side.SELL, qty=qty, avg="100.05"),
    )
    lighter = _StubExchange(
        name="lighter",
        submit_outcome=_ok(side=Side.BUY, qty=qty, avg="100.07"),
    )
    ex = _executor(aster, lighter)
    timeline = Timeline()
    report = await ex.execute(
        trade_id="t-1", legs=_legs(), qty=qty, timeline=timeline,
    )

    assert report.failure_reason is None
    assert [leg.kind for leg in report.legs] == [LegKind.ENTRY, LegKind.ENTRY]
    assert [leg.venue for leg in report.legs] == ["aster", "lighter"]
    # Each venue owns its `client_id_generator` (per-venue cid namespace),
    # so cross-venue legs both start at the seeded value — that's safe
    # because each driver's `_fill_tracker` is per-venue and keys won't
    # collide. A same-venue two-leg fire would see distinct cids from the
    # same counter (1000, 1001), which is what the original guard was for.
    assert aster.submit_calls[0]["client_id"] == str(_TEST_CID_SEED)
    assert lighter.submit_calls[0]["client_id"] == str(_TEST_CID_SEED)
    # send_ts_ms stamped onto each leg so the caller can derive its own record
    assert report.legs[0].send_ts_ms is not None and report.legs[0].send_ts_ms > 0
    # Cash-flow pair PnL: sell 100.05, buy 100.07, no fees → -0.02.
    assert pair_pnl_from_legs(report.legs[0], report.legs[1]) == Decimal("-0.02")


@pytest.mark.asyncio
async def test_realised_pnl_subtracts_fees() -> None:
    """The merged outcome carries `total_fee`; pair PnL nets fees out.
    Aster sells 100.10 with 0.03 fee; lighter buys 100.00 with 0 fee.
    Gross spread = 0.10; net PnL = 0.07."""
    qty = Decimal("1.0")
    aster = _StubExchange(
        name="aster",
        submit_outcome=_ok(side=Side.SELL, qty=qty, avg="100.10", fee="0.03"),
    )
    lighter = _StubExchange(
        name="lighter",
        submit_outcome=_ok(side=Side.BUY, qty=qty, avg="100.00"),
    )
    ex = _executor(aster, lighter)
    report = await ex.execute(
        trade_id="t-1", legs=_legs(), qty=qty, timeline=Timeline(),
    )
    assert report.failure_reason is None
    assert pair_pnl_from_legs(report.legs[0], report.legs[1]) == Decimal("0.07")
    assert report.legs[0].total_fee == Decimal("0.03")
    assert report.legs[1].total_fee == Decimal("0")


@pytest.mark.asyncio
async def test_failed_trade_has_no_realised_pnl() -> None:
    qty = Decimal("1.0")
    aster = _StubExchange(
        name="aster", submit_outcome=_bad(side=Side.SELL, qty=qty, msg="A down"),
    )
    lighter = _StubExchange(
        name="lighter", submit_outcome=_bad(side=Side.BUY, qty=qty, msg="L down"),
    )
    ex = _executor(aster, lighter)
    report = await ex.execute(
        trade_id="t-1", legs=_legs(), qty=qty, timeline=Timeline(),
    )
    assert report.failure_reason is not None
    # Both legs failed → no fills → derived PnL is None.
    assert pair_pnl_from_legs(report.legs[0], report.legs[1]) is None


# ---- live partial-failure quadrants --------------------------------------

@pytest.mark.asyncio
async def test_aster_ok_lighter_fail_no_auto_unwind() -> None:
    """The executor no longer auto-unwinds on partial failure — strategy
    owns reconcile via REST snapshot + targeted reduce-only flatten.
    All the executor does is build the failure_reason narrative."""
    qty = Decimal("1.0")
    aster = _StubExchange(
        name="aster",
        submit_outcome=_ok(side=Side.SELL, qty=qty, avg="100.05"),
    )
    lighter = _StubExchange(
        name="lighter",
        submit_outcome=_bad(side=Side.BUY, qty=qty, msg="sequencer rejected"),
    )
    ex = _executor(aster, lighter)
    report = await ex.execute(
        trade_id="t-1", legs=_legs(), qty=qty, timeline=Timeline(),
    )

    assert report.failure_reason is not None
    assert "lighter leg failed" in (report.failure_reason or "")
    assert "sequencer rejected" in (report.failure_reason or "")
    # No UNWIND leg — strategy will reconcile from REST truth.
    assert len(report.legs) == 2
    assert all(leg.kind is LegKind.ENTRY for leg in report.legs)
    # Successful leg was called exactly once (the entry) — no follow-up.
    assert len(aster.submit_calls) == 1


@pytest.mark.asyncio
async def test_lighter_ok_aster_fail_no_auto_unwind() -> None:
    """Mirror of the partial-failure case: lighter filled, aster failed.
    Strategy will reconcile; executor does not unwind."""
    qty = Decimal("1.0")
    aster = _StubExchange(
        name="aster",
        submit_outcome=_bad(side=Side.SELL, qty=qty, msg="aster timeout"),
    )
    lighter = _StubExchange(
        name="lighter",
        submit_outcome=_ok(side=Side.BUY, qty=qty, avg="100.07"),
    )
    ex = _executor(aster, lighter)
    report = await ex.execute(
        trade_id="t-1", legs=_legs(), qty=qty, timeline=Timeline(),
    )

    assert report.failure_reason is not None
    assert "aster leg failed" in (report.failure_reason or "")
    assert len(report.legs) == 2
    assert all(leg.kind is LegKind.ENTRY for leg in report.legs)
    assert len(lighter.submit_calls) == 1


@pytest.mark.asyncio
async def test_both_legs_fail_no_unwind() -> None:
    qty = Decimal("1.0")
    aster = _StubExchange(
        name="aster", submit_outcome=_bad(side=Side.SELL, qty=qty, msg="A down"),
    )
    lighter = _StubExchange(
        name="lighter", submit_outcome=_bad(side=Side.BUY, qty=qty, msg="L down"),
    )
    ex = _executor(aster, lighter)
    report = await ex.execute(
        trade_id="t-1", legs=_legs(), qty=qty, timeline=Timeline(),
    )

    assert report.failure_reason is not None
    assert report.failure_reason == "both legs failed"
    # No unwind leg — nothing to flatten
    assert all(leg.kind is LegKind.ENTRY for leg in report.legs)
    assert len(report.legs) == 2


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

    assert report.failure_reason is None
    assert report.legs[0].avg_price == Decimal("100.00")  # aster bid (SELL)
    assert report.legs[1].avg_price == Decimal("100.06")  # lighter ask (BUY)
    # No driver calls — paper short-circuits
    assert aster.submit_calls == [] and lighter.submit_calls == []


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


@pytest.mark.asyncio
async def test_paper_raises_on_empty_book() -> None:
    """`assert` would be stripped under python -O; explicit raise gives a
    clear error path regardless of optimization flags."""
    qty = Decimal("1.0")
    # bid + ask both present but capped: vwap on a missing side returns None
    empty_book = OrderBook(symbol=_SYM_A, bids=[], asks=[], ts_ms=0)
    aster = _StubExchange(name="aster", book_a=empty_book)
    lighter = _StubExchange(name="lighter", book_l=_book(_SYM_L))
    ex = _executor(aster, lighter, is_paper=True)
    with pytest.raises(RuntimeError, match="empty bids"):
        await ex.execute(
            trade_id="t-1", legs=_legs(), qty=qty, timeline=Timeline(),
        )


# ---- exception coercion from gather --------------------------------------

@pytest.mark.asyncio
async def test_unexpected_exception_coerced_to_failure_outcome() -> None:
    """A raise out of submit_and_await must NOT escape execute() —
    `return_exceptions=True` + `_coerce_outcome` convert it into a
    success=False outcome so `_handle_partial_failure` can run."""
    qty = Decimal("1.0")
    aster = _StubExchange(
        name="aster",
        submit_outcome=_ok(side=Side.SELL, qty=qty, avg="100.05"),
    )
    lighter = _StubExchange(
        name="lighter", submit_raises=RuntimeError("ws crashed"),
    )
    ex = _executor(aster, lighter)
    report = await ex.execute(
        trade_id="t-1", legs=_legs(), qty=qty, timeline=Timeline(),
    )
    # gather still resolved; raise was coerced to a fail outcome.
    assert report.failure_reason is not None
    assert "lighter leg failed" in (report.failure_reason or "")
    assert "ws crashed" in (report.failure_reason or "")
    # No auto-unwind appended — strategy will reconcile from REST.
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
        submit_outcome=_ok(side=Side.BUY, qty=qty, avg="100.02"),
    )
    lighter = _StubExchange(
        name="lighter",
        submit_outcome=_ok(side=Side.SELL, qty=qty, avg="100.04"),
    )
    ex = _executor(aster, lighter)
    report = await ex.execute(
        trade_id="t-1",
        legs=_legs(a_side=Side.BUY, b_side=Side.SELL),
        qty=qty, timeline=Timeline(),
    )

    assert report.failure_reason is None
    assert aster.submit_calls[0]["side"] is Side.BUY
    assert lighter.submit_calls[0]["side"] is Side.SELL


# ---- timeline marks survive ---------------------------------------------

@pytest.mark.asyncio
async def test_timeline_send_mark_set() -> None:
    """The executor stamps SEND so `lat_decision_send_ms` is computable.
    All other latencies derive from per-leg LegOutcome (fill_ts_ms - send_ts_ms)."""
    qty = Decimal("1.0")
    aster = _StubExchange(
        name="aster",
        submit_outcome=_ok(side=Side.SELL, qty=qty, avg="100.05"),
    )
    lighter = _StubExchange(
        name="lighter",
        submit_outcome=_ok(side=Side.BUY, qty=qty, avg="100.07"),
    )
    ex = _executor(aster, lighter)
    timeline = Timeline()
    await ex.execute(
        trade_id="t-1", legs=_legs(), qty=qty, timeline=timeline,
    )

    assert timeline.get(Phase.SEND) is not None


# ---- per-venue client_id_generator delegation ---------------------------

@pytest.mark.asyncio
async def test_executor_delegates_cid_to_venue_generator() -> None:
    """Executor must call each venue's `client_id_generator.next(side=...)`
    rather than mint cids itself. A pool-backed venue would inject a custom
    generator that returns its pre-staged COIs; this test verifies the
    plumbing using a sentinel-returning fake."""

    class _SentinelGen:
        def next(self, *, side: Side) -> str:
            return f"POOLED-{side.value}"

    qty = Decimal("1.0")
    aster = _StubExchange(
        name="aster",
        submit_outcome=_ok(side=Side.SELL, qty=qty, avg="100.05"),
    )
    lighter = _StubExchange(
        name="lighter",
        submit_outcome=_ok(side=Side.BUY, qty=qty, avg="100.07"),
    )
    # Pre-set generators; construct executor WITHOUT cid_seed so it doesn't
    # overwrite them.
    aster.client_id_generator = _SentinelGen()
    lighter.client_id_generator = _SentinelGen()
    ex = TwoLegExecutor(
        {"leg_a": aster, "leg_b": lighter}, _MARKETS,
        is_paper=False, max_levels=3,
    )
    await ex.execute(
        trade_id="t-1", legs=_legs(), qty=qty, timeline=Timeline(),
    )
    # leg_a fires SELL on aster; leg_b fires BUY on lighter.
    assert aster.submit_calls[0]["client_id"] == "POOLED-sell"
    assert lighter.submit_calls[0]["client_id"] == "POOLED-buy"
