"""Unit-tests for the taker fill model (`strategy/taker_fill_model.py`).

`compute_taker_fills` walks synthetic book depth to price a `qty`-sized market
order on each leg+side and gates on slippage-vs-mid. We assert the happy path
and both abort kinds.
"""

from __future__ import annotations

from decimal import Decimal

from perp_arb.core.types import BookLevel, OrderBook, Symbol
from perp_arb.strategy.taker_fill_model import (
    FillAbort,
    FillAbortKind,
    TakerFillParams,
    TakerFills,
    compute_taker_fills,
)

_SYM = Symbol(exchange="aster", raw="ETHUSDT", base="ETH", quote="USDT")


def _book(bids: list[tuple[str, str]], asks: list[tuple[str, str]]) -> OrderBook:
    return OrderBook(
        symbol=_SYM,
        bids=[BookLevel(Decimal(p), Decimal(s)) for p, s in bids],
        asks=[BookLevel(Decimal(p), Decimal(s)) for p, s in asks],
        ts_ms=0,
    )


def _params(*, qty: str = "1", max_levels: int = 3,
            max_slippage_bps: str = "100") -> TakerFillParams:
    return TakerFillParams(
        qty=Decimal(qty),
        max_levels=max_levels,
        max_slippage_bps=Decimal(max_slippage_bps),
    )


def test_happy_path_returns_top_of_book_fills() -> None:
    """qty fits in the top level → each fill is that level's price."""
    left = _book(bids=[("100.0", "10")], asks=[("100.1", "10")])
    right = _book(bids=[("99.0", "10")], asks=[("99.1", "10")])
    out = compute_taker_fills(_params(), left, right,
                              mid_left=Decimal("100.05"), mid_right=Decimal("99.05"))
    assert isinstance(out, TakerFills)
    assert out.left_sell == Decimal("100.0")    # market SELL walks the bids
    assert out.left_buy == Decimal("100.1")     # market BUY walks the asks
    assert out.right_sell == Decimal("99.0")
    assert out.right_buy == Decimal("99.1")


def test_no_depth_aborts_when_qty_exceeds_visible_size() -> None:
    """Only 0.1 visible but qty=1 → NO_DEPTH, no fills carried."""
    left = _book(bids=[("100.0", "0.1")], asks=[("100.1", "0.1")])
    right = _book(bids=[("99.0", "10")], asks=[("99.1", "10")])
    out = compute_taker_fills(_params(qty="1"), left, right,
                              mid_left=Decimal("100.05"), mid_right=Decimal("99.05"))
    assert isinstance(out, FillAbort)
    assert out.kind is FillAbortKind.NO_DEPTH
    assert out.fills is None


def test_slippage_aborts_and_carries_fills() -> None:
    """A mid far from the book makes vwap-vs-mid exceed max_slippage_bps →
    SLIPPAGE, and the computed fills are carried for the Decision record."""
    left = _book(bids=[("100.0", "10")], asks=[("100.1", "10")])
    right = _book(bids=[("99.0", "10")], asks=[("99.1", "10")])
    # mid_left deliberately ~10% away from the ~100 book → far over 100 bps.
    out = compute_taker_fills(_params(max_slippage_bps="100"), left, right,
                              mid_left=Decimal("110.0"), mid_right=Decimal("99.05"))
    assert isinstance(out, FillAbort)
    assert out.kind is FillAbortKind.SLIPPAGE
    assert out.fills is not None
    assert out.fills.left_sell == Decimal("100.0")
