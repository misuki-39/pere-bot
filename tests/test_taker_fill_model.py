"""Unit-tests for the taker fill model (`strategy/taker_fill_model.py`).

`compute_taker_fills` walks synthetic book depth to price a `qty`-sized market
order on each leg+side. Depth protection is `max_levels` — a fill that needs
more levels than that aborts as NO_DEPTH. No separate slippage gate: the
signal layer's edge formula already operates on the VWAPs and rejects
unprofitable books via the fees+min-profit threshold.
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


def _params(*, qty: str = "1", max_levels: int = 3) -> TakerFillParams:
    return TakerFillParams(qty=Decimal(qty), max_levels=max_levels)


def test_happy_path_returns_top_of_book_fills() -> None:
    """qty fits in the top level → each fill is that level's price."""
    left = _book(bids=[("100.0", "10")], asks=[("100.1", "10")])
    right = _book(bids=[("99.0", "10")], asks=[("99.1", "10")])
    out = compute_taker_fills(_params(), left, right)
    assert isinstance(out, TakerFills)
    assert out.left_sell == Decimal("100.0")    # market SELL walks the bids
    assert out.left_buy == Decimal("100.1")     # market BUY walks the asks
    assert out.right_sell == Decimal("99.0")
    assert out.right_buy == Decimal("99.1")


def test_no_depth_aborts_when_qty_exceeds_visible_size() -> None:
    """Only 0.1 visible but qty=1 → NO_DEPTH."""
    left = _book(bids=[("100.0", "0.1")], asks=[("100.1", "0.1")])
    right = _book(bids=[("99.0", "10")], asks=[("99.1", "10")])
    out = compute_taker_fills(_params(qty="1"), left, right)
    assert isinstance(out, FillAbort)
    assert out.kind is FillAbortKind.NO_DEPTH
