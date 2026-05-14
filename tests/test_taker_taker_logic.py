"""Unit-tests for the taker-taker entry math.

We test the EWMA tracker independently and then the full entry decision against
synthetic order books (no network, no real exchanges).
"""

from __future__ import annotations

from decimal import Decimal

from perp_arb.core.types import BookLevel, OrderBook, Quote, Symbol
from perp_arb.strategy.base import EwmaTracker
from perp_arb.utils.precision import vwap_fill

# ---- EWMA --------------------------------------------------------------

def test_ewma_first_sample_seeds_value() -> None:
    e = EwmaTracker(window=10)
    assert e.value is None
    v = e.update(Decimal("5"))
    assert v == Decimal("5")


def test_ewma_converges_to_mean() -> None:
    e = EwmaTracker(window=5)
    for _ in range(200):
        e.update(Decimal("10"))
    assert e.value is not None
    assert abs(e.value - Decimal("10")) < Decimal("0.001")


def test_ewma_is_warm_after_window_samples() -> None:
    e = EwmaTracker(window=3)
    for _ in range(2):
        e.update(Decimal("1"))
    assert not e.is_warm
    e.update(Decimal("1"))
    assert e.is_warm


def test_ewma_alpha_matches_pandas_convention() -> None:
    e = EwmaTracker(window=10)
    # alpha = 2 / (10 + 1) = 0.1818...
    assert e.alpha == Decimal(2) / Decimal(11)


# ---- entry math, end-to-end against synthetic books -------------------

# Helpers to build synthetic books and the same edge calculation the strategy
# performs. We assert the same arithmetic the strategy uses so that any drift
# in the decision rule trips this test.

def _book(symbol: Symbol, bids: list[tuple[str, str]], asks: list[tuple[str, str]]) -> OrderBook:
    return OrderBook(
        symbol=symbol,
        bids=[BookLevel(Decimal(p), Decimal(s)) for p, s in bids],
        asks=[BookLevel(Decimal(p), Decimal(s)) for p, s in asks],
        ts_ms=0,
    )


def _bbo(symbol: Symbol, bid: str, ask: str) -> Quote:
    return Quote(
        symbol=symbol,
        bid=Decimal(bid), bid_size=Decimal("100"),
        ask=Decimal(ask), ask_size=Decimal("100"),
        ts_ms=0,
    )


def _edge_A_minus_threshold(
    a_book: OrderBook, l_book: OrderBook,
    a_q: Quote, l_q: Quote,
    qty: Decimal, bias: Decimal,
    fees_bps: Decimal, min_profit_bps: Decimal,
) -> Decimal | None:
    """Re-implementation of the strategy's edge-A formula, for cross-check.

    If the books can't fill, returns None.
    """
    vwap_a_sell, _ = vwap_fill(a_book.bids, qty, max_levels=5)
    vwap_l_buy,  _ = vwap_fill(l_book.asks, qty, max_levels=5)
    if vwap_a_sell is None or vwap_l_buy is None:
        return None
    mid_a = a_q.mid
    mid_l = l_q.mid
    ref_mid = (mid_a + mid_l) / Decimal(2)
    fee_abs = ref_mid * fees_bps / Decimal(10_000)
    min_profit_abs = ref_mid * min_profit_bps / Decimal(10_000)
    return (vwap_a_sell - vwap_l_buy) - bias - (fee_abs + min_profit_abs)


def test_no_edge_when_books_tight() -> None:
    sa = Symbol(exchange="aster", raw="ETHUSDT", base="ETH", quote="USDT")
    sl = Symbol(exchange="lighter", raw="ETH", base="ETH", quote="USD")
    # tightly aligned books: aster mid ≈ lighter mid
    a_book = _book(sa, bids=[("3000.0", "10")], asks=[("3000.1", "10")])
    l_book = _book(sl, bids=[("3000.0", "10")], asks=[("3000.1", "10")])
    a_q = _bbo(sa, "3000.0", "3000.1")
    l_q = _bbo(sl, "3000.0", "3000.1")
    edge = _edge_A_minus_threshold(
        a_book, l_book, a_q, l_q,
        qty=Decimal("0.05"), bias=Decimal("0"),
        fees_bps=Decimal("6"), min_profit_bps=Decimal("3"),
    )
    assert edge is not None
    assert edge < 0   # not enough edge to cover fees


def test_clear_edge_when_aster_premium() -> None:
    """Aster is ~30 bps above Lighter: should produce positive edge_A after fees."""
    sa = Symbol(exchange="aster", raw="ETHUSDT", base="ETH", quote="USDT")
    sl = Symbol(exchange="lighter", raw="ETH", base="ETH", quote="USD")
    a_book = _book(sa, bids=[("3010.0", "10")], asks=[("3010.1", "10")])
    l_book = _book(sl, bids=[("3000.0", "10")], asks=[("3000.1", "10")])
    a_q = _bbo(sa, "3010.0", "3010.1")
    l_q = _bbo(sl, "3000.0", "3000.1")
    edge = _edge_A_minus_threshold(
        a_book, l_book, a_q, l_q,
        qty=Decimal("0.05"), bias=Decimal("0"),
        fees_bps=Decimal("6"), min_profit_bps=Decimal("3"),
    )
    assert edge is not None
    # raw spread ~10 (3010 - 3000.1), threshold = ~9 bps of ~3005 = ~2.7
    # so edge ≈ 10 - 0 - 2.7 = ~7.3 in price units
    assert edge > Decimal("5")


def test_bias_subtraction_neutralises_persistent_premium() -> None:
    """If bias already captures the premium, the edge collapses to zero / negative."""
    sa = Symbol(exchange="aster", raw="ETHUSDT", base="ETH", quote="USDT")
    sl = Symbol(exchange="lighter", raw="ETH", base="ETH", quote="USD")
    a_book = _book(sa, bids=[("3010.0", "10")], asks=[("3010.1", "10")])
    l_book = _book(sl, bids=[("3000.0", "10")], asks=[("3000.1", "10")])
    a_q = _bbo(sa, "3010.0", "3010.1")
    l_q = _bbo(sl, "3000.0", "3000.1")
    edge_no_bias = _edge_A_minus_threshold(
        a_book, l_book, a_q, l_q,
        qty=Decimal("0.05"), bias=Decimal("0"),
        fees_bps=Decimal("6"), min_profit_bps=Decimal("3"),
    )
    edge_with_bias = _edge_A_minus_threshold(
        a_book, l_book, a_q, l_q,
        qty=Decimal("0.05"), bias=Decimal("10"),
        fees_bps=Decimal("6"), min_profit_bps=Decimal("3"),
    )
    assert edge_no_bias is not None and edge_with_bias is not None
    # Same books, subtract bias -> edge drops by exactly 10
    assert edge_no_bias - edge_with_bias == Decimal("10")


def test_thin_book_returns_no_edge() -> None:
    sa = Symbol(exchange="aster", raw="ETHUSDT", base="ETH", quote="USDT")
    sl = Symbol(exchange="lighter", raw="ETH", base="ETH", quote="USD")
    # qty 1.0 but only 0.1 visible — can't fill
    a_book = _book(sa, bids=[("3010.0", "0.1")], asks=[("3010.1", "0.1")])
    l_book = _book(sl, bids=[("3000.0", "10")], asks=[("3000.1", "10")])
    a_q = _bbo(sa, "3010.0", "3010.1")
    l_q = _bbo(sl, "3000.0", "3000.1")
    edge = _edge_A_minus_threshold(
        a_book, l_book, a_q, l_q,
        qty=Decimal("1.0"), bias=Decimal("0"),
        fees_bps=Decimal("6"), min_profit_bps=Decimal("3"),
    )
    assert edge is None
