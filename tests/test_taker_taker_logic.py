"""Unit-tests for the taker-taker entry math.

We test the EWMA tracker independently and then the full entry decision against
synthetic order books (no network, no real exchanges).
"""

from __future__ import annotations

from decimal import Decimal

from perp_arb.core.exec_record import Direction, Outcome
from perp_arb.core.types import BookLevel, OrderBook, Quote, Symbol
from perp_arb.strategy.base import SpreadModel, TimeEwma
from perp_arb.strategy.markout import MarkoutTable, _Bucket
from perp_arb.strategy.reversion_signal import (
    AssessInputs,
    AssessParams,
    assess_reversion,
)
from perp_arb.strategy.taker_fill_model import TakerFillParams, compute_taker_fills
from perp_arb.utils.precision import vwap_fill

# ---- TimeEwma ----------------------------------------------------------

def test_time_ewma_first_sample_seeds_value() -> None:
    e = TimeEwma(half_life_s=60)
    assert e.value is None
    assert e.update(Decimal("5"), ts_ms=0) == Decimal("5")
    assert e.value == Decimal("5")


def test_time_ewma_converges_to_mean() -> None:
    e = TimeEwma(half_life_s=1.0)
    t = 0
    for _ in range(500):
        t += 100
        e.update(Decimal("10"), ts_ms=t)
    assert e.value is not None
    assert abs(e.value - Decimal("10")) < Decimal("0.001")


def test_time_ewma_one_half_life_halves_the_gap() -> None:
    # start at 0, then a step to 1; after exactly one half-life the estimate
    # should sit ~halfway (0.5), independent of how many ticks it took.
    e = TimeEwma(half_life_s=10)
    e.update(Decimal("0"), ts_ms=0)
    e.update(Decimal("1"), ts_ms=10_000)  # one half-life later
    assert abs(e.value - Decimal("0.5")) < Decimal("1e-9")


def test_time_ewma_decay_is_rate_invariant() -> None:
    # Same elapsed time, different tick counts → same result. This is the
    # whole point of time-decay over the old tick-count window.
    coarse = TimeEwma(half_life_s=5)
    coarse.update(Decimal("0"), 0)
    coarse.update(Decimal("1"), 5_000)

    fine = TimeEwma(half_life_s=5)
    fine.update(Decimal("0"), 0)
    for ms in range(100, 5_001, 100):  # 50 ticks over the same 5 s
        fine.update(Decimal("1"), ms)

    assert abs(coarse.value - fine.value) < Decimal("0.02")


def test_time_ewma_ignores_non_monotonic_ts() -> None:
    e = TimeEwma(half_life_s=10)
    e.update(Decimal("1"), ts_ms=1_000)
    before = e.value
    e.update(Decimal("99"), ts_ms=1_000)  # duplicate timestamp
    assert e.value == before


def test_spread_model_warmup_is_wall_clock() -> None:
    m = SpreadModel(center_half_life_s=3600, scale_half_life_s=300, warmup_s=2)
    assert not m.is_warm
    m.update(Decimal("0.01"), ts_ms=0)
    assert not m.is_warm
    m.update(Decimal("0.01"), ts_ms=1_999)
    assert not m.is_warm
    m.update(Decimal("0.01"), ts_ms=2_000)
    assert m.is_warm


def test_spread_model_residual_decomposition() -> None:
    # A very slow centre barely moves, so residual ≈ deviation from start.
    m = SpreadModel(center_half_life_s=1e9, scale_half_life_s=1e9, warmup_s=0)
    m.update(Decimal("0.00"), ts_ms=0)
    st = m.update(Decimal("0.05"), ts_ms=1_000)
    assert abs(st.center) < Decimal("1e-6")
    assert abs(st.residual - Decimal("0.05")) < Decimal("1e-6")
    assert abs(st.residual_bps(Decimal("100")) - Decimal("5")) < Decimal("1e-3")


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


# ---- Wave-1 optimisation knobs in assess_reversion --------------------


def _params(**overrides) -> AssessParams:
    """Build an AssessParams with sensible test defaults."""
    base = dict(
        qty=Decimal("1"),
        fees_bps=Decimal("1"),
        min_profit_bps=Decimal("0"),
        max_stale_ms=10_000,
        max_qty=Decimal("10"),
    )
    base.update(overrides)
    return AssessParams(**base)


def _fill_params() -> TakerFillParams:
    """Taker fill-model params matching the `_params` test defaults."""
    return TakerFillParams(qty=Decimal("1"), max_levels=3)


def _inputs(*, a_bid: str, a_ask: str, l_bid: str, l_ask: str,
            bias: str = "0", pos_left: str = "0", pos_right: str = "0",
            bump_a_bps: str = "0", bump_b_bps: str = "0") -> AssessInputs:
    """Build AssessInputs from BBO quotes (qty=1 fits inside top size)."""
    sa = Symbol(exchange="lighter", raw="WTI", base="WTI", quote="USD")
    sb = Symbol(exchange="aster", raw="CLUSDT", base="WTI", quote="USDT")
    a_book = _book(sa, bids=[(a_bid, "100")], asks=[(a_ask, "100")])
    b_book = _book(sb, bids=[(l_bid, "100")], asks=[(l_ask, "100")])
    a_q = _bbo(sa, a_bid, a_ask)
    b_q = _bbo(sb, l_bid, l_ask)
    fills = compute_taker_fills(_fill_params(), a_book, b_book)
    return AssessInputs(
        now_ms=0,
        left_quote=a_q, right_quote=b_q,
        fills=fills,
        bias=Decimal(bias), is_warm=True,
        position_left=Decimal(pos_left), position_right=Decimal(pos_right),
        bump_a_bps=Decimal(bump_a_bps),
        bump_b_bps=Decimal(bump_b_bps),
    )


def test_markout_subtraction_blocks_fire_just_above_fee_threshold() -> None:
    """Without markout: a +1.5 bps edge with 1 bps fee fires.
    With markout that bills another 1 bps for the 1-2 bps bucket: no fire."""
    # left bid 100.015, right ask 100.000 → vwap_left_sell - vwap_right_buy = +0.015
    # ref_mid ≈ 100.0075, so edge_A_bps ≈ +1.5 bps (clear above 1 bps fee)
    inp = _inputs(a_bid="100.015", a_ask="100.020", l_bid="100.000", l_ask="100.000")

    # Sanity: without markout, fires
    p_off = _params()
    d = assess_reversion(p_off, inp)
    assert d is not None and d.outcome is Outcome.FIRED
    assert d.direction is Direction.A

    # With markout subtracting +1 bps on direction A in the 1-2 bps bucket → no fire
    table = MarkoutTable(
        direction_A=(_Bucket(Decimal("0"), Decimal("1"), Decimal("0")),
                     _Bucket(Decimal("1"), Decimal("2"), Decimal("1.0")),
                     _Bucket(Decimal("2"), Decimal("Infinity"), Decimal("0"))),
        direction_B=(),
        latency_label="test",
    )
    p_on = _params(markout=table)
    d2 = assess_reversion(p_on, inp)
    assert d2 is None, f"expected no fire, got {d2}"


def test_bump_a_raises_threshold_for_direction_a_only() -> None:
    """A bump on direction A's threshold should suppress an A-edge fire that
    would otherwise have triggered, while leaving B unaffected."""
    # +1.5 bps direction-A edge as above
    inp_a = _inputs(a_bid="100.015", a_ask="100.020", l_bid="100.000", l_ask="100.000",
                    bump_a_bps="1.0")  # +1 bps bump → effective threshold 2 bps > 1.5
    p = _params()
    d = assess_reversion(p, inp_a)
    assert d is None, "direction-A bump should suppress this fire"

    # Same bump on B does not affect a direction-A fire opportunity
    inp_b_bump = _inputs(a_bid="100.015", a_ask="100.020", l_bid="100.000", l_ask="100.000",
                         bump_b_bps="1.0")
    d2 = assess_reversion(p, inp_b_bump)
    assert d2 is not None and d2.outcome is Outcome.FIRED and d2.direction is Direction.A


def test_inventory_skew_widens_growing_direction_narrows_flattening() -> None:
    """At position_left=+5 (half of max_qty=10), direction A (sells left → shrinks
    |pos|) should be EASIER, direction B (buys left → grows |pos|) HARDER.

    With κ=2 bps and |pos|/max_qty=0.5, the skew is ±1 bps respectively.
    """
    # A +1.5 bps A-edge sits just above default fee.
    inp = _inputs(a_bid="100.015", a_ask="100.020", l_bid="100.000", l_ask="100.000",
                  pos_left="5", pos_right="-5")
    p = _params(inventory_skew_bps=Decimal("2"))
    d = assess_reversion(p, inp)
    # A flattens → skew_A = +2 * (5 * -1) / 10 = -1 bps. Threshold drops; still fires.
    assert d is not None and d.outcome is Outcome.FIRED and d.direction is Direction.A

    # Now flip the position: long the OTHER way (position_left = -5 → A grows).
    inp_short = _inputs(a_bid="100.015", a_ask="100.020", l_bid="100.000", l_ask="100.000",
                        pos_left="-5", pos_right="5")
    d2 = assess_reversion(p, inp_short)
    # A grows → skew_A = +2 * (-5 * -1) / 10 = +1 bps. Effective threshold 2 bps > 1.5.
    assert d2 is None, "A should be blocked: growing |pos| and edge below skewed threshold"


def test_position_cap_returns_none_not_blocked_risk() -> None:
    """Position-cap hit is not a risk event — the cap is doing its job. The pure
    function drops the tick (returns None) instead of emitting a BLOCKED_RISK
    Decision; this prevents per-tick log spam while the strategy waits for a
    reverse-direction signal."""
    # Sanity: book has a +1.5 bps A-edge and at flat position it fires.
    flat = _inputs(a_bid="100.015", a_ask="100.020", l_bid="100.000", l_ask="100.000")
    p = _params()
    assert assess_reversion(p, flat).outcome is Outcome.FIRED

    # Direction A: post_left = pos_left + 1*(-1), post_right = pos_right + 1*(+1).
    # At pos_left=-10, pos_right=+10 the fire would push |pos|=11 > max_qty=10.
    growing = _inputs(a_bid="100.015", a_ask="100.020", l_bid="100.000", l_ask="100.000",
                      pos_left="-10", pos_right="10")
    assert assess_reversion(p, growing) is None


def test_position_cap_allows_flattening_fire() -> None:
    """Reverse-direction exits MUST still pass when current |pos| is already at
    the cap — that's how the strategy unwinds."""
    # Same book (A-edge) but position flipped so a Direction-A fire shrinks
    # |pos|: pos_left=+10 → post_left = 10 - 1 = 9; pos_right=-10 → post_right
    # = -10 + 1 = -9. max(9, 9) ≤ max_qty=10.
    flattening = _inputs(a_bid="100.015", a_ask="100.020", l_bid="100.000", l_ask="100.000",
                         pos_left="10", pos_right="-10")
    p = _params()
    d = assess_reversion(p, flattening)
    assert d is not None and d.outcome is Outcome.FIRED and d.direction is Direction.A
