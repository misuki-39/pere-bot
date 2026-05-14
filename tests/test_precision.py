"""round_to_tick / round_to_lot / vwap_fill — pure math, no I/O."""

from __future__ import annotations

from decimal import Decimal

from perp_arb.core.types import BookLevel
from perp_arb.utils.precision import round_to_lot, round_to_tick, vwap_fill

# ---- rounding ----------------------------------------------------------

def test_round_to_tick_basic() -> None:
    tick = Decimal("0.01")
    assert round_to_tick("3120.456", tick) == Decimal("3120.46")
    assert round_to_tick("3120.454", tick) == Decimal("3120.45")
    assert round_to_tick(Decimal("3120.5"), tick) == Decimal("3120.50")


def test_round_to_lot_floors() -> None:
    lot = Decimal("0.001")
    assert round_to_lot("0.0529", lot) == Decimal("0.052")
    assert round_to_lot("0.05", lot) == Decimal("0.050")


def test_round_to_lot_zero_lot_is_identity() -> None:
    assert round_to_lot("1.234567", Decimal("0")) == Decimal("1.234567")


# ---- vwap_fill ---------------------------------------------------------

def _levels(*prices_sizes: tuple[str, str]) -> list[BookLevel]:
    return [BookLevel(Decimal(p), Decimal(s)) for p, s in prices_sizes]


def test_vwap_single_level() -> None:
    bk = _levels(("100", "10"))
    vwap, used = vwap_fill(bk, Decimal("3"))
    assert vwap == Decimal("100")
    assert used == 1


def test_vwap_walks_levels() -> None:
    bk = _levels(("100", "2"), ("101", "5"), ("102", "100"))
    vwap, used = vwap_fill(bk, Decimal("4"))
    # take 2 @ 100, 2 @ 101 -> avg = (200 + 202) / 4 = 100.5
    assert vwap == Decimal("100.5")
    assert used == 2


def test_vwap_returns_none_if_insufficient_depth() -> None:
    bk = _levels(("100", "1"), ("101", "1"))
    vwap, used = vwap_fill(bk, Decimal("10"))
    assert vwap is None
    assert used == 2


def test_vwap_respects_max_levels() -> None:
    bk = _levels(("100", "1"), ("101", "1"), ("102", "1"), ("103", "10"))
    vwap, used = vwap_fill(bk, Decimal("4"), max_levels=3)
    assert vwap is None
    assert used == 3


def test_vwap_zero_qty_is_none() -> None:
    bk = _levels(("100", "1"))
    vwap, used = vwap_fill(bk, Decimal("0"))
    assert vwap is None
    assert used == 0


def test_vwap_skips_zero_size_levels() -> None:
    bk = _levels(("100", "0"), ("101", "5"))
    vwap, _ = vwap_fill(bk, Decimal("3"))
    assert vwap == Decimal("101")
