"""Unit tests for `core.pnl` — the cash-flow PnL primitive shared by
live (executor) and backtest. Mirrors the backtest's `apply_pair`
formula so any drift breaks here first."""

from __future__ import annotations

from decimal import Decimal

from perp_arb.core.exec_record import LegReport
from perp_arb.core.pnl import leg_cash_flow, pair_pnl, pair_pnl_from_legs
from perp_arb.core.types import Side


def _leg(*, exchange: str, side: Side, qty: str, price: str, fee: str = "0") -> LegReport:
    return LegReport(
        exchange=exchange, side=side.value,
        requested_qty=Decimal(qty), filled_qty=Decimal(qty),
        expected_price=Decimal(price), realized_price=Decimal(price),
        status="filled", success=True, fee=Decimal(fee),
    )


def test_leg_cash_flow_sell_in_buy_out() -> None:
    assert leg_cash_flow(Side.SELL, Decimal("100"), Decimal("1")) == Decimal("100")
    assert leg_cash_flow(Side.BUY, Decimal("100"), Decimal("1")) == Decimal("-100")


def test_pair_pnl_zero_when_flat_and_no_fees() -> None:
    # Sell @ 100 / Buy @ 100 → 0 spread, 0 fees → 0 PnL.
    assert pair_pnl(
        Side.SELL, Decimal("100"),
        Side.BUY, Decimal("100"),
        Decimal("1"), Decimal("0"),
    ) == Decimal("0")


def test_pair_pnl_captures_spread() -> None:
    # Sell @ 100.10 / Buy @ 100.00 → +0.10/unit × 1 qty = +0.10.
    assert pair_pnl(
        Side.SELL, Decimal("100.10"),
        Side.BUY, Decimal("100.00"),
        Decimal("1"), Decimal("0"),
    ) == Decimal("0.10")


def test_pair_pnl_subtracts_fees() -> None:
    # Same +0.10 gross, minus 0.04 fees → +0.06 net.
    assert pair_pnl(
        Side.SELL, Decimal("100.10"),
        Side.BUY, Decimal("100.00"),
        Decimal("1"), Decimal("0.04"),
    ) == Decimal("0.06")


def test_pair_pnl_from_legs_happy_path() -> None:
    # Aster sells 1 @ 100.10 with 0.03 fee; lighter buys 1 @ 100.00 with 0 fee.
    # gross = 100.10 - 100.00 = 0.10; fees = 0.03; net = 0.07.
    left = _leg(exchange="aster", side=Side.SELL, qty="1",
                price="100.10", fee="0.03")
    right = _leg(exchange="lighter", side=Side.BUY, qty="1",
                 price="100.00", fee="0")
    assert pair_pnl_from_legs(left, right) == Decimal("0.07")


def test_pair_pnl_from_legs_returns_none_when_realized_price_missing() -> None:
    left = _leg(exchange="aster", side=Side.SELL, qty="1", price="100")
    left.realized_price = None
    right = _leg(exchange="lighter", side=Side.BUY, qty="1", price="100")
    assert pair_pnl_from_legs(left, right) is None


def test_pair_pnl_from_legs_handles_asymmetric_partial_fills() -> None:
    # Left fills 1.0 @ 100.10; right fills only 0.5 @ 100.00.
    # left_cash  = +100.10 * 1.0 = +100.10
    # right_cash = -100.00 * 0.5 =  -50.00
    # net (no fees) = +50.10
    left = _leg(exchange="aster", side=Side.SELL, qty="1.0", price="100.10")
    right = _leg(exchange="lighter", side=Side.BUY, qty="0.5", price="100.00")
    assert pair_pnl_from_legs(left, right) == Decimal("50.10")
