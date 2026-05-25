"""Per-trade realized PnL for cash-flow-based spread arb.

Single source of the formula shared by live (`core/executor.py`) and
backtest (`backtest/pnl.py`). For mean-reverting arb where positions
flatten over time, the cumulative sum of pair-PnLs equals realized PnL.
"""

from __future__ import annotations

from decimal import Decimal

from .types import LegOutcome, Side


def leg_cash_flow(side: Side, price: Decimal, qty: Decimal) -> Decimal:
    """SELL brings cash in (+), BUY sends it out (−)."""
    return -Decimal(side.sign) * price * qty


def pair_pnl(
    left_side: Side, left_price: Decimal,
    right_side: Side, right_price: Decimal,
    qty: Decimal,
    fees: Decimal,
) -> Decimal:
    """Cash-flow PnL for one two-leg trade, net of fees."""
    return (
        leg_cash_flow(left_side, left_price, qty)
        + leg_cash_flow(right_side, right_price, qty)
        - fees
    )


def pair_pnl_from_legs(left: LegOutcome, right: LegOutcome) -> Decimal | None:
    """Live-side wrapper: build pair PnL from two filled `LegOutcome`s.

    Each leg's cash flow uses its own `filled_qty` so partial-fill asymmetry
    flows through correctly. Returns None if either leg lacks a realized
    price or fill — caller should not record PnL in that case.
    """
    left_price = left.avg_price
    right_price = right.avg_price
    if (
        left_price is None or left.filled_qty == 0 or left.side is None
        or right_price is None or right.filled_qty == 0 or right.side is None
    ):
        return None
    fees = left.total_fee + right.total_fee
    return (
        leg_cash_flow(left.side, left_price, left.filled_qty)
        + leg_cash_flow(right.side, right_price, right.filled_qty)
        - fees
    )
