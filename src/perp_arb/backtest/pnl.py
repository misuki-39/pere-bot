"""Per-venue signed position + realised PnL tracker.

Mirrors `taker_taker.SyntheticPosition` but venue-keyed by string so the
backtest is venue-agnostic (works for katana↔lighter, lighter↔aster, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from ..core.types import Side


@dataclass(slots=True)
class SyntheticPositions:
    """Signed position per venue + realised PnL across all fills."""
    sizes: dict[str, Decimal] = field(default_factory=dict)
    realised_pnl: Decimal = Decimal("0")

    def position(self, venue: str) -> Decimal:
        return self.sizes.get(venue, Decimal("0"))

    def apply_pair(
        self,
        left_venue: str, left_side: Side, left_price: Decimal,
        right_venue: str, right_side: Side, right_price: Decimal,
        qty: Decimal,
        fee_bps_per_leg: Decimal,
    ) -> Decimal:
        """Apply both legs atomically. Returns the leg-pair realised PnL in
        quote units.

        Per-leg PnL = side.sign * price * qty  (sell=+revenue, buy=−cost),
        minus the per-leg fee (bps of leg notional). The pair's net PnL is the
        sum across legs; positions update accordingly.
        """
        # cash flow per leg (sell brings cash in, buy sends it out)
        left_cash = -Decimal(left_side.sign) * left_price * qty
        right_cash = -Decimal(right_side.sign) * right_price * qty
        fees = (left_price + right_price) * qty * fee_bps_per_leg / Decimal(10_000)
        leg_pnl = left_cash + right_cash - fees

        self.sizes[left_venue] = self.sizes.get(left_venue, Decimal("0")) + qty * Decimal(left_side.sign)
        self.sizes[right_venue] = self.sizes.get(right_venue, Decimal("0")) + qty * Decimal(right_side.sign)
        self.realised_pnl += leg_pnl
        return leg_pnl
