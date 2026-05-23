"""Per-venue signed position + realised PnL tracker.

Mirrors `taker_taker.SyntheticPosition` but venue-keyed by string so the
backtest is venue-agnostic (works for katana↔lighter, lighter↔aster, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from ..core.pnl import pair_pnl
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

        Cash-flow PnL via the shared `core.pnl.pair_pnl` primitive; backtest
        synthesizes per-leg fees from a uniform bps knob, while live reads
        absolute fees from the WS stream. Same formula, different fee source.
        """
        fees = (left_price + right_price) * qty * fee_bps_per_leg / Decimal(10_000)
        leg_pnl = pair_pnl(
            left_side, left_price,
            right_side, right_price,
            qty, fees,
        )
        self.sizes[left_venue] = self.sizes.get(left_venue, Decimal("0")) + qty * Decimal(left_side.sign)
        self.sizes[right_venue] = self.sizes.get(right_venue, Decimal("0")) + qty * Decimal(right_side.sign)
        self.realised_pnl += leg_pnl
        return leg_pnl
