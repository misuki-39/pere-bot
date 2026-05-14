"""Decimal-safe rounding + VWAP helpers."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal

from ..core.types import BookLevel

BPS = Decimal(10_000)


def round_to_tick(price: Decimal | float | str, tick: Decimal) -> Decimal:
    """Snap a price to the nearest multiple of `tick` (banker's-style half-up)."""
    return Decimal(str(price)).quantize(tick, rounding=ROUND_HALF_UP)


def round_to_lot(qty: Decimal | float | str, lot: Decimal) -> Decimal:
    """Floor a quantity to the venue lot size (so we never over-order)."""
    q = Decimal(str(qty))
    if lot == 0:
        return q
    return (q // lot) * lot


def vwap_fill(
    levels: Iterable[BookLevel],
    qty: Decimal,
    *,
    max_levels: int | None = None,
) -> tuple[Decimal | None, int]:
    """Size-weighted average price to fill exactly `qty` across `levels`.

    Returns (vwap, levels_consumed). vwap is None if the visible book can't
    fill the requested qty within `max_levels`.
    """
    if qty <= 0:
        return None, 0
    remaining = qty
    notional = Decimal(0)
    consumed = 0
    for lvl in levels:
        if max_levels is not None and consumed >= max_levels:
            return None, consumed
        take = min(remaining, lvl.size)
        notional += take * lvl.price
        remaining -= take
        consumed += 1
        if remaining <= 0:
            return notional / qty, consumed
    return None, consumed
