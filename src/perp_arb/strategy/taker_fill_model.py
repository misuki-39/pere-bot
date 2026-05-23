"""Taker-execution fill model for the spread-reversion signal.

This module owns the part of the decision math that *assumes* both legs
execute as market-order takers: it walks the visible book depth to price a
`qty`-sized market order on each side. The reversion signal
(`reversion_signal.py`) consumes the four fill prices it produces but knows
nothing about how they were derived — swapping in a different execution
style means swapping this module, not the signal.

Depth protection is structural: `max_levels` caps the VWAP walk, so a fill
that would have to walk further than that returns `NO_DEPTH`. Economic
protection lives in the signal layer's fee+min-profit threshold, which sees
the same VWAPs and rejects unprofitable books automatically. No separate
"slippage vs mid" gate — that would double-count BBO half-spread, which is
already priced into the VWAPs the signal evaluates.

`compute_taker_fills` is pure: no I/O, no global state, never raises for
ordinary market states.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from ..core.types import OrderBook
from ..utils.precision import vwap_fill


@dataclass(frozen=True, slots=True)
class TakerFills:
    """Depth-aware VWAP fill prices for a `qty`-sized market order.

    `left_sell` is the price a market SELL of the left leg fills at (walking
    the bids); `left_buy` walks the asks; the right leg is analogous."""

    left_sell: Decimal
    left_buy: Decimal
    right_sell: Decimal
    right_buy: Decimal


class FillAbortKind(StrEnum):
    """Why the taker fill model declined to price the tick."""

    NO_DEPTH = "no_depth"      # qty does not fill within max_levels


@dataclass(frozen=True, slots=True)
class FillAbort:
    """A tick the taker fill model could not price (book too thin for qty
    within max_levels)."""

    kind: FillAbortKind


@dataclass(frozen=True, slots=True)
class TakerFillParams:
    """Static taker fill-model parameters; build once per strategy instance.

      qty:        order size each leg's market order must fill.
      max_levels: depth cap for the VWAP walk — a fill that needs more
                  levels than this is treated as no-depth.
    """

    qty: Decimal
    max_levels: int


def compute_taker_fills(
    p: TakerFillParams,
    left_book: OrderBook,
    right_book: OrderBook,
) -> TakerFills | FillAbort:
    """Price a `qty`-sized taker market order on all four leg+side combos.

    Returns `TakerFills` on success, or `FillAbort(NO_DEPTH)` if the qty
    doesn't fit within `max_levels` on any side. Pure; never raises for
    ordinary market states."""
    qty = p.qty
    vls, _ = vwap_fill(left_book.bids,  qty, max_levels=p.max_levels)
    vlb, _ = vwap_fill(left_book.asks,  qty, max_levels=p.max_levels)
    vrs, _ = vwap_fill(right_book.bids, qty, max_levels=p.max_levels)
    vrb, _ = vwap_fill(right_book.asks, qty, max_levels=p.max_levels)
    if vls is None or vlb is None or vrs is None or vrb is None:
        return FillAbort(FillAbortKind.NO_DEPTH)
    return TakerFills(left_sell=vls, left_buy=vlb, right_sell=vrs, right_buy=vrb)
