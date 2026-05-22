"""Taker-execution fill model for the spread-reversion signal.

This module owns the part of the decision math that *assumes* both legs
execute as market-order takers: it walks the visible book depth to price a
`qty`-sized market order on each side, and gates on per-leg slippage versus
mid. The reversion signal (`reversion_signal.py`) consumes the four fill
prices it produces but knows nothing about how they were derived — swapping
in a different execution style means swapping this module, not the signal.

`compute_taker_fills` is pure: no I/O, no global state, never raises for
ordinary market states. A book too thin to fill `qty`, or a fill too far
from mid, is reported as a `FillAbort`, not an exception.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from ..core.types import OrderBook
from ..utils.precision import BPS, vwap_fill


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
    SLIPPAGE = "slippage"      # a vwap-vs-mid deviation exceeds max_slippage_bps


@dataclass(frozen=True, slots=True)
class FillAbort:
    """A tick the taker fill model could not (or would not) price.

    `fills` is populated for `SLIPPAGE` — the four VWAPs are known and the
    signal layer records them on the abort `Decision` — and `None` for
    `NO_DEPTH`, where the book never filled."""

    kind: FillAbortKind
    fills: TakerFills | None = None


@dataclass(frozen=True, slots=True)
class TakerFillParams:
    """Static taker fill-model parameters; build once per strategy instance.

      qty:              order size each leg's market order must fill.
      max_levels:       depth cap for the VWAP walk — a fill that needs more
                        levels than this is treated as no-depth.
      max_slippage_bps: per-leg cap on |vwap - mid|; a wider fill aborts the
                        tick (stale / illiquid book guard).
    """

    qty: Decimal
    max_levels: int
    max_slippage_bps: Decimal


def compute_taker_fills(
    p: TakerFillParams,
    left_book: OrderBook,
    right_book: OrderBook,
    mid_left: Decimal,
    mid_right: Decimal,
) -> TakerFills | FillAbort:
    """Price a `qty`-sized taker market order on all four leg+side combos.

    Returns `TakerFills` on success, or a `FillAbort` when the book is too
    thin (`NO_DEPTH`) or a fill sits further from mid than `max_slippage_bps`
    (`SLIPPAGE`). Pure; never raises for ordinary market states."""
    qty = p.qty
    vls, _ = vwap_fill(left_book.bids,  qty, max_levels=p.max_levels)
    vlb, _ = vwap_fill(left_book.asks,  qty, max_levels=p.max_levels)
    vrs, _ = vwap_fill(right_book.bids, qty, max_levels=p.max_levels)
    vrb, _ = vwap_fill(right_book.asks, qty, max_levels=p.max_levels)
    if vls is None or vlb is None or vrs is None or vrb is None:
        return FillAbort(FillAbortKind.NO_DEPTH)

    fills = TakerFills(left_sell=vls, left_buy=vlb, right_sell=vrs, right_buy=vrb)

    slip = p.max_slippage_bps / BPS
    if (abs((fills.left_sell  - mid_left)  / mid_left)  > slip
            or abs((fills.left_buy   - mid_left)  / mid_left)  > slip
            or abs((fills.right_sell - mid_right) / mid_right) > slip
            or abs((fills.right_buy  - mid_right) / mid_right) > slip):
        return FillAbort(FillAbortKind.SLIPPAGE, fills=fills)

    return fills
