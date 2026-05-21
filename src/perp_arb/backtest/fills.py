"""Two fill models against the BBO + captured-VWAP data.

BBOFill: market taker eats one top level only. Used for small-capital
strategies (qty ≤ top size) where slippage past top is unrealistic.

VwapFill: uses the row's pre-computed `vwap_*_buy`/`vwap_*_sell` columns. The
capture computed them once at a fixed `capture_qty`, so this fill model
requires `intent.qty == capture_qty` exactly; any other qty is a hard error.

Both models also honour the row's `gates_passed` flag — if depth or staleness
gates failed at arrival, the order is rejected.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal, Protocol

from ..core.types import Side
from .dataset import BBORow
from .intents import FillEvent, FillModelKind, OrderIntent

VenueSide = Literal["left", "right"]


class FillModel(Protocol):
    kind: FillModelKind

    def try_fill(
        self,
        intent: OrderIntent,
        arrival_ts_ms: int,
        venue_row: BBORow,
        venue_side: VenueSide,
        capture_qty: Decimal,
    ) -> FillEvent: ...


def _reject(intent: OrderIntent, arrival_ts: int, fill_ts: int,
            error: str, kind: FillModelKind) -> FillEvent:
    return FillEvent(
        decision_id=intent.decision_id,
        venue=intent.venue,
        side=intent.side,
        requested_qty=intent.qty,
        filled_qty=None,
        realized_price=None,
        arrival_ts_ms=arrival_ts,
        fill_ts_ms=fill_ts,
        success=False,
        error=error,
        fill_model=kind,
    )


class BBOFill:
    """Top-of-book taker fill. Rejects when `qty > top-level size` — the
    capture has no L2 depth to walk into, so partial-on-top is not modelled."""

    kind = FillModelKind.BBO

    def try_fill(
        self,
        intent: OrderIntent,
        arrival_ts_ms: int,
        venue_row: BBORow,
        venue_side: VenueSide,
        capture_qty: Decimal,
    ) -> FillEvent:
        fill_ts = venue_row.ts_ms
        if not venue_row.gates_passed:
            return _reject(intent, arrival_ts_ms, fill_ts,
                           "gates_failed_at_arrival", self.kind)
        if intent.side is Side.BUY:
            price = venue_row.left_ask if venue_side == "left" else venue_row.right_ask
            size = venue_row.left_ask_size if venue_side == "left" else venue_row.right_ask_size
        else:
            price = venue_row.left_bid if venue_side == "left" else venue_row.right_bid
            size = venue_row.left_bid_size if venue_side == "left" else venue_row.right_bid_size
        if intent.qty > size:
            return _reject(intent, arrival_ts_ms, fill_ts,
                           f"exceeds_top_level_size (qty={intent.qty} > top_size={size})",
                           self.kind)
        return FillEvent(
            decision_id=intent.decision_id, venue=intent.venue, side=intent.side,
            requested_qty=intent.qty, filled_qty=intent.qty,
            realized_price=price,
            arrival_ts_ms=arrival_ts_ms, fill_ts_ms=fill_ts,
            success=True, fill_model=self.kind,
        )


class VwapFill:
    """Uses the row's pre-computed VWAP columns. Strict: requires
    `intent.qty == capture_qty` so the column is meaningful."""

    kind = FillModelKind.VWAP

    def try_fill(
        self,
        intent: OrderIntent,
        arrival_ts_ms: int,
        venue_row: BBORow,
        venue_side: VenueSide,
        capture_qty: Decimal,
    ) -> FillEvent:
        fill_ts = venue_row.ts_ms
        if not venue_row.gates_passed:
            return _reject(intent, arrival_ts_ms, fill_ts,
                           "gates_failed_at_arrival", self.kind)
        if intent.qty != capture_qty:
            return _reject(intent, arrival_ts_ms, fill_ts,
                           f"vwap_qty_mismatch (intent={intent.qty} capture={capture_qty})",
                           self.kind)
        if venue_side == "left":
            price = venue_row.vwap_left_buy if intent.side is Side.BUY else venue_row.vwap_left_sell
        else:
            price = venue_row.vwap_right_buy if intent.side is Side.BUY else venue_row.vwap_right_sell
        if price is None:
            return _reject(intent, arrival_ts_ms, fill_ts,
                           "vwap_unavailable_at_arrival", self.kind)
        return FillEvent(
            decision_id=intent.decision_id, venue=intent.venue, side=intent.side,
            requested_qty=intent.qty, filled_qty=intent.qty,
            realized_price=price,
            arrival_ts_ms=arrival_ts_ms, fill_ts_ms=fill_ts,
            success=True, fill_model=self.kind,
        )


def fill_model_for(kind: FillModelKind) -> FillModel:
    if kind is FillModelKind.BBO:
        return BBOFill()
    if kind is FillModelKind.VWAP:
        return VwapFill()
    raise ValueError(f"unknown fill model: {kind}")
