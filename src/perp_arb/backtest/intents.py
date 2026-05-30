"""Wire types passed between strategy → engine → fill-model → recorder."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from ..core.recording.decision import Decision
from ..core.types import Side


class FillModelKind(StrEnum):
    BBO = "bbo"
    VWAP = "vwap"


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """A strategy's request to send a market order at this tick.

    `decision` is the same `Decision` instance the strategy built in `on_tick`;
    the engine mutates its `timeline` and `legs` as the order is resolved, then
    calls `recorder.emit(decision)` once both legs land.
    """
    decision_id: str
    decision: Decision
    venue: str
    side: Side
    qty: Decimal
    expected_price: Decimal
    fill_model: FillModelKind
    sim_ts_ms: int


@dataclass(frozen=True, slots=True)
class FillEvent:
    """What a FillModel returns. `success=False` paths use `error` to convey
    the reject reason ("exceeds_top_level_size", "vwap_qty_mismatch", etc.)."""
    decision_id: str
    venue: str
    side: Side
    requested_qty: Decimal
    filled_qty: Decimal | None
    realized_price: Decimal | None
    arrival_ts_ms: int
    fill_ts_ms: int
    success: bool
    error: str | None = None
    fill_model: FillModelKind = FillModelKind.BBO


@dataclass(frozen=True, slots=True)
class PendingOrder:
    """Engine-internal queue entry: an intent waiting for its arrival_ts."""
    intent: OrderIntent
    arrival_ts_ms: int
