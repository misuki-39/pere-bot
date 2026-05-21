"""Tests for Aster's user-data fill dispatch.

The contract is: `_handle_order_trade_update` must emit PER-FILL DELTAS
(`o["l"]` / `o["L"]`) so the `_FillAccumulator`'s `+=` semantics produce
the correct totals across multi-event orders. Using cumulative `z` / `ap`
would silently double-count.
"""

from __future__ import annotations

from decimal import Decimal

from perp_arb.core.fill_tracker import _PerCidFillTracker
from perp_arb.core.types import (
    FillDelta,
    MarketInfo,
    Symbol,
)
from perp_arb.core.types import (
    TerminalFill as _FillAccumulator,
)
from perp_arb.exchanges.aster.client import AsterClient

_SYM = Symbol(exchange="aster", raw="CLUSDT", base="WTI", quote="USDT")
_MARKET = MarketInfo(
    symbol=_SYM, tick_size=Decimal("0.01"), lot_size=Decimal("1"),
    contract_id="CLUSDT",
)


def _trade_evt(*, last_qty: str, last_price: str, cum_qty: str,
               status: str = "PARTIALLY_FILLED", q: str = "10",
               trade_time: int = 1_700_000_000_500) -> dict:
    """Skeleton ORDER_TRADE_UPDATE — only the fields `_handle_order_trade_update`
    actually reads. `l`/`L` are per-fill deltas; `z` would be cumulative."""
    return {
        "e": "ORDER_TRADE_UPDATE", "E": trade_time,
        "T": trade_time,
        "o": {
            "s": "CLUSDT", "c": "cid-1", "S": "BUY",
            "i": 12345, "q": q, "p": "0",
            "X": status,
            "l": last_qty, "L": last_price, "z": cum_qty, "ap": "0",
        },
    }


def _make_client_with_market() -> tuple[AsterClient, list[FillDelta]]:
    c = AsterClient.__new__(AsterClient)
    c._fill_cbs = {}      # type: ignore[attr-defined]
    c._markets = {"CLUSDT": _MARKET}  # type: ignore[attr-defined]
    c._fill_tracker = _PerCidFillTracker()  # type: ignore[attr-defined]
    received: list[FillDelta] = []
    c._fill_cbs["CLUSDT"] = [received.append]   # type: ignore[index]
    return c, received


def test_aster_emits_per_fill_delta_not_cumulative() -> None:
    c, received = _make_client_with_market()
    c._handle_order_trade_update(_trade_evt(
        last_qty="0.4", last_price="100.00", cum_qty="0.4",
    ))
    c._handle_order_trade_update(_trade_evt(
        last_qty="0.6", last_price="100.10", cum_qty="1.0", status="FILLED",
    ))
    assert [d.qty for d in received] == [Decimal("0.4"), Decimal("0.6")]
    assert [d.price for d in received] == [Decimal("100.00"), Decimal("100.10")]


def test_aster_accumulator_round_trips_to_requested_qty() -> None:
    """End-to-end: handler delta emission + accumulator += → final qty == requested."""
    c, received = _make_client_with_market()
    c._handle_order_trade_update(_trade_evt(
        last_qty="0.4", last_price="100.00", cum_qty="0.4",
    ))
    c._handle_order_trade_update(_trade_evt(
        last_qty="0.6", last_price="100.10", cum_qty="1.0", status="FILLED",
    ))
    acc = _FillAccumulator()
    for d in received:
        acc.add(d)
    assert acc.filled_qty == Decimal("1.0")
    # size-weighted: (0.4 * 100.00 + 0.6 * 100.10) / 1.0 = 100.06
    assert acc.weighted_price_sum / acc.filled_qty == Decimal("100.06")


def test_aster_drops_non_fill_events_at_source() -> None:
    """NEW / OPEN events carry l=0 — adapter drops them, accumulator never
    sees a zero-qty delta. Eliminates a defensive guard structurally."""
    c, received = _make_client_with_market()
    c._handle_order_trade_update(_trade_evt(
        last_qty="0", last_price="0", cum_qty="0", status="NEW", q="10",
    ))
    assert received == []


def test_aster_uses_trade_time_not_event_time() -> None:
    """Prefer `T` (trade time) over `E` (event time) for ts_ms."""
    c, received = _make_client_with_market()
    evt = _trade_evt(
        last_qty="1.0", last_price="100.00", cum_qty="1.0", status="FILLED",
    )
    evt["E"] = 1_700_000_000_000
    evt["T"] = 1_700_000_000_500
    c._handle_order_trade_update(evt)
    assert received[0].ts_ms == 1_700_000_000_500


def test_aster_terminal_status_propagates_on_filled() -> None:
    """The final FILLED event carries terminal_status so the accumulator
    can short-circuit `is_complete` without summing the qty."""
    c, received = _make_client_with_market()
    c._handle_order_trade_update(_trade_evt(
        last_qty="1.0", last_price="100.00", cum_qty="1.0", status="FILLED",
    ))
    assert received[0].terminal_status is not None
    assert received[0].terminal_status.value == "filled"
