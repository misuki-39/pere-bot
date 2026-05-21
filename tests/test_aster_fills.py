"""Tests for Aster's user-data fill dispatch.

The contract is: `_handle_order_trade_update` must emit PER-FILL DELTAS
(`o["l"]` / `o["L"]`) so the `_FillAccumulator`'s `+=` semantics produce
the correct totals across multi-event orders. Using cumulative `z` / `ap`
would silently double-count.
"""

from __future__ import annotations

from decimal import Decimal

from perp_arb.core.types import MarketInfo, OrderInfo, Symbol
from perp_arb.exchanges.aster.client import AsterClient
from perp_arb.strategy.taker_taker import _FillAccumulator

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


def _make_client_with_market() -> tuple[AsterClient, list[OrderInfo]]:
    c = AsterClient.__new__(AsterClient)
    c._fill_cbs = {}      # type: ignore[attr-defined]
    c._markets = {"CLUSDT": _MARKET}  # type: ignore[attr-defined]
    received: list[OrderInfo] = []
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
    assert [info.filled_size for info in received] == [Decimal("0.4"), Decimal("0.6")]
    assert [info.avg_fill_price for info in received] == [
        Decimal("100.00"), Decimal("100.10"),
    ]


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
    for info in received:
        acc.add(info)
    assert acc.filled_qty == Decimal("1.0")
    # size-weighted: (0.4 * 100.00 + 0.6 * 100.10) / 1.0 = 100.06
    assert acc.avg_price == Decimal("100.06")


def test_aster_new_event_with_no_fill_does_not_poison() -> None:
    """A NEW / OPEN status event has l=0 (no fill yet). Must not bump filled_qty."""
    c, received = _make_client_with_market()
    c._handle_order_trade_update(_trade_evt(
        last_qty="0", last_price="0", cum_qty="0", status="NEW", q="10",
    ))
    acc = _FillAccumulator()
    for info in received:
        acc.add(info)
    assert acc.filled_qty == Decimal("0")


def test_aster_uses_trade_time_not_event_time_for_fills() -> None:
    """Prefer `T` (trade time) over `E` (event time) for `fill_ts_ms`."""
    c, received = _make_client_with_market()
    evt = _trade_evt(
        last_qty="1.0", last_price="100.00", cum_qty="1.0", status="FILLED",
    )
    evt["E"] = 1_700_000_000_000
    evt["T"] = 1_700_000_000_500
    c._handle_order_trade_update(evt)
    assert received[0].ts_ms == 1_700_000_000_500
