"""Unit tests for `LighterClient._on_market_event` — the account_market
WS → fill/position callback dispatch.

The real WS plumbing involves a signer, REST client, and live SDK
connection. These tests bypass all of that by constructing a public-only
client, priming its symbol map manually, and invoking `_on_market_event`
with synthetic payloads that mirror the actual wire shape observed via
scripts/lighter_ws_probe.py (see commit message / docs).
"""

from __future__ import annotations

from decimal import Decimal

from perp_arb.core.types import OrderSnapshot, OrderStatus, Position, Side, Symbol
from perp_arb.exchanges.lighter.client import LighterClient, _MarketMeta


def _client(account_index: int = 708718) -> LighterClient:
    """Public-only client wired up just enough to dispatch market events."""
    c = LighterClient(
        base_url="https://example.test",
        api_key_private_key=None,
        account_index=account_index,
        public_only=True,
    )
    sym = Symbol(exchange="lighter", raw="WTI", base="WTI", quote="USD")
    meta = _MarketMeta(
        market_index=145,
        base_multiplier=10**3,
        price_multiplier=10**3,
        symbol=sym,
    )
    c._meta_by_symbol["WTI"] = meta
    c._symbol_by_market_index[145] = "WTI"
    return c


# Captured from scripts/lighter_ws_probe.py: order goes open → filled in
# two updates, with `position: null` (only orders touched) and a final
# `position: {...}` update when the fill flips the net position.
def _order_open() -> dict:
    return {
        "order_id": "41376821495678292", "order_index": 41376821495678292,
        "client_order_id": "551535175282", "client_order_index": 551535175282,
        "market_index": 145, "owner_account_index": 708718,
        "initial_base_amount": "0.100", "remaining_base_amount": "0.100",
        "filled_base_amount": "0.000", "filled_quote_amount": "0.000000",
        "price": "100.712", "is_ask": False,
        "status": "open", "type": "limit",
        "transaction_time": 1779361394966700,
    }


def _order_filled() -> dict:
    return {
        "order_id": "41376821495678292", "order_index": 41376821495678292,
        "client_order_id": "551535175282", "client_order_index": 551535175282,
        "market_index": 145, "owner_account_index": 708718,
        "initial_base_amount": "0.100", "remaining_base_amount": "0.000",
        "filled_base_amount": "0.100", "filled_quote_amount": "10.071200",
        "price": "100.712", "is_ask": False,
        "status": "filled", "type": "limit",
        "transaction_time": 1779361398032308,
    }


def _position() -> dict:
    return {
        "market_id": 145, "symbol": "WTI",
        "sign": -1, "position": "0.100",
        "avg_entry_price": "100.930",
        "unrealized_pnl": "0.005300",
        "realized_pnl": "0.000000",
    }


# ---- orders -----------------------------------------------------------

def test_order_open_dispatches_orderinfo_with_open_status() -> None:
    c = _client()
    received: list[OrderSnapshot] = []
    c._fill_cbs["WTI"].append(received.append)

    c._on_market_event({"orders": [_order_open()], "position": None})

    assert len(received) == 1
    info = received[0]
    assert info.client_id == "551535175282"
    assert info.side is Side.BUY                # is_ask=False
    assert info.status is OrderStatus.OPEN
    assert info.filled_qty == Decimal("0.000")
    assert info.realized_price is None          # nothing filled yet
    # transaction_time microseconds → ms
    assert info.ts_ms == 1779361394966


def test_order_filled_dispatches_with_cumulative_avg_price() -> None:
    """realized_price = filled_quote / filled_base (cumulative semantics)."""
    c = _client()
    received: list[OrderSnapshot] = []
    c._fill_cbs["WTI"].append(received.append)

    c._on_market_event({"orders": [_order_filled()], "position": None})

    info = received[0]
    assert info.status is OrderStatus.FILLED
    assert info.filled_qty == Decimal("0.100")
    assert info.realized_price == Decimal("100.712")    # 10.0712 / 0.1


def test_order_with_unknown_market_index_drops() -> None:
    c = _client()
    received: list[OrderSnapshot] = []
    c._fill_cbs["WTI"].append(received.append)

    o = _order_open()
    o["market_index"] = 999      # not in symbol map
    c._on_market_event({"orders": [o]})

    assert received == []


def test_canceled_expired_maps_to_expired_status() -> None:
    """`canceled-expired` is semantically EXPIRED, not CANCELED."""
    c = _client()
    received: list[OrderSnapshot] = []
    c._fill_cbs["WTI"].append(received.append)

    o = _order_open()
    o["status"] = "canceled-expired"
    c._on_market_event({"orders": [o]})

    assert received[0].status is OrderStatus.EXPIRED


def test_canceled_variants_collapse_to_canceled() -> None:
    """`canceled-post-only` / `canceled-reduce-only` → CANCELED."""
    c = _client()
    received: list[OrderSnapshot] = []
    c._fill_cbs["WTI"].append(received.append)

    for variant in ("canceled", "canceled-post-only", "canceled-reduce-only"):
        o = _order_open()
        o["status"] = variant
        c._on_market_event({"orders": [o]})

    assert all(info.status is OrderStatus.CANCELED for info in received)


# ---- position ---------------------------------------------------------

def test_position_single_dict_routes_to_position_cb() -> None:
    """account_market pushes `position` as a single object (not a list)."""
    c = _client()
    received: list[Position] = []
    c._position_cbs["WTI"].append(received.append)

    c._on_market_event({"position": _position(), "orders": None})

    assert len(received) == 1
    assert received[0].size == Decimal("-0.100")        # sign=-1 * 0.100


def test_position_null_does_not_route() -> None:
    c = _client()
    received: list[Position] = []
    c._position_cbs["WTI"].append(received.append)

    c._on_market_event({"position": None, "orders": [_order_open()]})

    assert received == []


# ---- combined / edge cases -------------------------------------------

def test_orders_and_position_in_one_frame_both_route() -> None:
    """Snapshot frame can carry both; each goes to its own cb list."""
    c = _client()
    orders: list[OrderSnapshot] = []
    positions: list[Position] = []
    c._fill_cbs["WTI"].append(orders.append)
    c._position_cbs["WTI"].append(positions.append)

    c._on_market_event({
        "orders": [_order_filled()],
        "position": _position(),
    })

    assert len(orders) == 1
    assert len(positions) == 1


def test_no_callbacks_is_a_noop() -> None:
    c = _client()
    c._on_market_event({"orders": [_order_filled()], "position": _position()})
    # Should not raise.
