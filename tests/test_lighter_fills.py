"""Unit tests for `LighterClient._handle_trade` — the account_all → fill
callback dispatch.

The real WS plumbing involves a signer, REST client, and live SDK
connection. These tests bypass all of that by constructing a public-only
`LighterClient`, priming its symbol map manually, and invoking
`_on_account_event` with a synthetic payload that mirrors the Lighter
SDK's `Trade` schema (see lighter-sdk `models/trade.py`).
"""

from __future__ import annotations

from decimal import Decimal

from perp_arb.core.types import OrderInfo, OrderStatus, Side, Symbol
from perp_arb.exchanges.lighter.client import LighterClient, _MarketMeta


def _client(account_index: int = 42) -> LighterClient:
    """Public-only client wired up just enough to dispatch trades."""
    c = LighterClient(
        base_url="https://example.test",
        api_key_private_key=None,
        account_index=account_index,
        public_only=True,
    )
    sym = Symbol(exchange="lighter", raw="WTI", base="WTI", quote="USD")
    meta = _MarketMeta(
        market_index=7,
        base_multiplier=10**6,
        price_multiplier=10**4,
        symbol=sym,
    )
    c._meta_by_symbol["WTI"] = meta
    c._symbol_by_market_index[7] = "WTI"
    return c


def _trade(*, ask_account: int, bid_account: int,
           ask_client_id: int = 1001, bid_client_id: int = 2002,
           size: str = "1.0", price: str = "100.50",
           timestamp: int = 1_700_000_000_000) -> dict:
    """Mirrors the Lighter SDK Trade payload shape (subset we read)."""
    return {
        "trade_id": 99,
        "tx_hash": "0xdead",
        "type": "trade",
        "market_id": 7,
        "size": size,
        "price": price,
        "ask_id": 11,
        "bid_id": 22,
        "ask_client_id": ask_client_id,
        "bid_client_id": bid_client_id,
        "ask_account_id": ask_account,
        "bid_account_id": bid_account,
        "is_maker_ask": False,
        "block_height": 0,
        "timestamp": timestamp,
        "taker_fee": 1,
        "maker_fee": 0,
        "transaction_time": timestamp,
    }


def test_trade_where_we_sold_emits_sell_with_ask_client_id() -> None:
    c = _client(account_index=42)
    received: list[OrderInfo] = []
    c._fill_cbs["WTI"].append(received.append)

    c._on_account_event({
        "trades": {"99": _trade(ask_account=42, bid_account=99,
                                ask_client_id=12345, bid_client_id=67890)}
    })

    assert len(received) == 1
    info = received[0]
    assert info.side is Side.SELL
    assert info.client_id == "12345"           # our ask_client_id
    assert info.size == Decimal("1.0")
    assert info.avg_fill_price == Decimal("100.50")
    assert info.status is OrderStatus.FILLED
    assert info.ts_ms == 1_700_000_000_000


def test_trade_where_we_bought_emits_buy_with_bid_client_id() -> None:
    c = _client(account_index=42)
    received: list[OrderInfo] = []
    c._fill_cbs["WTI"].append(received.append)

    c._on_account_event({
        "trades": {"99": _trade(ask_account=99, bid_account=42,
                                ask_client_id=12345, bid_client_id=67890)}
    })

    assert len(received) == 1
    info = received[0]
    assert info.side is Side.BUY
    assert info.client_id == "67890"           # our bid_client_id


def test_trade_where_we_are_neither_side_drops_silently() -> None:
    """Defensive guard — should never happen on account_all/{my_account}
    but log + return rather than mis-attribute."""
    c = _client(account_index=42)
    received: list[OrderInfo] = []
    c._fill_cbs["WTI"].append(received.append)

    c._on_account_event({
        "trades": {"99": _trade(ask_account=1, bid_account=2)}
    })
    assert received == []


def test_trade_with_unknown_market_id_drops_silently() -> None:
    c = _client(account_index=42)
    received: list[OrderInfo] = []
    c._fill_cbs["WTI"].append(received.append)

    bad = _trade(ask_account=42, bid_account=99)
    bad["market_id"] = 999  # not in symbol map
    c._on_account_event({"trades": {"99": bad}})
    assert received == []


def test_account_event_routes_both_positions_and_trades() -> None:
    """Regression: adding the trades dispatch must not break position routing."""
    c = _client(account_index=42)
    pos_received: list = []
    trade_received: list[OrderInfo] = []
    c._position_cbs["WTI"].append(pos_received.append)
    c._fill_cbs["WTI"].append(trade_received.append)

    c._on_account_event({
        "positions": {
            "WTI": {
                "market_id": 7,
                "position": "5.0",
                "sign": 1,
                "avg_entry_price": "100.0",
                "unrealized_pnl": "0",
            }
        },
        "trades": {"99": _trade(ask_account=42, bid_account=99)},
    })
    assert len(pos_received) == 1
    assert len(trade_received) == 1


def test_no_callbacks_is_a_noop() -> None:
    """A trade arriving with zero registered subscribers must not raise."""
    c = _client(account_index=42)
    # No callbacks registered.
    c._on_account_event({"trades": {"99": _trade(ask_account=42, bid_account=99)}})
    # Should not raise.
