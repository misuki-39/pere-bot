"""Shared trading domain types used across all exchanges and strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1


class OrderStatus(StrEnum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNKNOWN = "unknown"

    @property
    def terminal(self) -> bool:
        return self in (
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )


@dataclass(frozen=True, slots=True)
class Symbol:
    """Cross-venue symbol identity. `raw` is the venue-specific ticker string."""

    exchange: str  # "aster" | "lighter"
    raw: str       # "ETHUSDT" on Aster, "ETH" on Lighter
    base: str      # "ETH"
    quote: str     # "USDT" / "USD"

    def __str__(self) -> str:
        return f"{self.exchange}:{self.raw}"


@dataclass(frozen=True, slots=True)
class MarketInfo:
    """Static market metadata. `contract_id` is what the venue REST/WS expects."""

    symbol: Symbol
    tick_size: Decimal
    lot_size: Decimal
    contract_id: str | int    # market_index (int) on Lighter, symbol string on Aster
    min_qty: Decimal = Decimal("0")


@dataclass(slots=True)
class BookLevel:
    price: Decimal
    size: Decimal


@dataclass(slots=True)
class OrderBook:
    """Top-N order book snapshot, sorted best-first."""

    symbol: Symbol
    bids: list[BookLevel] = field(default_factory=list)   # descending price
    asks: list[BookLevel] = field(default_factory=list)   # ascending price
    ts_ms: int = 0

    @property
    def best_bid(self) -> BookLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> BookLevel | None:
        return self.asks[0] if self.asks else None

    @property
    def mid(self) -> Decimal | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0].price + self.asks[0].price) / Decimal(2)


@dataclass(slots=True)
class Quote:
    """Top-of-book snapshot. The lightweight signal most strategies care about."""

    symbol: Symbol
    bid: Decimal
    bid_size: Decimal
    ask: Decimal
    ask_size: Decimal
    ts_ms: int

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / Decimal(2)

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid


@dataclass(slots=True)
class OrderResult:
    """Return value of place_market_order / cancel_order."""

    success: bool
    order_id: str | None = None
    client_id: str | None = None
    side: Side | None = None
    requested_size: Decimal | None = None
    filled_size: Decimal | None = None
    avg_price: Decimal | None = None
    status: OrderStatus = OrderStatus.UNKNOWN
    error_message: str | None = None
    latency_ms: int | None = None
    # Exchange-server clock (e.g. aster's `transactTime`). NOT our local
    # time — joinable to venue UIs / external trade logs.
    exchange_ts_ms: int | None = None


@dataclass(slots=True)
class OrderInfo:
    """Live order state as observed via REST poll or WS stream."""

    order_id: str
    client_id: str | None
    symbol: Symbol
    side: Side
    size: Decimal
    price: Decimal
    status: OrderStatus
    filled_size: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None
    ts_ms: int = 0


@dataclass(slots=True)
class Position:
    """Unified across venues. Signed size: positive = long, negative = short."""

    symbol: Symbol
    size: Decimal
    entry_price: Decimal = Decimal("0")
    unrealised_pnl: Decimal = Decimal("0")

    @property
    def is_flat(self) -> bool:
        return self.size == 0
