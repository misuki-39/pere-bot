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
    """Return value of place_market_order — the synchronous place ack."""

    success: bool
    client_id: str | None = None
    side: Side | None = None
    requested_qty: Decimal | None = None
    filled_qty: Decimal | None = None
    avg_price: Decimal | None = None
    status: OrderStatus = OrderStatus.UNKNOWN
    error_message: str | None = None
    latency_ms: int | None = None
    # Exchange-server clock (e.g. aster's `transactTime`). NOT our local
    # time — joinable to venue UIs / external trade logs.
    exchange_ts_ms: int | None = None


@dataclass(slots=True)
class OrderSnapshot:
    """Cumulative order-state observation: poll-style snapshot or per-order WS event
    (lighter `account_market.orders`).

    Semantics: `filled_qty` / `realized_price` are the order's RUNNING TOTALS
    at this instant. Consumer overwrites, never accumulates."""

    client_id: str | None
    symbol: Symbol
    side: Side
    size: Decimal
    price: Decimal
    status: OrderStatus
    filled_qty: Decimal = Decimal("0")
    realized_price: Decimal | None = None
    ts_ms: int = 0


@dataclass(slots=True)
class FillDelta:
    """Per-fill execution event: aster `ORDER_TRADE_UPDATE` `l`/`L` field.

    Each `FillDelta` represents ONE individual fill. Consumer sums them.
    Adapters MUST only construct this when `qty > 0` — empty events
    (NEW status, etc.) are non-emissions, not zero-qty deltas."""

    qty: Decimal
    price: Decimal
    ts_ms: int
    side: Side | None = None
    client_id: str | None = None
    # Carried so the accumulator can short-circuit waits when the venue
    # confirms the parent order is settled (FILLED / CANCELED / REJECTED /
    # EXPIRED). None when this delta isn't terminal.
    terminal_status: OrderStatus | None = None


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


@dataclass(slots=True)
class TerminalFill:
    """Per-`client_id` aggregate of WS fill events, the canonical "what
    actually filled" view that callers see.

    Absorbs two event types with different semantics:

      * `OrderSnapshot` — cumulative running totals (lighter
        `account_market.orders`); OVERWRITES filled_qty + weighted_price_sum.
      * `FillDelta` — per-fill event (aster `l`/`L`); ACCUMULATES qty * price.

    `last_status` is taken from any terminal-status signal so `is_complete`
    can short-circuit the wait the moment the venue confirms the parent
    order is settled (FILLED / CANCELED / REJECTED / EXPIRED)."""

    filled_qty: Decimal = Decimal("0")
    weighted_price_sum: Decimal = Decimal("0")
    last_ts_ms: int = 0
    last_status: OrderStatus | None = None

    def add(self, event: OrderSnapshot | FillDelta) -> None:
        if event.ts_ms:
            self.last_ts_ms = max(self.last_ts_ms, event.ts_ms)
        match event:
            case OrderSnapshot():
                if event.status.terminal:
                    self.last_status = event.status
                if event.filled_qty > 0:
                    self.filled_qty = event.filled_qty
                    if event.realized_price is not None:
                        self.weighted_price_sum = event.filled_qty * event.realized_price
            case FillDelta():
                if event.terminal_status is not None:
                    self.last_status = event.terminal_status
                # FillDelta adapter invariant: qty > 0 (non-fills dropped at source).
                self.filled_qty += event.qty
                self.weighted_price_sum += event.qty * event.price

    def is_complete(self, requested_qty: Decimal) -> bool:
        # Terminal status = no more fills coming (filled / canceled /
        # rejected / expired) → stop waiting. Otherwise fall back to qty
        # comparison for the trade-only path (account_orders unsubscribed
        # or lagging).
        if self.last_status is not None and self.last_status.terminal:
            return True
        return self.filled_qty >= requested_qty

    @property
    def avg_price(self) -> Decimal | None:
        if self.filled_qty == 0:
            return None
        return self.weighted_price_sum / self.filled_qty


@dataclass(slots=True)
class OrderOutcome:
    """Driver-layer return: synchronous place-ack + WS-tracked terminal fill.

    `ack` is whatever `place_market_order` returned — for aster this comes
    from an HTTP REST POST, for lighter from a signed WS tx ack; both are
    just "the venue acknowledged the submit" from the caller's POV.

    `fill is None` when the place ack failed (no point awaiting) OR the
    tracker timed out before any event arrived."""

    ack: OrderResult
    fill: TerminalFill | None
