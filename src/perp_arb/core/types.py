"""Shared trading domain types used across all exchanges and strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any, ClassVar


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


class LegKind(StrEnum):
    ENTRY = "entry"
    UNWIND = "unwind"


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
    # Per-fill commission in quote-currency units (aster `o.n`). 0 for
    # venues that don't surface a per-fill fee on the WS stream (lighter).
    fee: Decimal = Decimal("0")


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
class LegOutcome:
    """Unified post-execution state for one venue leg.

    Single leg-level dataclass used end-to-end: drivers populate ack-side
    fields from `place_market_order`; the WS fill tracker accumulates
    fill-side fields via `add()`; `submit_and_await` merges any WS data
    on top of the REST fallback; the executor stamps presentation fields
    (venue, expected_price, send_ts_ms, kind) after gather; the recorder
    serializes this directly to CSV. There is no separate "report"
    projection — what drivers / strategies / recorder see is the same
    object.

    Construct a fill state with `set_fill(qty, avg_price)`; never write
    `_weighted_price_sum` directly.

    `last_status` is taken from any terminal-status signal so
    `is_complete` can short-circuit the wait the moment the venue
    confirms the parent order is settled
    (FILLED / CANCELED / REJECTED / EXPIRED)."""

    # ack-side — populated by driver.place_market_order
    client_id: str | None = None
    side: Side | None = None
    requested_qty: Decimal | None = None
    success: bool = False
    status: OrderStatus = OrderStatus.UNKNOWN
    error_message: str | None = None
    # Exchange-server clock for the place-ack (aster's `transactTime`).
    # NOT our local time — joinable to venue UIs / external trade logs.
    exchange_ts_ms: int | None = None

    # fill-side — WS accumulator OR REST fallback (driver writes either)
    filled_qty: Decimal = Decimal("0")
    _weighted_price_sum: Decimal = Decimal("0")
    last_ts_ms: int = 0
    last_status: OrderStatus | None = None
    # Sum of per-fill commission across all FillDelta events for this cid
    # (quote-currency units). Snapshot-based venues (lighter) don't feed
    # this; it stays 0 — lighter is structurally zero-fee.
    total_fee: Decimal = Decimal("0")

    # presentation-side — stamped by executor after gather, used by recorder + analysis
    venue: str | None = None                # leg-label key into executor's `exchanges` dict
    expected_price: Decimal | None = None   # decision-time VWAP / unwind cost basis
    send_ts_ms: int | None = None           # local epoch ms at SEND mark (executor stamps)
    kind: LegKind | None = None

    def add(self, event: OrderSnapshot | FillDelta) -> None:
        """Absorb a WS event. `OrderSnapshot` (lighter cumulative state)
        OVERWRITES filled_qty + _weighted_price_sum; `FillDelta` (aster
        per-fill) ACCUMULATES qty * price * fee."""
        if event.ts_ms:
            self.last_ts_ms = max(self.last_ts_ms, event.ts_ms)
        match event:
            case OrderSnapshot():
                if event.status.terminal:
                    self.last_status = event.status
                # Atomic: only commit qty + price together. A snapshot
                # with filled_qty>0 but no realized_price is an
                # intermediate state — committing only filled_qty would
                # leave the accumulator with qty>0 / price_sum=0 and
                # `avg_price` would return Decimal('0') (fabricated $0
                # fill price) instead of None.
                if event.filled_qty > 0 and event.realized_price is not None:
                    self.filled_qty = event.filled_qty
                    self._weighted_price_sum = event.filled_qty * event.realized_price
            case FillDelta():
                if event.terminal_status is not None:
                    self.last_status = event.terminal_status
                # FillDelta adapter invariant: qty > 0 (non-fills dropped at source).
                self.filled_qty += event.qty
                self._weighted_price_sum += event.qty * event.price
                self.total_fee += event.fee
            case _:
                # Fail fast on unhandled event subtypes — silent fall-through
                # would leave fill-side fields zeroed, indistinguishable from
                # a WS timeout. New event types must opt in explicitly.
                raise TypeError(f"LegOutcome.add: unhandled event type {type(event).__name__}")

    def set_fill(self, qty: Decimal, avg_price: Decimal) -> None:
        """Synthesize a complete fill state from known qty + average price.
        Use this instead of writing the accumulator sum directly — keeps
        the internal storage detail (`_weighted_price_sum = qty * price`)
        in one place."""
        self.filled_qty = qty
        self._weighted_price_sum = qty * avg_price

    def merge_fill_from(self, other: LegOutcome) -> None:
        """Overlay another outcome's fill-side state onto self. Used by
        `submit_and_await` to promote WS-aggregated fills over REST-only
        ack data when WS observed at least as many fills."""
        self.filled_qty = other.filled_qty
        self._weighted_price_sum = other._weighted_price_sum
        self.last_ts_ms = other.last_ts_ms
        self.total_fee = other.total_fee

    def _clear_accumulator_status(self) -> None:
        """Drop the WS-accumulator's `last_status` signal once the merge
        in `submit_and_await` has promoted it onto `status`. Keeps the
        finalized outcome free of intermediate-stage state."""
        self.last_status = None

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
        return self._weighted_price_sum / self.filled_qty

    @property
    def fill_ts_ms(self) -> int | None:
        """Best timestamp for the actual fill on the venue's clock —
        prefer the WS last-event ts, fall back to the place-ack
        `exchange_ts_ms` when WS never delivered (REST-only fallback)."""
        return self.last_ts_ms or self.exchange_ts_ms

    # ----- CSV projection -----
    #
    # Owned here (not in csv_recorder) so the column set and the
    # field/property accessors stay in one place. Adding/renaming a
    # column is a single-file change.

    _CSV_FIELDS: ClassVar[tuple[str, ...]] = (
        "venue", "side", "requested_qty", "filled_qty",
        "expected_price", "realized_price", "status", "success",
        "error_message", "client_id", "total_fee",
        "send_ts_ms", "fill_ts_ms", "kind",
    )

    @classmethod
    def csv_header(cls) -> list[str]:
        return list(cls._CSV_FIELDS)

    def to_csv_row(self) -> list[Any]:
        """Project to CSV columns. Assert presentation fields are set —
        an unstamped outcome reaching the recorder is a bug, not a row."""
        assert self.venue is not None, "to_csv_row: venue not stamped"
        assert self.kind is not None, "to_csv_row: kind not stamped"
        return [
            self.venue,
            self.side.value if self.side is not None else "",
            self.requested_qty,
            # filled_qty=None on failure preserves the "no fill info"
            # vs "zero fill" distinction — both are recordable outcomes.
            self.filled_qty if self.success else None,
            self.expected_price,
            self.avg_price,
            self.status.value,
            self.success,
            self.error_message,
            self.client_id,
            self.total_fee,
            self.send_ts_ms,
            self.fill_ts_ms,
            self.kind.value,
        ]
