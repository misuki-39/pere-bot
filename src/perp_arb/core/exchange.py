"""Minimal async exchange interface. Concrete adapters live in exchanges/<venue>/."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from decimal import Decimal

from .types import (
    MarketInfo,
    OrderBook,
    OrderInfo,
    OrderResult,
    Position,
    Quote,
    Side,
)

QuoteCallback = Callable[[Quote], None]
OrderBookCallback = Callable[[OrderBook], None]
OrderUpdateCallback = Callable[[OrderInfo], None]
PositionCallback = Callable[[Position], None]


class BaseExchange(ABC):
    """Adapter contract every venue must implement.

    Keep this surface minimal: add methods (limit orders, batch ops) only when
    a strategy actually needs them. The reference perp-dex-tools BaseExchange
    fattened up and became hard to satisfy for new venues — don't repeat that.
    """

    name: str

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def load_market(self, raw_symbol: str) -> MarketInfo:
        """Resolve a venue-native ticker to a populated MarketInfo (tick, lot, id)."""

    @abstractmethod
    async def place_market_order(
        self,
        market: MarketInfo,
        side: Side,
        qty: Decimal,
        *,
        reduce_only: bool = False,
        client_id: str | None = None,
    ) -> OrderResult: ...

    @abstractmethod
    async def cancel_order(self, market: MarketInfo, order_id: str) -> OrderResult: ...

    @abstractmethod
    async def get_position(self, market: MarketInfo) -> Position: ...

    @abstractmethod
    async def get_order(self, market: MarketInfo, order_id: str) -> OrderInfo | None: ...

    @abstractmethod
    def subscribe_quotes(self, market: MarketInfo, cb: QuoteCallback) -> None:
        """Register a top-of-book callback. Fired every time BBO changes."""

    @abstractmethod
    def subscribe_book(self, market: MarketInfo, cb: OrderBookCallback) -> None:
        """Register a depth-aware callback. Fired on every depth update."""

    @abstractmethod
    def subscribe_fills(self, market: MarketInfo, cb: OrderUpdateCallback) -> None:
        """Register a fill / order-update callback from the user data stream."""

    @abstractmethod
    def subscribe_positions(self, market: MarketInfo, cb: PositionCallback) -> None:
        """Register a callback fired on every position update from the venue.

        Use this to track authoritative venue-reported position state (which
        also catches external changes: manual closes, liquidations, funding).
        """

    def book_ts(self, market: MarketInfo) -> int | None:
        """Wall-clock ms of the last update applied to a valid, in-sync local
        book for `market`; None until first sync.

        Concrete (non-abstract) so venues opt in. Recorders use this as a
        uniform liveness signal: a down/desynced feed applies no in-sync
        updates, so this timestamp ages out and the recorder can skip.
        """
        return None

    @abstractmethod
    def best_quote(self, market: MarketInfo) -> Quote | None:
        """Cached most-recent Quote (None until first WS frame)."""

    @abstractmethod
    def order_book(self, market: MarketInfo) -> OrderBook | None:
        """Cached most-recent OrderBook snapshot."""

    @abstractmethod
    def live_position(self, market: MarketInfo) -> Position | None:
        """Cached most-recent venue-reported position (None until first event)."""
