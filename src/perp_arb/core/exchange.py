"""Minimal async exchange interface. Concrete adapters live in exchanges/<venue>/."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from decimal import Decimal

from .fill_tracker import _PerCidFillTracker
from .types import (
    FillDelta,
    MarketInfo,
    OrderBook,
    OrderOutcome,
    OrderResult,
    OrderSnapshot,
    Position,
    Quote,
    Side,
    TerminalFill,
)

QuoteCallback = Callable[[Quote], None]
OrderBookCallback = Callable[[OrderBook], None]
# Either a cumulative-state snapshot (lighter account_market.orders,
# aster REST get_order) or a per-fill delta (aster ORDER_TRADE_UPDATE).
OrderUpdateCallback = Callable[[OrderSnapshot | FillDelta], None]
PositionCallback = Callable[[Position], None]


class BaseExchange(ABC):
    """Adapter contract every venue must implement.

    Keep this surface minimal: add methods (limit orders, batch ops) only when
    a strategy actually needs them. The reference perp-dex-tools BaseExchange
    fattened up and became hard to satisfy for new venues — don't repeat that.

    Concrete drivers MUST call `super().__init__()` and route their WS fill
    handler into `self._fill_tracker.on_event(ev)` so `submit_and_await`
    can resolve per-`client_id` terminal state. The tracker is an
    additional cid-keyed channel; the per-symbol `subscribe_fills`
    fan-out is unaffected.
    """

    name: str

    def __init__(self) -> None:
        self._fill_tracker = _PerCidFillTracker()

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
    async def get_order(self, market: MarketInfo, order_id: str) -> OrderSnapshot | None: ...

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

    # ----- per-`client_id` fill tracking primitives -----
    #
    # The three operations below are the lower-level primitives behind
    # `submit_and_await`. Callers that need to interleave bookkeeping
    # between REST ack and WS terminal-fill resolution (e.g. recording
    # per-leg latency marks the moment the venue acks the submit, even
    # while the WS event hasn't landed yet) use these directly instead
    # of the bundled `submit_and_await`.

    def register_fill_slot(self, client_id: str) -> None:
        """Whitelist `client_id` in the per-cid fill tracker BEFORE
        `place_market_order`. Required to prevent a fast fill (aster can
        fill before REST returns) from being dropped by the unknown-cid
        guard inside the WS dispatcher. Idempotent.

        Race-safety: `register_fill_slot` must run in the same
        synchronous stretch as the `place_market_order` call — no
        `await` between them. WS callbacks share the event loop, so a
        fill cannot land in that window.
        """
        self._fill_tracker.register(client_id)

    async def await_fill(
        self,
        client_id: str,
        requested_qty: Decimal,
        timeout_s: float,
    ) -> TerminalFill | None:
        """Wait up to `timeout_s` for accumulated fills on `client_id`
        to reach `requested_qty` or terminal status. Returns whatever
        landed (`None` if never registered)."""
        return await self._fill_tracker.await_terminal(
            client_id, requested_qty, timeout_s,
        )

    def release_fill_slot(self, client_id: str) -> None:
        """Drop the tracker slot. Idempotent. Pair with
        `register_fill_slot` in a try/finally so a raise during submit
        doesn't leak a slot."""
        self._fill_tracker.release(client_id)

    async def submit_and_await(
        self,
        market: MarketInfo,
        side: Side,
        qty: Decimal,
        *,
        client_id: str,
        timeout_s: float = 5.0,
        reduce_only: bool = False,
    ) -> OrderOutcome:
        """One-shot wrapper: register + submit + await + release.

        Use this when you don't need to interleave anything between
        REST ack and WS terminal fill. Most callers should use this.
        For finer-grained control (e.g. timeline marks between REST
        and fill), use the three primitives above.
        """
        self.register_fill_slot(client_id)
        try:
            rest = await self.place_market_order(
                market, side, qty,
                reduce_only=reduce_only, client_id=client_id,
            )
            if not rest.success:
                return OrderOutcome(rest=rest, fill=None)
            fill = await self.await_fill(client_id, qty, timeout_s)
            return OrderOutcome(rest=rest, fill=fill)
        finally:
            self.release_fill_slot(client_id)
