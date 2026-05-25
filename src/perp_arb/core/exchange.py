"""Minimal async exchange interface. Concrete adapters live in exchanges/<venue>/."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from decimal import Decimal

from .fill_tracker import _PerCidFillTracker
from .types import (
    FillDelta,
    LegOutcome,
    MarketInfo,
    OrderBook,
    OrderSnapshot,
    Position,
    Quote,
    Side,
)

QuoteCallback = Callable[[Quote], None]
OrderBookCallback = Callable[[OrderBook], None]
# Either a cumulative-state snapshot (lighter `account_market.orders`)
# or a per-fill delta (aster `ORDER_TRADE_UPDATE`).
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
        client_id: str,
        reduce_only: bool = False,
    ) -> LegOutcome:
        """Submit a market order and return whatever the venue's synchronous
        response provides — ack fields populated, fill-side populated only
        if the venue's REST/WS-sync reply carries them (aster's
        `executedQty`/`avgPrice` for synchronous-fill markets). WS fill
        events arrive separately via the per-cid tracker.

        `client_id` is mandatory — the WS fill tracker keys on it, so callers
        that want fill events to land in the correct slot must control the
        cid. For end-to-end submit + WS-fill resolution, use `submit_and_await`."""

    @abstractmethod
    async def get_position(self, market: MarketInfo) -> Position: ...

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
    # between the place ack and WS terminal-fill resolution (e.g. recording
    # per-leg latency marks the moment the venue acks the submit, even
    # while the WS event hasn't landed yet) use these directly instead
    # of the bundled `submit_and_await`.

    def register_fill_slot(self, client_id: str) -> None:
        """Whitelist `client_id` in the per-cid fill tracker BEFORE
        `place_market_order`. Required to prevent a fast fill (fills can
        land before the place ack returns on some venues) from being
        dropped by the unknown-cid guard inside the WS dispatcher.
        Idempotent.

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
    ) -> LegOutcome | None:
        """Wait up to `timeout_s` for accumulated fills on `client_id`
        to reach `requested_qty` or terminal status. Returns whatever
        landed (`None` if never registered). Only fill-side fields are
        populated — ack-side stays at defaults."""
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
    ) -> LegOutcome:
        """One-shot wrapper: register + submit + await + release.

        This is the **single merge point** between the synchronous REST/WS
        place-ack and the WS-aggregated fill stream. The driver's
        `place_market_order` returns a LegOutcome with ack fields plus any
        REST-reported fill data (aster's `executedQty`/`avgPrice` for
        synchronous-fill markets); we then overlay WS-aggregated fill data
        on top when the tracker observes equal-or-larger fill quantity.

        Overlay rule: WS only WINS when `ws.filled_qty >= outcome.filled_qty`.
        That preserves REST's synchronous-full-fill view (aster) when the
        WS stream only delivers a partial-with-terminal-status event (a
        single trailing FillDelta would otherwise downgrade a complete
        REST fill). For venues whose REST has no fill data (lighter:
        outcome.filled_qty=0), any ws.filled_qty>0 wins. Fee and ts come
        from the same source we accept the qty/price from, to keep the
        view internally consistent.

        Status reconciliation: when the WS stream observed a terminal
        status (FILLED / CANCELED / REJECTED / EXPIRED), it's the
        authoritative final state — promote it onto outcome.status so
        downstream filters (CSV `status` column) see the real outcome.
        """
        self.register_fill_slot(client_id)
        try:
            outcome = await self.place_market_order(
                market, side, qty,
                reduce_only=reduce_only, client_id=client_id,
            )
            if not outcome.success:
                return outcome
            ws = await self.await_fill(client_id, qty, timeout_s)
            if ws is not None:
                if ws.filled_qty >= outcome.filled_qty and ws.filled_qty > 0:
                    outcome.filled_qty = ws.filled_qty
                    outcome.weighted_price_sum = ws.weighted_price_sum
                    outcome.last_ts_ms = ws.last_ts_ms
                    outcome.total_fee = ws.total_fee
                if ws.last_status is not None:
                    outcome.last_status = ws.last_status
                    if ws.last_status.terminal:
                        outcome.status = ws.last_status
            return outcome
        finally:
            self.release_fill_slot(client_id)
