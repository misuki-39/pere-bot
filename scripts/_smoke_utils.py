"""Shared helpers for venue smoke scripts (aster, lighter, ...).

Kept inside `scripts/` (not the `perp_arb` package) because these helpers only
exist to support verification scripts, not the bot runtime.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from perp_arb.core.exchange import BaseExchange
from perp_arb.core.types import MarketInfo, OrderInfo, OrderStatus, Position

_log = logging.getLogger("smoke")


@dataclass
class WsObserver:
    """Collects fill + position events for a single round-trip leg.

    `arm()` clears the events before placing an order; the two `wait_*` helpers
    block until the matching WS frame arrives (or time out).
    """

    fill_timeout_s: float = 5.0
    position_timeout_s: float = 5.0

    fills: list[OrderInfo] = field(default_factory=list)
    positions: list[Position] = field(default_factory=list)
    _fill_event: asyncio.Event = field(default_factory=asyncio.Event)
    _position_event: asyncio.Event = field(default_factory=asyncio.Event)

    def on_fill(self, info: OrderInfo) -> None:
        self.fills.append(info)
        _log.info("WS FILL: order_id=%s status=%s filled=%s avg=%s",
                  info.order_id, info.status, info.filled_size, info.avg_fill_price)
        if info.status is OrderStatus.FILLED:
            self._fill_event.set()

    def on_position(self, pos: Position) -> None:
        self.positions.append(pos)
        _log.info("WS POSITION: size=%s entry=%s upnl=%s",
                  pos.size, pos.entry_price, pos.unrealised_pnl)
        self._position_event.set()

    def arm(self) -> None:
        self._fill_event.clear()
        self._position_event.clear()

    async def wait_fill(self) -> None:
        await asyncio.wait_for(self._fill_event.wait(), self.fill_timeout_s)

    async def wait_position(self) -> None:
        await asyncio.wait_for(self._position_event.wait(), self.position_timeout_s)


async def wait_for_quote(
    client: BaseExchange,
    market: MarketInfo,
    timeout_s: float = 8.0,
) -> None:
    """Poll `client.best_quote(market)` until non-None or timeout."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if client.best_quote(market) is not None:
            return
        await asyncio.sleep(0.1)
    raise RuntimeError("no quote received — public depth WS not connected?")
