"""KatanaClient — public-only BaseExchange for Katana Perps v1.

This phase only consumes public market data (spread feasibility study), so the
trading surface is intentionally unimplemented: order/position calls raise.
Mirrors the structure of the Lighter public path.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from decimal import Decimal

import aiohttp

from ...core.exchange import (
    BaseExchange,
    OrderBookCallback,
    OrderUpdateCallback,
    PositionCallback,
    QuoteCallback,
)
from ...core.types import (
    MarketInfo,
    OrderBook,
    OrderInfo,
    OrderResult,
    Position,
    Quote,
    Side,
    Symbol,
)
from .ws import KatanaPublicWs

_log = logging.getLogger(__name__)

_PUBLIC_ONLY = "katana is public_only — trading is not implemented this phase"


class KatanaClient(BaseExchange):
    name = "katana"

    def __init__(
        self,
        *,
        base_url: str,
        ws_url: str,
        public_only: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.ws_url = ws_url
        self.public_only = public_only
        self._session: aiohttp.ClientSession | None = None
        self._public_ws_by_symbol: dict[str, KatanaPublicWs] = {}

    # ---- lifecycle ----

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(trust_env=True)

    async def disconnect(self) -> None:
        for ws in self._public_ws_by_symbol.values():
            await ws.stop()
        if self._session is not None:
            with contextlib.suppress(Exception):
                await self._session.close()
            self._session = None

    # ---- markets / data ----

    async def load_market(self, raw_symbol: str) -> MarketInfo:
        assert self._session is not None, "call connect() before load_market()"
        async with self._session.get(
            f"{self.base_url}/v1/markets", timeout=aiohttp.ClientTimeout(total=15)
        ) as r:
            r.raise_for_status()
            body = await r.json(content_type=None)
        rows = body if isinstance(body, list) else body.get("data") or body.get("markets") or []
        for m in rows:
            if str(m.get("market")) != raw_symbol:
                continue
            symbol = Symbol(
                exchange="katana",
                raw=raw_symbol,
                base=str(m.get("baseAsset", raw_symbol.split("-")[0])),
                quote=str(m.get("quoteAsset", "USD")),
            )
            return MarketInfo(
                symbol=symbol,
                tick_size=Decimal(str(m["tickSize"])),
                lot_size=Decimal(str(m["stepSize"])),
                contract_id=raw_symbol,
                min_qty=Decimal(str(m.get("minimumPositionSize", "0"))),
            )
        raise RuntimeError(f"katana market {raw_symbol!r} not found in /v1/markets")

    def _ensure_public_ws(self, market: MarketInfo) -> KatanaPublicWs:
        key = market.symbol.raw
        ws = self._public_ws_by_symbol.get(key)
        if ws is None:
            ws = KatanaPublicWs(
                ws_url=self.ws_url,
                rest_base_url=self.base_url,
                symbol=market.symbol,
                market=str(market.contract_id),
            )
            self._public_ws_by_symbol[key] = ws
            asyncio.create_task(ws.start(), name=f"katana-pubws-{key}-start")
        return ws

    def subscribe_quotes(self, market: MarketInfo, cb: QuoteCallback) -> None:
        self._ensure_public_ws(market).add_quote_callback(cb)

    def subscribe_book(self, market: MarketInfo, cb: OrderBookCallback) -> None:
        self._ensure_public_ws(market).add_book_callback(cb)

    def subscribe_fills(self, market: MarketInfo, cb: OrderUpdateCallback) -> None:
        pass  # no user stream in public_only mode

    def subscribe_positions(self, market: MarketInfo, cb: PositionCallback) -> None:
        pass

    def best_quote(self, market: MarketInfo) -> Quote | None:
        ws = self._public_ws_by_symbol.get(market.symbol.raw)
        return ws.last_quote if ws else None

    def order_book(self, market: MarketInfo) -> OrderBook | None:
        ws = self._public_ws_by_symbol.get(market.symbol.raw)
        return ws.last_book if ws else None

    def live_position(self, market: MarketInfo) -> Position | None:
        return None

    def book_ts(self, market: MarketInfo) -> int | None:
        ws = self._public_ws_by_symbol.get(market.symbol.raw)
        return ws.last_update_ms if ws and ws.last_update_ms else None

    # ---- trading surface: unimplemented this phase ----

    async def place_market_order(
        self,
        market: MarketInfo,
        side: Side,
        qty: Decimal,
        *,
        reduce_only: bool = False,
        client_id: str | None = None,
    ) -> OrderResult:
        raise RuntimeError(_PUBLIC_ONLY)

    async def cancel_order(self, market: MarketInfo, order_id: str) -> OrderResult:
        raise RuntimeError(_PUBLIC_ONLY)

    async def get_order(self, market: MarketInfo, order_id: str) -> OrderInfo | None:
        raise RuntimeError(_PUBLIC_ONLY)

    async def get_position(self, market: MarketInfo) -> Position:
        raise RuntimeError(_PUBLIC_ONLY)
