"""Aster WebSocket: public depth stream + (optional) user-data stream."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable
from decimal import Decimal

import websockets

from ...core.types import BookLevel, OrderBook, Quote, Symbol
from ...utils.proxy import get_proxy_url
from ...utils.time import now_ms
from .rest import AsterRest

_log = logging.getLogger(__name__)

DepthCallback = Callable[[OrderBook], None]
QuoteCallback = Callable[[Quote], None]
UserEventCallback = Callable[[dict], None]


class AsterPublicWs:
    """Subscribes to `<symbol>@depth<levels>@<speed>` for partial book snapshots.

    Each frame is a self-contained snapshot of the top-N levels — we don't need
    to manage a diff-applied book.
    """

    def __init__(
        self,
        *,
        ws_url: str,
        symbol: Symbol,
        levels: int = 20,
        speed_ms: int = 100,
    ) -> None:
        self.ws_url = ws_url.rstrip("/")
        self.symbol = symbol
        self.levels = levels
        self.speed_ms = speed_ms
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

        self._book_cbs: list[DepthCallback] = []
        self._quote_cbs: list[QuoteCallback] = []

        self._last_book: OrderBook | None = None
        self._last_quote: Quote | None = None
        # wall-clock ms of the last received depth frame — feed liveness,
        # exposed via the client's book_ts(). 0 until the first frame.
        self._last_update_ms = 0

    @property
    def last_update_ms(self) -> int:
        return self._last_update_ms

    @property
    def stream_name(self) -> str:
        # Aster uses lowercase symbol in stream names.
        return f"{self.symbol.raw.lower()}@depth{self.levels}@{self.speed_ms}ms"

    def add_book_callback(self, cb: DepthCallback) -> None:
        self._book_cbs.append(cb)

    def add_quote_callback(self, cb: QuoteCallback) -> None:
        self._quote_cbs.append(cb)

    @property
    def last_book(self) -> OrderBook | None:
        return self._last_book

    @property
    def last_quote(self) -> Quote | None:
        return self._last_quote

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name=f"aster-ws-{self.symbol.raw}")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        url = f"{self.ws_url}/stream?streams={self.stream_name}"
        backoff = 1.0
        while not self._stop.is_set():
            try:
                _log.info("aster-ws connecting: %s", url)
                async with websockets.connect(url, ping_interval=20, proxy=get_proxy_url()) as ws:
                    backoff = 1.0
                    async for msg in ws:
                        if self._stop.is_set():
                            break
                        try:
                            self._handle(msg)
                        except Exception as e:  # noqa: BLE001
                            _log.exception("aster-ws handle error: %s", e)
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001
                _log.warning("aster-ws disconnected: %s — reconnecting in %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _handle(self, raw: str | bytes) -> None:
        # Combined-stream wraps as {"stream": "...", "data": {...depthUpdate...}}.
        payload = json.loads(raw)["data"]
        ts_ms = int(payload["E"])
        # @depth<N> stream returns sides pre-sorted (bids desc, asks asc).
        bids = [BookLevel(Decimal(p), s) for p, q in payload["b"] if (s := Decimal(q)) > 0]
        asks = [BookLevel(Decimal(p), s) for p, q in payload["a"] if (s := Decimal(q)) > 0]
        if not bids or not asks:
            return
        self._last_update_ms = now_ms()  # liveness: every valid frame

        prev = self._last_quote
        new_quote = (bids[0].price, bids[0].size, asks[0].price, asks[0].size)
        if prev is not None and (prev.bid, prev.bid_size, prev.ask, prev.ask_size) == new_quote:
            # top-of-book unchanged — skip the fan-out
            return

        self._last_book = OrderBook(symbol=self.symbol, bids=bids, asks=asks, ts_ms=ts_ms)
        self._last_quote = Quote(
            symbol=self.symbol,
            bid=bids[0].price, bid_size=bids[0].size,
            ask=asks[0].price, ask_size=asks[0].size,
            ts_ms=ts_ms,
        )
        for cb in self._book_cbs:
            cb(self._last_book)
        for cb in self._quote_cbs:
            cb(self._last_quote)


class AsterUserWs:
    """User-data stream via listenKey. Required for live order fills/updates."""

    def __init__(self, *, ws_url: str, rest: AsterRest) -> None:
        self.ws_url = ws_url.rstrip("/")
        self.rest = rest
        self._listen_key: str | None = None
        self._stop = asyncio.Event()
        self._connected = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._keepalive_task: asyncio.Task[None] | None = None
        self._cbs: list[UserEventCallback] = []

    def add_callback(self, cb: UserEventCallback) -> None:
        self._cbs.append(cb)

    async def start(self, *, connect_timeout_s: float = 10.0) -> None:
        """Start the user-data stream and block until the WS handshake completes.

        Returning early would race order placement against the WS connect — the
        first ORDER_TRADE_UPDATE would arrive before any subscriber exists.
        """
        self._listen_key = await self.rest.start_user_stream()
        _log.info("aster user-stream listenKey acquired")
        self._stop.clear()
        self._connected.clear()
        self._task = asyncio.create_task(self._run(), name="aster-user-ws")
        self._keepalive_task = asyncio.create_task(self._keepalive(), name="aster-user-keepalive")
        await asyncio.wait_for(self._connected.wait(), timeout=connect_timeout_s)

    async def stop(self) -> None:
        self._stop.set()
        for t in (self._task, self._keepalive_task):
            if t:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
        self._task = None
        self._keepalive_task = None
        with contextlib.suppress(Exception):
            await self.rest.close_user_stream()

    async def _keepalive(self) -> None:
        # PUT listenKey every ~50 min.
        while not self._stop.is_set():
            await asyncio.sleep(50 * 60)
            try:
                await self.rest.keepalive_user_stream()
            except Exception as e:  # noqa: BLE001
                _log.warning("aster listenKey keepalive failed: %s", e)

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            url = f"{self.ws_url}/ws/{self._listen_key}"
            try:
                async with websockets.connect(url, ping_interval=20, proxy=get_proxy_url()) as ws:
                    backoff = 1.0
                    _log.info("aster user-data WS connected")
                    self._connected.set()
                    async for msg in ws:
                        if self._stop.is_set():
                            break
                        try:
                            data = json.loads(msg if isinstance(msg, str) else msg.decode())
                        except json.JSONDecodeError:
                            continue
                        if data.get("e") == "listenKeyExpired":
                            _log.warning("aster listenKey expired; rotating")
                            self._listen_key = await self.rest.start_user_stream()
                            break
                        for cb in self._cbs:
                            try:
                                cb(data)
                            except Exception as e:  # noqa: BLE001
                                _log.exception("aster user-stream callback raised: %s", e)
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001
                _log.warning("aster user-ws disconnected: %s — reconnecting in %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
