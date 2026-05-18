"""Katana Perps public WebSocket adapter.

Katana's `l2orderbook` channel is **diff-only** (IDEX-v4 style): there is no
snapshot over the socket. The book is bootstrapped from the REST
`/v1/orderbook` snapshot and kept in sync via the per-message sequence `u`:

  C -> S {"method":"subscribe","subscriptions":[{"name":"l2orderbook","markets":["ETH-USD"]}]}
  S -> C {"type":"subscriptions","subscriptions":[...]}                      # ack
  S -> C {"type":"l2orderbook","data":{"m","t","u",                          # diff
          "b":[[price,size,orderCount],...],"a":[...],"lp","mp","ip"}}

`size == 0` removes a level. The REST snapshot carries `sequence`; diffs with
`u <= sequence` are stale and dropped, the rest applied in order. A jump in `u`
means missed messages — counted (`seq_gap_total`) and resynced from a fresh
snapshot. `reconnects` counts (re)connections. Both feed the spread monitor's
data-completeness columns.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import aiohttp
import orjson
import websockets

from ...core.types import BookLevel, OrderBook, Quote, Symbol
from ...utils.proxy import get_proxy_url
from ...utils.time import now_ms

_log = logging.getLogger(__name__)

BookCallback = Callable[[OrderBook], None]
QuoteCallback = Callable[[Quote], None]

_TOP_N = 20


class KatanaPublicWs:
    """Per-market diff stream reconciled against a REST snapshot."""

    def __init__(
        self,
        *,
        ws_url: str,
        rest_base_url: str,
        symbol: Symbol,
        market: str,
    ) -> None:
        self.ws_url = ws_url
        self.rest_base_url = rest_base_url.rstrip("/")
        self.symbol = symbol
        self.market = market
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

        self._book_cbs: list[BookCallback] = []
        self._quote_cbs: list[QuoteCallback] = []

        self._bids: dict[Decimal, Decimal] = {}
        self._asks: dict[Decimal, Decimal] = {}
        self._last_book: OrderBook | None = None
        self._last_quote: Quote | None = None

        self._synced = False
        self._buffer: list[dict[str, Any]] = []
        self._last_u: int | None = None

        # data-completeness counters (read by the spread monitor)
        self.reconnects = 0
        self.seq_gap_total = 0
        self.last_seq: int | None = None

    def add_book_callback(self, cb: BookCallback) -> None:
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
            self._task = asyncio.create_task(self._run(), name=f"katana-pubws-{self.market}")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        backoff = 1.0
        sub = orjson.dumps({
            "method": "subscribe",
            "subscriptions": [{"name": "l2orderbook", "markets": [self.market]}],
        }).decode()
        while not self._stop.is_set():
            try:
                _log.info("katana-pubws connecting: %s", self.ws_url)
                async with websockets.connect(
                    self.ws_url, ping_interval=20, open_timeout=30, proxy=get_proxy_url(),
                ) as ws:
                    backoff = 1.0
                    self.reconnects += 1
                    self._reset_book()
                    await ws.send(sub)
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            await self._handle(ws, raw)
                        except Exception:  # noqa: BLE001
                            _log.exception("katana-pubws _handle failed; raw=%r", raw)
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001
                _log.warning(
                    "katana-pubws disconnected: %s — reconnecting in %.1fs", e, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _reset_book(self) -> None:
        self._bids.clear()
        self._asks.clear()
        self._synced = False
        self._buffer.clear()
        self._last_u = None

    async def _handle(self, ws, raw: str | bytes) -> None:
        msg = orjson.loads(raw)
        if not isinstance(msg, dict):
            return
        t = msg.get("type")
        if t == "ping":
            await ws.send(orjson.dumps({"type": "pong"}).decode())
        elif t == "subscriptions":
            # ack: now safe to bootstrap the snapshot (diffs buffer until then)
            await self._bootstrap()
        elif t == "error":
            _log.error("katana-pubws server error: %s", msg.get("data"))
        elif t == "l2orderbook":
            self._on_diff(msg.get("data") or {})

    async def _bootstrap(self) -> None:
        """Fetch the REST snapshot and reconcile buffered diffs against it."""
        url = f"{self.rest_base_url}/v1/orderbook?market={self.market}&level=2"
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as s, s.get(url) as r:
                r.raise_for_status()
                snap = await r.json(content_type=None)
        except Exception as e:  # noqa: BLE001
            _log.warning("katana-pubws snapshot fetch failed: %s — will retry on next ack", e)
            return
        seq = int(snap["sequence"])
        self._bids = {
            p: sz for lvl in snap.get("bids", [])
            if (sz := Decimal(str(lvl[1]))) > 0
            for p in (Decimal(str(lvl[0])),)
        }
        self._asks = {
            p: sz for lvl in snap.get("asks", [])
            if (sz := Decimal(str(lvl[1]))) > 0
            for p in (Decimal(str(lvl[0])),)
        }
        self._last_u = seq
        # replay buffered diffs newer than the snapshot, in order
        for d in sorted(self._buffer, key=lambda d: int(d.get("u", 0))):
            if int(d.get("u", 0)) > seq:
                self._apply(d)
        self._buffer.clear()
        self._synced = True
        self._emit_book()

    def _on_diff(self, d: dict[str, Any]) -> None:
        u = int(d.get("u", 0))
        self.last_seq = u
        if not self._synced:
            self._buffer.append(d)
            return
        assert self._last_u is not None
        if u <= self._last_u:
            return  # stale / duplicate
        if u > self._last_u + 1:
            # missed messages — count them and resync from a fresh snapshot
            self.seq_gap_total += u - self._last_u - 1
            _log.warning(
                "katana-pubws sequence gap: %d -> %d (resync)", self._last_u, u,
            )
            self._reset_book()
            asyncio.create_task(self._bootstrap(), name=f"katana-resync-{self.market}")
            self._buffer.append(d)
            return
        self._apply(d)
        self._emit_book()

    def _apply(self, d: dict[str, Any]) -> None:
        for p, sz, _c in d.get("b", []):
            self._set(self._bids, Decimal(str(p)), Decimal(str(sz)))
        for p, sz, _c in d.get("a", []):
            self._set(self._asks, Decimal(str(p)), Decimal(str(sz)))
        self._last_u = int(d.get("u", 0))

    @staticmethod
    def _set(side: dict[Decimal, Decimal], price: Decimal, size: Decimal) -> None:
        if size == 0:
            side.pop(price, None)
        else:
            side[price] = size

    def _emit_book(self) -> None:
        top_bids = sorted(self._bids.items(), key=lambda x: x[0], reverse=True)[:_TOP_N]
        top_asks = sorted(self._asks.items(), key=lambda x: x[0])[:_TOP_N]
        if not top_bids or not top_asks:
            return
        prev = self._last_quote
        new_top = (top_bids[0][0], top_bids[0][1], top_asks[0][0], top_asks[0][1])
        if prev is not None and (prev.bid, prev.bid_size, prev.ask, prev.ask_size) == new_top:
            return
        bids = [BookLevel(p, s) for p, s in top_bids]
        asks = [BookLevel(p, s) for p, s in top_asks]
        ts = now_ms()
        self._last_book = OrderBook(symbol=self.symbol, bids=bids, asks=asks, ts_ms=ts)
        self._last_quote = Quote(
            symbol=self.symbol,
            bid=bids[0].price, bid_size=bids[0].size,
            ask=asks[0].price, ask_size=asks[0].size,
            ts_ms=ts,
        )
        for cb in self._book_cbs:
            cb(self._last_book)
        for cb in self._quote_cbs:
            cb(self._last_quote)
