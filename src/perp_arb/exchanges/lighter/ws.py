"""Lighter WebSocket adapters.

We bypass the official SDK's `WsClient` because it imports the deprecated
legacy `websockets.client.connect`, which the current Lighter server rejects
with HTTP 400. Two adapters here, both talking to `wss://<host>/stream`:

* `LighterPublicWs` — orderbook diff-merge per market.
* `LighterUserWs` — multiplexes user-data subscription (`account_all/<acct>`)
  with WS-based transaction submission (`jsonapi/sendtx`) on a single
  connection.

Protocol (verified against `wss://mainnet.zklighter.elliot.ai/stream`):

  C -> S {"type":"subscribe","channel":"order_book/0"}
  S -> C {"type":"connected","session_id":"..."}
  S -> C {"type":"subscribed/order_book","order_book":{"bids":[...],"asks":[...]}}
  S -> C {"type":"update/order_book","order_book":{"bids":[...],"asks":[...]}}
  S -> C {"type":"ping"} -> C -> S {"type":"pong"}

  C -> S {"type":"jsonapi/sendtx","data":{"id":"<req>","tx_type":N,"tx_info":{...}}}
  S -> C {"type":"jsonapi/sendtx", ...echoes "id" inside the reply...}
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import logging
import time
import uuid
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import orjson
import websockets

from ...core.logging import RateLimited
from ...core.types import BookLevel, OrderBook, Quote, Symbol
from ...utils.proxy import get_proxy_url
from ...utils.time import now_ms

_log = logging.getLogger(__name__)

BookCallback = Callable[[OrderBook], None]
QuoteCallback = Callable[[Quote], None]
AccountCallback = Callable[[dict[str, Any]], None]

_TOP_N = 20    # cap on book levels we expose downstream
# Lighter has no resync path (book only clears on a full socket reconnect), so
# an unpruned diff-merged dict grows for the whole multi-day run — memory plus
# a per-message O(n log n) sort. Keep only the best _BOOK_CAP levels.
_BOOK_CAP = 512


_STABLE_S = 30.0  # min session uptime to reset reconnect backoff to 1 s


def _trim(side: dict[Decimal, Decimal], *, keep_highest: bool) -> None:
    survivors = set(
        heapq.nlargest(_BOOK_CAP, side) if keep_highest
        else heapq.nsmallest(_BOOK_CAP, side)
    )
    for p in list(side):
        if p not in survivors:
            del side[p]


def _ws_url(base_url: str) -> str:
    host = base_url.replace("https://", "").replace("http://", "").rstrip("/")
    return f"wss://{host}/stream"


class LighterPublicWs:
    """Per-market orderbook stream. Diff-merged into a price→size dict."""

    def __init__(self, *, base_url: str, symbol: Symbol, market_index: int) -> None:
        self.ws_url = _ws_url(base_url)
        self.symbol = symbol
        self.market_index = market_index
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

        self._book_cbs: list[BookCallback] = []
        self._quote_cbs: list[QuoteCallback] = []

        self._bids: dict[Decimal, Decimal] = {}
        self._asks: dict[Decimal, Decimal] = {}
        self._last_book: OrderBook | None = None
        self._last_quote: Quote | None = None
        # wall-clock ms of the last applied book message — feed liveness,
        # exposed via the client's book_ts(). 0 until the first snapshot.
        self._last_update_ms = 0

    def add_book_callback(self, cb: BookCallback) -> None:
        self._book_cbs.append(cb)

    def add_quote_callback(self, cb: QuoteCallback) -> None:
        self._quote_cbs.append(cb)

    @property
    def last_update_ms(self) -> int:
        return self._last_update_ms

    @property
    def last_book(self) -> OrderBook | None:
        return self._last_book

    @property
    def last_quote(self) -> Quote | None:
        return self._last_quote

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(
                self._run(), name=f"lighter-pubws-{self.symbol.raw}",
            )

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        backoff = 1.0
        errlim = RateLimited(10.0)
        while not self._stop.is_set():
            t0: float | None = None
            try:
                _log.info("lighter-pubws connecting: %s", self.ws_url)
                async with websockets.connect(
                    self.ws_url, ping_interval=20, proxy=get_proxy_url(),
                ) as ws:
                    t0 = loop.time()
                    self._bids.clear()
                    self._asks.clear()
                    await ws.send(orjson.dumps({
                        "type": "subscribe",
                        "channel": f"order_book/{self.market_index}",
                    }).decode())
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            await self._handle(ws, raw)
                        except Exception:  # noqa: BLE001
                            if (n := errlim.tick(loop.time())) is not None:
                                _log.exception(
                                    "lighter-pubws _handle failed (%d in interval); raw=%r",
                                    n, raw,
                                )
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001
                lasted = (loop.time() - t0) if t0 is not None else 0.0
                backoff = 1.0 if lasted >= _STABLE_S else min(backoff * 2, 30.0)
                _log.warning(
                    "lighter-pubws disconnected: %s — reconnecting in %.1fs", e, backoff,
                )
                await asyncio.sleep(backoff)

    async def _handle(self, ws, raw: str | bytes) -> None:
        data = orjson.loads(raw)
        t = data.get("type")
        if t == "ping":
            await ws.send(orjson.dumps({"type": "pong"}).decode())
        elif t == "subscribed/order_book":
            ob = data["order_book"]
            self._bids = {
                p: s for lvl in ob.get("bids", [])
                if (s := Decimal(lvl["size"])) > 0
                for p in (Decimal(lvl["price"]),)
            }
            self._asks = {
                p: s for lvl in ob.get("asks", [])
                if (s := Decimal(lvl["size"])) > 0
                for p in (Decimal(lvl["price"]),)
            }
            self._last_update_ms = now_ms()
            self._emit_book()
        elif t == "update/order_book":
            ob = data["order_book"]
            _apply_diff(self._bids, ob.get("bids", []))
            _apply_diff(self._asks, ob.get("asks", []))
            self._last_update_ms = now_ms()
            self._emit_book()

    def _emit_book(self) -> None:
        bids, asks = self._bids, self._asks
        if not bids or not asks:
            return
        if len(bids) > _BOOK_CAP * 2:
            _trim(bids, keep_highest=True)
        if len(asks) > _BOOK_CAP * 2:
            _trim(asks, keep_highest=False)
        # Best bid/ask + dedup BEFORE building the top-N levels, so an
        # unchanged-BBO message (the common case) allocates nothing.
        best_bid = max(bids)
        best_ask = min(asks)
        new_top = (best_bid, bids[best_bid], best_ask, asks[best_ask])
        prev = self._last_quote
        if prev is not None and (prev.bid, prev.bid_size, prev.ask, prev.ask_size) == new_top:
            return
        top_bids = heapq.nlargest(_TOP_N, bids.items(), key=lambda kv: kv[0])
        top_asks = heapq.nsmallest(_TOP_N, asks.items(), key=lambda kv: kv[0])
        bid_levels = [BookLevel(p, s) for p, s in top_bids]
        ask_levels = [BookLevel(p, s) for p, s in top_asks]
        ts = now_ms()
        self._last_book = OrderBook(
            symbol=self.symbol, bids=bid_levels, asks=ask_levels, ts_ms=ts
        )
        self._last_quote = Quote(
            symbol=self.symbol,
            bid=bid_levels[0].price, bid_size=bid_levels[0].size,
            ask=ask_levels[0].price, ask_size=ask_levels[0].size,
            ts_ms=ts,
        )
        for cb in self._book_cbs:
            cb(self._last_book)
        for cb in self._quote_cbs:
            cb(self._last_quote)


class TxSubmitError(Exception):
    """Raised when the Lighter server rejects a jsonapi/sendtx submission."""


AuthTokenFactory = Callable[[], str]


class LighterUserWs:
    """User WS: subscribes to `account_all/<acct>` and submits txs via `jsonapi/sendtx`.

    Order/cancel txs are signed locally by SignerClient and shipped over this
    connection — same URL as the public stream but a separate socket so account
    traffic and market-data traffic don't compete in the same recv loop.

    `auth_token_factory` returns a fresh short-lived signed token (built with
    `SignerClient.create_auth_token_with_expiry`). The token is attached to the
    `account_all` subscribe — the server rejects the channel otherwise.
    """

    def __init__(
        self,
        *,
        base_url: str,
        account_index: int,
        auth_token_factory: AuthTokenFactory | None = None,
        subscribe_account_all: bool = True,
    ) -> None:
        self.ws_url = _ws_url(base_url)
        self.account_index = account_index
        self._auth_token_factory = auth_token_factory
        self._subscribe_account_all = subscribe_account_all
        self._stop = asyncio.Event()
        self._connected = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._ws: websockets.ClientConnection | None = None
        self._account_cbs: list[AccountCallback] = []
        self._market_cbs: list[AccountCallback] = []
        # Per-market opt-in subs. Recorded so `_run` re-subscribes them on
        # every reconnect.
        self._market_channels: list[str] = []
        self._pending_tx: dict[str, asyncio.Future[dict[str, Any]]] = {}

    def add_account_callback(self, cb: AccountCallback) -> None:
        self._account_cbs.append(cb)

    def add_market_callback(self, cb: AccountCallback) -> None:
        self._market_cbs.append(cb)

    async def subscribe_account_market(self, market_index: int) -> None:
        """Subscribe `account_market/<market>/<account>` — per-market stream
        bundling positions, orders, trades, and funding. Strict superset
        of `account_orders` for our purposes (orders carry cumulative qty
        + status + matching-engine `transaction_time`)."""
        channel = f"account_market/{market_index}/{self.account_index}"
        if channel in self._market_channels:
            return
        self._market_channels.append(channel)
        if self._ws is not None and self._connected.is_set():
            await self._send_subscribe(self._ws, channel)

    async def _send_subscribe(
        self, ws: websockets.ClientConnection, channel: str,
    ) -> None:
        msg = {"type": "subscribe", "channel": channel}
        if self._auth_token_factory is not None:
            msg["auth"] = self._auth_token_factory()
        await ws.send(orjson.dumps(msg).decode())

    async def start(self, *, connect_timeout_s: float = 10.0) -> None:
        """Open the WS and wait until the handshake + first subscribe land."""
        self._stop.clear()
        self._connected.clear()
        self._task = asyncio.create_task(self._run(), name="lighter-user-ws")
        await asyncio.wait_for(self._connected.wait(), timeout=connect_timeout_s)

    async def stop(self) -> None:
        self._stop.set()
        # Fail any in-flight txs so awaiters wake up.
        for fut in self._pending_tx.values():
            if not fut.done():
                fut.set_exception(TxSubmitError("user WS shutting down"))
        self._pending_tx.clear()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._ws = None

    async def send_tx(self, tx_type: int, tx_info: str, *, timeout_s: float = 10.0) -> dict[str, Any]:
        """Sign locally, submit over WS, await the server's matching reply."""
        # If the WS is mid-reconnect, wait for it to come back rather than fail.
        await asyncio.wait_for(self._connected.wait(), timeout=timeout_s)
        if self._ws is None:
            raise TxSubmitError("user WS not connected")
        req_id = f"pa-{uuid.uuid4().hex[:12]}-{int(time.time_ns())}"
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending_tx[req_id] = fut
        payload = {
            "type": "jsonapi/sendtx",
            "data": {
                "id": req_id,
                "tx_type": tx_type,
                "tx_info": orjson.loads(tx_info),
            },
        }
        try:
            await self._ws.send(orjson.dumps(payload).decode())
            return await asyncio.wait_for(fut, timeout=timeout_s)
        finally:
            self._pending_tx.pop(req_id, None)

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                _log.info("lighter-user-ws connecting: %s", self.ws_url)
                async with websockets.connect(
                    self.ws_url, ping_interval=20, proxy=get_proxy_url(),
                ) as ws:
                    self._ws = ws
                    backoff = 1.0
                    if self._subscribe_account_all:
                        await self._send_subscribe(
                            ws, f"account_all/{self.account_index}",
                        )
                    for ch in self._market_channels:
                        await self._send_subscribe(ws, ch)
                    self._connected.set()
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            await self._handle(ws, raw)
                        except Exception:  # noqa: BLE001
                            _log.exception("lighter-user-ws _handle failed; raw=%r", raw)
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001
                _log.warning(
                    "lighter-user-ws disconnected: %s — reconnecting in %.1fs", e, backoff,
                )
                self._ws = None
                self._connected.clear()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _handle(self, ws, raw: str | bytes) -> None:
        data = orjson.loads(raw)
        if not isinstance(data, dict):
            # Server sometimes drops plain-string heartbeats / error literals;
            # don't crash the recv loop on them.
            _log.debug("lighter-user-ws non-dict frame: %r", data)
            return
        t = data.get("type")
        if t == "ping":
            await ws.send(orjson.dumps({"type": "pong"}).decode())
        elif t == "connected":
            _log.debug("lighter-user-ws session_id=%s", data.get("session_id"))
        elif t == "jsonapi/sendtx":
            # Reply id is echoed at top-level or inside data — try both.
            req_id = data.get("id") or data.get("data", {}).get("id")
            fut = self._pending_tx.get(req_id) if req_id else None
            if fut is not None and not fut.done():
                fut.set_result(data)
            else:
                _log.warning("lighter-user-ws unmatched sendtx reply: %s", raw)
        elif t in ("subscribed/account_all", "update/account_all"):
            for cb in self._account_cbs:
                cb(data)
        elif t in ("subscribed/account_market", "update/account_market"):
            for cb in self._market_cbs:
                cb(data)
        else:
            _log.debug("lighter-user-ws unhandled frame type=%r", t)


def _apply_diff(side: dict[Decimal, Decimal], updates: list[dict[str, str]]) -> None:
    for lvl in updates:
        p = Decimal(lvl["price"])
        s = Decimal(lvl["size"])
        if s == 0:
            side.pop(p, None)
        else:
            side[p] = s
