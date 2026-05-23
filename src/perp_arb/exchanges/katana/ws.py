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
means missed messages — counted (`seq_gap_total`) and resynced.

Memory safety: while desynced the diff stream must NOT accumulate unbounded.
`_buffer` is a bounded `deque` (old diffs are useless anyway — a successful
bootstrap reseeds from a *fresh* snapshot at the current sequence), and
bootstrap is single-in-flight with a timed retry rather than one task per
sequence-gap.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import logging
from collections import deque
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import aiohttp
import orjson
import websockets

from ...core.logging import RateLimited
from ...core.types import BookLevel, OrderBook, Quote, Symbol
from ...utils.proxy import get_proxy_url
from ...utils.time import now_ms

_log = logging.getLogger(__name__)

BookCallback = Callable[[OrderBook], None]
QuoteCallback = Callable[[Quote], None]

_TOP_N = 20
_BUFFER_MAX = 4096       # bounded; a fresh snapshot makes older diffs irrelevant
_RESYNC_RETRY_S = 3.0    # snapshot-bootstrap retry cadence while desynced
# A session must stay up at least this long to count as "stable"; otherwise
# connect-then-drop flapping (e.g. a flaky proxy) keeps growing the backoff
# instead of resetting it to 1 s on every bare connect.
_STABLE_S = 30.0
# Diff venues only remove a level on an explicit size=0; levels that simply
# leave the top are never zeroed, so an unpruned book dict grows for the whole
# run (memory) and the per-message sort cost grows with it (CPU). Keep only the
# best _BOOK_CAP levels — far deeper than _TOP_N / any VWAP need — and only
# prune once it has drifted well past the cap (amortized O(1)).
_BOOK_CAP = 512


def _trim(side: dict[Decimal, Decimal], *, keep_highest: bool) -> None:
    survivors = set(
        heapq.nlargest(_BOOK_CAP, side) if keep_highest
        else heapq.nsmallest(_BOOK_CAP, side)
    )
    for p in list(side):
        if p not in survivors:
            del side[p]


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
        self._buffer: deque[dict[str, Any]] = deque(maxlen=_BUFFER_MAX)
        self._last_u: int | None = None
        self._gen = 0                       # bumped each (re)connect
        self._bootstrap_task: asyncio.Task[None] | None = None

        # wall-clock ms of the last update applied to the in-sync book —
        # liveness, exposed via the client's book_ts(). 0 until first sync.
        self._last_update_ms = 0
        # internal diagnostics only (drive the adapter's own log warnings)
        self.reconnects = 0
        self.seq_gap_total = 0
        self.last_seq: int | None = None

    @property
    def last_update_ms(self) -> int:
        return self._last_update_ms

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
        if self._bootstrap_task:
            self._bootstrap_task.cancel()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        backoff = 1.0
        errlim = RateLimited(10.0)
        sub = orjson.dumps({
            "method": "subscribe",
            "subscriptions": [{"name": "l2orderbook", "markets": [self.market]}],
        }).decode()
        while not self._stop.is_set():
            t0: float | None = None
            try:
                _log.info("katana-pubws connecting: %s", self.ws_url)
                async with websockets.connect(
                    self.ws_url, ping_interval=20, open_timeout=30, proxy=get_proxy_url(),
                ) as ws:
                    t0 = loop.time()
                    self.reconnects += 1
                    self._reset_book()
                    await ws.send(sub)
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            await self._handle(ws, raw)
                        except Exception:  # noqa: BLE001
                            if (n := errlim.tick(loop.time())) is not None:
                                _log.exception(
                                    "katana-pubws _handle failed (%d in interval); raw=%r",
                                    n, raw,
                                )
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001
                lasted = (loop.time() - t0) if t0 is not None else 0.0
                backoff = 1.0 if lasted >= _STABLE_S else min(backoff * 2, 30.0)
                _log.warning(
                    "katana-pubws disconnected: %s — reconnecting in %.1fs", e, backoff,
                )
                await asyncio.sleep(backoff)

    def _reset_book(self) -> None:
        self._gen += 1
        self._bids.clear()
        self._asks.clear()
        self._synced = False
        self._buffer.clear()
        self._last_u = None
        if self._bootstrap_task and not self._bootstrap_task.done():
            self._bootstrap_task.cancel()
        self._bootstrap_task = None

    async def _handle(self, ws, raw: str | bytes) -> None:
        msg = orjson.loads(raw)
        if not isinstance(msg, dict):
            return
        t = msg.get("type")
        if t == "ping":
            await ws.send(orjson.dumps({"type": "pong"}).decode())
        elif t == "subscriptions":
            self._schedule_bootstrap()
        elif t == "error":
            _log.error("katana-pubws server error: %s", msg.get("data"))
        elif t == "l2orderbook":
            self._on_diff(msg.get("data") or {})

    def _schedule_bootstrap(self) -> None:
        """Start a single bootstrap task; no-op if one is already running."""
        if self._bootstrap_task is not None and not self._bootstrap_task.done():
            return
        self._bootstrap_task = asyncio.create_task(
            self._bootstrap(self._gen), name=f"katana-bootstrap-{self.market}"
        )

    async def _bootstrap(self, gen: int) -> None:
        """Fetch the REST snapshot and reconcile; retry on a timer while desynced.

        Tied to the connection generation `gen` so a bootstrap from a previous
        socket cannot clobber a newer connection's state.
        """
        url = f"{self.rest_base_url}/v1/orderbook?market={self.market}&level=2"
        while not self._stop.is_set() and gen == self._gen and not self._synced:
            try:
                timeout = aiohttp.ClientTimeout(total=15)
                async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as s, \
                        s.get(url) as r:
                    r.raise_for_status()
                    snap = await r.json(content_type=None)
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001
                _log.warning(
                    "katana-pubws snapshot fetch failed (%s: %r) — retrying in %.1fs",
                    type(e).__name__, e, _RESYNC_RETRY_S,
                )
                await asyncio.sleep(_RESYNC_RETRY_S)
                continue
            if gen != self._gen:
                return  # reconnected underneath us
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
            for d in sorted(self._buffer, key=lambda d: int(d.get("u", 0))):
                if int(d.get("u", 0)) > seq:
                    self._apply(d)
            self._buffer.clear()
            self._synced = True
            self._last_update_ms = now_ms()  # liveness: book is now valid
            self._emit_book()
            return

    def _on_diff(self, d: dict[str, Any]) -> None:
        u = int(d.get("u", 0))
        self.last_seq = u
        if not self._synced:
            self._buffer.append(d)  # bounded deque: oldest auto-dropped
            self._schedule_bootstrap()
            return
        assert self._last_u is not None
        if u <= self._last_u:
            return  # stale / duplicate
        if u > self._last_u + 1:
            # missed messages — count, mark desynced, resync from a fresh snapshot
            self.seq_gap_total += u - self._last_u - 1
            _log.warning(
                "katana-pubws sequence gap: %d -> %d (resync)", self._last_u, u,
            )
            self._synced = False
            self._buffer.append(d)
            self._schedule_bootstrap()
            return
        self._apply(d)
        self._emit_book()

    def _apply(self, d: dict[str, Any]) -> None:
        for p, sz, _c in d.get("b", []):
            self._set(self._bids, Decimal(str(p)), Decimal(str(sz)))
        for p, sz, _c in d.get("a", []):
            self._set(self._asks, Decimal(str(p)), Decimal(str(sz)))
        self._last_u = int(d.get("u", 0))
        self._last_update_ms = now_ms()  # liveness: in-sync update applied

    @staticmethod
    def _set(side: dict[Decimal, Decimal], price: Decimal, size: Decimal) -> None:
        if size == 0:
            side.pop(price, None)
        else:
            side[price] = size

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
        for qcb in self._quote_cbs:
            qcb(self._last_quote)
