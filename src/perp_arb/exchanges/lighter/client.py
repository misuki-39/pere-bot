"""LighterClient — implements BaseExchange against Lighter.

Order placement is WS-based: SignerClient signs txs locally, LighterUserWs
ships them over `jsonapi/sendtx`. The REST `tx_api.send_tx` path is never
hit. User-data and tx-submit share one WS connection per account; orderbook
data uses a separate per-market WS.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import lighter

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
    OrderResult,
    OrderSnapshot,
    OrderStatus,
    Position,
    Quote,
    Side,
    Symbol,
)
from .ws import LighterPublicWs, LighterUserWs, TxSubmitError

_log = logging.getLogger(__name__)


@dataclass
class _MarketMeta:
    market_index: int
    base_multiplier: int   # 10**supported_size_decimals
    price_multiplier: int  # 10**supported_price_decimals
    symbol: Symbol         # cached so WS handlers reuse it instead of rebuilding


class LighterClient(BaseExchange):
    name = "lighter"

    def __init__(
        self,
        *,
        base_url: str,
        api_key_private_key: str | None = None,
        account_index: int = 0,
        api_key_index: int = 0,
        public_only: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_private_key = api_key_private_key
        self.account_index = account_index
        self.api_key_index = api_key_index
        self.public_only = public_only or not api_key_private_key

        self._api_client: lighter.ApiClient | None = None
        self._signer: lighter.SignerClient | None = None
        self._user_ws: LighterUserWs | None = None

        self._meta_by_symbol: dict[str, _MarketMeta] = {}
        self._symbol_by_market_index: dict[int, str] = {}
        self._public_ws_by_symbol: dict[str, LighterPublicWs] = {}

        self._fill_cbs: defaultdict[str, list[OrderUpdateCallback]] = defaultdict(list)
        self._position_cbs: defaultdict[str, list[PositionCallback]] = defaultdict(list)
        self._live_positions: dict[str, Position] = {}

    # ---- lifecycle ----

    async def connect(self) -> None:
        self._api_client = lighter.ApiClient(
            configuration=lighter.Configuration(host=self.base_url),
        )
        if self.public_only:
            return
        self._signer = lighter.SignerClient(
            url=self.base_url,
            account_index=self.account_index,
            api_private_keys={self.api_key_index: self.api_key_private_key},
        )
        err = self._signer.check_client()
        if err is not None:
            raise RuntimeError(f"lighter signer check_client failed: {err}")
        self._user_ws = LighterUserWs(
            base_url=self.base_url,
            account_index=self.account_index,
            auth_token_factory=self._make_auth_token,
            subscribe_account_all=False,   # account_market is the superset
        )
        self._user_ws.add_market_callback(self._on_market_event)
        await self._user_ws.start()

    def _make_auth_token(self) -> str:
        assert self._signer is not None
        token, err = self._signer.create_auth_token_with_expiry(api_key_index=self.api_key_index)
        if err:
            raise RuntimeError(f"lighter create_auth_token failed: {err}")
        return token

    async def disconnect(self) -> None:
        for ws in self._public_ws_by_symbol.values():
            await ws.stop()
        if self._user_ws is not None:
            await self._user_ws.stop()
            self._user_ws = None
        if self._signer is not None:
            with contextlib.suppress(Exception):
                await self._signer.close()
            self._signer = None
        if self._api_client is not None:
            with contextlib.suppress(Exception):
                await self._api_client.close()
            self._api_client = None

    # ---- markets / data ----

    async def load_market(self, raw_symbol: str) -> MarketInfo:
        api = await self._ensure_api()
        order_api = lighter.OrderApi(api)
        books = await order_api.order_books()
        for m in books.order_books:
            if m.symbol != raw_symbol:
                continue
            details_resp = await order_api.order_book_details(market_id=m.market_id)
            d = details_resp.order_book_details[0]
            tick = Decimal(1) / (Decimal(10) ** int(d.price_decimals))
            lot = Decimal(1) / (Decimal(10) ** int(m.supported_size_decimals))
            symbol = Symbol(exchange="lighter", raw=raw_symbol, base=raw_symbol, quote="USD")
            meta = _MarketMeta(
                market_index=int(m.market_id),
                base_multiplier=10 ** int(m.supported_size_decimals),
                price_multiplier=10 ** int(m.supported_price_decimals),
                symbol=symbol,
            )
            self._meta_by_symbol[raw_symbol] = meta
            self._symbol_by_market_index[meta.market_index] = raw_symbol
            if self._user_ws is not None:
                await self._user_ws.subscribe_account_market(meta.market_index)
            return MarketInfo(
                symbol=symbol,
                tick_size=tick,
                lot_size=lot,
                contract_id=meta.market_index,
                min_qty=Decimal("0"),
            )
        raise RuntimeError(f"lighter symbol {raw_symbol!r} not found")

    def _ensure_public_ws(self, market: MarketInfo) -> LighterPublicWs:
        key = market.symbol.raw
        ws = self._public_ws_by_symbol.get(key)
        if ws is None:
            ws = LighterPublicWs(
                base_url=self.base_url,
                symbol=market.symbol,
                market_index=int(market.contract_id),
            )
            self._public_ws_by_symbol[key] = ws
            asyncio.create_task(ws.start(), name=f"lighter-pubws-{key}-start")
        return ws

    def subscribe_quotes(self, market: MarketInfo, cb: QuoteCallback) -> None:
        self._ensure_public_ws(market).add_quote_callback(cb)

    def subscribe_book(self, market: MarketInfo, cb: OrderBookCallback) -> None:
        self._ensure_public_ws(market).add_book_callback(cb)

    def subscribe_fills(self, market: MarketInfo, cb: OrderUpdateCallback) -> None:
        self._fill_cbs[market.symbol.raw].append(cb)

    def subscribe_positions(self, market: MarketInfo, cb: PositionCallback) -> None:
        self._position_cbs[market.symbol.raw].append(cb)

    def live_position(self, market: MarketInfo) -> Position | None:
        return self._live_positions.get(market.symbol.raw)

    def best_quote(self, market: MarketInfo) -> Quote | None:
        ws = self._public_ws_by_symbol.get(market.symbol.raw)
        return ws.last_quote if ws else None

    def order_book(self, market: MarketInfo) -> OrderBook | None:
        ws = self._public_ws_by_symbol.get(market.symbol.raw)
        return ws.last_book if ws else None

    def book_ts(self, market: MarketInfo) -> int | None:
        ws = self._public_ws_by_symbol.get(market.symbol.raw)
        return ws.last_update_ms if ws and ws.last_update_ms else None

    # ---- orders ----

    async def place_market_order(
        self,
        market: MarketInfo,
        side: Side,
        qty: Decimal,
        *,
        reduce_only: bool = False,
        client_id: str | None = None,
    ) -> OrderResult:
        if self.public_only or self._signer is None or self._user_ws is None:
            raise RuntimeError("lighter is in public_only mode — cannot place orders")
        meta = self._meta_by_symbol.get(market.symbol.raw)
        if meta is None:
            raise RuntimeError("call load_market() before placing orders on lighter")

        is_ask = side is Side.SELL
        worst_price = _worst_acceptable_price(self.best_quote(market), is_ask)
        avg_price_int = int(worst_price * meta.price_multiplier)
        base_amount = int(qty * meta.base_multiplier)
        coi = int(time.time_ns() // 1_000) % 1_000_000_000
        client_id_str = client_id or f"lighter-{uuid.uuid4().hex[:12]}"

        tx_type, tx_info, _tx_hash, err = self._signer.sign_create_order(
            market_index=meta.market_index,
            client_order_index=coi,
            base_amount=base_amount,
            price=avg_price_int,
            is_ask=int(is_ask),
            order_type=self._signer.ORDER_TYPE_MARKET,
            time_in_force=self._signer.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
            reduce_only=int(reduce_only),
            order_expiry=self._signer.DEFAULT_IOC_EXPIRY,
            api_key_index=self.api_key_index,
        )
        if err is not None:
            return OrderResult(
                success=False, client_id=client_id_str, side=side,
                requested_qty=qty, error_message=f"sign: {err}",
            )

        t0 = time.monotonic()
        try:
            reply = await self._user_ws.send_tx(tx_type, tx_info)
        except (TimeoutError, TxSubmitError) as e:
            return OrderResult(
                success=False, client_id=client_id_str, side=side,
                requested_qty=qty, error_message=str(e),
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
        ok, err_msg = _sendtx_outcome(reply)
        if not ok:
            return OrderResult(
                success=False, client_id=client_id_str, side=side,
                requested_qty=qty, error_message=err_msg,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
        return OrderResult(
            success=True,
            order_id=str(coi),
            client_id=client_id_str,
            side=side,
            requested_qty=qty,
            status=OrderStatus.OPEN,    # final fill state arrives via account WS
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

    async def cancel_order(self, market: MarketInfo, order_id: str) -> OrderResult:
        if self.public_only or self._signer is None or self._user_ws is None:
            raise RuntimeError("lighter is in public_only mode — cannot cancel")
        meta = self._meta_by_symbol.get(market.symbol.raw)
        if meta is None:
            raise RuntimeError("call load_market() before cancelling on lighter")
        tx_type, tx_info, _tx_hash, err = self._signer.sign_cancel_order(
            market_index=meta.market_index,
            order_index=int(order_id),
            api_key_index=self.api_key_index,
        )
        if err is not None:
            return OrderResult(success=False, order_id=order_id, error_message=f"sign: {err}")
        try:
            reply = await self._user_ws.send_tx(tx_type, tx_info)
        except (TimeoutError, TxSubmitError) as e:
            return OrderResult(success=False, order_id=order_id, error_message=str(e))
        ok, err_msg = _sendtx_outcome(reply)
        if not ok:
            return OrderResult(success=False, order_id=order_id, error_message=err_msg)
        return OrderResult(success=True, order_id=order_id, status=OrderStatus.CANCELED)

    async def get_order(self, market: MarketInfo, order_id: str) -> OrderSnapshot | None:
        if self._signer is None:
            raise RuntimeError("lighter is in public_only mode — cannot get_order")
        meta = self._meta_by_symbol[market.symbol.raw]
        api = await self._ensure_api()
        order_api = lighter.OrderApi(api)
        resp = await order_api.account_active_orders(
            account_index=self.account_index,
            market_id=meta.market_index,
            auth=self._make_auth_token(),
        )
        for o in resp.orders:
            if str(o.order_index) == str(order_id):
                return OrderSnapshot(
                    order_id=str(o.order_index),
                    client_id=None,
                    symbol=market.symbol,
                    side=Side.SELL if o.is_ask else Side.BUY,
                    size=Decimal(str(o.initial_base_amount)),
                    price=Decimal(str(o.price)),
                    status=_status_from_str(o.status),
                    filled_qty=Decimal(str(o.filled_base_amount)),
                    realized_price=None,
                )
        return None

    async def get_position(self, market: MarketInfo) -> Position:
        api = await self._ensure_api()
        account_api = lighter.AccountApi(api)
        data = await account_api.account(by="index", value=str(self.account_index))
        meta = self._meta_by_symbol[market.symbol.raw]
        pos = Position(symbol=market.symbol, size=Decimal("0"))
        if data and data.accounts:
            # Normalize SDK pydantic positions → plain dict so the
            # downstream helpers see the same shape as WS payloads
            # (field names are identical: market_id / sign / position /
            #  avg_entry_price / unrealized_pnl).
            for p in (sdk_p.to_dict() for sdk_p in data.accounts[0].positions):
                if int(p["market_id"]) == meta.market_index:
                    pos = Position(
                        symbol=market.symbol,
                        size=_signed_size(p),
                        entry_price=Decimal(str(p.get("avg_entry_price") or 0)),
                        unrealised_pnl=Decimal(str(p.get("unrealized_pnl") or 0)),
                    )
                    break
        # Pattern A: WS owns the cache; REST only seeds. See docs/position_cache.md.
        self._live_positions.setdefault(market.symbol.raw, pos)
        return pos

    # ---- internals ----

    async def _ensure_api(self) -> lighter.ApiClient:
        if self._api_client is None:
            self._api_client = lighter.ApiClient(
                configuration=lighter.Configuration(host=self.base_url),
            )
        return self._api_client

    def _on_market_event(self, payload: dict[str, Any]) -> None:
        """`subscribed/account_market` (snapshot) / `update/account_market`
        (delta). Per-market channel bundling positions + orders + trades +
        funding. Each push has ONE non-null collection — we route the
        single `position` dict to position_cbs and each entry of `orders`
        to fill_cbs. `trades` is ignored: orders already carry cumulative
        `filled_base_amount` / `filled_quote_amount` + matching-engine
        `transaction_time`, making the per-trade stream redundant.
        """
        pos = payload.get("position")
        if isinstance(pos, dict):
            self._handle_position_update(pos)
        for o in payload.get("orders") or []:
            self._handle_order_update(o)

    def _handle_position_update(self, p: dict[str, Any]) -> None:
        raw_symbol = self._symbol_by_market_index.get(int(p["market_id"]))
        if raw_symbol is None:
            return
        meta = self._meta_by_symbol[raw_symbol]
        pos = Position(
            symbol=meta.symbol,
            size=_signed_size(p),
            entry_price=Decimal(p["avg_entry_price"]),
            unrealised_pnl=Decimal(p["unrealized_pnl"]),
        )
        self._live_positions[raw_symbol] = pos
        for cb in self._position_cbs.get(raw_symbol, ()):
            cb(pos)

    def _handle_order_update(self, o: dict[str, Any]) -> None:
        raw_symbol = self._symbol_by_market_index.get(int(o["market_index"]))
        if raw_symbol is None:
            return
        cbs = self._fill_cbs.get(raw_symbol)
        if not cbs:
            return
        snap = _order_to_snapshot(o, self._meta_by_symbol[raw_symbol].symbol)
        for cb in cbs:
            cb(snap)


def _worst_acceptable_price(q: Quote | None, is_ask: bool) -> Decimal:
    """±5% of mid as the slippage cap; extreme bounds when no quote is cached."""
    if q is not None:
        return q.mid * (Decimal("0.95") if is_ask else Decimal("1.05"))
    return Decimal("0.01") if is_ask else Decimal("1000000000")


def _signed_size(p: dict[str, Any]) -> Decimal:
    """Lighter encodes position as unsigned `position` + separate `sign` (+1/-1)."""
    size = Decimal(p["position"])
    return size if int(p["sign"]) >= 0 else -size


def _sendtx_outcome(reply: dict[str, Any]) -> tuple[bool, str]:
    """Parse a `jsonapi/sendtx` reply into (ok, error_message).

    Schema isn't formally documented. We treat `error` (anywhere) or a non-200
    `code` (top-level or inside `data`) as failure; everything else as success.
    """
    data = reply.get("data") or {}
    err = reply.get("error") or data.get("error")
    if err:
        return False, str(err)
    for d in (reply, data):
        code = d.get("code")
        if code is not None and int(code) != 200:
            return False, str(d.get("message") or d.get("msg") or f"sendtx code={code}")
    return True, ""


# Lighter has two distinct status vocabularies. We keep them as two
# functions because they really are different — collapsing into one map
# would obscure the venue's actual asymmetry. Compare with aster, which
# uses a single `_ASTER_STATUS_MAP` because its REST and WS share one
# vocabulary.
#
#   REST  `account_active_orders.status` →  UPPERCASE tokens, no variants.
#     Used by: `_status_from_str` (called from `get_order`).
#   WS    `account_market.orders.status` →  lowercase, with `canceled-*`
#     variants that fold into CANCELED / EXPIRED. Used by:
#     `_account_orders_status` (called from `_order_to_snapshot`).
_LIGHTER_STATUS_MAP = {
    "OPEN": OrderStatus.OPEN,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELED,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "REJECTED": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.EXPIRED,
}


def _status_from_str(s: str) -> OrderStatus:
    return _LIGHTER_STATUS_MAP.get(s.upper(), OrderStatus.UNKNOWN)


def _account_orders_status(s: str) -> OrderStatus:
    """`canceled-*` variants all collapse to CANCELED except
    `canceled-expired`, which is semantically EXPIRED."""
    s = (s or "").lower()
    if s == "pending":
        return OrderStatus.PENDING
    if s == "open":
        return OrderStatus.OPEN
    if s == "filled":
        return OrderStatus.FILLED
    if s == "canceled-expired":
        return OrderStatus.EXPIRED
    if s.startswith("canceled"):
        return OrderStatus.CANCELED
    return OrderStatus.UNKNOWN


def _order_to_snapshot(o: dict[str, Any], symbol: Symbol) -> OrderSnapshot:
    """Parse one entry of an `account_market.orders` array. `filled_base`
    and `filled_quote` are cumulative; avg = quote / base when base > 0.
    `transaction_time` is microseconds (lighter convention) → /1000 for ms."""
    filled_base = Decimal(o["filled_base_amount"])
    filled_quote = Decimal(o["filled_quote_amount"])
    avg_price = (filled_quote / filled_base) if filled_base > 0 else None
    return OrderSnapshot(
        order_id=str(o.get("order_id") or o["order_index"]),
        client_id=str(o["client_order_id"]),
        symbol=symbol,
        side=Side.SELL if o["is_ask"] else Side.BUY,
        size=Decimal(o["initial_base_amount"]),
        price=Decimal(o["price"]),
        status=_account_orders_status(o["status"]),
        filled_qty=filled_base,
        realized_price=avg_price,
        ts_ms=int(o["transaction_time"]) // 1000,
    )
