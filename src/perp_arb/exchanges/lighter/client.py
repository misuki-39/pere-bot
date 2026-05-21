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
    OrderInfo,
    OrderResult,
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
        )
        self._user_ws.add_account_callback(self._on_account_event)
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
                requested_size=qty, error_message=f"sign: {err}",
            )

        t0 = time.monotonic()
        try:
            reply = await self._user_ws.send_tx(tx_type, tx_info)
        except (TimeoutError, TxSubmitError) as e:
            return OrderResult(
                success=False, client_id=client_id_str, side=side,
                requested_size=qty, error_message=str(e),
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
        ok, err_msg = _sendtx_outcome(reply)
        if not ok:
            return OrderResult(
                success=False, client_id=client_id_str, side=side,
                requested_size=qty, error_message=err_msg,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
        return OrderResult(
            success=True,
            order_id=str(coi),
            client_id=client_id_str,
            side=side,
            requested_size=qty,
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

    async def get_order(self, market: MarketInfo, order_id: str) -> OrderInfo | None:
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
                return OrderInfo(
                    order_id=str(o.order_index),
                    client_id=None,
                    symbol=market.symbol,
                    side=Side.SELL if o.is_ask else Side.BUY,
                    size=Decimal(str(o.initial_base_amount)),
                    price=Decimal(str(o.price)),
                    status=_status_from_str(o.status),
                    filled_size=Decimal(str(o.filled_base_amount)),
                    avg_fill_price=None,
                )
        return None

    async def get_position(self, market: MarketInfo) -> Position:
        api = await self._ensure_api()
        account_api = lighter.AccountApi(api)
        data = await account_api.account(by="index", value=str(self.account_index))
        meta = self._meta_by_symbol[market.symbol.raw]
        pos = Position(symbol=market.symbol, size=Decimal("0"))
        if data and data.accounts:
            for p in data.accounts[0].positions:
                if int(p.market_id) == meta.market_index:
                    pos = Position(
                        symbol=market.symbol,
                        size=_signed_size(p),
                        entry_price=Decimal(str(getattr(p, "avg_entry_price", 0) or 0)),
                        unrealised_pnl=Decimal(str(getattr(p, "unrealized_pnl", 0) or 0)),
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

    def _on_account_event(self, payload: dict[str, Any]) -> None:
        """`subscribed/account_all` (snapshot) and `update/account_all` (delta).

        Both push `positions` and `trades` as dicts keyed by id-string.
        `positions` give the authoritative net-position state for the
        account; `trades` are the per-fill events that drive `_fill_cbs`
        subscribers (see Lighter SDK `models/trade.py` for the schema).
        Lighter does not push `orders` on this channel — fills must be
        reconstructed from the trades stream.
        """
        for p in _iter_dict_values(payload.get("positions")):
            self._handle_position_update(p)
        for t in _iter_dict_values(payload.get("trades")):
            self._handle_trade(t)

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

    def _handle_trade(self, t: dict[str, Any]) -> None:
        """One filled trade from the account_all stream. Disambiguates which
        side of the match is ours via `account_id` and emits an `OrderInfo`
        to all registered `subscribe_fills` callbacks.

        Trade schema source: lighter-sdk `models/trade.py`. The two fields
        we depend on are:
          - `ask_account_id` / `bid_account_id`: integer account indices,
            one of which == `self.account_index` for any trade pushed to us
          - `ask_client_id` / `bid_client_id`: the `client_order_index` we
            passed in at submit time, surfaced for the matching side
          - `timestamp`: server-side fill time (ms since epoch)
        """
        raw_symbol = self._symbol_by_market_index.get(int(t["market_id"]))
        if raw_symbol is None:
            return
        cbs = self._fill_cbs.get(raw_symbol)
        if not cbs:
            return
        meta = self._meta_by_symbol[raw_symbol]
        ask_acc = int(t["ask_account_id"])
        bid_acc = int(t["bid_account_id"])
        if ask_acc == self.account_index:
            our_side, our_client_id = Side.SELL, str(t["ask_client_id"])
        elif bid_acc == self.account_index:
            our_side, our_client_id = Side.BUY, str(t["bid_client_id"])
        else:
            # Should not happen — account_all is scoped to our account_id —
            # but guard rather than mis-attribute.
            _log.warning("lighter trade with no matching account_id: %s", t)
            return
        size = Decimal(t["size"])
        info = OrderInfo(
            order_id=str(t["trade_id"]),
            client_id=our_client_id,
            symbol=meta.symbol,
            side=our_side,
            size=size,
            price=Decimal(t["price"]),
            status=OrderStatus.FILLED,
            filled_size=size,
            avg_fill_price=Decimal(t["price"]),
            ts_ms=int(t["timestamp"]),
        )
        for cb in cbs:
            cb(info)


def _iter_dict_values(coll: Any) -> list[dict[str, Any]]:
    """Lighter's account_all uses dicts keyed by id-string for collections."""
    if not isinstance(coll, dict):
        return []
    return [v for v in coll.values() if isinstance(v, dict)]


def _worst_acceptable_price(q: Quote | None, is_ask: bool) -> Decimal:
    """±5% of mid as the slippage cap; extreme bounds when no quote is cached."""
    if q is not None:
        return q.mid * (Decimal("0.95") if is_ask else Decimal("1.05"))
    return Decimal("0.01") if is_ask else Decimal("1000000000")


def _signed_size(p: Any) -> Decimal:
    """Lighter encodes position as unsigned `position` + separate `sign` (+1/-1)."""
    if isinstance(p, dict):
        size = Decimal(p["position"])
        sign = int(p["sign"])
    else:
        size = Decimal(str(p.position))
        sign = int(p.sign)
    return size if sign >= 0 else -size


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
