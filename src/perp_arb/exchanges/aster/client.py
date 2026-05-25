"""AsterClient — implements BaseExchange against Aster V3."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from decimal import Decimal

from ...core.exchange import (
    BaseExchange,
    OrderBookCallback,
    OrderUpdateCallback,
    PositionCallback,
    QuoteCallback,
)
from ...core.types import (
    FillDelta,
    LegOutcome,
    MarketInfo,
    OrderBook,
    OrderStatus,
    Position,
    Quote,
    Side,
)
from .rest import AsterRest
from .signer import AsterSigner
from .ws import AsterPublicWs, AsterUserWs

_log = logging.getLogger(__name__)


_ASTER_STATUS_MAP: dict[str, OrderStatus] = {
    "NEW": OrderStatus.OPEN,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELED,
    "REJECTED": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.EXPIRED,
}


class AsterClient(BaseExchange):
    name = "aster"

    def __init__(
        self,
        *,
        rest_url: str,
        ws_url: str,
        signer: AsterSigner | None = None,
        public_only: bool = False,
    ) -> None:
        super().__init__()
        self.rest = AsterRest(base_url=rest_url, signer=signer)
        self.ws_url = ws_url
        self.signer = signer
        self.public_only = public_only or signer is None

        self._public_ws_by_symbol: dict[str, AsterPublicWs] = {}
        self._user_ws: AsterUserWs | None = None

        self._fill_cbs: defaultdict[str, list[OrderUpdateCallback]] = defaultdict(list)
        self._position_cbs: defaultdict[str, list[PositionCallback]] = defaultdict(list)
        # raw symbol -> MarketInfo, populated by load_market; used to rehydrate
        # Symbol on incoming user-data events.
        self._markets: dict[str, MarketInfo] = {}
        # raw symbol -> latest Position, fed by both get_position (REST) and
        # ACCOUNT_UPDATE (WS) so live_position() reflects whichever source ran last.
        self._live_positions: dict[str, Position] = {}

        self._user_event_handlers = {
            "ORDER_TRADE_UPDATE": self._handle_order_trade_update,
            "ACCOUNT_UPDATE": self._handle_account_update,
        }

    # ----- lifecycle -----

    async def connect(self) -> None:
        await self.rest.__aenter__()
        if not self.public_only:
            self._user_ws = AsterUserWs(ws_url=self.ws_url, rest=self.rest)
            self._user_ws.add_callback(self._on_user_event)
            await self._user_ws.start()

    async def disconnect(self) -> None:
        for ws in self._public_ws_by_symbol.values():
            await ws.stop()
        if self._user_ws is not None:
            await self._user_ws.stop()
            self._user_ws = None
        await self.rest.close()

    # ----- markets / data -----

    async def load_market(self, raw_symbol: str) -> MarketInfo:
        m = await self.rest.load_market(raw_symbol)
        self._markets[raw_symbol] = m
        return m

    def _ensure_public_ws(self, market: MarketInfo) -> AsterPublicWs:
        key = market.symbol.raw
        ws = self._public_ws_by_symbol.get(key)
        if ws is None:
            ws = AsterPublicWs(ws_url=self.ws_url, symbol=market.symbol)
            self._public_ws_by_symbol[key] = ws
            asyncio.create_task(ws.start(), name=f"aster-pubws-{key}-start")
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

    # ----- orders -----

    async def place_market_order(
        self,
        market: MarketInfo,
        side: Side,
        qty: Decimal,
        *,
        client_id: str,
        reduce_only: bool = False,
    ) -> LegOutcome:
        if self.public_only:
            raise RuntimeError("aster is in public_only mode — cannot place orders")
        try:
            resp = await self.rest.place_order(
                symbol=str(market.contract_id),
                side="BUY" if side is Side.BUY else "SELL",
                order_type="MARKET",
                quantity=qty,
                reduce_only=reduce_only,
                new_client_order_id=client_id,
            )
        except Exception as e:  # noqa: BLE001
            _log.warning("aster order failed: %s", e)
            return LegOutcome(
                success=False,
                client_id=client_id,
                side=side,
                requested_qty=qty,
                error_message=str(e),
            )
        # `transactTime` (Binance-fork convention) is the exchange-server
        # millisecond timestamp of the order action. Aster's REST also
        # returns `executedQty`/`avgPrice` for market orders (fills are
        # synchronous), which we store as a fallback in case the WS
        # ORDER_TRADE_UPDATE never lands; `submit_and_await` overlays WS
        # data on top when it does.
        #
        # Only populate the fill-side fields when BOTH executedQty>0 AND
        # avgPrice>0 are present — otherwise _weighted_price_sum=qty*0=0
        # would fabricate a `realized_price=0` (via the avg_price property)
        # downstream, blowing up PnL math with a $0 fill price. `is not None`
        # over `or`-fallback because Decimal('0') is falsy in Python.
        rest_filled = _dec(resp.get("executedQty"))
        rest_avg = _dec(resp.get("avgPrice"))
        outcome = LegOutcome(
            success=True,
            client_id=client_id,
            side=side,
            requested_qty=qty,
            status=_ASTER_STATUS_MAP.get(resp["status"], OrderStatus.UNKNOWN),
            exchange_ts_ms=int(resp["transactTime"]) if resp.get("transactTime") else None,
        )
        if (
            rest_filled is not None and rest_avg is not None
            and rest_filled > 0 and rest_avg > 0
        ):
            outcome.set_fill(rest_filled, rest_avg)
        return outcome

    async def get_position(self, market: MarketInfo) -> Position:
        # Assumes the account is in one-way (non-hedge) position mode:
        # `positionAmt` is already signed (long > 0, short < 0). Hedge
        # mode is not supported — it would split into LONG/SHORT rows
        # with non-negative `positionAmt` + a `positionSide` field, and
        # the seed path would silently take the wrong sign.
        rows = await self.rest.get_position_risk(str(market.contract_id))
        pos = Position(symbol=market.symbol, size=Decimal("0"))
        for row in rows:
            if row["symbol"] == market.symbol.raw:
                pos = Position(
                    symbol=market.symbol,
                    size=Decimal(row["positionAmt"]),
                    entry_price=Decimal(row["entryPrice"]),
                    unrealised_pnl=Decimal(row["unRealizedProfit"]),
                )
                break
        # Pattern A: WS owns the cache. REST only seeds when no ACCOUNT_UPDATE
        # has fired yet; if WS won the race, setdefault is a no-op. See
        # docs/position_cache.md.
        self._live_positions.setdefault(market.symbol.raw, pos)
        return pos

    # ----- user-data event fan-out -----

    def _on_user_event(self, data: dict) -> None:
        event_type = data.get("e")
        if not isinstance(event_type, str):
            return
        handler = self._user_event_handlers.get(event_type)
        if handler is not None:
            handler(data)

    def _handle_order_trade_update(self, data: dict) -> None:
        o = data["o"]
        raw_symbol = o["s"]
        market = self._markets.get(raw_symbol)
        if market is None:
            _log.warning("aster ORDER_TRADE_UPDATE for unloaded symbol %s", raw_symbol)
            return
        # `l` / `L` = THIS event's per-fill delta (Binance-fork convention).
        # Non-fill events (NEW / pure-CANCELED) carry l=0 and are dropped at
        # the source — the accumulator only sees real fills, so its
        # `+=` semantics need no defensive size>0 guard.
        last_qty = Decimal(o["l"]) if o.get("l") else Decimal("0")
        if last_qty <= 0:
            return
        status = _ASTER_STATUS_MAP.get(o["X"], OrderStatus.UNKNOWN)
        # `n` is the per-fill commission (Binance-fork ORDER_TRADE_UPDATE);
        # `N` is the asset (USDT on aster perp markets, treated as ≈ USD for
        # cash-flow PnL math). Absent on non-fill events but we've already
        # short-circuited those above.
        fee = Decimal(o["n"]) if o.get("n") else Decimal("0")
        delta = FillDelta(
            qty=last_qty,
            price=Decimal(o["L"]),
            # `T` is the trade time on fills; `E` is event time (fallback).
            ts_ms=int(data.get("T") or data["E"]),
            side=Side.BUY if o["S"] == "BUY" else Side.SELL,
            client_id=o.get("c"),
            terminal_status=status if status.terminal else None,
            fee=fee,
        )
        # Internal: feed the per-cid tracker (submit_and_await's awaitable).
        # Must precede the per-symbol fan-out: once the strategy stops
        # subscribing, `_fill_cbs[raw_symbol]` is empty but the tracker
        # still needs the event.
        self._fill_tracker.on_event(delta)
        for cb in self._fill_cbs.get(raw_symbol, ()):
            cb(delta)

    def _handle_account_update(self, data: dict) -> None:
        # Cache + fan-out one Position per known symbol in `a.P[]`. Schema:
        # futures-api-v3.md:4140-4202.
        for p in data["a"]["P"]:
            raw_symbol = p["s"]
            market = self._markets.get(raw_symbol)
            if market is None:
                continue
            pos = Position(
                symbol=market.symbol,
                size=Decimal(p["pa"]),
                entry_price=Decimal(p["ep"]),
                unrealised_pnl=Decimal(p["up"]),
            )
            self._live_positions[raw_symbol] = pos
            for cb in self._position_cbs.get(raw_symbol, ()):
                cb(pos)


def _dec(v: object) -> Decimal | None:
    if v is None or v == "":
        return None
    return Decimal(str(v))
