"""Aster V3 REST client. Signed endpoints use AsterSigner."""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any

import aiohttp

from ...core.types import MarketInfo, Symbol
from ...utils.retry import query_retry
from .signer import AsterSigner

_log = logging.getLogger(__name__)


class AsterRestError(Exception):
    pass


class AsterRest:
    def __init__(
        self,
        *,
        base_url: str,
        signer: AsterSigner | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.signer = signer
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> AsterRest:
        if self._session is None:
            # DIAG (temp): wrap the connector to log every new TCP open.
            # aiohttp only calls _wrap_create_connection on miss, so pooled
            # reuses are silent — a `NEW-TCP` line means a real handshake
            # happened on that REST call. Remove once the prewarm question
            # is settled.
            connector = aiohttp.TCPConnector()
            _orig_wrap = connector._wrap_create_connection
            async def _wrap_diag(*a: Any, **k: Any) -> Any:
                t0 = time.perf_counter()
                transport, proto = await _orig_wrap(*a, **k)
                dt_ms = (time.perf_counter() - t0) * 1000
                peer = transport.get_extra_info("peername")
                _log.info("aster NEW-TCP peer=%s handshake_ms=%.1f", peer, dt_ms)
                return transport, proto
            connector._wrap_create_connection = _wrap_diag  # type: ignore[method-assign]
            self._session = aiohttp.ClientSession(connector=connector, trust_env=True)
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    @property
    def _sess(self) -> aiohttp.ClientSession:
        assert self._session is not None, "AsterRest used before connect() / __aenter__"
        return self._session

    # ---------------- public (unsigned) ----------------

    @query_retry(reraise=True)
    async def get_exchange_info(self) -> dict[str, Any]:
        async with self._sess.get(f"{self.base_url}/fapi/v3/exchangeInfo") as r:
            r.raise_for_status()
            return await r.json()

    async def load_market(self, raw_symbol: str) -> MarketInfo:
        info = await self.get_exchange_info()
        for s in info["symbols"]:
            if s["symbol"] != raw_symbol:
                continue
            tick = Decimal("0")
            step = Decimal("0")
            min_qty = Decimal("0")
            for f in s["filters"]:
                if f["filterType"] == "PRICE_FILTER":
                    tick = Decimal(f["tickSize"])
                elif f["filterType"] == "LOT_SIZE":
                    step = Decimal(f["stepSize"])
                    min_qty = Decimal(f.get("minQty", "0"))
            return MarketInfo(
                symbol=Symbol(
                    exchange="aster",
                    raw=raw_symbol,
                    base=s["baseAsset"],
                    quote=s["quoteAsset"],
                ),
                tick_size=tick,
                lot_size=step,
                contract_id=raw_symbol,
                min_qty=min_qty,
            )
        raise AsterRestError(f"symbol {raw_symbol!r} not found in exchangeInfo")

    # ---------------- signed helpers ----------------

    def _require_signer(self) -> AsterSigner:
        if self.signer is None:
            raise AsterRestError("This call requires a signed request, but no signer is configured")
        return self.signer

    async def _signed(
        self,
        method: str,
        path: str,
        params: dict[str, Any],
    ) -> Any:
        """Single signed-request helper. Returns parsed JSON (dict or list)."""
        s = self._require_signer()
        req = s.sign(params)
        url = f"{self.base_url}{path}"
        if method in ("POST", "PUT"):
            kwargs: dict[str, Any] = {
                "data": req.form_body,
                "headers": {"Content-Type": "application/x-www-form-urlencoded"},
            }
        else:
            url = f"{url}?{req.query_string}"
            kwargs = {}
        async with self._sess.request(method, url, **kwargs) as r:
            if r.status >= 400:
                raise AsterRestError(f"{method} {path} -> {r.status}: {await r.text()}")
            return await r.json()

    # ---------------- order ops ----------------

    async def place_order(
        self,
        symbol: str,
        side: str,                       # "BUY" | "SELL"
        order_type: str,                 # "MARKET" | "LIMIT" | ...
        quantity: Decimal,
        *,
        price: Decimal | None = None,
        time_in_force: str | None = None,
        reduce_only: bool = False,
        new_client_order_id: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": str(quantity),
        }
        if price is not None:
            params["price"] = str(price)
        if time_in_force:
            params["timeInForce"] = time_in_force
        if reduce_only:
            params["reduceOnly"] = "true"
        if new_client_order_id:
            params["newClientOrderId"] = new_client_order_id
        return await self._signed("POST", "/fapi/v3/order", params)

    async def get_position_risk(self, symbol: str) -> list[dict[str, Any]]:
        # /fapi/v3/positionRisk returns a JSON array.
        return await self._signed("GET", "/fapi/v3/positionRisk", {"symbol": symbol})

    async def query_order(
        self,
        symbol: str,
        *,
        orig_client_order_id: str,
    ) -> dict[str, Any]:
        """Lookup an order by `newClientOrderId` (`origClientOrderId` in the
        query API). Used to disambiguate `POST /order` 400 timeouts: the
        server may reject the response while the order is already in the
        matching engine. Raises AsterRestError when the order is not
        found (caller treats that as a confirmed non-execution)."""
        return await self._signed(
            "GET", "/fapi/v3/order",
            {"symbol": symbol, "origClientOrderId": orig_client_order_id},
        )

    # ---------------- user data stream ----------------

    async def start_user_stream(self) -> str:
        r = await self._signed("POST", "/fapi/v3/listenKey", {})
        return r["listenKey"]

    async def keepalive_user_stream(self) -> None:
        await self._signed("PUT", "/fapi/v3/listenKey", {})

    async def close_user_stream(self) -> None:
        await self._signed("DELETE", "/fapi/v3/listenKey", {})
