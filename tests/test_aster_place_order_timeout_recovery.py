"""Tests for `AsterClient.place_market_order`'s 400/timeout disambiguation.

Aster occasionally returns `400 ... "msg":"The request has timed out."`
for a request that DID reach the matching engine — the response just
didn't make it back. Treating that as a definitive non-execution causes
spurious unwinds on the hedge leg (the bug we hit on 2026-05-25). The
client recovers by looking up the order via `GET /fapi/v3/order` and
re-reading its state. These tests pin down that recovery contract.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from perp_arb.core.types import MarketInfo, Side, Symbol
from perp_arb.exchanges.aster.client import AsterClient
from perp_arb.exchanges.aster.rest import AsterRestError

_SYM = Symbol(exchange="aster", raw="CLUSDT", base="WTI", quote="USDT")
_MARKET = MarketInfo(
    symbol=_SYM, tick_size=Decimal("0.01"), lot_size=Decimal("1"),
    contract_id="CLUSDT",
)


class _StubRest:
    """Stand-in for `AsterRest`. Each behaviour is configured at construction
    so individual tests stay focused — `place_raises` controls whether
    place_order succeeds or raises, `query_raises` likewise for query_order."""

    def __init__(
        self,
        *,
        place_raises: Exception | None = None,
        place_resp: dict | None = None,
        query_raises: Exception | None = None,
        query_resp: dict | None = None,
    ) -> None:
        self.place_raises = place_raises
        self.place_resp = place_resp
        self.query_raises = query_raises
        self.query_resp = query_resp
        self.query_calls: list[dict] = []

    async def place_order(self, **kw):
        if self.place_raises is not None:
            raise self.place_raises
        return self.place_resp

    async def query_order(self, symbol: str, *, orig_client_order_id: str):
        self.query_calls.append(
            {"symbol": symbol, "orig_client_order_id": orig_client_order_id},
        )
        if self.query_raises is not None:
            raise self.query_raises
        return self.query_resp


def _make_client(rest: _StubRest) -> AsterClient:
    c = AsterClient.__new__(AsterClient)
    c.rest = rest
    c.public_only = False
    return c


@pytest.mark.asyncio
async def test_place_timeout_recovered_when_query_finds_filled_order() -> None:
    """The defining case: place_order 400/timeout, queryOrder sees the
    order FILLED — outcome must be success with real fill data."""
    rest = _StubRest(
        place_raises=AsterRestError(
            "POST /fapi/v3/order -> 400: "
            "{\"timestamp\":1,\"path\":\"/fapi/v3/order\","
            "\"msg\":\"The request has timed out.\"}",
        ),
        query_resp={
            "clientOrderId": "cid-1",
            "status": "FILLED",
            "executedQty": "0.12",
            "avgPrice": "91.50",
            "updateTime": 1_700_000_000_000,
        },
    )
    c = _make_client(rest)
    out = await c.place_market_order(
        _MARKET, Side.BUY, Decimal("0.12"), client_id="cid-1",
    )
    assert out.success is True
    assert out.filled_qty == Decimal("0.12")
    assert out.avg_price == Decimal("91.50")
    assert out.exchange_ts_ms == 1_700_000_000_000
    assert rest.query_calls == [
        {"symbol": "CLUSDT", "orig_client_order_id": "cid-1"},
    ]


@pytest.mark.asyncio
async def test_place_timeout_falls_through_to_fail_when_query_raises() -> None:
    """Conservative fallback: if queryOrder itself errors (order not found,
    or another timeout), we cannot confirm execution — report fail."""
    rest = _StubRest(
        place_raises=AsterRestError(
            "POST /fapi/v3/order -> 400: \"msg\":\"The request has timed out.\"",
        ),
        query_raises=AsterRestError(
            "GET /fapi/v3/order -> 400: \"Order does not exist.\"",
        ),
    )
    c = _make_client(rest)
    out = await c.place_market_order(
        _MARKET, Side.BUY, Decimal("0.12"), client_id="cid-1",
    )
    assert out.success is False
    assert out.filled_qty == Decimal("0")
    assert "timed out" in (out.error_message or "")
    # queryOrder was still attempted before falling back.
    assert len(rest.query_calls) == 1


@pytest.mark.asyncio
async def test_non_timeout_error_skips_query_and_reports_fail() -> None:
    """Other 4xx/5xx errors (validation, auth, rate-limit) are unambiguous
    non-executions — no need to recover, no queryOrder call."""
    rest = _StubRest(
        place_raises=AsterRestError(
            "POST /fapi/v3/order -> 400: \"msg\":\"Invalid quantity.\"",
        ),
    )
    c = _make_client(rest)
    out = await c.place_market_order(
        _MARKET, Side.BUY, Decimal("0.12"), client_id="cid-1",
    )
    assert out.success is False
    assert rest.query_calls == []


@pytest.mark.asyncio
async def test_place_timeout_query_finds_new_status_returned_as_success() -> None:
    """If queryOrder reports the order exists but isn't yet FILLED (e.g.
    NEW / PARTIALLY_FILLED), the order is in the engine — surface as
    success so `submit_and_await`'s WS leg can resolve the final state.
    fill-side fields stay unset because executedQty is 0 (skip price
    fabrication)."""
    rest = _StubRest(
        place_raises=AsterRestError(
            "POST /fapi/v3/order -> 400: \"msg\":\"The request has timed out.\"",
        ),
        query_resp={
            "clientOrderId": "cid-1",
            "status": "NEW",
            "executedQty": "0",
            "avgPrice": "0",
            "updateTime": 1_700_000_000_000,
        },
    )
    c = _make_client(rest)
    out = await c.place_market_order(
        _MARKET, Side.BUY, Decimal("0.12"), client_id="cid-1",
    )
    assert out.success is True
    assert out.filled_qty == Decimal("0")
