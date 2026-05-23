"""Unit tests for `_PerCidFillTracker` — the driver-layer per-client_id
fill aggregator that succeeds the old strategy-level `_on_fill` +
`_await_fill` pair.

Tests pin three invariants:
  * Unknown `client_id`s are silently dropped (whitelist semantics).
  * `OrderSnapshot` overwrite vs `FillDelta` accumulate semantics are
    preserved (inherited from `LegOutcome.add`).
  * `await_terminal` short-circuits on terminal-status OR completed qty;
    otherwise returns whatever landed by the timeout.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from perp_arb.core.fill_tracker import _PerCidFillTracker
from perp_arb.core.types import FillDelta, OrderSnapshot, OrderStatus, Side, Symbol

_SYM = Symbol(exchange="aster", raw="ETHUSDT", base="ETH", quote="USDT")


def _delta(qty: str, price: str, *, cid: str = "cid-1", ts: int = 1000,
           terminal: OrderStatus | None = None) -> FillDelta:
    return FillDelta(
        qty=Decimal(qty), price=Decimal(price), ts_ms=ts,
        side=Side.BUY, client_id=cid, terminal_status=terminal,
    )


def _snap(filled: str, *, cid: str = "cid-1", status: OrderStatus,
          avg: str = "100", ts: int = 1000) -> OrderSnapshot:
    return OrderSnapshot(
        client_id=cid, symbol=_SYM, side=Side.BUY,
        size=Decimal("1.0"), price=Decimal("100"), status=status,
        filled_qty=Decimal(filled), realized_price=Decimal(avg), ts_ms=ts,
    )


@pytest.mark.asyncio
async def test_register_then_event_then_await_returns_immediately() -> None:
    """Happy path: cid registered, terminal event arrives, await returns."""
    tr = _PerCidFillTracker()
    tr.register("cid-1")
    tr.on_event(_delta("1.0", "100", terminal=OrderStatus.FILLED))
    fill = await tr.await_terminal("cid-1", Decimal("1.0"), timeout_s=1.0)
    assert fill is not None
    assert fill.filled_qty == Decimal("1.0")
    assert fill.last_status is OrderStatus.FILLED


@pytest.mark.asyncio
async def test_unknown_cid_event_is_dropped() -> None:
    """Whitelist semantics: events for non-registered cids never accumulate."""
    tr = _PerCidFillTracker()
    tr.register("cid-1")
    tr.on_event(_delta("0.5", "100", cid="stale-other-session"))
    fill = await tr.await_terminal("cid-1", Decimal("1.0"), timeout_s=0.05)
    # Registered cid never saw an event → empty (not None — slot exists but
    # nothing accumulated).
    assert fill is None or fill.filled_qty == Decimal("0")


@pytest.mark.asyncio
async def test_await_unregistered_cid_returns_none() -> None:
    """Calling await on a cid that was never registered → None, no hang."""
    tr = _PerCidFillTracker()
    fill = await tr.await_terminal("never-seen", Decimal("1.0"), timeout_s=0.05)
    assert fill is None


@pytest.mark.asyncio
async def test_qty_completion_short_circuits_timeout() -> None:
    """Accumulated qty ≥ requested → return without waiting for the deadline."""
    tr = _PerCidFillTracker()
    tr.register("cid-1")
    tr.on_event(_delta("1.0", "100"))   # no terminal status, qty reaches target
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    fill = await tr.await_terminal("cid-1", Decimal("1.0"), timeout_s=5.0)
    assert (loop.time() - t0) < 0.1
    assert fill is not None and fill.filled_qty == Decimal("1.0")


@pytest.mark.asyncio
async def test_timeout_returns_partial() -> None:
    """REST succeeded but WS lags — return whatever (partial) landed."""
    tr = _PerCidFillTracker()
    tr.register("cid-1")
    tr.on_event(_delta("0.3", "100"))
    fill = await tr.await_terminal("cid-1", Decimal("1.0"), timeout_s=0.05)
    assert fill is not None
    assert fill.filled_qty == Decimal("0.3")


@pytest.mark.asyncio
async def test_snapshot_overwrite_vs_delta_accumulate() -> None:
    """Mixed-stream behaviour matches LegOutcome.add: snapshot overwrites
    qty, delta accumulates qty. Order shouldn't matter to final state."""
    tr = _PerCidFillTracker()
    tr.register("cid-1")
    tr.on_event(_delta("0.4", "100.00"))
    tr.on_event(_snap("1.0", avg="100.05", status=OrderStatus.FILLED))
    fill = await tr.await_terminal("cid-1", Decimal("1.0"), timeout_s=0.5)
    assert fill is not None
    assert fill.filled_qty == Decimal("1.0")            # snapshot overwrote
    assert fill.avg_price == Decimal("100.05")


@pytest.mark.asyncio
async def test_release_drops_state_and_subsequent_events() -> None:
    """After release, the cid is no longer whitelisted — late events drop."""
    tr = _PerCidFillTracker()
    tr.register("cid-1")
    tr.on_event(_delta("1.0", "100", terminal=OrderStatus.FILLED))
    await tr.await_terminal("cid-1", Decimal("1.0"), timeout_s=0.1)
    tr.release("cid-1")
    # Late-arriving event must drop (not raise, not pollute internal state)
    tr.on_event(_delta("0.5", "200"))
    assert "cid-1" not in tr._events
    assert "cid-1" not in tr._fills


@pytest.mark.asyncio
async def test_register_is_idempotent() -> None:
    """Re-registering an active cid must not clobber accumulated fills.
    Defends against driver-side retries / pre-register-twice paths."""
    tr = _PerCidFillTracker()
    tr.register("cid-1")
    tr.on_event(_delta("0.4", "100"))
    tr.register("cid-1")    # idempotent
    tr.on_event(_delta("0.6", "100"))
    fill = await tr.await_terminal("cid-1", Decimal("1.0"), timeout_s=0.5)
    assert fill is not None and fill.filled_qty == Decimal("1.0")


@pytest.mark.asyncio
async def test_concurrent_cids_dont_interfere() -> None:
    """Two parallel orders on the same tracker resolve independently."""
    tr = _PerCidFillTracker()
    tr.register("cid-a")
    tr.register("cid-b")
    tr.on_event(_delta("1.0", "100", cid="cid-a", terminal=OrderStatus.FILLED))
    tr.on_event(_delta("2.0", "200", cid="cid-b", terminal=OrderStatus.FILLED))
    fa, fb = await asyncio.gather(
        tr.await_terminal("cid-a", Decimal("1.0"), timeout_s=0.5),
        tr.await_terminal("cid-b", Decimal("2.0"), timeout_s=0.5),
    )
    assert fa is not None and fa.filled_qty == Decimal("1.0") and fa.avg_price == Decimal("100")
    assert fb is not None and fb.filled_qty == Decimal("2.0") and fb.avg_price == Decimal("200")
