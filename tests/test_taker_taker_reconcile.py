"""Tests for the strategy-owned partial-failure reconcile flow.

After `_fire` fails (one or both legs), the strategy runs
`_reconcile_after_failure(target_a, target_b)`:

  1. REST-snapshot both venues via `session.snapshot_position`.
  2. For each leg whose snap differs from the pre-fire target by more
     than `max(min_qty, lot_size)`, place a reduce-only market order
     to flatten the residual.
  3. On success: clear the pending overlay and the reconcile flag.
  4. On failure (snapshot raises, or rebalance submit fails): set
     `_reconcile_pending=True` and remember the target so the next
     `_evaluate` after cooldown expiry retries.

`max_consecutive_failures` remains the hard backstop — retries are
gated by cooldown, not by busy-loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from perp_arb.core.types import (
    LegOutcome,
    MarketInfo,
    OrderStatus,
    Position,
    Side,
    Symbol,
)
from perp_arb.strategy.taker_taker import (
    SyntheticPosition,
    TakerTakerArbitrage,
)

_SYM_A = Symbol(exchange="lighter", raw="WTI", base="WTI", quote="USDC")
_SYM_B = Symbol(exchange="aster", raw="CLUSDT", base="WTI", quote="USDT")
_MARKETS = {
    "leg_a": MarketInfo(
        symbol=_SYM_A, tick_size=Decimal("0.01"), lot_size=Decimal("0.01"),
        contract_id="WTI", min_qty=Decimal("0.01"),
    ),
    "leg_b": MarketInfo(
        symbol=_SYM_B, tick_size=Decimal("0.01"), lot_size=Decimal("0.01"),
        contract_id="CLUSDT", min_qty=Decimal("0.01"),
    ),
}


def _ok_outcome(side: Side, qty: Decimal) -> LegOutcome:
    out = LegOutcome(
        success=True, client_id="cid", side=side,
        requested_qty=qty, status=OrderStatus.FILLED,
    )
    out.set_fill(qty, Decimal("100.00"))
    return out


def _fail_outcome(side: Side, qty: Decimal, msg: str) -> LegOutcome:
    return LegOutcome(
        success=False, side=side, requested_qty=qty, error_message=msg,
    )


class _CountingCidGen:
    def __init__(self) -> None:
        self._n = 9000

    def next(self, *, side: Side) -> str:  # noqa: ARG002
        self._n += 1
        return str(self._n)


@dataclass
class _StubExchange:
    """Records every submit_and_await call so tests can assert on the
    reconcile orders the strategy issued. `submit_outcomes` is a queue
    of pre-canned outcomes the stub returns per call (FIFO)."""
    name: str
    submit_outcomes: list[LegOutcome] = field(default_factory=list)
    submit_calls: list[dict[str, Any]] = field(default_factory=list)
    # Reconcile path asks the venue for a fresh cid via this generator
    # (matches the real `BaseExchange.client_id_generator` contract).
    client_id_generator: Any = field(default_factory=_CountingCidGen)

    def live_position(self, _market: MarketInfo) -> Position | None:
        return None

    async def submit_and_await(
        self, market: MarketInfo, side: Side, qty: Decimal,
        *, client_id: str, timeout_s: float = 5.0, reduce_only: bool = False,
    ) -> LegOutcome:
        self.submit_calls.append({
            "venue": self.name, "side": side, "qty": qty,
            "reduce_only": reduce_only, "client_id": client_id,
        })
        if not self.submit_outcomes:
            return _ok_outcome(side, qty)
        return self.submit_outcomes.pop(0)


class _StubExecutor:
    """Minimal `_executor` surface — strategy only reads `_next_cid()`
    for reconcile cid generation."""
    def __init__(self) -> None:
        self._n = 9000

    def _next_cid(self) -> str:
        self._n += 1
        return f"cid-{self._n}"


@dataclass
class _StubSession:
    is_paper: bool = False
    # snap_a / snap_b are the values `snapshot_position` returns for
    # leg_a / leg_b respectively. `raise_on` selects which leg's call
    # should raise instead of return.
    snap_a: Decimal = Decimal("0")
    snap_b: Decimal = Decimal("0")
    raise_on: str | None = None  # "a", "b", or None
    snapshot_calls: list[str] = field(default_factory=list)

    async def snapshot_position(
        self, exchange: _StubExchange, market: MarketInfo,
    ) -> Decimal:
        leg = "a" if market.symbol == _SYM_A else "b"
        self.snapshot_calls.append(leg)
        if self.raise_on == leg:
            raise RuntimeError("REST blew up")
        return self.snap_a if leg == "a" else self.snap_b


class _StubRisk:
    """Tracks record_failure / record_success calls so tests can verify
    cooldown was armed (or not). The real RiskManager is exercised in
    tests/test_risk_manager.py."""
    def __init__(self) -> None:
        self.failure_calls: list[str] = []
        self.success_calls: list[int] = []

    def can_trade(self) -> tuple[bool, str | None]:
        return True, None

    def record_failure(self, reason: str) -> None:
        self.failure_calls.append(reason)

    def record_success(self, *, leg_latency_ms: int) -> None:
        self.success_calls.append(leg_latency_ms)

    def record_pnl(self, _pnl: Decimal) -> None: ...


def _mk_strategy(
    *, snap_a: Decimal, snap_b: Decimal,
    raise_on: str | None = None,
    submit_outcomes_a: list[LegOutcome] | None = None,
    submit_outcomes_b: list[LegOutcome] | None = None,
) -> tuple[TakerTakerArbitrage, _StubExchange, _StubExchange, _StubSession]:
    s = TakerTakerArbitrage.__new__(TakerTakerArbitrage)
    leg_a = _StubExchange(name="lighter", submit_outcomes=submit_outcomes_a or [])
    leg_b = _StubExchange(name="aster", submit_outcomes=submit_outcomes_b or [])
    session = _StubSession(
        is_paper=False, snap_a=snap_a, snap_b=snap_b, raise_on=raise_on,
    )
    s.exchanges = {"leg_a": leg_a, "leg_b": leg_b}
    s.markets = _MARKETS
    s.session = session
    s.cfg = SimpleNamespace(strategy=SimpleNamespace(qty=Decimal("0.12"), mode="live"))
    s._position = SyntheticPosition()
    s._pending_a = Decimal("0")
    s._pending_b = Decimal("0")
    s._paper_pos_a = Decimal("0")
    s._paper_pos_b = Decimal("0")
    s._reconcile_pending = False
    s._reconcile_target = None
    s._risk = _StubRisk()
    s._executor = _StubExecutor()
    s._inflight_cap = 0
    s._inflight_dir = {}
    s._throttle_enabled = False
    s._recorder = None
    return s, leg_a, leg_b, session


# ---- happy + edge cases of the sync → balance → cooldown flow ----


@pytest.mark.asyncio
async def test_reconcile_clean_no_rebalance_order() -> None:
    """Sync confirms target == snap on BOTH legs → no rebalance order
    issued. `_reconcile_pending` stays False."""
    target = Decimal("-0.24")
    s, leg_a, leg_b, _sess = _mk_strategy(
        snap_a=target, snap_b=-target,
    )
    await s._reconcile_after_failure(target, -target)
    assert leg_a.submit_calls == []
    assert leg_b.submit_calls == []
    assert s._reconcile_pending is False
    assert s._reconcile_target is None
    assert s._pending_a == Decimal("0")
    assert s._pending_b == Decimal("0")


@pytest.mark.asyncio
async def test_reconcile_leg_a_imbalanced_places_one_reduce_only() -> None:
    """target_a=0, snap_a=-0.12 → BUY 0.12 reduce-only on leg_a; leg_b is
    already at target so no order there."""
    s, leg_a, leg_b, _ = _mk_strategy(
        snap_a=Decimal("-0.12"), snap_b=Decimal("0"),
    )
    await s._reconcile_after_failure(Decimal("0"), Decimal("0"))
    assert len(leg_a.submit_calls) == 1
    call = leg_a.submit_calls[0]
    assert call["side"] is Side.BUY
    assert call["qty"] == Decimal("0.12")
    assert call["reduce_only"] is True
    assert leg_b.submit_calls == []
    assert s._reconcile_pending is False


@pytest.mark.asyncio
async def test_reconcile_both_legs_imbalanced_places_two_orders() -> None:
    """The asymmetric partial fill: snap differs on both legs."""
    s, leg_a, leg_b, _ = _mk_strategy(
        snap_a=Decimal("-0.12"), snap_b=Decimal("+0.12"),
    )
    await s._reconcile_after_failure(Decimal("0"), Decimal("0"))
    assert len(leg_a.submit_calls) == 1
    assert len(leg_b.submit_calls) == 1
    # leg_a is -0.12 (short) → BUY to flatten
    assert leg_a.submit_calls[0]["side"] is Side.BUY
    # leg_b is +0.12 (long) → SELL to flatten
    assert leg_b.submit_calls[0]["side"] is Side.SELL
    assert all(c["reduce_only"] for c in leg_a.submit_calls + leg_b.submit_calls)
    assert s._reconcile_pending is False


@pytest.mark.asyncio
async def test_reconcile_sub_lot_diff_is_treated_as_clean() -> None:
    """Dust threshold = max(min_qty, lot_size) = 0.01. A diff smaller
    than that should be ignored — never submit a sub-lot order."""
    s, leg_a, leg_b, _ = _mk_strategy(
        snap_a=Decimal("-0.005"), snap_b=Decimal("0"),
    )
    await s._reconcile_after_failure(Decimal("0"), Decimal("0"))
    assert leg_a.submit_calls == []
    assert leg_b.submit_calls == []
    assert s._reconcile_pending is False


# ---- contingency: any step failure → set pending, defer to next cycle ----


@pytest.mark.asyncio
async def test_reconcile_rest_snapshot_raise_sets_pending() -> None:
    """REST snapshot raises → no rebalance attempted, _reconcile_pending
    marked, target stored for retry."""
    s, leg_a, leg_b, _ = _mk_strategy(
        snap_a=Decimal("0"), snap_b=Decimal("0"), raise_on="a",
    )
    await s._reconcile_after_failure(Decimal("-1"), Decimal("+1"))
    assert leg_a.submit_calls == []
    assert leg_b.submit_calls == []
    assert s._reconcile_pending is True
    assert s._reconcile_target == (Decimal("-1"), Decimal("+1"))


@pytest.mark.asyncio
async def test_reconcile_rebalance_returns_fail_sets_pending() -> None:
    """Snapshot succeeds, but the reduce-only submit returns
    success=False → mark pending for next-cycle retry."""
    s, leg_a, _leg_b, _ = _mk_strategy(
        snap_a=Decimal("-0.12"), snap_b=Decimal("0"),
        submit_outcomes_a=[
            _fail_outcome(Side.BUY, Decimal("0.12"), "venue rejected"),
        ],
    )
    await s._reconcile_after_failure(Decimal("0"), Decimal("0"))
    # rebalance was attempted on leg_a
    assert len(leg_a.submit_calls) == 1
    assert s._reconcile_pending is True
    assert s._reconcile_target == (Decimal("0"), Decimal("0"))


@pytest.mark.asyncio
async def test_reconcile_rebalance_raise_sets_pending() -> None:
    """If submit_and_await raises directly (network error mid-call), the
    reconcile path catches it and marks pending."""
    @dataclass
    class _RaisingExchange(_StubExchange):
        async def submit_and_await(  # type: ignore[override]
            self, *_args: Any, **_kw: Any,
        ) -> LegOutcome:
            self.submit_calls.append({"raised": True})
            raise RuntimeError("conn reset")

    s = TakerTakerArbitrage.__new__(TakerTakerArbitrage)
    leg_a = _RaisingExchange(name="lighter")
    leg_b = _StubExchange(name="aster")
    s.exchanges = {"leg_a": leg_a, "leg_b": leg_b}
    s.markets = _MARKETS
    s.session = _StubSession(
        is_paper=False, snap_a=Decimal("-0.12"), snap_b=Decimal("0"),
    )
    s.cfg = SimpleNamespace(strategy=SimpleNamespace(qty=Decimal("0.12"), mode="live"))
    s._position = SyntheticPosition()
    s._pending_a = Decimal("0")
    s._pending_b = Decimal("0")
    s._paper_pos_a = Decimal("0")
    s._paper_pos_b = Decimal("0")
    s._reconcile_pending = False
    s._reconcile_target = None
    s._risk = _StubRisk()
    s._executor = _StubExecutor()
    s._inflight_cap = 0
    s._inflight_dir = {}
    s._throttle_enabled = False
    s._recorder = None

    await s._reconcile_after_failure(Decimal("0"), Decimal("0"))
    assert s._reconcile_pending is True


# ---- evaluate-time retry: pending reconcile re-runs after cooldown ----


@pytest.mark.asyncio
async def test_evaluate_retries_pending_reconcile_when_cooldown_clear() -> None:
    """When `_reconcile_pending=True` and `can_trade()` returns OK
    (cooldown has expired), the next `_evaluate` re-runs the reconcile
    BEFORE any normal signal evaluation. If retry succeeds, pending
    clears."""
    s, leg_a, _leg_b, _ = _mk_strategy(
        snap_a=Decimal("-0.12"), snap_b=Decimal("0"),
    )
    s._reconcile_pending = True
    s._reconcile_target = (Decimal("0"), Decimal("0"))
    # _StubRisk.can_trade() returns (True, None) by default → simulate
    # cooldown expired.

    # Patch out the normal-signal path: _evaluate should NOT touch
    # order_book / best_quote when handling a pending reconcile. We give
    # the stubs `order_book` / `best_quote` callables that raise so any
    # accidental fall-through trips the test.
    for ex in (s._leg_a(), s._leg_b()):
        ex.order_book = _raises  # type: ignore[method-assign]
        ex.best_quote = _raises  # type: ignore[method-assign]

    await s._evaluate()
    assert s._reconcile_pending is False
    assert len(leg_a.submit_calls) == 1
    assert s._risk.failure_calls == []  # retry succeeded, no new failure


@pytest.mark.asyncio
async def test_evaluate_skips_normal_path_while_cooldown_pending() -> None:
    """If reconcile is pending AND cooldown still active, _evaluate
    returns immediately without touching books or quotes."""
    s, leg_a, _leg_b, _ = _mk_strategy(
        snap_a=Decimal("0"), snap_b=Decimal("0"),
    )
    s._reconcile_pending = True
    s._reconcile_target = (Decimal("0"), Decimal("0"))
    s._risk.can_trade = lambda: (False, "cooldown 12.0s")  # type: ignore[method-assign]

    # No books / quotes; if _evaluate falls through it would crash.
    await s._evaluate()
    # No retry was attempted
    assert leg_a.submit_calls == []
    assert s._reconcile_pending is True


@pytest.mark.asyncio
async def test_evaluate_retry_still_failing_records_new_failure() -> None:
    """Retry sync still raises → reconcile stays pending → _evaluate
    calls record_failure (re-arming cooldown) before returning."""
    s, leg_a, _leg_b, _ = _mk_strategy(
        snap_a=Decimal("0"), snap_b=Decimal("0"), raise_on="b",
    )
    s._reconcile_pending = True
    s._reconcile_target = (Decimal("0"), Decimal("0"))

    await s._evaluate()
    assert s._reconcile_pending is True
    # New failure recorded → cooldown re-armed by record_failure
    assert s._risk.failure_calls == ["reconcile retry failed"]
    assert leg_a.submit_calls == []


# ---- helpers ---------------------------------------------------------


def _raises(*_a: Any, **_k: Any) -> None:
    raise AssertionError(
        "normal-signal path was reached while a reconcile was pending",
    )
