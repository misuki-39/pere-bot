"""Race-condition tests for the WS-as-truth position overlay in
TakerTakerArbitrage._fire.

The strategy maintains a `_pending_*` overlay so the next eval after a
fired trade reads the predicted state before WS pushes ACCOUNT_UPDATE.
The danger: WS can land DURING the executor await (especially on the
aster 400/timeout → queryOrder recovery path, which adds ~200ms to the
await). If we always bumped `pending += delta` after the await, a WS
event that arrived in the meantime would have already cleared pending,
and the post-fire bump would double-count against live_position until
the *next* trade's ACCOUNT_UPDATE wiped it.

The fix is to bump only the portion NOT yet absorbed:
    gap = delta - (post - pre)
where `pre`/`post` are `live_position().size` before/after the await.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest

from perp_arb.core.exec_record import (
    Decision,
    Direction,
    Verdict,
    Phase,
    Timeline,
)
from perp_arb.core.executor import ExecutionResult
from perp_arb.core.types import (
    LegKind,
    LegOutcome,
    MarketInfo,
    OrderStatus,
    Position,
    Quote,
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
        contract_id="WTI",
    ),
    "leg_b": MarketInfo(
        symbol=_SYM_B, tick_size=Decimal("0.01"), lot_size=Decimal("0.01"),
        contract_id="CLUSDT",
    ),
}


@dataclass
class _StubExchange:
    """Bare-bones BaseExchange surface for the strategy: only `live_position`
    is read in the pending-overlay code path.

    `live_size` is the field tests mutate to simulate WS pushing a new
    position during the executor await.
    """
    name: str
    symbol: Symbol
    live_size: Decimal = Decimal("0")

    def live_position(self, _market: MarketInfo) -> Position | None:
        return Position(symbol=self.symbol, size=self.live_size)


class _StubRisk:
    """Replaces RiskManager to keep tests focused on overlay math."""
    def record_success(self, *, leg_latency_ms: int) -> None: ...
    def record_failure(self, _reason: str) -> None: ...
    def record_pnl(self, _pnl: Decimal) -> None: ...


@dataclass
class _StubSession:
    is_paper: bool = False


def _mk_legs(qty: Decimal) -> list[LegOutcome]:
    """Two successful ENTRY legs at $100 with no fees."""
    out: list[LegOutcome] = []
    for venue, side in (("lighter", Side.SELL), ("aster", Side.BUY)):
        lg = LegOutcome(
            success=True, client_id=f"cid-{venue}", side=side,
            requested_qty=qty, status=OrderStatus.FILLED,
            exchange_ts_ms=1_700_000_000_000,
        )
        lg.set_fill(qty, Decimal("100.00"))
        lg.venue = venue
        lg.send_ts_ms = 1_700_000_000_000
        lg.kind = LegKind.ENTRY
        out.append(lg)
    return out


def _mk_strategy(
    *, qty: Decimal, leg_a_live: Decimal = Decimal("0"),
    leg_b_live: Decimal = Decimal("0"),
) -> tuple[TakerTakerArbitrage, _StubExchange, _StubExchange]:
    """Construct a TakerTakerArbitrage with only the attributes `_fire`
    actually reads — bypass __init__ to keep the fixture small."""
    s = TakerTakerArbitrage.__new__(TakerTakerArbitrage)
    leg_a = _StubExchange(name="lighter", symbol=_SYM_A, live_size=leg_a_live)
    leg_b = _StubExchange(name="aster", symbol=_SYM_B, live_size=leg_b_live)
    s.exchanges = {"leg_a": leg_a, "leg_b": leg_b}
    s.markets = _MARKETS
    s.session = _StubSession(is_paper=False)

    # `cfg.strategy.qty`. Use SimpleNamespace to dodge the full AppCfg.
    from types import SimpleNamespace
    s.cfg = SimpleNamespace(strategy=SimpleNamespace(
        qty=qty, mode="live",
    ))
    s._position = SyntheticPosition()
    s._pending_a = Decimal("0")
    s._pending_b = Decimal("0")
    s._paper_pos_a = Decimal("0")
    s._paper_pos_b = Decimal("0")
    s._risk = _StubRisk()
    s._inflight_cap = 0
    s._inflight_dir = {}
    s._throttle_enabled = False
    s._recorder = None
    return s, leg_a, leg_b


def _mk_decision(qty: Decimal) -> Decision:
    return Decision(
        decision_id="d-test",
        ts_ms=1_700_000_000_000,
        mid_left=Decimal("100"), mid_right=Decimal("100"),
        left_quote_ts_ms=1_700_000_000_000, right_quote_ts_ms=1_700_000_000_000,
        bias=Decimal("0"),
        vwap_left_sell=Decimal("100"), vwap_left_buy=Decimal("100"),
        vwap_right_sell=Decimal("100"), vwap_right_buy=Decimal("100"),
        edge_bps=Decimal("0"), direction=Direction.A, outcome=Verdict.FIRED,
        timeline=Timeline(),
    )


def _mk_quote() -> Quote:
    """Minimal quote stub for `_fire`'s per-leg context stamping."""
    return Quote(
        symbol=_SYM_A,
        bid=Decimal("100"), bid_size=Decimal("1"),
        ask=Decimal("100"), ask_size=Decimal("1"),
        ts_ms=1_700_000_000_000,
    )


class _StubExecutor:
    """Replace `_executor`. `on_execute` is invoked synchronously inside
    `execute()` AFTER the strategy snapshots `pre_a`/`pre_b` but BEFORE
    we return the ExecutionResult — that's the race window."""

    def __init__(self, qty: Decimal, *, on_execute=None, success: bool = True):
        self.qty = qty
        self.on_execute = on_execute or (lambda: None)
        self.success = success

    async def execute(self, *, trade_id, legs, qty, timeline):
        timeline.mark(Phase.SEND)
        self.on_execute()
        return ExecutionResult(
            legs=_mk_legs(qty),
            failure_reason=None if self.success else "stub: leg failed",
        )


# ---- the three race outcomes ------------------------------------------


@pytest.mark.asyncio
async def test_pending_gets_full_delta_when_ws_did_not_arrive() -> None:
    """Fast path: WS ACCOUNT_UPDATE hasn't landed by the time the executor
    returns. `pending` carries the full predicted delta until WS arrives."""
    qty = Decimal("0.12")
    s, _a, _b = _mk_strategy(qty=qty)
    # Executor does NOT mutate live_size — simulating WS hasn't arrived.
    s._executor = _StubExecutor(qty)
    await s._fire(_mk_decision(qty), _mk_quote(), _mk_quote())
    # Direction A: leg_a (lighter) sells, leg_b (aster) buys.
    assert s._pending_a == -qty
    assert s._pending_b == +qty
    # _pos_a/_b = live(0) + pending → match the predicted delta
    assert s._pos_a() == -qty
    assert s._pos_b() == +qty


@pytest.mark.asyncio
async def test_pending_stays_zero_when_ws_already_absorbed_during_await() -> None:
    """The race we're guarding against. WS ACCOUNT_UPDATE lands during the
    executor await (e.g. aster 400/timeout → queryOrder takes ~200ms,
    plenty of time for ACCOUNT_UPDATE to arrive). post-pre == delta, so
    gap == 0 and pending must NOT be bumped — otherwise _pos_x() would
    double-count delta until the next trade clears it."""
    qty = Decimal("0.12")
    s, leg_a, leg_b = _mk_strategy(qty=qty)

    def ws_lands_during_await() -> None:
        # Simulate ACCOUNT_UPDATE pushing the new position into the
        # driver cache while the executor is mid-flight.
        leg_a.live_size = -qty  # leg_a (lighter) sold qty
        leg_b.live_size = +qty  # leg_b (aster) bought qty

    s._executor = _StubExecutor(qty, on_execute=ws_lands_during_await)
    await s._fire(_mk_decision(qty), _mk_quote(), _mk_quote())
    # Critical assertion: pending stays at 0 because WS already absorbed
    # the delta. live_position alone reflects the position.
    assert s._pending_a == Decimal("0")
    assert s._pending_b == Decimal("0")
    assert s._pos_a() == -qty
    assert s._pos_b() == +qty


@pytest.mark.asyncio
async def test_pending_partial_overlay_when_ws_partially_caught_up() -> None:
    """Mid-window: WS observed a smaller delta than expected by the time
    the executor returns. Overlay carries the remaining gap so the next
    eval sees the predicted state."""
    qty = Decimal("0.12")
    s, leg_a, leg_b = _mk_strategy(qty=qty)
    partial = Decimal("0.05")  # WS pushed only a partial fill so far

    def ws_partial_during_await() -> None:
        leg_a.live_size = -partial
        leg_b.live_size = +partial

    s._executor = _StubExecutor(qty, on_execute=ws_partial_during_await)
    await s._fire(_mk_decision(qty), _mk_quote(), _mk_quote())
    # gap_a = delta_a - (post - pre) = -qty - (-partial - 0) = -(qty - partial)
    assert s._pending_a == -(qty - partial)
    assert s._pending_b == +(qty - partial)
    # Effective view still matches the full predicted delta.
    assert s._pos_a() == -qty
    assert s._pos_b() == +qty


@pytest.mark.asyncio
async def test_pending_not_bumped_on_executor_failure() -> None:
    """Failure path: no prediction to overlay. live_position carries
    whatever actually happened on the venues."""
    qty = Decimal("0.12")
    s, _a, _b = _mk_strategy(qty=qty)
    s._executor = _StubExecutor(qty, success=False)
    await s._fire(_mk_decision(qty), _mk_quote(), _mk_quote())
    assert s._pending_a == Decimal("0")
    assert s._pending_b == Decimal("0")
    assert s._pos_a() == Decimal("0")
    assert s._pos_b() == Decimal("0")
