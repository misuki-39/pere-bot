"""RiskManager — halt, consecutive-failure, daily-loss-cap, day rollover.

Position-cap enforcement is in the pure decision function
(`strategy.reversion_signal.assess_reversion`), tested in
`test_taker_taker_logic.py`.
"""

from __future__ import annotations

from decimal import Decimal

from perp_arb.core.config import RiskCfg
from perp_arb.risk.manager import RiskManager


def _mk(**risk_overrides) -> RiskManager:
    cfg = RiskCfg(**risk_overrides) if risk_overrides else RiskCfg()
    return RiskManager(cfg)


def test_allows_trade_when_no_gates_tripped() -> None:
    r = _mk()
    ok, _ = r.can_trade()
    assert ok


def test_halts_after_max_consecutive_failures() -> None:
    r = _mk(max_consecutive_failures=2)
    r.record_failure("leg-1")
    ok, _ = r.can_trade()
    assert ok                    # one failure is still ok
    r.record_failure("leg-2")    # second failure triggers halt
    ok, reason = r.can_trade()
    assert not ok
    assert reason is not None and "halt" in reason.lower()


def test_recovers_failure_counter_on_success() -> None:
    r = _mk(max_consecutive_failures=3)
    r.record_failure("leg-1")
    r.record_failure("leg-2")
    r.record_success(leg_latency_ms=120)
    assert r.state.consecutive_failures == 0


def test_daily_loss_cap_halts() -> None:
    r = _mk(daily_loss_cap_usd=Decimal("10"))
    r.record_pnl(Decimal("-15"))
    ok, reason = r.can_trade()
    assert not ok
    assert reason is not None
