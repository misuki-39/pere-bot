"""RiskManager — direction-aware position cap, failure counting, day rollover."""

from __future__ import annotations

from decimal import Decimal

from perp_arb.core.config import RiskCfg
from perp_arb.risk.manager import RiskManager


def _mk(max_qty: str = "0.5", **risk_overrides) -> RiskManager:
    cfg = RiskCfg(**risk_overrides) if risk_overrides else RiskCfg()
    return RiskManager(cfg, max_qty=Decimal(max_qty))


def test_allows_first_entry_under_cap() -> None:
    r = _mk()
    ok, _ = r.can_trade(post_trade_abs_position=Decimal("0.05"))
    assert ok


def test_blocks_when_post_trade_exceeds_cap() -> None:
    r = _mk(max_qty="0.5")
    ok, reason = r.can_trade(post_trade_abs_position=Decimal("0.55"))
    assert not ok
    assert reason is not None and "position cap" in reason


def test_allows_reverse_entry_that_reduces_exposure() -> None:
    """Loop: at max_qty, a reverse-direction fire reduces |pos| to 0.

    This is the load-bearing property for using reverse-entry as exit.
    """
    r = _mk(max_qty="0.5")
    # Pre-trade |pos| = 0.5 (at cap). A reverse-direction fire of qty=0.5
    # brings post-trade |pos| to 0 — must be allowed.
    ok, _ = r.can_trade(post_trade_abs_position=Decimal("0"))
    assert ok


def test_halts_after_max_consecutive_failures() -> None:
    r = _mk(max_consecutive_failures=2)
    r.record_failure("leg-1")
    ok, _ = r.can_trade(post_trade_abs_position=Decimal("0"))
    assert ok                    # one failure is still ok
    r.record_failure("leg-2")    # second failure triggers halt
    ok, reason = r.can_trade(post_trade_abs_position=Decimal("0"))
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
    ok, reason = r.can_trade(post_trade_abs_position=Decimal("0"))
    assert not ok
    assert reason is not None
