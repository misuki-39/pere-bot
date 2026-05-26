"""RiskManager — halt, cooldown, consecutive-failure, daily-loss-cap, day rollover.

Position-cap enforcement is in the pure decision function
(`strategy.reversion_signal.assess_reversion`), tested in
`test_taker_taker_logic.py`.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from perp_arb.core.config import RiskCfg
from perp_arb.risk.manager import RiskManager


def _mk(**risk_overrides) -> RiskManager:
    # cooldown_s defaults to 0 so legacy non-cooldown tests still exercise
    # the consecutive-failures / halt narratives in isolation. Cooldown-
    # focused tests below set it explicitly.
    risk_overrides.setdefault("cooldown_s", 0.0)
    cfg = RiskCfg(**risk_overrides)
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


# ---- cooldown gate ---------------------------------------------------


def test_record_failure_arms_cooldown() -> None:
    """A single failure with cooldown_s>0 sets cooldown_until_ms in the
    future and blocks `can_trade` with a "cooldown" reason."""
    r = _mk(cooldown_s=30.0, max_consecutive_failures=10)
    r.record_failure("net blip")
    assert r.state.cooldown_until_ms > 0
    ok, reason = r.can_trade()
    assert not ok
    assert reason is not None and "cooldown" in reason


def test_can_trade_unblocks_after_cooldown_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once `now_ms()` passes cooldown_until_ms the gate releases. We
    don't actually sleep — we patch the now_ms used by `can_trade`."""
    r = _mk(cooldown_s=30.0, max_consecutive_failures=10)
    r.record_failure("blip")
    expiry = r.state.cooldown_until_ms

    from perp_arb.risk import manager as risk_mod
    monkeypatch.setattr(risk_mod, "now_ms", lambda: expiry + 1)
    ok, _ = r.can_trade()
    assert ok


def test_cooldown_armed_even_when_halt_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure that finally crosses max_consecutive_failures still
    leaves cooldown_until_ms set. Halt takes precedence in `can_trade`'s
    reason text, but the cooldown bookkeeping must be intact (e.g. for
    observability + future unhalt restart paths)."""
    r = _mk(cooldown_s=30.0, max_consecutive_failures=2)
    r.record_failure("first")
    r.record_failure("second — trips halt")
    assert r.state.halted is True
    assert r.state.cooldown_until_ms > 0
    ok, reason = r.can_trade()
    assert not ok
    assert reason is not None and "halt" in reason.lower()


def test_cooldown_zero_means_no_gate() -> None:
    """`cooldown_s=0` keeps pre-refactor semantics: a single failure
    leaves `can_trade` open (subject to consecutive-failures cap)."""
    r = _mk(cooldown_s=0.0, max_consecutive_failures=5)
    r.record_failure("blip")
    ok, _ = r.can_trade()
    assert ok
