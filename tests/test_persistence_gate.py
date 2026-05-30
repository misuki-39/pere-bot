"""Unit tests for the edge-persistence confirmation gate."""

from __future__ import annotations

from decimal import Decimal

from perp_arb.core.exec_record import Decision, Direction, Verdict
from perp_arb.strategy.persistence_gate import PersistenceGate, PersistenceParams


def _decision(
    ts_ms: int,
    *,
    outcome: Verdict = Verdict.FIRED,
    direction: Direction | None = Direction.A,
    mid_left: str = "100",
    mid_right: str = "100",
) -> Decision:
    return Decision(
        decision_id=f"d-{ts_ms}",
        ts_ms=ts_ms,
        mid_left=Decimal(mid_left),
        mid_right=Decimal(mid_right),
        left_quote_ts_ms=ts_ms,
        right_quote_ts_ms=ts_ms,
        direction=direction,
        outcome=outcome,
    )


_ENABLED = PersistenceParams(enabled=True, t_confirm_ms=400, n_confirm=4,
                             drift_max_bps=Decimal("1.0"))


def test_disabled_gate_is_identity() -> None:
    gate = PersistenceGate(PersistenceParams(enabled=False))
    d = _decision(1000)
    assert gate.admit(d, left_ticked=True, right_ticked=False) is d
    # ...even on the very first FIRED tick — no suppression when disabled.


def test_fired_suppressed_until_time_and_count_met() -> None:
    gate = PersistenceGate(_ENABLED)
    # ticks within the window or below the update count → suppressed
    for ts in (1000, 1100, 1200, 1300):           # 4 updates, but <400ms old
        assert gate.admit(_decision(ts), left_ticked=True, right_ticked=False) is None
    # 500ms old AND 5 updates AND no drift → confirmed, fires
    out = gate.admit(_decision(1500), left_ticked=True, right_ticked=False)
    assert out is not None and out.outcome is Verdict.FIRED


def test_only_one_fire_per_run() -> None:
    gate = PersistenceGate(_ENABLED)
    for ts in (1000, 1100, 1200, 1300):
        gate.admit(_decision(ts), left_ticked=True, right_ticked=False)
    assert gate.admit(_decision(1500), left_ticked=True, right_ticked=False) is not None
    # the run already fired — further FIRED ticks on it are suppressed
    assert gate.admit(_decision(1600), left_ticked=True, right_ticked=False) is None


def test_non_fired_resets_the_run() -> None:
    gate = PersistenceGate(_ENABLED)
    for ts in (1000, 1100, 1200):
        gate.admit(_decision(ts), left_ticked=True, right_ticked=False)
    # a None tick (edge died) resets the run and passes through
    assert gate.admit(None, left_ticked=True, right_ticked=False) is None
    # the run restarts: a fresh FIRED is young again, so it is suppressed
    assert gate.admit(_decision(1300), left_ticked=True, right_ticked=False) is None
    # an abort passes straight through (telemetry) and also resets
    abort = _decision(1350, outcome=Verdict.ABORT_STALE, direction=None)
    assert gate.admit(abort, left_ticked=True, right_ticked=False) is abort


def test_direction_flip_restarts_the_run() -> None:
    gate = PersistenceGate(_ENABLED)
    for ts in (1000, 1100, 1200, 1300):
        gate.admit(_decision(ts, direction=Direction.A),
                   left_ticked=True, right_ticked=False)
    # direction flips → new run, young again → suppressed despite the elapsed time
    out = gate.admit(_decision(1500, direction=Direction.B),
                     left_ticked=True, right_ticked=False)
    assert out is None


def test_adverse_drift_rejects_a_confirmed_run() -> None:
    gate = PersistenceGate(_ENABLED)
    # run starts with mid_left = 100
    for ts in (1000, 1100, 1200, 1300):
        gate.admit(_decision(ts), left_ticked=True, right_ticked=False)
    # confirmed by time+count, but mid_left fell ~2bps → adverse for Direction.A
    out = gate.admit(_decision(1500, mid_left="99.98"),
                     left_ticked=True, right_ticked=False)
    assert out is None


def test_both_venues_ticking_counts_two_updates() -> None:
    gate = PersistenceGate(PersistenceParams(enabled=True, t_confirm_ms=0,
                                             n_confirm=4, drift_max_bps=Decimal("1.0")))
    # t_confirm=0, so only the update count gates. Two ticks × 2 venues = 4.
    assert gate.admit(_decision(1000), left_ticked=True, right_ticked=True) is None
    out = gate.admit(_decision(1001), left_ticked=True, right_ticked=True)
    assert out is not None
