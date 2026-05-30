"""Persistence-confirmation gate — a stateful temporal filter on FIRED decisions.

Sits between the signal layer (`assess_reversion`) and execution. The
baseline fires the instant the edge clears threshold — the tick maximally
enriched for one-tick winner's-curse noise. This gate suppresses a FIRED
decision until its edge has *survived* `t_confirm_ms` across `n_confirm` venue
updates, and the venue mids have not drifted adversely over the run (the
signature of sustained one-way flow rather than a real dislocation).

It is the live wiring of "Strategy 3 — Edge-Persistence Confirmation" from the
2026-05-22 strategy search (OOS +$0.36/day vs the −$6/day taker-taker
baseline). See memory `strategy-search-wti-2026-05-22`.

Layering: the gate is a pure, *stateful but I/O-free* deep module — no async,
no clock (the caller supplies `ts_ms` via the `Decision`), no venue knowledge.
The same instance type is used by the live strategy and the backtest so the
two cannot diverge. When disabled, `admit` is an identity pass-through —
"disabled" is not a branch the caller has to handle.

What it does NOT do: magnitude filtering. Whether an edge is *big enough* is
`assess_reversion`'s job (`fees_bps + min_profit_bps`); a tick reaches this
gate as FIRED only once it already cleared that. The gate adds the orthogonal
*duration* axis — has the edge lasted long enough to be real.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ..core.decision import Decision, Direction, Verdict
from ..utils.precision import BPS


@dataclass(frozen=True, slots=True)
class PersistenceParams:
    """Gate configuration. `enabled=False` = identity pass-through."""

    enabled: bool = False
    t_confirm_ms: int = 400          # edge must survive at least this long
    n_confirm: int = 6               # …across at least this many venue updates
    drift_max_bps: Decimal = Decimal("1.0")  # reject runs with adverse mid-drift


@dataclass(slots=True)
class _Run:
    """One direction's continuously-FIRED edge run."""

    direction: Direction
    start_ts_ms: int
    start_mid_left: Decimal
    start_mid_right: Decimal
    updates: int = 0
    fired: bool = False


class PersistenceGate:
    """Temporal survivorship filter. One active run at a time, keyed to the
    direction the signal currently wants to fire."""

    __slots__ = ("_p", "_run")

    def __init__(self, params: PersistenceParams) -> None:
        self._p = params
        self._run: _Run | None = None

    def admit(
        self,
        candidate: Decision | None,
        *,
        left_ticked: bool,
        right_ticked: bool,
    ) -> Decision | None:
        """Feed the per-tick assessment; get back what the caller should act on.

        - gate disabled → `candidate` returned unchanged.
        - `candidate` is None or a non-FIRED outcome (abort / blocked) → the
          run is reset and `candidate` is passed straight through (so aborts
          still reach telemetry).
        - `candidate` is FIRED but its edge has not yet been confirmed →
          returns None (the fire is suppressed; the run keeps accumulating).
        - `candidate` is FIRED and the run just satisfied persistence + the
          drift check → returns `candidate` (fire it). A run fires at most once.
        """
        if not self._p.enabled:
            return candidate
        if candidate is None or candidate.outcome is not Verdict.FIRED:
            self._run = None
            return candidate

        assert candidate.direction is not None
        run = self._run
        if run is None or run.direction is not candidate.direction:
            run = self._run = _Run(
                direction=candidate.direction,
                start_ts_ms=candidate.ts_ms,
                start_mid_left=candidate.mid_left,
                start_mid_right=candidate.mid_right,
            )
        run.updates += int(left_ticked) + int(right_ticked)

        if run.fired:
            return None                                  # one fire per run
        if candidate.ts_ms - run.start_ts_ms < self._p.t_confirm_ms:
            return None                                  # not old enough
        if run.updates < self._p.n_confirm:
            return None                                  # not enough updates
        if self._drift_adverse(run, candidate):
            return None                                  # one-way flow — skip
        run.fired = True
        return candidate

    def _drift_adverse(self, run: _Run, d: Decision) -> bool:
        """True if either leg's venue mid drifted against the trade by more
        than `drift_max_bps` over the run — sustained one-way flow, not a
        mean-reverting dislocation. Direction.A sells left / buys right, so
        adverse = left fell or right rose; Direction.B is the mirror."""
        drift_l = (d.mid_left - run.start_mid_left) / d.mid_left * BPS
        drift_r = (d.mid_right - run.start_mid_right) / d.mid_right * BPS
        limit = self._p.drift_max_bps
        if run.direction is Direction.A:
            return drift_l < -limit or drift_r > limit
        return drift_l > limit or drift_r < -limit
