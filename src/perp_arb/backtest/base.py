"""Backtest strategy ABC + supporting context types.

Mirrors `perp_arb.strategy.base` (which holds the *live* strategy ABC and
shared EWMA primitives) — same naming, different concern. Concrete BT
strategy classes live under `backtest/strategies/` and inherit from the
`BacktestStrategy` ABC defined here.

Strategies are *pure synchronous*: `on_tick(snapshot, view)` returns a list
of `OrderIntent`s (possibly empty), and `on_fill(fill, view)` is a no-op
hook for strategies that want to react to resolution. No asyncio, no
exchange clients.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal

from ..core.recording.recorder import Recorder
from ..strategy.persistence_gate import PersistenceParams
from .fills import FillModelKind
from .intents import FillEvent, OrderIntent
from .snapshot import MarketSnapshot


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Static parameters + the recorder handle. One per backtest run.

    `fees_bps` is the *round-trip* fee bps (matches the live YAML convention —
    see configs/taker_taker_eth.yaml). The engine separately uses
    `EngineConfig.fee_bps_per_leg` for actual PnL fee deduction; the two are
    related by `fee_bps_per_leg = fees_bps / 2` unless overridden.
    """
    capture_qty: Decimal
    fees_bps: Decimal                          # round-trip, used for decision threshold
    min_profit_bps: Decimal
    max_stale_ms: int
    bias_halflife_s: float
    scale_halflife_s: float
    warmup_seconds: float
    max_qty: Decimal
    left_venue: str
    right_venue: str
    fill_model: FillModelKind
    recorder: Recorder
    # Optional Wave-1 optimisation knobs. All default-off so the legacy
    # strategy behaviour is preserved when the YAML omits these fields.
    inventory_skew_bps: Decimal = Decimal(0)   # κ for AS-style threshold widener (open side)
    inventory_skew_close_bps: Decimal = Decimal(0)   # κ for the close side (0 = no exit-easing)
    in_flight_cap_per_direction: int = 0       # 0 = unlimited; K = at most K same-dir entries pending
    persistence: PersistenceParams = PersistenceParams()   # edge-persistence gate (off by default)


@dataclass(slots=True)
class EngineView:
    """Read-only proxy over engine state for the strategy.

    `position(venue)` returns the *committed* exposure — settled fills plus
    any signed quantity currently in-flight (scheduled but not yet resolved).
    This matches the live model where `await self._fire(d)` serialises ticks,
    so by the time the next decision is assessed the in-flight delta has
    already been folded in. The backtest doesn't serialise, so it has to
    expose in-flight explicitly or `max_qty` is unenforceable under latency.
    """
    _positions: dict[str, Decimal] = field(default_factory=dict)
    _in_flight: dict[str, Decimal] = field(default_factory=dict)
    _sim_now_ms: int = 0
    _pending_count: int = 0

    def position(self, venue: str) -> Decimal:
        return (self._positions.get(venue, Decimal("0"))
                + self._in_flight.get(venue, Decimal("0")))

    def sim_now_ms(self) -> int:
        return self._sim_now_ms

    def pending_orders(self) -> int:
        return self._pending_count


class BacktestStrategy(ABC):
    """Sync strategy ABC.

    Lifecycle:
      __init__(ctx)
      → on_tick(snap, view)  [N rows]
      → on_fill(fill, view)  [each resolved leg, possibly after on_tick]
      → on_end(view)         [once]
    """
    name: str

    def __init__(self, ctx: StrategyContext) -> None:
        self.ctx = ctx

    @abstractmethod
    def on_tick(self, snap: MarketSnapshot, view: EngineView) -> list[OrderIntent]: ...

    def on_fill(self, fill: FillEvent, view: EngineView) -> None:  # noqa: B027
        """Default no-op. Strategies override to react to fills."""

    def on_end(self, view: EngineView) -> None:  # noqa: B027
        """Default no-op."""
