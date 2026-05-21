"""BacktestStrategy ABC + supporting context types.

Strategies are *pure synchronous*: `on_tick(snapshot, view)` returns a list
of `OrderIntent`s (possibly empty), and `on_fill(fill, view)` is a no-op
hook for strategies that want to react to resolution. No asyncio, no
exchange clients.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal

from ..core.exec_record import ExecutionRecorder
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
    max_slippage_bps: Decimal
    max_stale_ms: int
    bias_halflife_s: float
    scale_halflife_s: float
    warmup_seconds: float
    max_qty: Decimal
    left_venue: str
    right_venue: str
    fill_model: FillModelKind
    recorder: ExecutionRecorder


@dataclass(slots=True)
class EngineView:
    """Read-only proxy over engine state for the strategy."""
    _positions: dict[str, Decimal] = field(default_factory=dict)
    _sim_now_ms: int = 0
    _pending_count: int = 0

    def position(self, venue: str) -> Decimal:
        return self._positions.get(venue, Decimal("0"))

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
