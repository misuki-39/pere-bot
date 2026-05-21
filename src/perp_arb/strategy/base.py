"""Strategy base class + shared bookkeeping helpers."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

from ..core.config import AppCfg
from ..core.exchange import BaseExchange
from ..core.types import MarketInfo
from ..utils.precision import BPS

_log = logging.getLogger(__name__)

_LN2 = Decimal(2).ln()


class BaseStrategy(ABC):
    """Async run-to-completion strategy.

    Concrete strategies own their evaluation cadence — most subscribe to
    venue WS callbacks and run their loop on the asyncio event loop directly.
    """

    name: str

    def __init__(
        self,
        cfg: AppCfg,
        exchanges: dict[str, BaseExchange],
        markets: dict[str, MarketInfo],
    ) -> None:
        self.cfg = cfg
        self.exchanges = exchanges
        self.markets = markets
        self._stop = asyncio.Event()

    @abstractmethod
    async def run(self) -> None: ...

    async def stop(self) -> None:
        self._stop.set()

    def _venue(self, name: str) -> BaseExchange:
        return self.exchanges[name]

    def _market(self, name: str) -> MarketInfo:
        return self.markets[name]

    def _leg_a(self) -> BaseExchange:
        return self.exchanges["leg_a"]

    def _leg_b(self) -> BaseExchange:
        return self.exchanges["leg_b"]

    def _leg_a_market(self) -> MarketInfo:
        return self.markets["leg_a"]

    def _leg_b_market(self) -> MarketInfo:
        return self.markets["leg_b"]


class TimeEwma:
    """EWMA whose decay is driven by wall-clock elapsed time, not sample count.

    Irregular sampling (the BBO stream runs 1.7-4.2 ticks/s with multi-second
    gaps) makes a fixed sample-count window a moving target in real time. A
    half-life in seconds is regime-invariant:

        alpha = 1 - exp(-ln2 * dt / half_life)

    so one half-life of elapsed time always discounts the old value by 50%,
    whatever the tick rate. Large gaps self-attenuate (a 26s gap at HL=1h
    moves the estimate <0.5%), which is exactly the robustness we want for a
    slow center.
    """

    def __init__(self, half_life_s: float) -> None:
        if half_life_s <= 0:
            raise ValueError("half_life_s must be > 0")
        self.half_life_s = Decimal(str(half_life_s))
        self.value: Decimal | None = None
        self._last_ts_ms: int | None = None

    def update(self, x: Decimal, ts_ms: int) -> Decimal:
        if self.value is None or self._last_ts_ms is None:
            self.value = x
            self._last_ts_ms = ts_ms
            return x
        dt_s = Decimal(ts_ms - self._last_ts_ms) / Decimal(1000)
        self._last_ts_ms = ts_ms
        if dt_s <= 0:  # non-monotonic / duplicate-timestamp tick: ignore decay
            return self.value
        alpha = Decimal(1) - (-_LN2 * dt_s / self.half_life_s).exp()
        self.value = alpha * x + (Decimal(1) - alpha) * self.value
        return self.value

    def bump(self, delta: Decimal, ts_ms: int) -> None:
        """Add `delta` on top of the current value and stamp `ts_ms` as the
        decay anchor. Unlike `update`, this does not EWMA-blend — callers
        use it when they want an exact step (e.g. throttle seeding)."""
        current = self.value if self.value is not None else Decimal(0)
        self.value = current + delta
        self._last_ts_ms = ts_ms


@dataclass(frozen=True)
class SpreadState:
    """One evaluation's view of the spread decomposed into the two timescales
    the strategy actually cares about."""

    center: Decimal      # slow inter-venue center (the "bias")
    residual: Decimal    # spread - center: the fast, mean-reverting tradeable
    scale: Decimal       # running stdev of the residual (dispersion diagnostic)

    def residual_bps(self, ref: Decimal) -> Decimal:
        return self.residual / ref * BPS


class SpreadModel:
    """Decomposes (mid_left - mid_right) into a slow center + fast residual.

    The data says the spread is a slowly-wandering center (hours, ~5-8 bps
    range, an intraday session effect) plus a strongly mean-reverting residual
    (AR(1) half-life ~2 s). The strategy trades the residual and bets on it
    reverting to the center. Therefore:

      * center half-life must be FAR slower than the ~2 s reversion, or the
        center eats the very signal we trade (a fast EWMA flatters its own
        residual to near zero). Hours-scale is correct.
      * scale tracks residual dispersion on a minutes half-life — a dispersion
        diagnostic only, never the entry gate: it spikes during a dislocation
        burst, so a scale-relative measure would shrink exactly when the
        absolute opportunity is largest. Volume-farming gating stays on
        absolute bps.
    """

    def __init__(
        self,
        center_half_life_s: float,
        scale_half_life_s: float,
        warmup_s: float,
    ) -> None:
        self._center = TimeEwma(center_half_life_s)
        self._resid_sq = TimeEwma(scale_half_life_s)  # EWMA of residual**2
        self._warmup_ms = int(warmup_s * 1000)
        self._first_ts_ms: int | None = None
        self._last_ts_ms: int | None = None

    def update(self, spread: Decimal, ts_ms: int) -> SpreadState:
        if self._first_ts_ms is None:
            self._first_ts_ms = ts_ms
        self._last_ts_ms = ts_ms
        center = self._center.update(spread, ts_ms)
        residual = spread - center
        resid_sq = self._resid_sq.update(residual * residual, ts_ms)
        scale = resid_sq.sqrt() if resid_sq > 0 else Decimal(0)
        return SpreadState(center=center, residual=residual, scale=scale)

    @property
    def is_warm(self) -> bool:
        if self._first_ts_ms is None or self._last_ts_ms is None:
            return False
        return (self._last_ts_ms - self._first_ts_ms) >= self._warmup_ms
