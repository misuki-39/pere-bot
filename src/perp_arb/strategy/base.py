"""Strategy base class + shared bookkeeping helpers."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

_LN2 = Decimal(2).ln()

from ..core.config import AppCfg
from ..core.exchange import BaseExchange
from ..core.types import MarketInfo

_log = logging.getLogger(__name__)


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

    def _aster(self) -> BaseExchange:
        return self.exchanges["aster"]

    def _lighter(self) -> BaseExchange:
        return self.exchanges["lighter"]

    def _aster_market(self) -> MarketInfo:
        return self.markets["aster"]

    def _lighter_market(self) -> MarketInfo:
        return self.markets["lighter"]


class TimeEwma:
    """EWMA whose decay is driven by wall-clock elapsed time, not sample count.

    Irregular sampling (the BBO stream runs 1.7-4.2 ticks/s with multi-second
    gaps) makes a fixed sample-count window a moving target in real time. A
    half-life in seconds is regime-invariant:

        alpha = 1 - exp(-ln2 * dt / half_life)

    so one half-life of elapsed time always discounts the old value by 50%,
    whatever the tick rate. Large gaps self-attenuate (a 26s gap at HL=1h
    moves the estimate <0.5%), which is exactly the robustness we want for a
    slow centre.
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


@dataclass(frozen=True)
class SpreadState:
    """One evaluation's view of the spread decomposed into the two timescales
    the strategy actually cares about."""

    center: Decimal      # slow inter-venue centre (the "bias")
    residual: Decimal    # spread - center: the fast, mean-reverting tradeable
    scale: Decimal       # running stdev of the residual (dispersion, diag/z)

    def residual_bps(self, ref: Decimal) -> Decimal:
        return self.residual / ref * Decimal(10_000)

    @property
    def zscore(self) -> Decimal:
        return self.residual / self.scale if self.scale > 0 else Decimal(0)


class SpreadModel:
    """Decomposes (mid_a - mid_l) into a slow centre + fast residual.

    The data says the spread is a slowly-wandering centre (hours, ~5-8 bps
    range, an intraday session effect) plus a strongly mean-reverting residual
    (AR(1) half-life ~2 s). The strategy trades the residual and bets on it
    reverting to the centre. Therefore:

      * centre half-life must be FAR slower than the ~2 s reversion, or the
        centre eats the very signal we trade (a fast EWMA flatters its own
        residual to near zero). Hours-scale is correct.
      * scale tracks residual dispersion on a minutes half-life. It is for
        diagnostics and an optional outlier clamp ONLY — never the entry gate:
        dispersion spikes during a dislocation burst, so z-score shrinks
        exactly when the absolute opportunity is largest. Volume-farming
        gating must stay on absolute bps.
    """

    def __init__(
        self,
        center_half_life_s: float,
        scale_half_life_s: float,
        warmup_s: float,
    ) -> None:
        self._center = TimeEwma(center_half_life_s)
        self._var = TimeEwma(scale_half_life_s)  # EWMA of residual**2
        self._warmup_ms = int(warmup_s * 1000)
        self._first_ts_ms: int | None = None
        self._last_ts_ms: int | None = None

    def update(self, spread: Decimal, ts_ms: int) -> SpreadState:
        if self._first_ts_ms is None:
            self._first_ts_ms = ts_ms
        self._last_ts_ms = ts_ms
        center = self._center.update(spread, ts_ms)
        residual = spread - center
        var = self._var.update(residual * residual, ts_ms)
        scale = var.sqrt() if var > 0 else Decimal(0)
        return SpreadState(center=center, residual=residual, scale=scale)

    @property
    def is_warm(self) -> bool:
        if self._first_ts_ms is None or self._last_ts_ms is None:
            return False
        return (self._last_ts_ms - self._first_ts_ms) >= self._warmup_ms
