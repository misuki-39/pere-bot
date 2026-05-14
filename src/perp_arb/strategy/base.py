"""Strategy base class + shared bookkeeping helpers."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from decimal import Decimal

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


class EwmaTracker:
    """Single-variable EWMA: stable, monotone, no buffer to manage.

    alpha = 2 / (window + 1)  — same convention as pandas / TradingView.
    """

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.alpha = Decimal(2) / Decimal(window + 1)
        self.value: Decimal | None = None
        self.samples = 0

    def update(self, x: Decimal) -> Decimal:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (Decimal(1) - self.alpha) * self.value
        self.samples += 1
        return self.value

    @property
    def is_warm(self) -> bool:
        # Conservative: require at least one full window of samples.
        return self.samples >= self.window
