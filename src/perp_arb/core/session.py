"""Strategy runtime session: collapses live-vs-paper divergence.

A `Session` is injected into every strategy at construction. It answers
the three questions whose answer differs between live and paper runs:

  * `is_paper` — selects synth vs real fills inside the executor
  * `preflight(cfg)` — runs before `build_exchanges` opens sockets; live
    validates creds, paper is a no-op. Both also assert that the cfg's
    declared `mode` matches the session class — i.e. the session and
    the rest of the wiring (factory, executor) agree on which mode the
    bot is actually running.
  * `snapshot_position(exchange, market)` — fetches the current venue
    position so the bot's local tracker reflects real inventory on
    startup (paper has no inventory; returns 0)

This is the single boundary between mode and strategy code. The strategy
never inspects `RunMode` directly — it just calls into its session.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from .config import AppCfg, RunMode, require_live_creds
from .exchange import BaseExchange
from .types import MarketInfo


class Session(ABC):
    """Bundle of strategy-runtime concerns that diverge by mode."""

    @property
    @abstractmethod
    def is_paper(self) -> bool:
        """True for paper, False for live. Enforced as abstract so a new
        subclass that forgets to declare it can't be instantiated."""

    @abstractmethod
    async def preflight(self, cfg: AppCfg) -> None:
        """Runs before `build_exchanges`. Live validates that credentials
        are real; paper is a no-op. Both assert the cfg mode matches the
        session class. Raise to abort startup before any sockets open."""

    @abstractmethod
    async def snapshot_position(
        self, exchange: BaseExchange, market: MarketInfo,
    ) -> Decimal:
        """Signed current position size for (exchange, market). Long > 0,
        short < 0. Paper has no real inventory: returns Decimal(0)."""


class LiveSession(Session):
    # Class-var override clears the abstract `is_paper` property — CPython
    # accepts any attribute (incl. plain value) as satisfying an abstract
    # property slot on subclasses.
    is_paper = False

    async def preflight(self, cfg: AppCfg) -> None:
        if cfg.strategy.mode is not RunMode.LIVE:
            raise ValueError(
                f"LiveSession requires cfg.strategy.mode=LIVE, "
                f"got {cfg.strategy.mode.value!r}"
            )
        require_live_creds(cfg)

    async def snapshot_position(
        self, exchange: BaseExchange, market: MarketInfo,
    ) -> Decimal:
        pos = await exchange.get_position(market)
        return pos.size


class PaperSession(Session):
    is_paper = True

    async def preflight(self, cfg: AppCfg) -> None:
        if cfg.strategy.mode is not RunMode.PAPER:
            raise ValueError(
                f"PaperSession requires cfg.strategy.mode=PAPER, "
                f"got {cfg.strategy.mode.value!r}"
            )

    async def snapshot_position(
        self, exchange: BaseExchange, market: MarketInfo,
    ) -> Decimal:
        return Decimal(0)
