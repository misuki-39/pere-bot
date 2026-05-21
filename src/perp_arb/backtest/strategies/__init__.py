"""Strategy registry. CLI passes `--strategy <id>`; we map it to a factory."""

from __future__ import annotations

from collections.abc import Callable

from ..strategy import BacktestStrategy, StrategyContext
from .taker_taker_bt import TakerTakerBT

_REGISTRY: dict[str, Callable[[StrategyContext], BacktestStrategy]] = {
    TakerTakerBT.name: TakerTakerBT,
}


def build_strategy(strategy_id: str, ctx: StrategyContext) -> BacktestStrategy:
    if strategy_id not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown strategy '{strategy_id}'. known: {known}")
    return _REGISTRY[strategy_id](ctx)


__all__ = ["TakerTakerBT", "build_strategy"]
