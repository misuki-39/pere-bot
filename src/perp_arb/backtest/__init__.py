"""Event-driven backtest framework for cross-venue perp arbitrage strategies.

Replays Parquet captures (one row per tick, both venues' BBO + capture-qty
VWAP, per-leg source timestamps) and simulates order placement with per-venue
submit-delay. Strategies are pure Python — no asyncio, no exchange clients —
and produce the same `Decision`/`LegOutcome` output as live so the existing
analyze_exec.py script works on either.
"""

from __future__ import annotations

from .base import BacktestStrategy, EngineView, StrategyContext
from .dataset import BBORow, load_capture
from .engine import Engine, EngineConfig, EngineSummary
from .fills import BBOFill, FillModel, FillModelKind, VwapFill
from .intents import FillEvent, OrderIntent
from .latency import BookIndex, LatencyModel
from .snapshot import MarketSnapshot

__all__ = [
    "BBOFill",
    "BBORow",
    "BacktestStrategy",
    "BookIndex",
    "Engine",
    "EngineConfig",
    "EngineSummary",
    "EngineView",
    "FillEvent",
    "FillModel",
    "FillModelKind",
    "LatencyModel",
    "MarketSnapshot",
    "OrderIntent",
    "StrategyContext",
    "VwapFill",
    "load_capture",
]
