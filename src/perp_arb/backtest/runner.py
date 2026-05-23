"""Glue: load capture, instantiate strategy, run engine, write summary.

Used by both the `runbt` CLI and the test suite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from ..core.exec_record import ExecutionRecorder
from ..strategy.persistence_gate import PersistenceParams
from .base import StrategyContext
from .dataset import load_capture
from .engine import Engine, EngineConfig, EngineSummary, write_summary
from .strategies import build_strategy

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StrategyParams:
    """Subset of live `StrategyCfg` the backtest cares about. Parsed from
    YAML or set directly by tests."""
    qty: Decimal
    fees_bps: Decimal
    min_profit_bps: Decimal
    max_stale_ms: int
    bias_halflife_s: float
    scale_halflife_s: float
    warmup_seconds: float
    max_qty: Decimal
    # Optional Wave-1 knobs. Default = legacy behaviour.
    markout_table_path: Path | None = None
    inventory_skew_bps: Decimal = Decimal(0)
    throttle_bump_bps: Decimal = Decimal(0)
    throttle_halflife_s: float = 3.0
    in_flight_cap_per_direction: int = 0       # 0 = unlimited (current behaviour)
    persistence: PersistenceParams = PersistenceParams()   # edge-persistence gate


def build_context(
    params: StrategyParams,
    cfg: EngineConfig,
    left_venue: str,
    right_venue: str,
    recorder: ExecutionRecorder,
) -> StrategyContext:
    if params.qty != cfg.capture_qty:
        raise ValueError(
            f"strategy qty ({params.qty}) must equal capture_qty ({cfg.capture_qty}) — "
            f"VwapFill is strict; either re-capture at the new qty or change --capture-qty."
        )
    return StrategyContext(
        capture_qty=cfg.capture_qty,
        fees_bps=params.fees_bps,
        min_profit_bps=params.min_profit_bps,
        max_stale_ms=params.max_stale_ms,
        bias_halflife_s=params.bias_halflife_s,
        scale_halflife_s=params.scale_halflife_s,
        warmup_seconds=params.warmup_seconds,
        max_qty=params.max_qty,
        left_venue=left_venue,
        right_venue=right_venue,
        fill_model=cfg.fill_model,
        recorder=recorder,
        markout_table_path=params.markout_table_path,
        inventory_skew_bps=params.inventory_skew_bps,
        throttle_bump_bps=params.throttle_bump_bps,
        throttle_halflife_s=params.throttle_halflife_s,
        in_flight_cap_per_direction=params.in_flight_cap_per_direction,
        persistence=params.persistence,
    )


def run_backtest(cfg: EngineConfig, params: StrategyParams) -> EngineSummary:
    """End-to-end: load → run → persist. Returns the summary."""
    rows = load_capture(cfg.data_root)
    left_venue = rows[0].left_venue
    right_venue = rows[0].right_venue
    # consistency check: venue names should not change across the capture
    for r in rows:
        if r.left_venue != left_venue or r.right_venue != right_venue:
            raise RuntimeError(
                f"venue names changed mid-capture: expected ({left_venue}, {right_venue}) "
                f"got ({r.left_venue}, {r.right_venue}) at ts={r.ts_ms}"
            )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    recorder = ExecutionRecorder(cfg.out_dir, run_ts=run_ts, strategy_id=cfg.strategy_id)
    try:
        ctx = build_context(params, cfg, left_venue, right_venue, recorder)
        strategy = build_strategy(cfg.strategy_id, ctx)
        engine = Engine(rows, strategy, cfg, ctx)
        _log.info(
            "backtest start: rows=%d venues=(%s,%s) qty=%s fill=%s strategy=%s",
            len(rows), left_venue, right_venue, cfg.capture_qty,
            cfg.fill_model, cfg.strategy_id,
        )
        summary = engine.run(recorder)
    finally:
        recorder.close()
    write_summary(summary, cfg.out_dir / "summary.json")
    _log.info("%s", summary.pretty())
    return summary
