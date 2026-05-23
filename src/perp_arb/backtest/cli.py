"""`runbt` console script.

Loads a config (reusing live's strategy YAML schema), parses backtest-only
CLI flags (data dir, latency, fill model, capture qty), runs the backtest.
"""

from __future__ import annotations

import argparse
import logging
from decimal import Decimal
from pathlib import Path

import yaml

from ..strategy.persistence_gate import PersistenceParams
from .engine import EngineConfig
from .fills import FillModelKind
from .latency import LatencyModel
from .runner import StrategyParams, run_backtest


def _parse_delays(s: str) -> dict[str, int]:
    """Parse 'venueA=ms,venueB=ms' into a dict."""
    if not s:
        return {}
    out: dict[str, int] = {}
    for kv in s.split(","):
        k, v = kv.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def _load_params(config_path: Path, override_qty: Decimal | None) -> StrategyParams:
    """Reuse live's YAML schema (subset of fields we need).

    The Wave-1 optimisation knobs are read from an optional `optimisations:`
    block; legacy configs that omit it get the default-off behaviour.
    """
    raw = yaml.safe_load(config_path.read_text())
    qty = override_qty if override_qty is not None else Decimal(str(raw["qty"]))
    opt = raw.get("optimisations", {}) or {}
    markout_path = opt.get("markout_table_path")
    pc = opt.get("persistence_confirm", {}) or {}
    persistence = PersistenceParams(
        enabled=bool(pc.get("enabled", False)),
        t_confirm_ms=int(pc.get("t_confirm_ms", 400)),
        n_confirm=int(pc.get("n_confirm", 6)),
        drift_max_bps=Decimal(str(pc.get("drift_max_bps", "1.0"))),
    )
    return StrategyParams(
        qty=qty,
        fees_bps=Decimal(str(raw.get("fees_bps", 0))),
        min_profit_bps=Decimal(str(raw.get("min_profit_bps", 0))),
        max_stale_ms=int(raw.get("max_stale_ms", 200)),
        bias_halflife_s=float(raw.get("bias_halflife_s", 3600)),
        scale_halflife_s=float(raw.get("scale_halflife_s", 300)),
        warmup_seconds=float(raw.get("warmup_seconds", 180)),
        max_qty=Decimal(str(raw.get("max_qty", raw["qty"] * 100))),
        markout_table_path=Path(markout_path) if markout_path else None,
        inventory_skew_bps=Decimal(str(opt.get("inventory_skew_bps", 0))),
        throttle_bump_bps=Decimal(str(opt.get("throttle_bump_bps", 0))),
        throttle_halflife_s=float(opt.get("throttle_halflife_s", 3.0)),
        in_flight_cap_per_direction=int(opt.get("in_flight_cap_per_direction", 0)),
        persistence=persistence,
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="runbt",
                                description="Run a backtest against captured BBO+VWAP Parquet.")
    p.add_argument("--data", required=True, type=Path,
                   help="Capture root dir (Hive-partitioned spread_*).")
    p.add_argument("--strategy", required=True,
                   help="Strategy id (e.g. taker_taker).")
    p.add_argument("--config", required=True, type=Path,
                   help="YAML config (subset of live's strategy schema).")
    p.add_argument("--out", required=True, type=Path,
                   help="Output dir for decisions/legs CSVs + summary.json.")
    p.add_argument("--capture-qty", required=True, type=lambda s: Decimal(s),
                   help="The qty the recorder used when pre-computing VWAP columns.")
    p.add_argument("--submit-delay", default="",
                   help="Per-venue submit delay in ms, e.g. 'aster=120,lighter=40'.")
    p.add_argument("--fill-model", default="vwap", choices=["bbo", "vwap"],
                   help="Fill model applied to both legs (v1).")
    p.add_argument("--fee-bps", type=lambda s: Decimal(s), default=None,
                   help="Override per-leg fee bps (default: config 'fees_bps' / 2).")
    p.add_argument("--qty", type=lambda s: Decimal(s), default=None,
                   help="Override strategy qty (must equal --capture-qty).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s.%(msecs)03d %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    params = _load_params(args.config, override_qty=args.qty)
    fee_bps_per_leg = args.fee_bps if args.fee_bps is not None else params.fees_bps / Decimal(2)
    cfg = EngineConfig(
        data_root=args.data,
        out_dir=args.out,
        capture_qty=args.capture_qty,
        latency=LatencyModel(submit_delay_ms=_parse_delays(args.submit_delay)),
        fill_model=FillModelKind(args.fill_model),
        fee_bps_per_leg=fee_bps_per_leg,
        strategy_id=args.strategy,
    )
    run_backtest(cfg, params)
