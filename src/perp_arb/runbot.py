"""CLI entrypoint. Wires AppCfg -> exchanges -> strategy -> asyncio.run.

Usage:
    uv run runbot --config configs/taker_taker_eth.yaml [--mode paper|live]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys

from dotenv import load_dotenv

from .core.config import AppCfg, RunMode, load_app_config, require_live_creds
from .core.exchange import BaseExchange
from .core.logging import setup_logging
from .core.types import MarketInfo
from .exchanges.factory import build_exchanges, required_legs
from .strategy.base import BaseStrategy
from .strategy.spread_monitor import SpreadMonitor
from .strategy.taker_taker import TakerTakerArbitrage

_log = logging.getLogger("perp_arb.runbot")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="runbot", description="perp-arb")
    p.add_argument("--config", required=True, help="Path to strategy YAML config")
    p.add_argument(
        "--mode",
        choices=[m.value for m in RunMode],
        default=None,
        help="Override `mode:` in the YAML config",
    )
    p.add_argument("--env-file", default=".env", help="dotenv file to load")
    return p.parse_args()


def _build_strategy(
    cfg: AppCfg,
    exchanges: dict[str, BaseExchange],
    markets: dict[str, MarketInfo],
) -> BaseStrategy:
    name = cfg.strategy.strategy
    if name == "spread_monitor":
        return SpreadMonitor(cfg, exchanges, markets)
    if name == "taker_taker":
        return TakerTakerArbitrage(cfg, exchanges, markets)
    raise ValueError(f"unknown strategy: {name!r}")


async def _async_main(cfg: AppCfg) -> int:
    setup_logging(cfg.runtime.log_dir, cfg.runtime.log_level, run_tag=cfg.strategy.strategy)
    _log.info(
        "starting strategy=%s mode=%s pair=%s qty=%s max_qty=%s",
        cfg.strategy.strategy, cfg.strategy.mode, cfg.strategy.pair.base,
        cfg.strategy.qty, cfg.strategy.max_qty,
    )

    if cfg.strategy.mode is RunMode.LIVE:
        require_live_creds(cfg)

    exchanges = build_exchanges(cfg)
    try:
        await asyncio.gather(*(ex.connect() for ex in exchanges.values()))

        needed = required_legs(cfg)  # leg label -> native symbol
        loaded = await asyncio.gather(
            *(exchanges[leg].load_market(sym) for leg, sym in needed.items())
        )
        markets = dict(zip(needed.keys(), loaded, strict=True))
        for leg, m in markets.items():
            _log.info(
                "market resolved: %s (%s)=%s tick=%s lot=%s",
                leg, exchanges[leg].name, m.symbol.raw, m.tick_size, m.lot_size,
            )

        strategy = _build_strategy(cfg, exchanges, markets)

        loop = asyncio.get_running_loop()
        stop_evt = asyncio.Event()

        def _on_signal(sig: int) -> None:
            _log.warning("signal %d received; shutting down", sig)
            stop_evt.set()

        # SIGHUP too: on a VPS a dropped SSH (if the job wasn't disowned) would
        # otherwise kill the process abruptly and skip the writer's clean close.
        sigs = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGHUP"):
            sigs.append(signal.SIGHUP)
        for sig in sigs:
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, _on_signal, sig)

        strat_task = asyncio.create_task(strategy.run(), name="strategy")
        await stop_evt.wait()
        await strategy.stop()
        strat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await strat_task
    finally:
        for ex in exchanges.values():
            with contextlib.suppress(Exception):
                await ex.disconnect()
        _log.info("clean shutdown complete")
    return 0


def main() -> None:
    args = _parse_args()
    load_dotenv(args.env_file)   # silently no-ops if file is missing
    mode_override = RunMode(args.mode) if args.mode else None
    cfg = load_app_config(args.config, mode_override=mode_override)
    sys.exit(asyncio.run(_async_main(cfg)))


if __name__ == "__main__":
    main()
