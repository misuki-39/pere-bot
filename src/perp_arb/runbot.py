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

from .core.config import AppCfg, RunMode, load_app_config
from .core.exchange import BaseExchange
from .core.logging import setup_logging
from .core.session import LiveSession, PaperSession, Session
from .core.types import MarketInfo
from .exchanges.factory import build_exchanges, required_legs
from .exchanges.lighter.client import LighterClient
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


_STRATEGIES: dict[str, type[BaseStrategy]] = {
    SpreadMonitor.name: SpreadMonitor,
    TakerTakerArbitrage.name: TakerTakerArbitrage,
}


def _build_strategy(
    cfg: AppCfg,
    exchanges: dict[str, BaseExchange],
    markets: dict[str, MarketInfo],
    session: Session,
) -> BaseStrategy:
    try:
        cls = _STRATEGIES[cfg.strategy.strategy]
    except KeyError:
        known = ", ".join(sorted(_STRATEGIES))
        raise ValueError(
            f"unknown strategy {cfg.strategy.strategy!r}. known: {known}"
        ) from None
    return cls(cfg, exchanges, markets, session)


async def _maybe_enable_lighter_presign_pool(
    cfg: AppCfg,
    exchanges: dict[str, BaseExchange],
    markets: dict[str, MarketInfo],
) -> None:
    """If the optimisation is enabled in config AND the strategy actually
    binds a lighter leg, spin up the pre-signed pool on that LighterClient
    after its market is loaded. No-op for spread_monitor / non-lighter pairs
    / paper sessions running without a signer.
    """
    pool_cfg = cfg.strategy.optimisations.lighter_presign_pool
    if not pool_cfg.enabled:
        return
    for leg, ex in exchanges.items():
        if not isinstance(ex, LighterClient):
            continue
        # public_only LighterClient (paper without creds) has no signer —
        # the pool would have nothing to sign with. Skip silently.
        if ex.public_only:
            _log.info("lighter presign pool: skipped — leg=%s is public_only", leg)
            continue
        await ex.enable_presign_pool(
            markets[leg],
            qty=cfg.strategy.qty,
            refresh_interval_s=pool_cfg.refresh_interval_s,
            drift_threshold_bps=pool_cfg.drift_threshold_bps,
        )
        _log.info(
            "lighter presign pool: enabled leg=%s refresh=%.0fs drift_threshold=%sbps",
            leg, pool_cfg.refresh_interval_s, pool_cfg.drift_threshold_bps,
        )


async def _async_main(cfg: AppCfg) -> int:
    setup_logging(cfg.runtime.log_dir, cfg.runtime.log_level, run_tag=cfg.strategy.strategy)
    _log.info(
        "starting strategy=%s mode=%s pair=%s qty=%s max_qty=%s",
        cfg.strategy.strategy, cfg.strategy.mode, cfg.strategy.pair.base,
        cfg.strategy.qty, cfg.strategy.max_qty,
    )

    # Single mode dispatch: every other strategy-runtime divergence
    # (creds preflight, executor synth, position seeding) is answered by
    # the session, not by inspecting `mode` again.
    session: Session = (
        LiveSession() if cfg.strategy.mode is RunMode.LIVE else PaperSession()
    )
    await session.preflight(cfg)

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

        await _maybe_enable_lighter_presign_pool(cfg, exchanges, markets)

        strategy = _build_strategy(cfg, exchanges, markets, session)

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
        # A crash in strategy.run() (e.g. snapshot_position raising at startup)
        # must unblock the main wait so the exception surfaces — without this
        # callback, stop_evt is only set by a signal and the bot would hang
        # silently on a dead task.
        strat_task.add_done_callback(lambda _t: stop_evt.set())
        try:
            await stop_evt.wait()
        finally:
            await strategy.stop()
            if not strat_task.done():
                strat_task.cancel()
            # CancelledError is the normal shutdown path; any real exception
            # re-raises out of _async_main → asyncio.run → non-zero exit.
            with contextlib.suppress(asyncio.CancelledError):
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
