"""Build BaseExchange adapters from AppCfg."""

from __future__ import annotations

from collections.abc import Callable

from ..core.config import AppCfg, RunMode
from ..core.exchange import BaseExchange
from .aster.client import AsterClient
from .aster.signer import AsterSigner
from .katana.client import KatanaClient
from .lighter.client import LighterClient


def build_aster(cfg: AppCfg) -> AsterClient:
    # Anything other than LIVE runs as public-only — no signer, no order ops.
    public_only = cfg.strategy.mode is not RunMode.LIVE
    signer = None
    if not public_only:
        signer = AsterSigner(
            user=cfg.aster.user,
            signer=cfg.aster.signer,
            signer_privkey=cfg.aster.signer_privkey,
            chain_id=cfg.aster.chain_id,
        )
    return AsterClient(
        rest_url=cfg.aster.rest_url,
        ws_url=cfg.aster.ws_url,
        signer=signer,
        public_only=public_only,
    )


def build_lighter(cfg: AppCfg) -> LighterClient:
    public_only = cfg.strategy.mode is not RunMode.LIVE
    return LighterClient(
        base_url=cfg.lighter.base_url,
        api_key_private_key=None if public_only else cfg.lighter.api_key_private_key,
        account_index=cfg.lighter.account_index,
        api_key_index=cfg.lighter.api_key_index,
        public_only=public_only,
    )


def build_katana(cfg: AppCfg) -> KatanaClient:
    # Public market data only this phase — no creds, always public_only.
    return KatanaClient(
        base_url=cfg.katana.base_url,
        ws_url=cfg.katana.ws_url,
        public_only=True,
    )


_BUILDERS: dict[str, Callable[[AppCfg], BaseExchange]] = {
    "aster": build_aster,
    "lighter": build_lighter,
    "katana": build_katana,
}


def required_venues(cfg: AppCfg) -> dict[str, str]:
    """Map venue name -> venue-native symbol for the venues this run needs.

    `taker_taker` always needs aster + lighter. `spread_monitor` uses
    `monitor_pair` if set, else defaults to the same aster/lighter pair.
    """
    s = cfg.strategy
    if s.strategy == "spread_monitor" and s.monitor_pair is not None:
        left, right = s.monitor_pair
        return {left.venue: left.symbol, right.venue: right.symbol}
    return {"aster": s.pair.aster_symbol, "lighter": s.pair.lighter_symbol}


def build_exchanges(cfg: AppCfg) -> dict[str, BaseExchange]:
    return {name: _BUILDERS[name](cfg) for name in required_venues(cfg)}
