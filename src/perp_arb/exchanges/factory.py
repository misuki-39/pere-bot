"""Build BaseExchange adapters from AppCfg."""

from __future__ import annotations

from collections.abc import Callable

from ..core.config import AppCfg, RunMode, VenueLeg
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


def _bound_legs(cfg: AppCfg) -> tuple[VenueLeg, VenueLeg]:
    s = cfg.strategy
    if s.strategy == "spread_monitor" and s.monitor_pair is not None:
        return s.monitor_pair
    return s.pair.leg_a, s.pair.leg_b


def required_legs(cfg: AppCfg) -> dict[str, str]:
    """Map leg label ("leg_a"/"leg_b") -> venue-native symbol to load.

    The strategy refers to its two venues only as leg_a / leg_b; this
    function and `build_exchanges` are the two boundary points that turn
    that abstract binding into concrete venue+symbol pairs.
    """
    a, b = _bound_legs(cfg)
    return {"leg_a": a.symbol, "leg_b": b.symbol}


def build_exchanges(cfg: AppCfg) -> dict[str, BaseExchange]:
    """Build `{leg_a, leg_b} -> BaseExchange`. Each leg is wired to the
    venue driver named by config (`PairCfg.leg_a.venue` etc)."""
    a, b = _bound_legs(cfg)
    return {"leg_a": _BUILDERS[a.venue](cfg), "leg_b": _BUILDERS[b.venue](cfg)}
