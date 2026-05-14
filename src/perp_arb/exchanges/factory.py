"""Build BaseExchange adapters from AppCfg."""

from __future__ import annotations

from ..core.config import AppCfg, RunMode
from ..core.exchange import BaseExchange
from .aster.client import AsterClient
from .aster.signer import AsterSigner
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


def build_exchanges(cfg: AppCfg) -> dict[str, BaseExchange]:
    return {
        "aster": build_aster(cfg),
        "lighter": build_lighter(cfg),
    }
