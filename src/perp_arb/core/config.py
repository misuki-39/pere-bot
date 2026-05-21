"""Pydantic-based config schema. Loaded from YAML + .env at startup."""

from __future__ import annotations

import os
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class RunMode(StrEnum):
    PAPER = "paper"     # full decision math, orders are no-ops with synthetic fills
    LIVE = "live"       # real orders


_PLACEHOLDER_ADDR = "0x" + "0" * 40
_PLACEHOLDER_PK = "0x" + "0" * 64


class PairCfg(BaseModel):
    base: str
    aster_symbol: str
    lighter_symbol: str


class VenueLeg(BaseModel):
    """One side of a generic spread-monitor pair."""

    venue: str   # "aster" | "lighter" | "katana"
    symbol: str  # venue-native ticker, e.g. "ETH-USD" on katana


class RiskCfg(BaseModel):
    max_consecutive_failures: int = 3
    max_leg_latency_ms: int = 500
    daily_loss_cap_usd: Decimal = Decimal("50")
    min_free_margin_usd: Decimal = Decimal("0")


class OptimisationsCfg(BaseModel):
    """Wave-1 optional knobs. All default-off = identical pre-Wave-1 behaviour.

    Live `taker_taker` consumes these via `taker_taker.__init__`. The pure
    decision function `assess_taker_taker` already accepts the corresponding
    `AssessParams` / `AssessInputs` fields, so the live wiring is just
    plumbing.

    `inventory_skew_bps` is intentionally NOT exposed here — the backtest
    sweep showed it is alone-bad / combined-good with markout; rolling
    markout calibration must stabilise first.
    """
    # Per-(direction, edge-bucket) adverse-selection table built offline by
    # `scripts/build_markout_table.py`. None ⇒ MarkoutTable.disabled().
    # Path is relative to the cwd at runbot startup.
    markout_table_path: Path | None = None

    # Same-direction threshold throttle. After each FILLED on direction X,
    # raise X's threshold by `throttle_bump_bps`; decay back over half-life
    # `throttle_halflife_s`. bump=0 disables.
    throttle_bump_bps: Decimal = Decimal("0")
    throttle_halflife_s: float = Field(default=3.0, gt=0)

    # Per-direction in-flight cap. K=0 disables. K=1 = at most one outstanding
    # entry of that direction at a time. Note: live's `_evaluating` gate in
    # `taker_taker._schedule_eval` already serializes evaluation, so this is
    # in practice a no-op in live; kept for parity with the backtest path and
    # as a forward-safety belt if the gate is ever relaxed.
    in_flight_cap_per_direction: int = Field(default=0, ge=0)


class StrategyCfg(BaseModel):
    strategy: Literal["spread_monitor", "taker_taker"]
    mode: RunMode = RunMode.PAPER
    pair: PairCfg
    # spread_monitor only. Absent ⇒ default to the aster/lighter pair above so
    # existing configs keep working. (left, right).
    monitor_pair: tuple[VenueLeg, VenueLeg] | None = None

    # sizing
    qty: Decimal = Field(gt=0)
    max_qty: Decimal = Field(gt=0)

    # entry math
    fees_bps: Decimal = Decimal("6")
    min_profit_bps: Decimal = Decimal("3")
    # Spread center ("bias") is a wall-clock half-life EWMA. It must be far
    # slower than the ~2 s residual reversion so it tracks the slow intraday
    # center without eating the tradeable signal — hours-scale by default.
    bias_halflife_s: float = Field(default=3600.0, gt=0)
    # Residual dispersion half-life (diagnostic only, never the entry gate).
    # Minutes-scale.
    scale_halflife_s: float = Field(default=300.0, gt=0)
    warmup_seconds: int = 180

    # depth / staleness gates
    max_levels: int = 3
    max_slippage_bps: Decimal = Decimal("5")
    max_stale_ms: int = 200

    risk: RiskCfg = RiskCfg()
    optimisations: OptimisationsCfg = OptimisationsCfg()

    @model_validator(mode="after")
    def _validate_sizes(self) -> StrategyCfg:
        if self.max_qty < self.qty:
            raise ValueError("max_qty must be >= qty")
        # Fail fast if a markout table path is configured but missing — easier
        # to debug at startup than at the first FIRED tick.
        mp = self.optimisations.markout_table_path
        if mp is not None and not Path(mp).exists():
            raise ValueError(
                f"optimisations.markout_table_path does not exist: {mp}"
            )
        return self


class AsterCreds(BaseModel):
    user: str                  # main account address (0x...)
    signer: str                # API wallet address
    signer_privkey: str        # API wallet private key (hex)
    rest_url: str = "https://fapi.asterdex.com"
    ws_url: str = "wss://fstream.asterdex.com"
    chain_id: int = 1666       # 1666 mainnet, 714 testnet (EIP-712 domain)

    @property
    def is_placeholder(self) -> bool:
        return (
            self.user == _PLACEHOLDER_ADDR
            or self.signer == _PLACEHOLDER_ADDR
            or self.signer_privkey == _PLACEHOLDER_PK
        )


class LighterCreds(BaseModel):
    api_key_private_key: str
    account_index: int = 0
    api_key_index: int = 0
    base_url: str = "https://mainnet.zklighter.elliot.ai"

    @property
    def is_placeholder(self) -> bool:
        return self.api_key_private_key == _PLACEHOLDER_PK


class KatanaCreds(BaseModel):
    # Public market data only — no API key required for /v1/markets, /v1/orderbook
    # or the l2orderbook WS channel.
    base_url: str = "https://api-perps.katana.network"
    ws_url: str = "wss://websocket-perps.katana.network/v1"


class RuntimeCfg(BaseModel):
    log_dir: Path = Path("./logs")
    log_level: str = "INFO"


class AppCfg(BaseModel):
    strategy: StrategyCfg
    aster: AsterCreds
    lighter: LighterCreds
    katana: KatanaCreds = KatanaCreds()
    runtime: RuntimeCfg = RuntimeCfg()


class MissingCredsError(RuntimeError):
    """Raised when LIVE mode is requested but credentials are placeholders."""


def load_app_config(yaml_path: str | Path, *, mode_override: RunMode | None = None) -> AppCfg:
    """Load YAML + env. Credentials are always optional at load time — they fall
    back to placeholders so paper / spread_monitor can run without keys.

    For LIVE mode, call `require_live_creds(cfg)` to validate before connecting.
    """

    yaml_path = Path(yaml_path)
    raw = yaml.safe_load(yaml_path.read_text())

    strategy_cfg = StrategyCfg.model_validate(raw)
    if mode_override is not None:
        strategy_cfg = strategy_cfg.model_copy(update={"mode": mode_override})

    aster = AsterCreds(
        user=os.getenv("ASTER_USER") or _PLACEHOLDER_ADDR,
        signer=os.getenv("ASTER_SIGNER") or _PLACEHOLDER_ADDR,
        signer_privkey=os.getenv("ASTER_SIGNER_PRIVKEY") or _PLACEHOLDER_PK,
        rest_url=os.getenv("ASTER_REST_URL") or "https://fapi.asterdex.com",
        ws_url=os.getenv("ASTER_WS_URL") or "wss://fstream.asterdex.com",
        chain_id=int(os.getenv("ASTER_CHAIN_ID") or "1666"),
    )
    lighter = LighterCreds(
        api_key_private_key=os.getenv("LIGHTER_API_KEY_PRIVATE_KEY") or _PLACEHOLDER_PK,
        account_index=int(os.getenv("LIGHTER_ACCOUNT_INDEX") or "0"),
        api_key_index=int(os.getenv("LIGHTER_API_KEY_INDEX") or "0"),
        base_url=os.getenv("LIGHTER_BASE_URL") or "https://mainnet.zklighter.elliot.ai",
    )
    katana = KatanaCreds(
        base_url=os.getenv("KATANA_BASE_URL") or "https://api-perps.katana.network",
        ws_url=os.getenv("KATANA_WS_URL") or "wss://websocket-perps.katana.network/v1",
    )
    runtime = RuntimeCfg(
        log_dir=Path(os.getenv("LOG_DIR") or "./logs"),
        log_level=os.getenv("LOG_LEVEL") or "INFO",
    )

    return AppCfg(
        strategy=strategy_cfg, aster=aster, lighter=lighter, katana=katana, runtime=runtime,
    )


def require_live_creds(cfg: AppCfg) -> None:
    """Raise MissingCredsError if any credential is still a placeholder.

    Called only when mode == LIVE; paper / spread_monitor tolerate placeholders.
    """
    missing: list[str] = []
    if cfg.aster.is_placeholder:
        missing.append("aster")
    if cfg.lighter.is_placeholder:
        missing.append("lighter")
    if missing:
        raise MissingCredsError(
            f"LIVE mode requires real credentials for: {', '.join(missing)}. "
            f"Set them in .env (see .env.example)."
        )
