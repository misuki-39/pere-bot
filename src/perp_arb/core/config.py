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


class VenueLeg(BaseModel):
    """One leg of a two-venue strategy. Names which venue driver to spin
    up and which native symbol to trade on it."""

    venue: str   # driver key — must be in factory `_BUILDERS` (aster|lighter|katana)
    symbol: str  # venue-native ticker, e.g. "ETHUSDT" on aster, "ETH-USD" on katana


class PairCfg(BaseModel):
    """The two venues a two-venue strategy is bound to. The strategy
    refers to them only as leg_a / leg_b; the actual venue mapping lives
    here in config (and in `monitor_pair` for spread_monitor)."""

    base: str
    leg_a: VenueLeg
    leg_b: VenueLeg


class RiskCfg(BaseModel):
    max_consecutive_failures: int = 3
    max_leg_latency_ms: int = 500
    daily_loss_cap_usd: Decimal = Decimal("50")
    min_free_margin_usd: Decimal = Decimal("0")
    # Soft cooldown applied after every `record_failure`. While
    # `cooldown_until_ms > now`, `can_trade()` blocks new fires; the
    # strategy's _reconcile_after_failure runs synchronously inside _fire's
    # failure branch BEFORE this cooldown is armed (see plan
    # agile-waddling-beacon). max_consecutive_failures remains the hard
    # backstop — repeated failures still ratchet toward halt.
    cooldown_s: float = Field(default=60.0, ge=0)


class PersistenceConfirmCfg(BaseModel):
    """Edge-persistence confirmation gate (2026-05-22 strategy search,
    "Strategy 3"). Suppresses a FIRED decision until its edge has survived
    `t_confirm_ms` across `n_confirm` venue updates with no adverse mid-drift.
    `enabled=False` ⇒ the gate is an identity pass-through."""

    enabled: bool = False
    t_confirm_ms: int = Field(default=400, ge=0)
    n_confirm: int = Field(default=6, ge=1)
    drift_max_bps: Decimal = Decimal("1.0")


class LighterPreSignPoolCfg(BaseModel):
    enabled: bool = False
    refresh_interval_s: float = Field(default=240.0, gt=0)
    drift_threshold_bps: Decimal = Field(default=Decimal("200"), gt=0)


class OptimisationsCfg(BaseModel):
    """Wave-1 optional knobs. All default-off = identical pre-Wave-1 behaviour.

    Live `taker_taker` consumes these via `taker_taker.__init__`. The pure
    decision function `assess_reversion` already accepts the corresponding
    `AssessParams` / `AssessInputs` fields, so the live wiring is just
    plumbing.

    `inventory_skew_bps` / `inventory_skew_close_bps` widen the entry
    threshold as |position| grows (κ_open) and narrow it as |position|
    shrinks (κ_close). Default 0 / None = off. See
    docs/inventory_skew_wti_2026-05-24.md for BT-derived recommendations.
    """
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

    # Edge-persistence confirmation gate. Disabled by default.
    persistence_confirm: PersistenceConfirmCfg = PersistenceConfirmCfg()

    # Avellaneda-Stoikov-style inventory skew (κ in bps per unit of
    # |position|/max_qty). κ_open raises the entry threshold when the trade
    # GROWS |position|; κ_close lowers it when the trade FLATTENS. κ_open=0
    # disables the widener entirely. κ_close=None recovers symmetric behaviour
    # (uses κ_open for both sides); =0 disables exit-easing while keeping
    # entry-tightening. BT-recommended on WTI: κ_open=15, κ_close=5
    # (regime-conditional: +22% on drift, slight loss on calm — capacity-bound
    # tool). See docs/inventory_skew_wti_2026-05-24.md.
    inventory_skew_bps: Decimal = Decimal("0")
    inventory_skew_close_bps: Decimal | None = None

    # Lighter pre-signed order pool. Pre-signs both BUY and SELL with the
    # same nonce, refreshes every `refresh_interval_s` OR when the remaining
    # slippage buffer falls below `drift_threshold_bps` (worst price is set
    # at ±5% of sign-time mid, i.e. 500 bps buffer; default trigger of 200
    # bps means refresh after mid drifts ~300 bps). Disabled by default.
    lighter_presign_pool: LighterPreSignPoolCfg = Field(
        default_factory=LighterPreSignPoolCfg,
    )


class StrategyCfg(BaseModel):
    strategy: Literal["spread_monitor", "taker_taker"]
    mode: RunMode = RunMode.PAPER
    pair: PairCfg
    # spread_monitor only. Absent ⇒ fall back to the `pair` legs above.
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
    max_stale_ms: int = 200

    risk: RiskCfg = RiskCfg()
    optimisations: OptimisationsCfg = OptimisationsCfg()

    @model_validator(mode="after")
    def _validate_sizes(self) -> StrategyCfg:
        if self.max_qty < self.qty:
            raise ValueError("max_qty must be >= qty")
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


class TursoCfg(BaseModel):
    """libSQL/Turso cloud-sync settings for the live SQLite recorder.

    The recorder always writes a local SQLite file (the on-box source of
    truth); when `enabled`, a background task replicates new rows to Turso.
    `url` / `auth_token` come from env (placeholders when absent) so paper
    runs load without a token. `enabled=False` = local-only, no network, no
    libsql dependency touched at runtime."""

    enabled: bool = False
    url: str = ""                       # libsql://<db>.turso.io  (from TURSO_DATABASE_URL)
    auth_token: str = ""                # from TURSO_AUTH_TOKEN
    db_path: Path = Path("./logs/live.db")   # local SQLite source of truth
    sync_interval_s: float = 5.0        # background push cadence
    sync_batch_rows: int = 500          # max rows per remote batch

    @property
    def is_placeholder(self) -> bool:
        return not self.url or not self.auth_token


class RuntimeCfg(BaseModel):
    log_dir: Path = Path("./logs")
    log_level: str = "INFO"
    turso: TursoCfg = TursoCfg()


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
    # Runtime knobs come from the YAML `runtime:` block (non-secret) merged
    # with env (secrets + dir overrides). Turso secrets are env-only so paper
    # configs stay credential-free.
    runtime_raw = raw.get("runtime") or {}
    turso_raw = runtime_raw.get("turso") or {}
    turso = TursoCfg(
        enabled=bool(turso_raw.get("enabled", False)),
        url=os.getenv("TURSO_DATABASE_URL") or "",
        auth_token=os.getenv("TURSO_AUTH_TOKEN") or "",
        db_path=Path(turso_raw.get("db_path") or os.getenv("LIVE_DB_PATH") or "./logs/live.db"),
        sync_interval_s=float(turso_raw.get("sync_interval_s", 5.0)),
        sync_batch_rows=int(turso_raw.get("sync_batch_rows", 500)),
    )
    runtime = RuntimeCfg(
        log_dir=Path(os.getenv("LOG_DIR") or "./logs"),
        log_level=os.getenv("LOG_LEVEL") or "INFO",
        turso=turso,
    )

    return AppCfg(
        strategy=strategy_cfg, aster=aster, lighter=lighter, katana=katana, runtime=runtime,
    )


def require_live_creds(cfg: AppCfg) -> None:
    """Raise MissingCredsError if any credential is still a placeholder.

    Only checks venues actually referenced by the bound legs — a config
    that uses only aster doesn't need lighter creds (and vice versa).

    Called only when mode == LIVE; paper / spread_monitor tolerate placeholders.
    """
    venues_in_use = {cfg.strategy.pair.leg_a.venue, cfg.strategy.pair.leg_b.venue}
    creds_by_venue: dict[str, AsterCreds | LighterCreds] = {
        "aster": cfg.aster, "lighter": cfg.lighter,
    }
    missing = [
        v for v in venues_in_use
        if v in creds_by_venue and creds_by_venue[v].is_placeholder
    ]
    if missing:
        raise MissingCredsError(
            f"LIVE mode requires real credentials for: {', '.join(sorted(missing))}. "
            f"Set them in .env (see .env.example)."
        )
