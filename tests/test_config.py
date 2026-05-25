"""Config loading + schema validation."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from perp_arb.core.config import (
    MissingCredsError,
    RunMode,
    StrategyCfg,
    load_app_config,
    require_live_creds,
)

SAMPLE_YAML = """
strategy: taker_taker
mode: paper
pair:
  base: ETH
  leg_a: { venue: aster,   symbol: ETHUSDT }
  leg_b: { venue: lighter, symbol: ETH }
qty: 0.05
max_qty: 0.5
fees_bps: 6
min_profit_bps: 3
bias_halflife_s: 3600
scale_halflife_s: 300
warmup_seconds: 180
max_levels: 3
max_stale_ms: 200
risk:
  max_consecutive_failures: 3
  max_leg_latency_ms: 500
  daily_loss_cap_usd: 50
"""


@pytest.fixture(autouse=True)
def _clean_creds_env(monkeypatch) -> None:
    """Ensure every test starts with no real credentials in the environment."""
    for k in (
        "ASTER_USER", "ASTER_SIGNER", "ASTER_SIGNER_PRIVKEY",
        "ASTER_REST_URL", "ASTER_WS_URL",
        "LIGHTER_API_KEY_PRIVATE_KEY", "LIGHTER_ACCOUNT_INDEX", "LIGHTER_API_KEY_INDEX",
        "LIGHTER_BASE_URL", "LOG_DIR", "LOG_LEVEL",
    ):
        monkeypatch.delenv(k, raising=False)


def test_strategy_cfg_validates_max_qty_gte_qty() -> None:
    raw = yaml.safe_load(SAMPLE_YAML)
    raw["qty"] = "1.0"
    raw["max_qty"] = "0.5"
    with pytest.raises(ValueError):
        StrategyCfg.model_validate(raw)


def test_strategy_cfg_parses_paper_mode() -> None:
    cfg = StrategyCfg.model_validate(yaml.safe_load(SAMPLE_YAML))
    assert cfg.strategy == "taker_taker"
    assert cfg.mode is RunMode.PAPER
    assert cfg.qty == Decimal("0.05")
    assert cfg.max_qty == Decimal("0.5")


def test_load_app_config_uses_placeholders_when_env_absent(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(SAMPLE_YAML)

    cfg = load_app_config(cfg_path)
    assert cfg.strategy.mode is RunMode.PAPER
    assert cfg.aster.is_placeholder
    assert cfg.lighter.is_placeholder


def test_load_app_config_picks_up_real_env(tmp_path: Path, monkeypatch) -> None:
    real_addr = "0x1111111111111111111111111111111111111111"
    real_pk = "0x" + "ab" * 32
    monkeypatch.setenv("ASTER_USER", real_addr)
    monkeypatch.setenv("ASTER_SIGNER", real_addr)
    monkeypatch.setenv("ASTER_SIGNER_PRIVKEY", real_pk)
    monkeypatch.setenv("LIGHTER_API_KEY_PRIVATE_KEY", real_pk)
    monkeypatch.setenv("LIGHTER_ACCOUNT_INDEX", "42")

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(SAMPLE_YAML)
    cfg = load_app_config(cfg_path)

    assert cfg.aster.user == real_addr
    assert not cfg.aster.is_placeholder
    assert not cfg.lighter.is_placeholder
    assert cfg.lighter.account_index == 42


def test_require_live_creds_raises_when_placeholder(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(SAMPLE_YAML)
    cfg = load_app_config(cfg_path, mode_override=RunMode.LIVE)
    with pytest.raises(MissingCredsError):
        require_live_creds(cfg)


def test_require_live_creds_passes_with_real_env(tmp_path: Path, monkeypatch) -> None:
    real_addr = "0x1111111111111111111111111111111111111111"
    real_pk = "0x" + "ab" * 32
    monkeypatch.setenv("ASTER_USER", real_addr)
    monkeypatch.setenv("ASTER_SIGNER", real_addr)
    monkeypatch.setenv("ASTER_SIGNER_PRIVKEY", real_pk)
    monkeypatch.setenv("LIGHTER_API_KEY_PRIVATE_KEY", real_pk)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(SAMPLE_YAML)
    cfg = load_app_config(cfg_path, mode_override=RunMode.LIVE)
    require_live_creds(cfg)   # should not raise


def test_mode_override_replaces_yaml_mode(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(SAMPLE_YAML)
    cfg = load_app_config(cfg_path, mode_override=RunMode.LIVE)
    assert cfg.strategy.mode is RunMode.LIVE


# ---- Wave-1 optimisations block ----

def test_optimisations_defaults_when_block_absent() -> None:
    cfg = StrategyCfg.model_validate(yaml.safe_load(SAMPLE_YAML))
    opt = cfg.optimisations
    assert opt.markout_table_path is None
    assert opt.throttle_bump_bps == Decimal("0")
    assert opt.throttle_halflife_s == 3.0
    assert opt.in_flight_cap_per_direction == 0
    # inventory skew default-off: κ_open=0 disables widener; κ_close=None
    # means "symmetric — fall back to κ_open" rather than "explicitly 0".
    assert opt.inventory_skew_bps == Decimal("0")
    assert opt.inventory_skew_close_bps is None


def test_optimisations_inventory_skew_round_trips() -> None:
    """The live boundary that the stale 'intentionally NOT exposed' comment
    used to gate. Catches regression of the config → AssessParams plumbing."""
    raw = yaml.safe_load(SAMPLE_YAML)
    raw["optimisations"] = {
        "inventory_skew_bps": "15",
        "inventory_skew_close_bps": "5",
    }
    cfg = StrategyCfg.model_validate(raw)
    assert cfg.optimisations.inventory_skew_bps == Decimal("15")
    assert cfg.optimisations.inventory_skew_close_bps == Decimal("5")


def test_optimisations_block_fully_populated(tmp_path: Path) -> None:
    # write a fake markout JSON so the validator passes
    table = tmp_path / "m.json"
    table.write_text('{"direction_A":{"buckets":[]},"direction_B":{"buckets":[]}}')
    raw = yaml.safe_load(SAMPLE_YAML)
    raw["optimisations"] = {
        "markout_table_path": str(table),
        "throttle_bump_bps": "2",
        "throttle_halflife_s": 5.0,
        "in_flight_cap_per_direction": 1,
    }
    cfg = StrategyCfg.model_validate(raw)
    assert cfg.optimisations.markout_table_path == table
    assert cfg.optimisations.throttle_bump_bps == Decimal("2")
    assert cfg.optimisations.throttle_halflife_s == 5.0
    assert cfg.optimisations.in_flight_cap_per_direction == 1


def test_optimisations_missing_markout_path_raises() -> None:
    raw = yaml.safe_load(SAMPLE_YAML)
    raw["optimisations"] = {"markout_table_path": "/tmp/__no_such_file__.json"}
    with pytest.raises(ValueError, match="markout_table_path does not exist"):
        StrategyCfg.model_validate(raw)


def test_spread_monitor_config_still_validates_without_optimisations() -> None:
    """Regression: existing spread_monitor YAMLs omit `optimisations:` and
    must keep parsing cleanly (the field is defaulted)."""
    sm_yaml = """
strategy: spread_monitor
mode: paper
pair:
  base: WTI
  leg_a: { venue: aster,   symbol: CLUSDT }
  leg_b: { venue: lighter, symbol: WTI }
monitor_pair:
  - { venue: lighter, symbol: WTI }
  - { venue: aster,   symbol: CLUSDT }
qty: 1
max_qty: 10
fees_bps: 1
bias_halflife_s: 3600
scale_halflife_s: 300
max_levels: 3
"""
    cfg = StrategyCfg.model_validate(yaml.safe_load(sm_yaml))
    assert cfg.strategy == "spread_monitor"
    # default block fills in
    assert cfg.optimisations.markout_table_path is None
