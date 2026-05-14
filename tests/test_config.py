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
  aster_symbol: ETHUSDT
  lighter_symbol: ETH
qty: 0.05
max_qty: 0.5
fees_bps: 6
min_profit_bps: 3
bias_window_ticks: 600
warmup_seconds: 180
max_levels: 3
max_slippage_bps: 5
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
