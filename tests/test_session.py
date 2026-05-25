"""LiveSession / PaperSession behaviour."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from perp_arb.core.config import (
    MissingCredsError,
    RunMode,
    load_app_config,
)
from perp_arb.core.exchange import BaseExchange
from perp_arb.core.session import LiveSession, PaperSession
from perp_arb.core.types import (
    LegOutcome,
    MarketInfo,
    OrderBook,
    Position,
    Quote,
    Symbol,
)

# Mirror tests/test_config.py: same YAML body and same env-cleanup fixture.
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
    for k in (
        "ASTER_USER", "ASTER_SIGNER", "ASTER_SIGNER_PRIVKEY",
        "ASTER_REST_URL", "ASTER_WS_URL",
        "LIGHTER_API_KEY_PRIVATE_KEY", "LIGHTER_ACCOUNT_INDEX", "LIGHTER_API_KEY_INDEX",
        "LIGHTER_BASE_URL", "LOG_DIR", "LOG_LEVEL",
    ):
        monkeypatch.delenv(k, raising=False)


def _market() -> MarketInfo:
    return MarketInfo(
        symbol=Symbol(exchange="aster", raw="ETHUSDT", base="ETH", quote="USDT"),
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.001"),
        contract_id="ETHUSDT",
    )


class _FakeExchange(BaseExchange):
    """BaseExchange-compliant fake. Only `get_position` is exercised here;
    every other abstract raises NotImplementedError so any future drift
    in the contract surfaces as a test failure, not a silent skip."""

    name = "fake"

    def __init__(
        self,
        size: Decimal = Decimal(0),
        raises: Exception | None = None,
    ) -> None:
        super().__init__()
        self._size = size
        self._raises = raises
        self.calls = 0

    async def get_position(self, market: MarketInfo) -> Position:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return Position(symbol=market.symbol, size=self._size)

    async def connect(self) -> None: raise NotImplementedError
    async def disconnect(self) -> None: raise NotImplementedError
    async def load_market(self, raw_symbol: str) -> MarketInfo: raise NotImplementedError
    async def place_market_order(  # type: ignore[override]
        self, market, side, qty, *, client_id: str, reduce_only=False,
    ) -> LegOutcome: raise NotImplementedError
    def subscribe_quotes(self, market: MarketInfo, cb: Callable) -> None: raise NotImplementedError
    def subscribe_book(self, market: MarketInfo, cb: Callable) -> None: raise NotImplementedError
    def subscribe_fills(self, market: MarketInfo, cb: Callable) -> None: raise NotImplementedError
    def subscribe_positions(self, market: MarketInfo, cb: Callable) -> None: raise NotImplementedError
    def best_quote(self, market: MarketInfo) -> Quote | None: raise NotImplementedError
    def order_book(self, market: MarketInfo) -> OrderBook | None: raise NotImplementedError
    def live_position(self, market: MarketInfo) -> Position | None: raise NotImplementedError


# ---- PaperSession ----

async def test_paper_snapshot_returns_zero_without_calling_exchange() -> None:
    s = PaperSession()
    ex = _FakeExchange(size=Decimal("99"))  # would be wrong if it were called
    out = await s.snapshot_position(ex, _market())
    assert out == Decimal(0)
    assert ex.calls == 0


# ---- LiveSession ----

async def test_live_snapshot_delegates_to_exchange_get_position() -> None:
    s = LiveSession()
    ex = _FakeExchange(size=Decimal("-0.42"))
    out = await s.snapshot_position(ex, _market())
    assert out == Decimal("-0.42")
    assert ex.calls == 1


async def test_live_snapshot_propagates_exception() -> None:
    """A REST failure on get_position must propagate; the strategy uses
    this to refuse trading with unknown inventory at startup."""
    s = LiveSession()
    ex = _FakeExchange(raises=RuntimeError("aster rest 503"))
    with pytest.raises(RuntimeError, match="aster rest 503"):
        await s.snapshot_position(ex, _market())


# ---- Mode-vs-session invariant (preflight assertions) ----

async def test_live_preflight_rejects_paper_cfg(tmp_path: Path) -> None:
    """LiveSession on a PAPER cfg is a wiring bug; preflight must refuse
    so factory + executor + session can't disagree on mode."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(SAMPLE_YAML)
    cfg = load_app_config(cfg_path)  # SAMPLE_YAML is mode=paper
    with pytest.raises(ValueError, match="LiveSession requires"):
        await LiveSession().preflight(cfg)


async def test_paper_preflight_rejects_live_cfg(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(SAMPLE_YAML)
    cfg = load_app_config(cfg_path, mode_override=RunMode.LIVE)
    with pytest.raises(ValueError, match="PaperSession requires"):
        await PaperSession().preflight(cfg)


async def test_live_preflight_raises_on_placeholder_creds(tmp_path: Path) -> None:
    """Once the mode check passes, LiveSession.preflight must still
    enforce real credentials — defends the wiring through require_live_creds."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(SAMPLE_YAML)
    cfg = load_app_config(cfg_path, mode_override=RunMode.LIVE)
    with pytest.raises(MissingCredsError):
        await LiveSession().preflight(cfg)
