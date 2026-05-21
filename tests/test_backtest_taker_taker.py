"""End-to-end smoke for the TakerTakerBT backtest strategy on synthetic data.

Verifies the *zero-latency canary*: with a captured VWAP-edge column and a
strategy that fires on any positive edge, the realised round-trip PnL per
fired decision should equal `(edge_bps - threshold) * notional / 1e4` to
within rounding. Effectively proves the strategy reads the right VWAP
columns and the engine applies fees correctly.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from perp_arb.backtest.dataset import BBORow
from perp_arb.backtest.engine import Engine, EngineConfig
from perp_arb.backtest.fills import FillModelKind
from perp_arb.backtest.latency import LatencyModel
from perp_arb.backtest.runner import build_context
from perp_arb.backtest.strategies import TakerTakerBT
from perp_arb.core.exec_record import ExecutionRecorder


def _row(ts: int, mid_left: str, mid_right: str, *,
         vwap_left_sell: str | None = None, vwap_left_buy: str | None = None,
         vwap_right_sell: str | None = None, vwap_right_buy: str | None = None) -> BBORow:
    ml, mr = Decimal(mid_left), Decimal(mid_right)
    ls = Decimal(vwap_left_sell or mid_left)
    lb = Decimal(vwap_left_buy or mid_left)
    rs = Decimal(vwap_right_sell or mid_right)
    rb = Decimal(vwap_right_buy or mid_right)
    return BBORow(
        ts_ms=ts,
        left_venue="lighter", right_venue="aster",
        left_bid=ls, left_bid_size=Decimal("100"),
        left_ask=lb, left_ask_size=Decimal("100"),
        right_bid=rs, right_bid_size=Decimal("100"),
        right_ask=rb, right_ask_size=Decimal("100"),
        mid_left=ml, mid_right=mr,
        raw_spread=ml - mr, bias_ewma=Decimal("0"),
        vwap_left_sell=ls, vwap_left_buy=lb,
        vwap_right_sell=rs, vwap_right_buy=rb,
        edge_A_bps=None, edge_B_bps=None,
        gates_passed=True,
        left_ts_ms=ts, right_ts_ms=ts, gap_ms=0,
    )


def test_taker_taker_zero_latency_captures_clean_edge(tmp_path: Path) -> None:
    """EWMA bias seeds to the first sample, so we need a *neutral* warmup
    phase (spread = 0) before introducing the edge — otherwise the bias
    perfectly cancels the static dislocation and the strategy never fires.

    Setup:
      - Phase 1 (200 ticks, 100ms cadence): both venues quote 100.10. spread=0.
        Bias EWMA converges to 0.
      - Phase 2 (50 ticks): left jumps to 100.20, right stays at 100.00. The
        bias half-life is 1e9s so the bias barely moves over 5s; the edge
        stays roughly equal to the dislocation.
    """
    neutral = [_row(t, "100.10", "100.10",
                    vwap_left_sell="100.10", vwap_left_buy="100.10",
                    vwap_right_sell="100.10", vwap_right_buy="100.10")
               for t in range(1000, 1000 + 200 * 100, 100)]
    edge_start = neutral[-1].ts_ms + 100
    edged = [_row(t, "100.20", "100.00",
                  vwap_left_sell="100.20", vwap_left_buy="100.20",
                  vwap_right_sell="100.00", vwap_right_buy="100.00")
             for t in range(edge_start, edge_start + 50 * 100, 100)]
    rows = neutral + edged

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rec = ExecutionRecorder(out_dir, run_ts="ZL", strategy_id="taker_taker")

    from perp_arb.backtest.runner import StrategyParams
    params = StrategyParams(
        qty=Decimal("1"), fees_bps=Decimal("2"), min_profit_bps=Decimal("2"),
        max_slippage_bps=Decimal("100"), max_stale_ms=10_000,
        bias_halflife_s=1e9, scale_halflife_s=300, warmup_seconds=0,
        max_qty=Decimal("1000"),
    )
    cfg = EngineConfig(
        data_root=tmp_path, out_dir=out_dir,
        capture_qty=Decimal("1"),
        latency=LatencyModel(submit_delay_ms={}),
        fill_model=FillModelKind.VWAP,
        fee_bps_per_leg=Decimal("1"),                # 2 bps round-trip
        strategy_id="taker_taker",
    )
    ctx = build_context(params, cfg, left_venue="lighter", right_venue="aster", recorder=rec)
    strat = TakerTakerBT(ctx)
    engine = Engine(rows, strat, cfg, ctx)
    summary = engine.run(rec)
    rec.close()

    # Phase 1 produces no intents (spread = 0 = bias → edge < threshold);
    # phase 2 fires on every tick: 50 decisions × 2 legs = 100 intents.
    assert summary.intents_emitted == 100, summary
    assert summary.fills_succeeded == 100
    assert summary.fills_rejected == 0
    assert summary.decisions_emitted == 50
    # Each round-trip: sell(100.20) - buy(100.00) = +0.20, minus 1 bps fee
    # on each leg: 0.20 * 1e-4 ≈ 0.02 per leg. Net ≈ 0.18 / RT × 50 ≈ 9.
    assert summary.realised_pnl > Decimal("8")
    assert summary.realised_pnl < Decimal("10")


def test_taker_taker_qty_mismatch_fails_fast() -> None:
    """build_context rejects when strategy qty != capture_qty."""
    from perp_arb.backtest.runner import StrategyParams
    params = StrategyParams(
        qty=Decimal("2"), fees_bps=Decimal("2"), min_profit_bps=Decimal("2"),
        max_slippage_bps=Decimal("100"), max_stale_ms=10_000,
        bias_halflife_s=3600, scale_halflife_s=300, warmup_seconds=0,
        max_qty=Decimal("1000"),
    )
    cfg = EngineConfig(
        data_root=Path("/dev/null"), out_dir=Path("/tmp"),
        capture_qty=Decimal("1"),                    # mismatch
        latency=LatencyModel(submit_delay_ms={}),
        fill_model=FillModelKind.VWAP,
        fee_bps_per_leg=Decimal("1"),
        strategy_id="taker_taker",
    )
    import pytest
    with pytest.raises(ValueError, match="capture_qty"):
        build_context(params, cfg, left_venue="lighter", right_venue="aster",
                      recorder=None)  # type: ignore[arg-type]
