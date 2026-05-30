"""Engine: book-at-arrival (BookIndex), atomic-pair fills, position update.

Uses an in-memory rows list (no parquet I/O); only the engine itself, plus
a tiny custom BacktestStrategy.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from perp_arb.backtest.base import BacktestStrategy, EngineView, StrategyContext
from perp_arb.backtest.dataset import BBORow
from perp_arb.backtest.engine import Engine, EngineConfig, EngineSummary
from perp_arb.backtest.fills import FillModelKind
from perp_arb.backtest.intents import OrderIntent
from perp_arb.backtest.latency import BookIndex, LatencyModel
from perp_arb.backtest.snapshot import MarketSnapshot
from perp_arb.core.exec_record import Decision, ExecutionRecorder, Verdict
from perp_arb.core.types import Side


def _row(ts_ms: int, left_ts: int, right_ts: int, *,
         left_bid: str = "100.00", left_ask: str = "100.10",
         right_bid: str = "100.05", right_ask: str = "100.15") -> BBORow:
    return BBORow(
        ts_ms=ts_ms,
        left_venue="lighter", right_venue="aster",
        left_bid=Decimal(left_bid), left_bid_size=Decimal("10"),
        left_ask=Decimal(left_ask), left_ask_size=Decimal("10"),
        right_bid=Decimal(right_bid), right_bid_size=Decimal("10"),
        right_ask=Decimal(right_ask), right_ask_size=Decimal("10"),
        mid_left=(Decimal(left_bid) + Decimal(left_ask)) / 2,
        mid_right=(Decimal(right_bid) + Decimal(right_ask)) / 2,
        raw_spread=Decimal("0"), bias_ewma=Decimal("0"),
        vwap_left_sell=Decimal(left_bid), vwap_left_buy=Decimal(left_ask),
        vwap_right_sell=Decimal(right_bid), vwap_right_buy=Decimal(right_ask),
        edge_A_bps=None, edge_B_bps=None,
        gates_passed=True,
        left_ts_ms=left_ts, right_ts_ms=right_ts,
        gap_ms=0,
    )


# ---- BookIndex --------------------------------------------------------

def test_book_index_skips_rows_where_source_ts_did_not_advance() -> None:
    rows = [
        _row(1000, 1000, 950),    # left ticked
        _row(1100, 1000, 1100),   # right ticked, left held
        _row(1200, 1200, 1100),   # left ticked
        _row(1300, 1200, 1300),   # right ticked, left held
    ]
    left_idx = BookIndex.build(rows, "left")
    assert left_idx.ts_array == [1000, 1200]
    right_idx = BookIndex.build(rows, "right")
    assert right_idx.ts_array == [950, 1100, 1300]


def test_book_index_book_at_returns_latest_freshness() -> None:
    rows = [_row(1000, 1000, 1000), _row(1100, 1100, 1100), _row(1200, 1200, 1200)]
    idx = BookIndex.build(rows, "left")
    assert idx.book_at(999).ts_ms == 1000   # before first → first row
    assert idx.book_at(1000).ts_ms == 1000
    assert idx.book_at(1050).ts_ms == 1000  # not yet advanced
    assert idx.book_at(1100).ts_ms == 1100
    assert idx.book_at(99999).ts_ms == 1200


# ---- engine atomic-pair semantics -------------------------------------

class _AlwaysFireStrategy(BacktestStrategy):
    """Fires direction A every tick with VwapFill at sim_ts = snap.ts_ms."""
    name = "test_always_fire"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.intents_built = 0

    def on_tick(self, snap: MarketSnapshot, view: EngineView) -> list[OrderIntent]:
        d = Decision(
            decision_id=f"d-{self.intents_built}",
            ts_ms=snap.ts_ms,
            mid_left=snap.left_quote.mid, mid_right=snap.right_quote.mid,
            left_quote_ts_ms=snap.left_ts_ms, right_quote_ts_ms=snap.right_ts_ms,
            outcome=Verdict.FIRED,
        )
        self.intents_built += 1
        return [
            OrderIntent(
                decision_id=d.decision_id, decision=d,
                venue=self.ctx.left_venue, side=Side.SELL,
                qty=self.ctx.capture_qty, expected_price=snap.vwap_left_sell,
                fill_model=self.ctx.fill_model, sim_ts_ms=snap.ts_ms,
            ),
            OrderIntent(
                decision_id=d.decision_id, decision=d,
                venue=self.ctx.right_venue, side=Side.BUY,
                qty=self.ctx.capture_qty, expected_price=snap.vwap_right_buy,
                fill_model=self.ctx.fill_model, sim_ts_ms=snap.ts_ms,
            ),
        ]


def _ctx(out_dir: Path, *, max_qty: str = "1000",
         fill_model: FillModelKind = FillModelKind.VWAP,
         strategy_id: str = "test_always_fire") -> tuple[StrategyContext, ExecutionRecorder]:
    rec = ExecutionRecorder(out_dir, run_ts="TEST", strategy_id=strategy_id)
    return StrategyContext(
        capture_qty=Decimal("1"),
        fees_bps=Decimal("0"),
        min_profit_bps=Decimal("0"),
        max_stale_ms=10_000,
        bias_halflife_s=3600,
        scale_halflife_s=300,
        warmup_seconds=0,
        max_qty=Decimal(max_qty),
        left_venue="lighter", right_venue="aster",
        fill_model=fill_model,
        recorder=rec,
    ), rec


def _engine(rows: list[BBORow], ctx: StrategyContext, *,
            submit_delay: dict[str, int] | None = None,
            fill_model: FillModelKind = FillModelKind.VWAP,
            fee_bps_per_leg: Decimal = Decimal("0")) -> tuple[Engine, _AlwaysFireStrategy]:
    cfg = EngineConfig(
        data_root=Path("/dev/null"),
        out_dir=Path("/tmp"),
        capture_qty=Decimal("1"),
        latency=LatencyModel(submit_delay_ms=submit_delay or {}),
        fill_model=fill_model,
        fee_bps_per_leg=fee_bps_per_leg,
        strategy_id="test_always_fire",
    )
    strat = _AlwaysFireStrategy(ctx)
    return Engine(rows, strat, cfg, ctx), strat


def test_zero_latency_atomic_pair_updates_position_and_pnl(tmp_path) -> None:
    """Each tick: sell left @ vwap_left_sell, buy right @ vwap_right_buy.
    With zero latency, both fill against the same row's VWAPs.
    """
    rows = [_row(t, t, t) for t in (1000, 1100, 1200)]
    ctx, rec = _ctx(tmp_path)
    engine, strat = _engine(rows, ctx)
    summary = engine.run(rec)
    rec.close()

    assert summary.intents_emitted == 6           # 3 ticks * 2 legs
    assert summary.fills_succeeded == 6
    assert summary.fills_rejected == 0
    assert summary.decisions_emitted == 3
    # left: SELL × 3 = -3 ; right: BUY × 3 = +3
    assert engine.positions.position("lighter") == Decimal("-3")
    assert engine.positions.position("aster") == Decimal("3")
    # PnL per round: sell(100.00) + buy(-100.15) per ETH = -0.15 per leg-pair
    assert summary.realised_pnl == Decimal("-0.45")
    # duration_ms = last_ts - first_ts; max_qty plumbed from ctx; max_qty=1000
    # is way above the 3-tick exposure so no pin samples.
    assert summary.duration_ms == 200
    assert summary.max_qty == Decimal("1000")
    assert summary.ticks_pinned == {"lighter": 0, "aster": 0}


def test_eod_unfilled_when_arrival_past_last_tick(tmp_path) -> None:
    rows = [_row(1000, 1000, 1000)]
    ctx, rec = _ctx(tmp_path)
    engine, _ = _engine(rows, ctx, submit_delay={"lighter": 500, "aster": 500})
    summary = engine.run(rec)
    rec.close()
    # both legs scheduled with arrival=1500 > last row ts=1000 → eod_unfilled
    assert summary.fills_succeeded == 0
    assert summary.fills_rejected == 2
    assert summary.reject_reasons.get("eod_unfilled") == 2
    # decision still emits (atomic-pair: both legs resolved as rejects)
    assert summary.decisions_emitted == 1
    # no position change on rejects
    assert engine.positions.position("lighter") == Decimal("0")
    assert engine.positions.position("aster") == Decimal("0")


def test_adverse_selection_lookup_picks_book_at_arrival(tmp_path) -> None:
    """submit_delay_right=200 should make right leg fill against the row at
    venue source ts ≥ arrival_ts. With ts spacing 100ms, fire at 1000 →
    right arrival 1200 → fills against right_ask at ts=1200 (the moved book).
    """
    rows = [
        _row(1000, 1000, 1000, left_ask="100.10", right_ask="100.15"),
        _row(1100, 1100, 1100, left_ask="100.10", right_ask="100.15"),
        # at ts=1200 the right book has moved up — adverse for a BUY
        _row(1200, 1200, 1200, left_ask="100.10", right_ask="100.50"),
        _row(1300, 1300, 1300, left_ask="100.10", right_ask="100.50"),
    ]
    ctx, rec = _ctx(tmp_path, fill_model=FillModelKind.BBO, strategy_id="test_fire_once")

    # Only fire on the FIRST row, so we can read a clean per-fill price.
    class FireOnce(_AlwaysFireStrategy):
        def on_tick(self, snap, view):
            if self.intents_built == 0:
                return super().on_tick(snap, view)
            return []

    cfg = EngineConfig(
        data_root=Path("/dev/null"),
        out_dir=tmp_path,
        capture_qty=Decimal("1"),
        latency=LatencyModel(submit_delay_ms={"lighter": 0, "aster": 200}),
        fill_model=FillModelKind.BBO,
        fee_bps_per_leg=Decimal("0"),
        strategy_id="test_fire_once",
    )
    engine = Engine(rows, FireOnce(ctx), cfg, ctx)
    summary = engine.run(rec)
    rec.close()

    # find the right-leg fill price
    legs_file = (tmp_path / "legs_test_fire_once_TEST.csv")
    text = legs_file.read_text().splitlines()
    header = text[0].split(",")
    leg_rows = [dict(zip(header, line.split(","), strict=True)) for line in text[1:]]
    right_leg = next(r for r in leg_rows if r["venue"] == "aster")
    # BUY right hit the MOVED ask (100.50) — proves the lookup found the post-arrival book
    assert Decimal(right_leg["realized_price"]) == Decimal("100.50")
    # SELL left arrived at the same ts (0 delay) → fills against original bid
    left_leg = next(r for r in leg_rows if r["venue"] == "lighter")
    assert Decimal(left_leg["realized_price"]) == Decimal("100.00")
    assert summary.decisions_emitted == 1


def test_summary_ticks_pinned_counts_pre_strategy_pin(tmp_path) -> None:
    """max_qty=2; AlwaysFire fires 3 ticks ignoring the cap. Pin sampling
    happens AFTER the pre-strategy drain — so by tick 3 the position is ±2
    (settled from tick 2) and both legs count as pinned exactly once."""
    rows = [_row(t, t, t) for t in (1000, 1100, 1200)]
    ctx, rec = _ctx(tmp_path, max_qty="2")
    engine, _ = _engine(rows, ctx)
    summary = engine.run(rec)
    rec.close()
    # tick 1: pos=0 (no pin). tick 2: pos=±1 (no pin, |1|<2). tick 3: pos=±2 (PIN).
    assert summary.ticks_pinned == {"lighter": 1, "aster": 1}


def test_summary_pretty_smoke() -> None:
    """`EngineSummary.pretty()` renders all the load-bearing fields. Format
    is judged by humans — assert key substrings, not full string equality."""
    s = EngineSummary(
        rows_processed=1000, intents_emitted=20, fills_succeeded=20,
        fills_rejected=0, decisions_emitted=10, realised_pnl=Decimal("1.2345"),
        final_positions={"lighter": Decimal("-3"), "aster": Decimal("3")},
        fires_dir_a=7, fires_dir_b=3,
        ticks_pinned={"lighter": 100, "aster": 0},
        duration_ms=3_600_000, max_qty=Decimal("10"),
    )
    text = s.pretty()
    assert "backtest done" in text
    assert "A=7" in text and "B=3" in text
    assert "1.2345" in text
    assert "/day" in text and "/filled-pair" in text
    assert "lighter=100/1000" in text                   # pin row
    assert "lighter=-3" in text and "aster=3" in text   # final pos row
    assert "cap=±10" in text
