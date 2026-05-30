"""BBOFill + VwapFill semantics: accept, reject, gate failure."""

from __future__ import annotations

from decimal import Decimal

from perp_arb.backtest.dataset import BBORow
from perp_arb.backtest.fills import BBOFill, FillModelKind, VwapFill
from perp_arb.backtest.intents import OrderIntent
from perp_arb.core.exec_record import Decision, Verdict
from perp_arb.core.types import Side


def _row(*, gates: bool = True,
         left_ask_size: str = "5", left_ask: str = "100.10",
         left_bid_size: str = "5", left_bid: str = "100.00") -> BBORow:
    return BBORow(
        ts_ms=2000,
        left_venue="lighter", right_venue="aster",
        left_bid=Decimal(left_bid), left_bid_size=Decimal(left_bid_size),
        left_ask=Decimal(left_ask), left_ask_size=Decimal(left_ask_size),
        right_bid=Decimal("100.05"), right_bid_size=Decimal("5"),
        right_ask=Decimal("100.15"), right_ask_size=Decimal("5"),
        mid_left=Decimal("100.05"), mid_right=Decimal("100.10"),
        raw_spread=Decimal("0.05"), bias_ewma=Decimal("0"),
        vwap_left_sell=Decimal("99.99"), vwap_left_buy=Decimal("100.11"),
        vwap_right_sell=Decimal("100.04"), vwap_right_buy=Decimal("100.16"),
        edge_A_bps=None, edge_B_bps=None,
        gates_passed=gates,
        left_ts_ms=2000, right_ts_ms=1950, gap_ms=0,
    )


def _intent(side: Side, qty: str, fm: FillModelKind = FillModelKind.BBO) -> OrderIntent:
    d = Decision(
        decision_id="d-test", ts_ms=1900,
        mid_left=Decimal("100.05"), mid_right=Decimal("100.10"),
        left_quote_ts_ms=1900, right_quote_ts_ms=1900,
        outcome=Verdict.FIRED,
    )
    return OrderIntent(
        decision_id=d.decision_id, decision=d,
        venue="lighter", side=side, qty=Decimal(qty),
        expected_price=Decimal("100.00"),
        fill_model=fm, sim_ts_ms=1900,
    )


def test_bbo_fill_buy_fills_at_ask_within_size() -> None:
    fill = BBOFill().try_fill(
        _intent(Side.BUY, "3"), arrival_ts_ms=2000,
        venue_row=_row(), venue_side="left", capture_qty=Decimal("1"),
    )
    assert fill.success
    assert fill.realized_price == Decimal("100.10")
    assert fill.filled_qty == Decimal("3")


def test_bbo_fill_rejects_when_qty_exceeds_top_size() -> None:
    fill = BBOFill().try_fill(
        _intent(Side.BUY, "10"), arrival_ts_ms=2000,
        venue_row=_row(), venue_side="left", capture_qty=Decimal("1"),
    )
    assert not fill.success
    assert "exceeds_top_level_size" in (fill.error or "")


def test_bbo_fill_rejects_when_gates_failed() -> None:
    fill = BBOFill().try_fill(
        _intent(Side.BUY, "1"), arrival_ts_ms=2000,
        venue_row=_row(gates=False), venue_side="left", capture_qty=Decimal("1"),
    )
    assert not fill.success
    assert fill.error == "gates_failed_at_arrival"


def test_vwap_fill_requires_exact_capture_qty() -> None:
    fill = VwapFill().try_fill(
        _intent(Side.BUY, "2"), arrival_ts_ms=2000,
        venue_row=_row(), venue_side="left", capture_qty=Decimal("1"),
    )
    assert not fill.success
    assert "vwap_qty_mismatch" in (fill.error or "")


def test_vwap_fill_uses_pre_computed_column() -> None:
    # qty=capture_qty=1, BUY left → vwap_left_buy = 100.11
    fill = VwapFill().try_fill(
        _intent(Side.BUY, "1"), arrival_ts_ms=2000,
        venue_row=_row(), venue_side="left", capture_qty=Decimal("1"),
    )
    assert fill.success
    assert fill.realized_price == Decimal("100.11")

    # SELL left → vwap_left_sell = 99.99
    fill_sell = VwapFill().try_fill(
        _intent(Side.SELL, "1"), arrival_ts_ms=2000,
        venue_row=_row(), venue_side="left", capture_qty=Decimal("1"),
    )
    assert fill_sell.success
    assert fill_sell.realized_price == Decimal("99.99")


def test_vwap_fill_right_side_reads_right_columns() -> None:
    fill = VwapFill().try_fill(
        _intent(Side.SELL, "1"), arrival_ts_ms=2000,
        venue_row=_row(), venue_side="right", capture_qty=Decimal("1"),
    )
    assert fill.success
    assert fill.realized_price == Decimal("100.04")  # vwap_right_sell
