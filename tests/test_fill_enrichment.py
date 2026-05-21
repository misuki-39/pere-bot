"""Unit tests for the live fill-enrichment plumbing:

  - `_FillAccumulator` aggregates partial trade events into a single
    size-weighted avg price + last-ts;
  - `_leg_report_with_fill` prefers WS fill data over REST when present;
  - `LegReport.from_result` propagates `OrderResult.exchange_ts_ms` to
    `LegReport.fill_ts_ms` (the aster transactTime path);
  - The recorder writes the new `send_ts_ms` + `fill_ts_ms` columns.

These tests do not exercise the asyncio `_await_fill` loop directly — the
loop just wraps the accumulator + Event primitives, which are covered
here and in stdlib.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from perp_arb.core.exec_record import (
    Decision,
    Direction,
    ExecutionRecorder,
    LegReport,
    Outcome,
    _decision_header,
    _leg_header,
)
from perp_arb.core.types import (
    FillDelta,
    OrderResult,
    OrderSnapshot,
    OrderStatus,
    Side,
    Symbol,
)
from perp_arb.strategy.taker_taker import _FillAccumulator

_SYM = Symbol(exchange="aster", raw="CLUSDT", base="WTI", quote="USDT")


# ---- _FillAccumulator --------------------------------------------------

def _delta(qty: str, price: str, ts: int, *, client_id: str = "x",
           terminal: OrderStatus | None = None) -> FillDelta:
    return FillDelta(
        qty=Decimal(qty), price=Decimal(price), ts_ms=ts,
        side=Side.BUY, client_id=client_id,
        terminal_status=terminal,
    )


def _snapshot(*, filled: str, avg: str, status: OrderStatus, ts: int = 0,
              client_id: str = "x") -> OrderSnapshot:
    return OrderSnapshot(
        order_id="o-1", client_id=client_id, symbol=_SYM, side=Side.BUY,
        size=Decimal("1.0"), price=Decimal("100"),
        status=status,
        filled_size=Decimal(filled),
        avg_fill_price=Decimal(avg) if avg else None,
        ts_ms=ts,
    )


def test_accumulator_single_delta() -> None:
    acc = _FillAccumulator()
    acc.add(_delta("1.0", "100.00", ts=1000))
    assert acc.filled_qty == Decimal("1.0")
    assert acc.weighted_price_sum / acc.filled_qty == Decimal("100.00")
    assert acc.last_ts_ms == 1000


def test_accumulator_aggregates_partial_deltas() -> None:
    """Two partial fills at different prices → size-weighted avg, latest ts."""
    acc = _FillAccumulator()
    acc.add(_delta("0.4", "100.00", ts=1000))
    acc.add(_delta("0.6", "100.10", ts=1050))
    assert acc.filled_qty == Decimal("1.0")
    # (0.4 * 100.00 + 0.6 * 100.10) / 1.0 = 100.06
    assert acc.weighted_price_sum / acc.filled_qty == Decimal("100.06")
    assert acc.last_ts_ms == 1050


def test_accumulator_complete_via_terminal_status() -> None:
    """FILLED status from account_orders short-circuits the qty check."""
    acc = _FillAccumulator()
    acc.add(_snapshot(filled="0.4", avg="100.00", status=OrderStatus.OPEN))
    assert not acc.is_complete(Decimal("1.0"))
    acc.add(_snapshot(filled="1.0", avg="100.05", status=OrderStatus.FILLED))
    assert acc.is_complete(Decimal("1.0"))


def test_accumulator_complete_on_cancel_with_partial_fill() -> None:
    """Cancel after a partial fill — terminal status short-circuits the wait."""
    acc = _FillAccumulator()
    acc.add(_snapshot(filled="0.3", avg="100.00", status=OrderStatus.CANCELED))
    assert acc.is_complete(Decimal("1.0"))
    assert acc.filled_qty == Decimal("0.3")


def test_accumulator_qty_fallback_when_status_absent() -> None:
    """Trade-delta only path (no snapshot stream): exact qty comparison."""
    acc = _FillAccumulator()
    acc.add(_delta("0.9995", "100", ts=1))
    assert not acc.is_complete(Decimal("1.0"))
    acc.add(_delta("0.0005", "100", ts=2))
    assert acc.is_complete(Decimal("1.0"))


def test_accumulator_snapshot_overwrites_delta() -> None:
    """Trade delta arrives first; snapshot arrives second and wins on qty/price.
    ts is taken from whichever event carries it (delta does, snapshot may not)."""
    acc = _FillAccumulator()
    acc.add(_delta("0.4", "100.00", ts=1000))
    acc.add(_snapshot(filled="1.0", avg="100.06", status=OrderStatus.FILLED))
    assert acc.filled_qty == Decimal("1.0")
    assert acc.weighted_price_sum / acc.filled_qty == Decimal("100.06")
    assert acc.last_ts_ms == 1000


def test_accumulator_delta_terminal_status_propagates() -> None:
    """Aster's final FILLED delta carries terminal_status → is_complete True."""
    acc = _FillAccumulator()
    acc.add(_delta("1.0", "100.00", ts=1, terminal=OrderStatus.FILLED))
    assert acc.is_complete(Decimal("2.0"))   # filled < requested but status terminal


# ---- _leg_report_with_fill --------------------------------------------

def _rest_ok(*, avg_price: str = "100.00", exchange_ts: int | None = None) -> OrderResult:
    return OrderResult(
        success=True,
        order_id="rest-1",
        client_id="x",
        side=Side.BUY,
        requested_size=Decimal("1.0"),
        filled_size=Decimal("1.0"),
        avg_price=Decimal(avg_price),
        status=OrderStatus.FILLED,
        latency_ms=50,
        exchange_ts_ms=exchange_ts,
    )


def test_build_uses_rest_data_when_no_fill() -> None:
    """No WS fill (aster timeout / pure REST path): REST is the sole source."""
    rest = _rest_ok(avg_price="100.00", exchange_ts=1_700_000_000_000)
    leg = LegReport.build(
        venue="aster", side=Side.BUY, qty=Decimal("1.0"),
        expected=Decimal("99.95"), rest=rest, latency_ms=50,
    )
    assert leg.realized_price == Decimal("100.00")
    assert leg.filled_qty == Decimal("1.0")
    assert leg.fill_ts_ms == 1_700_000_000_000


def test_build_prefers_ws_fill_over_rest_when_both_present() -> None:
    """WS fill is the matching-engine's authoritative view — wins on price,
    qty, and ts. REST stays the source for status / order_id / etc."""
    rest = _rest_ok(avg_price="100.00", exchange_ts=1_700_000_000_000)
    acc = _FillAccumulator()
    acc.add(_delta("1.0", "100.05", ts=1_700_000_000_500))
    leg = LegReport.build(
        venue="aster", side=Side.BUY, qty=Decimal("1.0"),
        expected=Decimal("99.95"), rest=rest, fill=acc, latency_ms=50,
    )
    assert leg.realized_price == Decimal("100.05")
    assert leg.filled_qty == Decimal("1.0")
    assert leg.fill_ts_ms == 1_700_000_000_500
    assert leg.order_id == "rest-1"    # REST still supplies these


def test_build_falls_back_when_accumulator_empty() -> None:
    """Accumulator with zero fills (timed out) → REST data unchanged."""
    rest = _rest_ok(avg_price="100.00", exchange_ts=1_700_000_000_000)
    leg = LegReport.build(
        venue="aster", side=Side.BUY, qty=Decimal("1.0"),
        expected=Decimal("99.95"), rest=rest, fill=_FillAccumulator(),
        latency_ms=50,
    )
    assert leg.realized_price == Decimal("100.00")
    assert leg.fill_ts_ms == 1_700_000_000_000


def test_build_lighter_path_ws_is_only_price_source() -> None:
    """Lighter REST returns submit-ack only (no avg_price / filled_size).
    WS fill is the only realized-data source for that leg."""
    rest = OrderResult(
        success=True, order_id="seq-1", client_id="x", side=Side.SELL,
        requested_size=Decimal("1.0"), status=OrderStatus.OPEN, latency_ms=200,
    )
    acc = _FillAccumulator()
    acc.add(_delta("1.0", "100.20", ts=1_700_000_001_000))
    leg = LegReport.build(
        venue="lighter", side=Side.SELL, qty=Decimal("1.0"),
        expected=Decimal("100.25"), rest=rest, fill=acc, latency_ms=200,
    )
    assert leg.realized_price == Decimal("100.20")
    assert leg.filled_qty == Decimal("1.0")
    assert leg.fill_ts_ms == 1_700_000_001_000


def test_build_propagates_exchange_ts_ms_from_rest() -> None:
    """Aster's transactTime path: rest.exchange_ts_ms → fill_ts_ms when no WS fill."""
    leg = LegReport.build(
        venue="aster", side=Side.BUY, qty=Decimal("1.0"),
        expected=Decimal("99.95"),
        rest=_rest_ok(exchange_ts=1_700_000_000_777), latency_ms=50,
    )
    assert leg.fill_ts_ms == 1_700_000_000_777


# ---- recorder CSV header contains the new columns ---------------------

def test_decision_header_contains_send_ts_ms() -> None:
    assert "send_ts_ms" in _decision_header()


def test_leg_header_contains_fill_ts_ms() -> None:
    assert "fill_ts_ms" in _leg_header()


def test_recorder_writes_send_ts_ms_to_csv(tmp_path: Path) -> None:
    """End-to-end: a FIRED Decision with send_ts_ms + a LegReport with
    fill_ts_ms should round-trip through the CSV writer."""
    rec = ExecutionRecorder(tmp_path, run_ts="TEST", strategy_id="taker_taker")
    d = Decision(
        decision_id="d-test",
        ts_ms=1_700_000_000_000,
        mid_left=Decimal("100"), mid_right=Decimal("100"),
        left_quote_ts_ms=1_700_000_000_000,
        right_quote_ts_ms=1_700_000_000_000,
        direction=Direction.A,
        outcome=Outcome.FIRED,
        send_ts_ms=1_700_000_000_010,
    )
    d.legs.append(LegReport(
        exchange="aster", side="buy",
        requested_qty=Decimal("1.0"), filled_qty=Decimal("1.0"),
        expected_price=Decimal("100"), realized_price=Decimal("100.05"),
        status="filled", success=True,
        fill_ts_ms=1_700_000_000_050,
    ))
    rec.emit(d)
    rec.close()

    decisions_csv = next(tmp_path.glob("decisions_*.csv")).read_text().splitlines()
    legs_csv = next(tmp_path.glob("legs_*.csv")).read_text().splitlines()
    assert "send_ts_ms" in decisions_csv[0]
    assert "1700000000010" in decisions_csv[1]
    assert "fill_ts_ms" in legs_csv[0]
    assert "1700000000050" in legs_csv[1]
