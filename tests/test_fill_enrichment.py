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
from perp_arb.core.types import OrderInfo, OrderResult, OrderStatus, Side, Symbol
from perp_arb.strategy.taker_taker import _FillAccumulator, _leg_report_with_fill

_SYM = Symbol(exchange="aster", raw="CLUSDT", base="WTI", quote="USDT")


# ---- _FillAccumulator --------------------------------------------------

def _fill_info(size: str, price: str, ts: int, *, client_id: str = "x") -> OrderInfo:
    return OrderInfo(
        order_id="t-1",
        client_id=client_id,
        symbol=_SYM,
        side=Side.BUY,
        size=Decimal(size),
        price=Decimal(price),
        status=OrderStatus.FILLED,
        filled_size=Decimal(size),
        avg_fill_price=Decimal(price),
        ts_ms=ts,
    )


def test_accumulator_single_fill() -> None:
    acc = _FillAccumulator()
    acc.add(_fill_info("1.0", "100.00", ts=1000))
    assert acc.filled_qty == Decimal("1.0")
    assert acc.avg_price == Decimal("100.00")
    assert acc.last_ts_ms == 1000
    assert acc.fills_count == 1


def test_accumulator_aggregates_partials_size_weighted() -> None:
    """Two partial fills at different prices → size-weighted avg, latest ts."""
    acc = _FillAccumulator()
    acc.add(_fill_info("0.4", "100.00", ts=1000))
    acc.add(_fill_info("0.6", "100.10", ts=1050))
    assert acc.filled_qty == Decimal("1.0")
    # (0.4 * 100.00 + 0.6 * 100.10) / 1.0 = 100.06
    assert acc.avg_price == Decimal("100.06")
    assert acc.last_ts_ms == 1050  # "fill complete" = last partial
    assert acc.fills_count == 2


def test_accumulator_ignores_zero_or_negative_size() -> None:
    """Defensive: malformed WS event shouldn't poison the accumulator."""
    acc = _FillAccumulator()
    acc.add(_fill_info("0", "100.00", ts=1000))
    assert acc.fills_count == 0
    assert acc.filled_qty == Decimal("0")


def test_accumulator_is_complete_within_tolerance() -> None:
    """The 0.1 % slack absorbs base_multiplier rounding."""
    acc = _FillAccumulator()
    acc.add(_fill_info("0.9995", "100", ts=1))   # 0.05% short of 1.0
    assert acc.is_complete(Decimal("1.0"))

    acc2 = _FillAccumulator()
    acc2.add(_fill_info("0.99", "100", ts=1))   # 1% short — NOT complete
    assert not acc2.is_complete(Decimal("1.0"))


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


def test_leg_report_uses_rest_data_when_no_fill_event() -> None:
    """Aster timeout / lighter not yet wired → fall back to REST data."""
    rest = _rest_ok(avg_price="100.00", exchange_ts=1_700_000_000_000)
    leg = _leg_report_with_fill(
        venue="aster", side=Side.BUY, qty=Decimal("1.0"),
        expected=Decimal("99.95"), rest=rest, latency_ms=50, fill=None,
    )
    assert leg.realized_price == Decimal("100.00")
    assert leg.filled_qty == Decimal("1.0")
    assert leg.fill_ts_ms == 1_700_000_000_000   # from REST transactTime


def test_leg_report_prefers_ws_fill_over_rest_when_both_present() -> None:
    """WS fill is the matching-engine's authoritative view — wins on price,
    qty, and ts. (REST values may be stale or partial.)"""
    rest = _rest_ok(avg_price="100.00", exchange_ts=1_700_000_000_000)
    acc = _FillAccumulator()
    acc.add(_fill_info("1.0", "100.05", ts=1_700_000_000_500))
    leg = _leg_report_with_fill(
        venue="aster", side=Side.BUY, qty=Decimal("1.0"),
        expected=Decimal("99.95"), rest=rest, latency_ms=50, fill=acc,
    )
    assert leg.realized_price == Decimal("100.05")     # WS wins
    assert leg.filled_qty == Decimal("1.0")
    assert leg.fill_ts_ms == 1_700_000_000_500         # WS wins


def test_leg_report_falls_back_when_ws_accumulator_empty() -> None:
    """An accumulator with 0 fills (timed out before any event) should
    not overwrite REST data."""
    rest = _rest_ok(avg_price="100.00", exchange_ts=1_700_000_000_000)
    empty = _FillAccumulator()
    leg = _leg_report_with_fill(
        venue="aster", side=Side.BUY, qty=Decimal("1.0"),
        expected=Decimal("99.95"), rest=rest, latency_ms=50, fill=empty,
    )
    assert leg.realized_price == Decimal("100.00")
    assert leg.fill_ts_ms == 1_700_000_000_000


def test_leg_report_lighter_path_only_ws_data_available() -> None:
    """Lighter REST gives only submit-ack: filled_size=None, avg_price=None.
    The WS fill event is the *only* source of realized data."""
    rest = OrderResult(
        success=True, order_id="seq-1", client_id="x",
        side=Side.SELL, requested_size=Decimal("1.0"),
        filled_size=None, avg_price=None,        # ← submit-ack only
        status=OrderStatus.OPEN, latency_ms=200,
        exchange_ts_ms=None,
    )
    acc = _FillAccumulator()
    acc.add(_fill_info("1.0", "100.20", ts=1_700_000_001_000))
    leg = _leg_report_with_fill(
        venue="lighter", side=Side.SELL, qty=Decimal("1.0"),
        expected=Decimal("100.25"), rest=rest, latency_ms=200, fill=acc,
    )
    assert leg.realized_price == Decimal("100.20")
    assert leg.filled_qty == Decimal("1.0")
    assert leg.fill_ts_ms == 1_700_000_001_000


# ---- LegReport.from_result propagates exchange_ts_ms ------------------

def test_legreport_from_result_propagates_exchange_ts_ms() -> None:
    """Aster's transactTime path: REST sets exchange_ts_ms → LegReport.fill_ts_ms."""
    rest = _rest_ok(exchange_ts=1_700_000_000_777)
    leg = LegReport.from_result(
        "aster", Side.BUY, Decimal("1.0"),
        expected=Decimal("99.95"), r=rest, latency_ms=50,
    )
    assert leg.fill_ts_ms == 1_700_000_000_777


def test_legreport_from_result_handles_missing_exchange_ts() -> None:
    rest = _rest_ok(exchange_ts=None)
    leg = LegReport.from_result(
        "aster", Side.BUY, Decimal("1.0"),
        expected=Decimal("99.95"), r=rest, latency_ms=50,
    )
    assert leg.fill_ts_ms is None


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
