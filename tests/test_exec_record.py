"""Unit tests for execution telemetry: derived headers, single-emit, and the
two-file normalisation. No network, no strategy."""

from __future__ import annotations

import csv
from decimal import Decimal

import perp_arb.core.exec_record as er
from perp_arb.core.exec_record import (
    Decision,
    Direction,
    ExecutionRecorder,
    Outcome,
    Phase,
    Timeline,
    _decision_header,
    _leg_header,
)
from perp_arb.core.types import LegKind, LegOutcome, OrderStatus, Side


def test_timeline_span_none_until_both_marks(monkeypatch) -> None:
    clock = {"t": 1000}
    monkeypatch.setattr(er, "mono_ms", lambda: clock["t"])
    tl = Timeline()
    assert tl.span("decision", "send") is None
    tl.mark("decision")
    assert tl.span("decision", "send") is None
    clock["t"] = 1175
    tl.mark("send")
    assert tl.span("decision", "send") == 175
    assert tl.span("send", "decision") == -175


def test_headers_are_derived_from_dataclasses() -> None:
    dh = _decision_header()
    assert "timeline" not in dh and "legs" not in dh
    assert dh[:2] == ["decision_id", "ts_ms"]
    assert dh[-1] == "lat_decision_send_ms"
    assert "realised_pnl" in dh
    lh = _leg_header()
    assert lh[:2] == ["decision_id", "ts_ms"]
    assert "expected_price" in lh and "realized_price" in lh
    assert "send_ts_ms" in lh and "fill_ts_ms" in lh
    assert "latency_ms" not in lh   # legacy mixed-clock derived column dropped
    assert "total_fee" in lh
    assert "venue" in lh and "error_message" in lh


def _read(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_fired_decision_emits_one_decision_row_and_two_leg_rows(tmp_path, monkeypatch) -> None:
    clock = {"t": 0}
    monkeypatch.setattr(er, "mono_ms", lambda: clock["t"])
    rec = ExecutionRecorder(tmp_path, run_ts="TEST")

    d = Decision(
        decision_id="d-abc", ts_ms=111, mid_left=Decimal("100"), mid_right=Decimal("100.05"),
        left_quote_ts_ms=110, right_quote_ts_ms=109,
        bias=Decimal("-0.03"), vwap_left_sell=Decimal("100.01"),
        vwap_left_buy=Decimal("100.02"), vwap_right_sell=Decimal("100.04"),
        vwap_right_buy=Decimal("100.06"), edge_bps=Decimal("2.5"),
        direction=Direction.B, outcome=Outcome.FIRED,
        realised_pnl=Decimal("0.012"),
    )
    clock["t"] = 5
    d.timeline.mark(Phase.DECISION)
    clock["t"] = 12
    d.timeline.mark(Phase.SEND)
    leg_a = LegOutcome(
        client_id="x", side=Side.BUY, requested_qty=Decimal("0.6"),
        success=True, status=OrderStatus.FILLED,
        total_fee=Decimal("0.018"),
        venue="aster", kind=LegKind.ENTRY, expected_price=Decimal("100.02"),
        send_ts_ms=1_700_000_000_000, last_ts_ms=1_700_000_000_120,
    )
    leg_a.set_fill(Decimal("0.6"), Decimal("100.03"))
    leg_b = LegOutcome(
        client_id="y", side=Side.SELL, requested_qty=Decimal("0.6"),
        success=True, status=OrderStatus.FILLED,
        venue="lighter", kind=LegKind.ENTRY, expected_price=Decimal("100.04"),
        send_ts_ms=1_700_000_000_000, last_ts_ms=1_700_000_000_330,
    )
    leg_b.set_fill(Decimal("0.6"), Decimal("100.038"))
    d.legs = [leg_a, leg_b]
    rec.emit(d)
    rec.close()

    dec = _read(tmp_path / "decisions_taker_taker_TEST.csv")
    legs = _read(tmp_path / "legs_taker_taker_TEST.csv")
    assert dec[0] == _decision_header()
    assert len(dec) == 2  # header + 1
    row = dict(zip(dec[0], dec[1], strict=True))
    assert row["decision_id"] == "d-abc"
    assert row["outcome"] == "FIRED"
    assert row["lat_decision_send_ms"] == "7"
    assert row["realised_pnl"] == "0.012"

    assert legs[0] == _leg_header()
    assert len(legs) == 3  # header + 2 legs
    l0 = dict(zip(legs[0], legs[1], strict=True))
    assert l0["decision_id"] == "d-abc" and l0["venue"] == "aster"
    assert l0["expected_price"] == "100.02" and l0["realized_price"] == "100.03"
    assert l0["total_fee"] == "0.018"
    assert l0["kind"] == LegKind.ENTRY  # StrEnum serialises to its value
    l1 = dict(zip(legs[0], legs[2], strict=True))
    assert l1["venue"] == "lighter" and l1["total_fee"] == "0"


def test_abort_decision_emits_row_with_no_legs(tmp_path) -> None:
    rec = ExecutionRecorder(tmp_path, run_ts="AB")
    # only the always-known fields; the rest default — proves early aborts
    # (pre-edge) can be recorded without fabricating values.
    d = Decision(
        decision_id="d-x", ts_ms=9, mid_left=Decimal("100"), mid_right=Decimal("100"),
        left_quote_ts_ms=1, right_quote_ts_ms=2,
        outcome=Outcome.ABORT_STALE, abort_reason="quote older than max_stale_ms",
    )
    rec.emit(d)
    rec.close()
    dec = _read(tmp_path / "decisions_taker_taker_AB.csv")
    legs = _read(tmp_path / "legs_taker_taker_AB.csv")
    assert len(dec) == 2 and len(legs) == 1  # decision recorded, no leg rows
    row = dict(zip(dec[0], dec[1], strict=True))
    assert row["outcome"] == "ABORT_STALE"
    assert row["lat_decision_send_ms"] == ""  # never marked → no span
    assert row["direction"] == "" and row["edge_bps"] == "0.0"
