"""Tests for the live SQLite recorder (core/recording/sqlite_recorder.py).

Covers: outcome routing into the three-table model, local round-trip of the
per-leg context fields, ON CONFLICT idempotency, and the Turso sync cursor
advancing against a fake client. No network — the local SQLite path is the
unit under test; the remote push is exercised via a stub client.
"""

from __future__ import annotations

import sqlite3
from decimal import Decimal

from perp_arb.core.config import TursoCfg
from perp_arb.core.recording.decision import Decision, Direction, Phase, Timeline, Verdict
from perp_arb.core.recording.sqlite_recorder import SqliteRecorder
from perp_arb.core.types import LegKind, LegOutcome, OrderStatus, Side

_RUN = "20260530T120000Z"


def _cfg(tmp_path, *, enabled=False) -> TursoCfg:
    return TursoCfg(enabled=enabled, db_path=tmp_path / "live.db")


def _recorder(tmp_path, **kw) -> SqliteRecorder:
    return SqliteRecorder(
        _RUN, strategy_id="taker_taker", mode="paper",
        config_json='{"qty": "0.12"}', turso=_cfg(tmp_path, **kw),
    )


def _leg(venue: str, side: Side, *, success=True, error=None) -> LegOutcome:
    lg = LegOutcome(
        success=success, client_id=f"cid-{venue}", side=side,
        requested_qty=Decimal("0.12"),
        status=OrderStatus.FILLED if success else OrderStatus.REJECTED,
        exchange_ts_ms=1_700_000_000_900, error_message=error,
    )
    if success:
        lg.set_fill(Decimal("0.12"), Decimal("88.62"))
    lg.venue = venue
    lg.send_ts_ms = 1_700_000_000_500
    lg.last_ts_ms = 1_700_000_000_810
    lg.kind = LegKind.ENTRY
    # decision-time per-venue context (stamped by the live strategy)
    lg.bbo_bid = Decimal("88.6150")
    lg.bbo_ask = Decimal("88.6200")
    lg.quote_ts_ms = 1_700_000_000_400
    lg.position_before = Decimal("0.24")
    lg.bbo_bid_size = Decimal("5")
    lg.bbo_ask_size = Decimal("7")
    return lg


def _decision(outcome: Verdict, *, did="d-1", legs=None, failure_reason=None) -> Decision:
    tl = Timeline()
    tl.mark_at(Phase.DECISION, 100)
    tl.mark_at(Phase.SEND, 101)
    d = Decision(
        decision_id=did, ts_ms=1_700_000_000_000,
        mid_left=Decimal("88.6175"), mid_right=Decimal("88.6250"),
        left_quote_ts_ms=1_700_000_000_400, right_quote_ts_ms=1_700_000_000_300,
        bias=Decimal("0.0107"), edge_bps=Decimal("1.94"),
        direction=Direction.B if outcome is Verdict.FIRED else None,
        outcome=outcome, timeline=tl,
        failure_reason=failure_reason,
    )
    if legs is not None:
        d.send_ts_ms = legs[0].send_ts_ms
    return d


def _emit_fired(rec, d, legs) -> None:
    """Helper: a fired trade records its header then its legs (two writes)."""
    rec.emit(d)
    rec.emit_legs(d.decision_id, d.ts_ms, legs)


def _rows(tmp_path, table: str) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(tmp_path / "live.db"))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
    finally:
        conn.close()


# ----- routing ----------------------------------------------------------------


def test_fired_routes_to_trades_and_legs(tmp_path):
    rec = _recorder(tmp_path)
    legs = [_leg("lighter", Side.BUY), _leg("aster", Side.SELL)]
    _emit_fired(rec, _decision(Verdict.FIRED, legs=legs), legs)
    trades, legrows, rejs = (_rows(tmp_path, t) for t in ("trades", "legs", "rejections"))
    assert len(trades) == 1 and len(legrows) == 2 and len(rejs) == 0
    t = trades[0]
    assert t["decision_id"] == "d-1"
    assert t["direction"] == "B"
    assert t["success"] == 1
    assert t["failure_reason"] is None
    assert t["lat_decision_send_ms"] == 1          # SEND(101) - DECISION(100)
    # per-leg context round-trips losslessly as TEXT
    lg = legrows[0]
    assert lg["venue"] == "lighter"
    assert lg["position_before"] == "0.24"
    assert lg["bbo_bid_size"] == "5" and lg["bbo_ask_size"] == "7"
    assert lg["bbo_bid"] == "88.6150" and lg["bbo_ask"] == "88.6200"
    assert lg["realized_price"] == "88.62"


def test_abort_routes_to_rejections_only(tmp_path):
    rec = _recorder(tmp_path)
    rec.emit(_decision(Verdict.ABORT_STALE, did="d-rej"))
    trades, legrows, rejs = (_rows(tmp_path, t) for t in ("trades", "legs", "rejections"))
    assert len(trades) == 0 and len(legrows) == 0 and len(rejs) == 1
    assert rejs[0]["outcome"] == "ABORT_STALE"
    assert rejs[0]["edge_bps"] == "1.9"            # quantized to 0.1


def test_partial_failure_routes_to_trades_with_failure(tmp_path):
    rec = _recorder(tmp_path)
    legs = [_leg("lighter", Side.BUY),
            _leg("aster", Side.SELL, success=False, error="timeout")]
    d = _decision(Verdict.FIRED, legs=legs, failure_reason="aster: timeout")
    _emit_fired(rec, d, legs)
    trades, legrows = _rows(tmp_path, "trades"), _rows(tmp_path, "legs")
    assert len(trades) == 1 and len(legrows) == 2
    assert trades[0]["success"] == 0
    assert "aster: timeout" in trades[0]["failure_reason"]
    # failed leg stores no filled_qty (None), preserving the no-fill distinction
    failed = [r for r in legrows if r["venue"] == "aster"][0]
    assert failed["filled_qty"] is None and failed["success"] == 0


def test_idempotent_on_duplicate_decision_id(tmp_path):
    rec = _recorder(tmp_path)
    legs = [_leg("lighter", Side.BUY), _leg("aster", Side.SELL)]
    _emit_fired(rec, _decision(Verdict.FIRED, legs=legs), legs)
    _emit_fired(rec, _decision(Verdict.FIRED, legs=legs), legs)  # same decision_id -> ignored
    assert len(_rows(tmp_path, "trades")) == 1
    assert len(_rows(tmp_path, "legs")) == 2


def test_run_row_recorded(tmp_path):
    _recorder(tmp_path)
    runs = _rows(tmp_path, "runs")
    assert len(runs) == 1
    assert runs[0]["run_id"] == _RUN
    assert runs[0]["mode"] == "paper"
    assert runs[0]["config_json"] == '{"qty": "0.12"}'


# ----- Turso sync cursor (fake client, no network) ----------------------------


class _FakeClient:
    def __init__(self):
        self.batches: list[list] = []

    async def batch(self, stmts):
        self.batches.append(list(stmts))


async def test_push_table_advances_cursor(tmp_path):
    rec = _recorder(tmp_path, enabled=True)
    rec._client = _FakeClient()
    for i in range(3):
        rec.emit(_decision(Verdict.ABORT_NO_DEPTH, did=f"d-{i}"))

    await rec._push_table("rejections")
    # all 3 pushed, cursor at the last rowid
    assert len(rec._client.batches) == 1
    assert len(rec._client.batches[0]) == 3
    cur = rec._conn.execute(
        "SELECT last_synced_id FROM sync_state WHERE table_name='rejections'"
    ).fetchone()[0]
    assert cur == 3

    # a second push with no new rows is a no-op (no extra batch)
    await rec._push_table("rejections")
    assert len(rec._client.batches) == 1

    # a new row advances only by the delta
    rec.emit(_decision(Verdict.ABORT_NO_DEPTH, did="d-3"))
    await rec._push_table("rejections")
    assert len(rec._client.batches) == 2
    assert len(rec._client.batches[1]) == 1


async def test_local_only_when_turso_disabled(tmp_path):
    rec = _recorder(tmp_path, enabled=False)
    await rec.start()                       # no client, no task
    assert rec._client is None and rec._sync_task is None
    rec.emit(_decision(Verdict.ABORT_STALE, did="d-x"))
    assert len(_rows(tmp_path, "rejections")) == 1
    await rec.aclose()
