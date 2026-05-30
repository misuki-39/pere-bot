"""Integration test: real Turso (libSQL) round-trip for SqliteRecorder.

Exercises the actual network sync path (not a fake client): emit decisions
through the recorder, push to Turso, then read them back with an INDEPENDENT
client to confirm the rows + values landed. Also checks remote ON CONFLICT
idempotency. Cleans up its own rows (keyed by a unique run_id) on exit.

Skipped unless BOTH are true:
  1. the `turso` extra is installed   →  uv sync --extra turso
  2. creds are in the env:
       TURSO_DATABASE_URL=libsql://<db>.turso.io
       TURSO_AUTH_TOKEN=<token>

Run:  TURSO_DATABASE_URL=... TURSO_AUTH_TOKEN=... \
        uv run --extra turso pytest tests/test_turso_integration.py -q
"""
from __future__ import annotations

import os
import uuid
from decimal import Decimal

import pytest

from perp_arb.core.config import TursoCfg
from perp_arb.core.exec_record import Decision, Direction, Phase, Timeline, Verdict
from perp_arb.core.record_sink import _SYNCED_TABLES, SqliteRecorder, _http_url
from perp_arb.core.types import LegKind, LegOutcome, OrderStatus, Side

libsql_client = pytest.importorskip(
    "libsql_client", reason="install the turso extra: uv sync --extra turso"
)
_URL = os.getenv("TURSO_DATABASE_URL")
_TOKEN = os.getenv("TURSO_AUTH_TOKEN")
pytestmark = pytest.mark.skipif(
    not (_URL and _TOKEN),
    reason="set TURSO_DATABASE_URL + TURSO_AUTH_TOKEN to run the Turso integration test",
)


def _leg(venue: str, side: Side, px: str) -> LegOutcome:
    lg = LegOutcome(success=True, client_id=f"cid-{venue}", side=side,
                    requested_qty=Decimal("0.12"), status=OrderStatus.FILLED,
                    exchange_ts_ms=1_700_000_000_900)
    lg.set_fill(Decimal("0.12"), Decimal(px))
    lg.venue = venue
    lg.send_ts_ms = 1_700_000_000_500
    lg.last_ts_ms = 1_700_000_000_810
    lg.kind = LegKind.ENTRY
    lg.mid = Decimal(px)
    lg.quote_ts_ms = 1_700_000_000_400
    lg.position_before = Decimal("0.24")
    lg.bbo_bid_size = Decimal("5")
    lg.bbo_ask_size = Decimal("7")
    return lg


def _fired(run_id: str) -> Decision:
    tl = Timeline()
    tl.mark_at(Phase.DECISION, 100)
    tl.mark_at(Phase.SEND, 102)
    d = Decision(
        decision_id=f"{run_id}-f0", ts_ms=1_700_000_000_000,
        mid_left=Decimal("88.61"), mid_right=Decimal("88.63"),
        left_quote_ts_ms=1, right_quote_ts_ms=1, bias=Decimal("0.0107"),
        edge_bps=Decimal("1.94"), direction=Direction.B, outcome=Verdict.FIRED,
        timeline=tl, thr_throttle_bps=Decimal("0.5"),
    )
    d.legs = [_leg("lighter", Side.BUY, "88.61"), _leg("aster", Side.SELL, "88.63")]
    d.send_ts_ms = d.legs[0].send_ts_ms
    return d


def _reject(run_id: str) -> Decision:
    return Decision(
        decision_id=f"{run_id}-r0", ts_ms=1_700_000_000_050,
        mid_left=Decimal("88.6"), mid_right=Decimal("88.6"),
        left_quote_ts_ms=1, right_quote_ts_ms=1, outcome=Verdict.ABORT_NO_DEPTH,
        abort_reason="qty does not fill within max_levels", edge_bps=Decimal("0.4"),
        timeline=Timeline(),
    )


async def _delete_run(run_id: str) -> None:
    client = libsql_client.create_client(url=_http_url(_URL), auth_token=_TOKEN)
    try:
        for t in (*_SYNCED_TABLES, "runs"):
            await client.execute(f"DELETE FROM {t} WHERE run_id=?", [run_id])
    finally:
        await client.close()


async def test_turso_roundtrip(tmp_path):
    run_id = "ITEST-" + uuid.uuid4().hex[:12]
    cfg = TursoCfg(
        enabled=True, url=_URL, auth_token=_TOKEN,
        db_path=tmp_path / "live.db",
        sync_interval_s=3600,  # background loop won't race; we push manually
    )
    rec = SqliteRecorder(run_id, strategy_id="itest", mode="paper",
                         config_json='{"qty": "0.12"}', turso=cfg)
    await rec.start()
    try:
        rec.emit(_fired(run_id))
        rec.emit(_reject(run_id))
        await rec._push_all()  # deterministic: remote DDL bootstrap + push

        client = libsql_client.create_client(url=_http_url(_URL), auth_token=_TOKEN)
        try:
            # trades: 1 row, success, direction, lat round-tripped
            rs = await client.execute(
                "SELECT decision_id, direction, success, lat_decision_send_ms, "
                "thr_throttle_bps FROM trades WHERE run_id=?", [run_id])
            assert len(rs.rows) == 1
            row = rs.rows[0]
            assert row[0] == f"{run_id}-f0"
            assert row[1] == "B"
            assert row[2] == 1
            assert row[3] == 2          # SEND(102) - DECISION(100)
            assert row[4] == "0.5"

            # legs: 2 rows; per-venue context lands losslessly as TEXT
            rs = await client.execute(
                "SELECT venue, side, realized_price, position_before, bbo_bid_size "
                "FROM legs WHERE run_id=? ORDER BY venue", [run_id])
            assert len(rs.rows) == 2
            aster, lighter = rs.rows[0], rs.rows[1]
            assert aster[0] == "aster" and lighter[0] == "lighter"
            assert lighter[2] == "88.61"          # realized_price
            assert lighter[3] == "0.24"           # position_before
            assert lighter[4] == "5"              # bbo_bid_size

            # rejection: 1 row, no legs
            rs = await client.execute(
                "SELECT outcome, edge_bps FROM rejections WHERE run_id=?", [run_id])
            assert len(rs.rows) == 1
            assert rs.rows[0][0] == "ABORT_NO_DEPTH"
            assert rs.rows[0][1] == "0.4"

            # run header
            rs = await client.execute(
                "SELECT mode, config_json FROM runs WHERE run_id=?", [run_id])
            assert len(rs.rows) == 1
            assert rs.rows[0][0] == "paper"

            # idempotency: force a re-push of already-synced rows; remote
            # INSERT OR IGNORE must not duplicate.
            rec._conn.execute("UPDATE sync_state SET last_synced_id=0")
            rec._conn.commit()
            await rec._push_all()
            rs = await client.execute(
                "SELECT COUNT(*) FROM legs WHERE run_id=?", [run_id])
            assert rs.rows[0][0] == 2
            rs = await client.execute(
                "SELECT COUNT(*) FROM trades WHERE run_id=?", [run_id])
            assert rs.rows[0][0] == 1
        finally:
            await client.close()
    finally:
        await rec.aclose()
        await _delete_run(run_id)
