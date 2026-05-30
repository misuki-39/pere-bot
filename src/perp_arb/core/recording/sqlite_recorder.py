"""Live execution telemetry sink: local SQLite source-of-truth + Turso sync.

The `Recorder` backend for the **live** path (the backtest uses the CSV
`CsvRecorder`). One `Decision` per evaluated tick is routed by outcome into a
normalised three-table model:

* `rejections` — ticks we declined to fire (ABORT_* / BLOCKED_RISK). Diagnostic,
  no legs.
* `trades`     — ticks we fired (incl. partial failures). Pair-level header.
* `legs`       — per-leg execution detail (2 rows per trade); carries all
  per-venue data (price, fill, fee, decision-time mid / quote-ts / position /
  depth).

Write architecture (local-first):

* `emit()` runs on the event loop and does a synchronous INSERT into a local
  SQLite file (WAL, `synchronous=NORMAL`) — ~µs, never touches the network,
  never drops rows. This is the on-box source of truth.
* When Turso is enabled, a background asyncio task replicates new rows to the
  cloud in batches, tracked by a local-only `sync_state` cursor. A network
  stall leaves rows safely in local SQLite to be retried — nothing is lost.

Money/size are stored as TEXT (lossless `Decimal`, matching the project's
Parquet/CSV convention); timestamps and booleans as INTEGER.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .decision import Decision, Verdict
from .recorder import Recorder

if TYPE_CHECKING:
    from ..config import TursoCfg
    from ..types import LegOutcome

_log = logging.getLogger(__name__)


# ----- schema -----------------------------------------------------------------
#
# `_SCHEMA` is the single source of truth: per synced table, the ordered
# (column, sqltype) pairs + the UNIQUE key. Everything else — the INSERT/SELECT
# column tuples, the local DDL (with an autoincrement `id` driving the sync
# cursor), and the remote DDL (no `id`; the remote assigns its own and dedupes
# on the UNIQUE key) — is generated from it, so local and remote can't drift.

_SCHEMA: dict[str, dict] = {
    "rejections": {
        "cols": (("run_id", "TEXT"), ("decision_id", "TEXT"), ("ts_ms", "INTEGER"),
                 ("outcome", "TEXT"), ("abort_reason", "TEXT"), ("edge_bps", "TEXT"),
                 ("bias", "TEXT")),
        "unique": "decision_id",
    },
    "trades": {
        "cols": (("run_id", "TEXT"), ("decision_id", "TEXT"), ("ts_ms", "INTEGER"),
                 ("direction", "TEXT"), ("edge_bps", "TEXT"), ("bias", "TEXT"),
                 ("send_ts_ms", "INTEGER"), ("lat_decision_send_ms", "INTEGER"),
                 ("realised_pnl", "TEXT"), ("success", "INTEGER"),
                 ("failure_reason", "TEXT"), ("thr_throttle_bps", "TEXT")),
        "unique": "decision_id",
    },
    "legs": {
        "cols": (("run_id", "TEXT"), ("decision_id", "TEXT"), ("ts_ms", "INTEGER"),
                 ("venue", "TEXT"), ("side", "TEXT"), ("requested_qty", "TEXT"),
                 ("filled_qty", "TEXT"), ("expected_price", "TEXT"),
                 ("realized_price", "TEXT"), ("status", "TEXT"), ("success", "INTEGER"),
                 ("error_message", "TEXT"), ("client_id", "TEXT"), ("total_fee", "TEXT"),
                 ("send_ts_ms", "INTEGER"), ("fill_ts_ms", "INTEGER"), ("kind", "TEXT"),
                 ("mid", "TEXT"), ("quote_ts_ms", "INTEGER"), ("position_before", "TEXT"),
                 ("bbo_bid_size", "TEXT"), ("bbo_ask_size", "TEXT")),
        "unique": "decision_id, venue, kind",
    },
}
# Data tables replicated to Turso (runs is pushed once at start, not cursor-synced).
_SYNCED_TABLES = tuple(_SCHEMA)


def _cols(table: str) -> tuple[str, ...]:
    return tuple(c for c, _ in _SCHEMA[table]["cols"])


def _coldefs(table: str) -> str:
    return ", ".join(f"{c} {t}" for c, t in _SCHEMA[table]["cols"])


_RUNS_DDL = ("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, "
             "strategy_id TEXT, mode TEXT, started_at_ms INTEGER, config_json TEXT);")
_SYNC_STATE_DDL = ("CREATE TABLE IF NOT EXISTS sync_state (table_name TEXT PRIMARY KEY,"
                   " last_synced_id INTEGER NOT NULL DEFAULT 0);")

# Local DDL: runs + (id-prefixed synced tables) + the local-only sync cursor.
_LOCAL_DDL = "\n".join([
    _RUNS_DDL,
    *(f"CREATE TABLE IF NOT EXISTS {t} (id INTEGER PRIMARY KEY AUTOINCREMENT, "
      f"{_coldefs(t)}, UNIQUE({_SCHEMA[t]['unique']}));" for t in _SYNCED_TABLES),
    _SYNC_STATE_DDL,
])
# Remote DDL: runs + synced tables without `id` (remote assigns its own).
_REMOTE_DDL = (
    _RUNS_DDL,
    *(f"CREATE TABLE IF NOT EXISTS {t} ({_coldefs(t)}, "
      f"UNIQUE({_SCHEMA[t]['unique']}));" for t in _SYNCED_TABLES),
)


def _txt(v: object) -> str | None:
    """Lossless TEXT for money/size: Decimal → fixed-point, else str."""
    if v is None:
        return None
    if isinstance(v, Decimal):
        return format(v, "f")
    return str(v)


def _int(v: int | bool | None) -> int | None:
    if v is None:
        return None
    return int(v)  # bool → 0/1 too


def _http_url(url: str) -> str:
    """Map a Turso `libsql://` (or `wss://`) URL to the HTTP transport the
    client speaks reliably. Pass-through for `http(s)://`."""
    for src, dst in (("libsql://", "https://"), ("wss://", "https://"), ("ws://", "http://")):
        if url.startswith(src):
            return dst + url[len(src):]
    return url


class SqliteRecorder(Recorder):
    """Live telemetry recorder. `emit` / `emit_legs` are the write entry points
    (the `Recorder` contract).

    Construct synchronously (opens the local DB, creates tables, records the
    run); call `await start()` to spin up the Turso sync task and `await
    aclose()` on shutdown — the lifecycle is backend-specific and not part of
    the `Recorder` contract."""

    def __init__(
        self,
        run_id: str,
        *,
        strategy_id: str,
        mode: str,
        config_json: str,
        turso: TursoCfg,
    ) -> None:
        self._run_id = run_id
        self._turso = turso
        db_path = Path(turso.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_LOCAL_DDL)
        self._conn.execute(
            "INSERT OR IGNORE INTO sync_state (table_name, last_synced_id) "
            "VALUES " + ",".join(f"('{t}', 0)" for t in _SYNCED_TABLES)
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO runs (run_id, strategy_id, mode, started_at_ms, "
            "config_json) VALUES (?,?,?,?,?)",
            (run_id, strategy_id, mode, _run_started_ms(run_id), config_json),
        )
        self._conn.commit()
        self._sync_task: asyncio.Task[None] | None = None
        self._client: Any = None
        self._remote_ready = False
        _log.info("sqlite recorder -> %s (turso=%s)", db_path,
                  "on" if turso.enabled else "off")

    # ----- write (hot path, synchronous) -----

    def emit(self, d: Decision) -> None:
        """Record one Decision header by outcome. FIRED → trades; everything
        else (ABORT_* / BLOCKED_RISK) → rejections. Legs are recorded separately
        via `emit_legs`. Synchronous local write."""
        if d.outcome is Verdict.FIRED:
            self._insert("trades", _cols("trades"), self._trade_row(d))
        else:
            self._insert("rejections", _cols("rejections"), self._rejection_row(d))
        self._conn.commit()

    def emit_legs(self, decision_id: str, ts_ms: int, legs: Sequence[LegOutcome]) -> None:
        """Record the per-leg execution detail for a fired trade. Separate from
        `emit` (own commit) — a trade header with its legs briefly absent is a
        harmless, detectable audit gap, not a corrupted trade."""
        for leg in legs:
            self._insert("legs", _cols("legs"), self._leg_row(decision_id, ts_ms, leg))
        self._conn.commit()

    def _insert(self, table: str, cols: tuple[str, ...], row: tuple) -> None:
        placeholders = ",".join("?" * len(cols))
        self._conn.execute(
            f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({placeholders})",
            row,
        )

    def _rejection_row(self, d: Decision) -> tuple:
        return (
            self._run_id, d.decision_id, _int(d.ts_ms), d.outcome.value,
            d.abort_reason, _txt(d.edge_bps.quantize(Decimal("0.1"))), _txt(d.bias),
        )

    def _trade_row(self, d: Decision) -> tuple:
        success = d.failure_reason is None
        failure = d.failure_reason
        return (
            self._run_id, d.decision_id, _int(d.ts_ms),
            d.direction.value if d.direction else None,
            _txt(d.edge_bps.quantize(Decimal("0.1"))), _txt(d.bias),
            _int(d.send_ts_ms), _int(d.timeline.latencies()["lat_decision_send_ms"]),
            _txt(d.realised_pnl), int(success), failure, _txt(d.thr_throttle_bps),
        )

    def _leg_row(self, decision_id: str, ts_ms: int, lg: LegOutcome) -> tuple:
        return (
            self._run_id, decision_id, _int(ts_ms),
            lg.venue, lg.side.value if lg.side else None, _txt(lg.requested_qty),
            _txt(lg.filled_qty if lg.success else None), _txt(lg.expected_price),
            _txt(lg.avg_price), lg.status.value, int(lg.success), lg.error_message,
            lg.client_id, _txt(lg.total_fee), _int(lg.send_ts_ms), _int(lg.fill_ts_ms),
            lg.kind.value if lg.kind else None,
            _txt(lg.mid), _int(lg.quote_ts_ms), _txt(lg.position_before),
            _txt(lg.bbo_bid_size), _txt(lg.bbo_ask_size),
        )

    # ----- Turso sync (background) -----

    async def start(self) -> None:
        if not self._turso.enabled:
            return
        if self._turso.is_placeholder:
            _log.warning("turso enabled but url/auth_token missing — running local-only")
            return
        try:
            import libsql_client  # lazy: only a hard dep when actually syncing
        except ImportError:
            _log.error("turso enabled but libsql-client not installed — local-only")
            return
        # create_client only constructs — no network here, so a remote that's
        # unreachable at startup can never crash the bot. The remote schema
        # bootstrap happens inside the retrying sync loop.
        #
        # Force the HTTP transport: Turso hands out `libsql://` URLs, which this
        # client maps to the legacy Hrana-over-WebSocket protocol that current
        # Turso servers reject (WS 400). `https://` uses Hrana-over-HTTP, which
        # works.
        url = _http_url(self._turso.url)
        self._client = libsql_client.create_client(
            url=url, auth_token=self._turso.auth_token
        )
        self._remote_ready = False
        self._sync_task = asyncio.create_task(self._sync_loop(), name="turso-sync")
        _log.info("turso sync started -> %s", url)

    async def aclose(self) -> None:
        if self._sync_task is not None:
            self._sync_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._sync_task
            self._sync_task = None
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.close()
            self._client = None
        self._conn.commit()
        self._conn.close()

    async def _sync_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._turso.sync_interval_s)
                await self._push_all()
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await self._push_all()  # best-effort final drain
            raise

    async def _push_all(self) -> None:
        # First successful pass creates the remote schema + pushes the run row.
        # Folded into the retry loop so a network blip at startup just defers it.
        if not self._remote_ready:
            try:
                await self._client.batch(list(_REMOTE_DDL))
                await self._push_runs()
                self._remote_ready = True
            except Exception:  # noqa: BLE001 — retry next interval, never crash
                _log.warning("turso remote bootstrap failed (will retry)", exc_info=True)
                return
        for table in _SYNCED_TABLES:
            try:
                await self._push_table(table)
            except Exception:  # noqa: BLE001 — a remote stall must not kill the loop
                _log.warning("turso push failed for %s (will retry)", table, exc_info=True)
                return  # leave cursor; retry the whole batch next interval

    async def _push_table(self, table: str) -> None:
        cols = _cols(table)
        cur = self._conn.execute(
            "SELECT last_synced_id FROM sync_state WHERE table_name=?", (table,)
        ).fetchone()[0]
        rows = self._conn.execute(
            f"SELECT id,{','.join(cols)} FROM {table} WHERE id>? ORDER BY id LIMIT ?",
            (cur, self._turso.sync_batch_rows),
        ).fetchall()
        if not rows:
            return
        placeholders = ",".join("?" * len(cols))
        sql = f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
        stmts = [(sql, list(r[1:])) for r in rows]
        await self._client.batch(stmts)
        new_cur = rows[-1][0]
        self._conn.execute(
            "UPDATE sync_state SET last_synced_id=? WHERE table_name=?", (new_cur, table)
        )
        self._conn.commit()
        _log.debug("turso pushed %d rows to %s (cursor=%d)", len(rows), table, new_cur)

    async def _push_runs(self) -> None:
        row = self._conn.execute(
            "SELECT run_id,strategy_id,mode,started_at_ms,config_json FROM runs "
            "WHERE run_id=?", (self._run_id,)
        ).fetchone()
        if row is not None:
            await self._client.batch([(
                "INSERT OR IGNORE INTO runs (run_id,strategy_id,mode,started_at_ms,"
                "config_json) VALUES (?,?,?,?,?)", list(row),
            )])


def _run_started_ms(run_id: str) -> int | None:
    """Parse the run-ts label (YYYYMMDDTHHMMSSZ) to epoch ms; None if unparseable."""
    from datetime import UTC, datetime
    try:
        dt = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None
    return int(dt.timestamp() * 1000)
