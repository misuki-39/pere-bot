"""Multi-day capture primitive: queue-decoupled, hourly-rotated Parquet.

Designed for unattended runs of several days. Two properties matter:

* The producer (a WS callback on the event loop) must never block on disk.
  `submit()` is a non-blocking `put_nowait`; if the writer falls behind and the
  bounded queue fills, rows are dropped and counted rather than stalling the
  socket recv loop.
* A crash/restart must not cost more than the in-flight batch. Each wall-clock
  hour is written to its own file and finalized on rotation, so every closed
  hour is a complete, independently-readable Parquet file:

      <root>/date=YYYY-MM-DD/HH.parquet

Numeric fields are stored as strings (lossless for `Decimal`, matching the
project's CSV convention); the analysis step casts as needed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_log = logging.getLogger(__name__)


def _bucket(ts_ms: int) -> tuple[str, str]:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H")


class RotatingParquetWriter:
    """Append rows via `submit()`; a background task batches them to Parquet."""

    def __init__(
        self,
        root: Path,
        schema: pa.Schema,
        *,
        rotation_minutes: int = 60,
        batch_size: int = 2000,
        queue_max: int = 100_000,
    ) -> None:
        self.root = Path(root)
        self.schema = schema
        self.batch_size = batch_size
        # rotation_minutes is honored at hour granularity for the on-disk path;
        # values <60 just rotate more often within the hour partition dir.
        self._rotation_min = max(1, rotation_minutes)
        self._q: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=queue_max)
        self._task: asyncio.Task[None] | None = None
        self._writer: pq.ParquetWriter | None = None
        self._cur_key: tuple[str, str, int] | None = None
        self.dropped = 0
        self._last_drop_log = 0.0

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="parquet-writer")

    def submit(self, row: dict) -> None:
        """Non-blocking. Drops + counts on a full queue — never blocks the caller."""
        try:
            self._q.put_nowait(row)
        except asyncio.QueueFull:
            self.dropped += 1
            loop_t = asyncio.get_event_loop().time()
            if loop_t - self._last_drop_log >= 10.0:
                self._last_drop_log = loop_t
                _log.warning("parquet writer behind: dropped=%d (queue full)", self.dropped)

    async def close(self) -> None:
        if self._task is None:
            return
        await self._q.put(None)  # sentinel: drain then stop
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        self._finalize()

    # ---- writer task ----

    async def _run(self) -> None:
        batch: list[dict] = []
        try:
            while True:
                row = await self._q.get()
                if row is None:
                    self._flush(batch)
                    return
                batch.append(row)
                if len(batch) >= self.batch_size or self._q.empty():
                    self._flush(batch)
                    batch = []
        except asyncio.CancelledError:
            self._flush(batch)
            raise
        except Exception:  # noqa: BLE001 — a writer crash must not be silent
            _log.exception("parquet writer task failed")
            raise

    def _slot(self, ts_ms: int) -> tuple[str, str, int]:
        date, hour = _bucket(ts_ms)
        minute = datetime.fromtimestamp(ts_ms / 1000, tz=UTC).minute
        sub = minute // self._rotation_min if self._rotation_min < 60 else 0
        return date, hour, sub

    def _path(self, key: tuple[str, str, int]) -> Path:
        date, hour, sub = key
        name = f"{hour}.parquet" if sub == 0 and self._rotation_min >= 60 else f"{hour}-{sub:02d}.parquet"
        return self.root / f"date={date}" / name

    def _flush(self, batch: list[dict]) -> None:
        if not batch:
            return
        # Capture is time-ordered, so equal slots form contiguous runs: write
        # each run as one row group, rotating the file when the slot changes.
        run: list[dict] = []
        run_key: tuple[str, str, int] | None = None
        names = self.schema.names
        for row in batch:
            key = self._slot(int(row["ts_ms"]))
            if run and key != run_key:
                self._write_run(run_key, run, names)
                run = []
            run_key = key
            run.append(row)
        if run:
            self._write_run(run_key, run, names)

    def _write_run(self, key: tuple[str, str, int] | None, rows: list[dict], names: list[str]) -> None:
        if key != self._cur_key:
            self._finalize()
            path = self._path(key)  # type: ignore[arg-type]
            path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(str(path), self.schema)
            self._cur_key = key
            _log.info("parquet rotate -> %s", path)
        tbl = pa.Table.from_pylist(
            [{n: _s(r.get(n)) for n in names} for r in rows], schema=self.schema
        )
        assert self._writer is not None
        self._writer.write_table(tbl)

    def _finalize(self) -> None:
        if self._writer is not None:
            with contextlib.suppress(Exception):
                self._writer.close()
            self._writer = None


def _s(v: object) -> object:
    """Stringify Decimals/floats losslessly; pass through int/bool/None/str."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return f"{v:.10g}"
    return format(v, "f")  # Decimal -> full fixed-point text
