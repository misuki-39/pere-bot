"""Multi-day capture primitive: queue-decoupled, hourly Parquet.

Designed for unattended runs of several days. Two properties matter:

* The producer (a WS callback on the event loop) must never block on disk.
  `submit()` is a non-blocking `put_nowait`; if the writer falls behind and the
  bounded queue fills, rows are dropped and counted rather than stalling the
  socket recv loop or growing memory without bound. Rows are buffered and
  written in batches (by size or a time bound) as one Parquet row group per
  flush — never one row group per row — and the blocking pyarrow write/close
  runs in a worker thread so a slow disk or the hourly close cannot freeze the
  event loop.
* A clean shutdown loses nothing; an abnormal kill costs at most the
  currently-open hour. Each wall-clock hour is its own file, finalized on
  rotation or on `close()`:

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

_CLOSE = object()  # queue sentinel: drain remaining batch then stop
_FLUSH = object()  # internal: time-bound flush tick


def _hour_key(ts_ms: int) -> tuple[str, str]:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H")


class RotatingParquetWriter:
    """Append rows via `submit()`; a background task batches them to Parquet."""

    def __init__(
        self,
        root: Path,
        schema: pa.Schema,
        *,
        batch_size: int = 2000,
        flush_interval_s: float = 5.0,
        queue_max: int = 50_000,
    ) -> None:
        self.root = Path(root)
        self.schema = schema
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self._q: asyncio.Queue[object] = asyncio.Queue(maxsize=queue_max)
        self._task: asyncio.Task[None] | None = None
        self._writer: pq.ParquetWriter | None = None
        self._cur_key: tuple[str, str] | None = None
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
        with contextlib.suppress(asyncio.QueueFull):
            self._q.put_nowait(_CLOSE)  # drain remaining batch then stop
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        await asyncio.to_thread(self._finalize)

    # ---- writer task ----

    async def _run(self) -> None:
        """Buffer rows; flush a batch on size OR a time bound. The blocking
        pyarrow write/close runs in a thread so a slow disk or the hourly
        close can never freeze the event loop (and the WS feeds with it)."""
        loop = asyncio.get_event_loop()
        batch: list[dict] = []
        deadline = loop.time() + self.flush_interval_s
        try:
            while True:
                # Block indefinitely when idle; otherwise wake to flush by the
                # time bound even if batch_size hasn't been reached.
                timeout = None if not batch else max(0.0, deadline - loop.time())
                try:
                    item = await asyncio.wait_for(self._q.get(), timeout)
                except TimeoutError:
                    item = _FLUSH
                if item is _CLOSE:
                    await self._flush_async(batch)
                    return
                if item is not _FLUSH:
                    batch.append(item)  # type: ignore[arg-type]
                if batch and (item is _FLUSH or len(batch) >= self.batch_size):
                    await self._flush_async(batch)
                    batch = []
                    deadline = loop.time() + self.flush_interval_s
        except asyncio.CancelledError:
            self._flush(batch)  # best-effort, synchronous on shutdown
            raise
        except Exception:  # noqa: BLE001 — a writer crash must not be silent
            _log.exception("parquet writer task failed")
            raise

    async def _flush_async(self, batch: list[dict]) -> None:
        if batch:
            await asyncio.to_thread(self._flush, batch)

    def _path(self, key: tuple[str, str]) -> Path:
        date, hour = key
        return self.root / f"date={date}" / f"{hour}.parquet"

    def _flush(self, batch: list[dict]) -> None:
        if not batch:
            return
        # Capture is time-ordered, so equal hours form contiguous runs: write
        # each run as one row group, rotating the file when the hour changes.
        run: list[dict] = []
        run_key: tuple[str, str] | None = None
        names = self.schema.names
        for row in batch:
            key = _hour_key(int(row["ts_ms"]))
            if run and key != run_key:
                self._write_run(run_key, run, names)
                run = []
            run_key = key
            run.append(row)
        if run:
            self._write_run(run_key, run, names)

    def _write_run(
        self, key: tuple[str, str] | None, rows: list[dict], names: list[str]
    ) -> None:
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
