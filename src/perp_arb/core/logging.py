"""Centralised logging + CSV trade ledger.

Separates two concerns:
  * Standard Python logging via stdlib (per-strategy file + stderr).
  * Structured append-only CSVs: one for tick-level spread data (monitor mode)
    and one for fills (paper/live modes).
"""

from __future__ import annotations

import atexit
import csv
import logging
import queue
import sys
from datetime import UTC, datetime
from decimal import Decimal
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Any, TextIO

_listener: QueueListener | None = None


def _stop_listener() -> None:
    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None

_DEFAULT_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class RateLimited:
    """Counts events; lets the caller emit at most once per `every_s`.

    Stops a per-message failure (a malformed-frame storm at tick rate, days
    long) from flooding the log even with rotation. `tick(now)` returns the
    number of events since the last emit when it's time to log again, else
    None.
    """

    def __init__(self, every_s: float = 10.0) -> None:
        self.every_s = every_s
        self._count = 0
        self._last = 0.0

    def tick(self, now: float) -> int | None:
        self._count += 1
        if now - self._last >= self.every_s:
            self._last = now
            n, self._count = self._count, 0
            return n
        return None


def setup_logging(log_dir: Path, level: str = "INFO", run_tag: str = "run") -> Path:
    """Install stderr + rotating-file handlers behind a QueueListener.

    Hot-path callers (`_log.info(...)` during order firing) only enqueue
    a LogRecord (~5µs); a background thread does the actual stderr + file
    writes. Without this, every INFO line on the fire path blocks on two
    synchronous write() syscalls (50-300µs each), and we sit right above
    a real `ws.send`.
    """
    global _listener

    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{run_tag}_{ts}.log"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # avoid duplicate handlers if setup_logging is called twice in the same process
    for h in list(root.handlers):
        root.removeHandler(h)
    _stop_listener()

    fmt = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)

    # Rotating, not plain: a multi-day run that hits a reconnect / bad-frame
    # warning storm must not fill a small VPS disk (a full disk stalls the
    # Parquet writer thread). 50 MB × 5 ≈ 250 MB hard cap.
    fh = RotatingFileHandler(
        log_path, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    fh.setFormatter(fmt)

    # Unbounded queue: a stalled listener thread should not drop log records
    # silently. Memory bound is naturally enforced by the disk-side rotation
    # (50 MB × 5) — a runaway log storm rotates files, it doesn't grow RAM.
    log_queue: queue.Queue[Any] = queue.Queue(-1)
    root.addHandler(QueueHandler(log_queue))

    _listener = QueueListener(log_queue, sh, fh, respect_handler_level=False)
    _listener.start()
    atexit.register(_stop_listener)

    # silence noisy upstream libs unless we explicitly want their detail
    for noisy in ("websockets", "aiohttp", "lighter"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return log_path


class CsvWriter:
    """Append-only CSV writer. Thread-safe via a Lock."""

    def __init__(self, path: Path, header: list[str]):
        self.path = path
        self.header = header
        self._lock = Lock()
        self._fp: TextIO | None = None
        # `csv.writer` is a factory function whose returned _writer object is
        # not in any public type — Any here matches what cpython exposes.
        self._writer: Any = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists()
        self._fp = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.writer(self._fp)
        if is_new:
            self._writer.writerow(header)
            self._fp.flush()

    def write(self, row: list) -> None:
        with self._lock:
            assert self._writer is not None
            assert self._fp is not None
            self._writer.writerow([_format(v) for v in row])
            # flush() only (no os.fsync): userspace → page cache, ~1µs, CPU
            # only. This is load-bearing — taker_taker records a row on every
            # gate-aborted/blocked tick; a per-row fsync here would turn that
            # into hot-path I/O. Keep it flush-not-fsync.
            self._fp.flush()

    def close(self) -> None:
        with self._lock:
            if self._fp is not None:
                self._fp.close()
                self._fp = None
                self._writer = None


def _format(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, Decimal):
        # keep full precision; strip trailing zeros only after decimal point
        s = format(v, "f")
        return s
    if isinstance(v, float):
        return f"{v:.10g}"
    return str(v)


SPREAD_CSV_HEADER = [
    "ts_ms",
    "aster_bid", "aster_bid_size", "aster_ask", "aster_ask_size",
    "lighter_bid", "lighter_bid_size", "lighter_ask", "lighter_ask_size",
    "mid_aster", "mid_lighter", "raw_spread", "bias_ewma",
    "vwap_a_sell", "vwap_a_buy", "vwap_l_sell", "vwap_l_buy",
    "edge_A_bps", "edge_B_bps", "gates_passed",
]
# taker_taker telemetry headers live with their dataclasses in
# core.recording.csv_recorder (derived from fields, so they cannot drift).
