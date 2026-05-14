"""Time helpers — keep monotonic vs wall-clock concerns explicit."""

from __future__ import annotations

import time
from datetime import UTC, datetime


def now_ms() -> int:
    """Wall-clock UTC milliseconds. Use for timestamps stored in logs / orders."""
    return int(time.time() * 1000)


def now_us() -> int:
    """Wall-clock UTC microseconds. Aster V3 requires us-precision nonces."""
    return int(time.time() * 1_000_000)


def mono_ms() -> int:
    """Monotonic milliseconds. Use for latency measurement, not for log timestamps."""
    return int(time.monotonic() * 1000)


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
