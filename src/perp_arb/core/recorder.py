"""The telemetry recorder seam.

`Recorder` is the write contract shared by both backends — the CSV
`CsvRecorder` (backtest) and the SQLite `SqliteRecorder` (live). Recorder-
agnostic consumers (the backtest engine, `StrategyContext`) depend on this ABC,
never on a concrete backend, so the two are interchangeable.

The contract is only the two write methods. Lifecycle is deliberately *not*
here: a batch CSV sink closes synchronously (`close()`), a streaming SQLite sink
opens/closes asynchronously with a background cloud-sync task
(`start()`/`aclose()`). Those are different abstractions and stay on the
concrete classes; whoever constructs a recorder owns its lifecycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .decision import Decision
from .types import LegOutcome


class Recorder(ABC):
    """Persist one evaluated `Decision` and, for fired trades, its legs."""

    @abstractmethod
    def emit(self, d: Decision) -> None:
        """Record one decision header (routed by `d.outcome`)."""

    @abstractmethod
    def emit_legs(self, decision_id: str, ts_ms: int, legs: Sequence[LegOutcome]) -> None:
        """Record the per-leg execution detail for a fired trade."""
