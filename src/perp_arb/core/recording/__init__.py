"""Execution telemetry: the decision domain model + pluggable recorders.

`Decision` (and its value types) is the pre-trade record the strategy produces;
`Recorder` is the write seam both backends implement — `CsvRecorder` (backtest)
and `SqliteRecorder` (live). Importers depend on this package, not the
individual modules, so the split into domain / seam / backends stays internal.
"""

from __future__ import annotations

from .csv_recorder import CsvRecorder
from .decision import Decision, Direction, Phase, Timeline, Verdict
from .recorder import Recorder
from .sqlite_recorder import SqliteRecorder

__all__ = [
    "CsvRecorder",
    "Decision",
    "Direction",
    "Phase",
    "Recorder",
    "SqliteRecorder",
    "Timeline",
    "Verdict",
]
