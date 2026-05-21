"""Latency model + per-venue BookIndex.

We model only `submit_delay_ms` (bot → venue match latency) because the
capture's `ts_ms` already encodes the recorder's feed delay. Adverse
selection is realised by `BookIndex.book_at(arrival_ts)`: given the moment
an order lands at venue V, find the freshest BBORow whose V-source ts ≤
arrival_ts. Looking up by venue-source ts (not by `ts_ms`) avoids leaking
forward information from rows where only the *other* venue ticked.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from .dataset import BBORow


@dataclass(frozen=True, slots=True)
class LatencyModel:
    """Per-venue constant submit delay. Missing venues default to 0."""
    submit_delay_ms: dict[str, int]

    def arrival_ts(self, venue: str, decision_ts_ms: int) -> int:
        return decision_ts_ms + self.submit_delay_ms.get(venue, 0)


class BookIndex:
    """Indexes rows by a venue's source-ts so we can answer
    "freshest book V saw at or before arrival_ts" in O(log n).

    Only rows where the venue's source ts actually advanced are kept; that
    way duplicate `*_ts_ms` (from ticks where the OTHER venue moved) don't
    appear as fake "tick" events for this side.
    """

    __slots__ = ("ts_array", "rows")

    def __init__(self, ts_array: list[int], rows: list[BBORow]) -> None:
        self.ts_array = ts_array
        self.rows = rows

    @classmethod
    def build(cls, all_rows: list[BBORow], side: str) -> BookIndex:
        ts_array: list[int] = []
        rows: list[BBORow] = []
        last_ts: int | None = None
        for r in all_rows:
            t = r.left_ts_ms if side == "left" else r.right_ts_ms
            if last_ts is None or t != last_ts:
                ts_array.append(t)
                rows.append(r)
                last_ts = t
        return cls(ts_array, rows)

    def book_at(self, arrival_ts_ms: int) -> BBORow:
        """The freshest row whose venue source ts ≤ arrival_ts_ms. If
        arrival_ts_ms is earlier than the first tick, returns the first row
        (the engine treats that as the only book it could have seen)."""
        i = bisect_right(self.ts_array, arrival_ts_ms) - 1
        if i < 0:
            return self.rows[0]
        return self.rows[i]
