"""Markout lookup table for adverse-selection-adjusted edge.

The offline script `scripts/markout_analysis.py` measures, on a historical
capture, the per-direction expected adverse price-drift between decision and
fill (under a given latency profile). The strategy then subtracts this
expected drift from the raw decision-time edge, so the threshold check becomes:

    fire if  raw_edge_bps  >  fees_bps + min_profit_bps + E[markout_bps]

We bucket by raw edge size because adverse drift grows with edge — the bigger
the dislocation, the more of it tends to bleed off during the latency window.
Lookup is bucket-based, not interpolated: that mirrors how the offline
estimator was built (mean per bucket), avoiding the false precision of
interpolating across small-sample bins.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True, slots=True)
class _Bucket:
    """Half-open [lo, hi) edge-bps bucket with its mean adverse markout."""

    lo: Decimal
    hi: Decimal           # Decimal("inf") for the open-ended last bucket
    markout_bps: Decimal  # >0 = adverse (subtract from edge)


@dataclass(frozen=True, slots=True)
class MarkoutTable:
    """Per-direction list of buckets. Direction names match `Direction.A` /
    `Direction.B` from `core.exec_record`.

    `latency_label` is a free-form string identifying the latency profile the
    table was built under (e.g., "lighter=350,aster=50") — for logs / sanity
    checks, not used in computation.
    """

    direction_A: tuple[_Bucket, ...]
    direction_B: tuple[_Bucket, ...]
    latency_label: str = ""

    @classmethod
    def disabled(cls) -> MarkoutTable:
        """All-zero markout — opt-out (strategy behaves identically to pre-markout)."""
        return cls(direction_A=(), direction_B=(), latency_label="disabled")

    @classmethod
    def from_json(cls, path: Path) -> MarkoutTable:
        """Load output of `scripts/markout_analysis.py`.

        Buckets with n=0 (insufficient samples) get 0-bps markout so they don't
        spuriously block fires — the offline tooling logs n per bucket for the
        operator to inspect.
        """
        raw = json.loads(Path(path).read_text())

        def _parse(side: str) -> tuple[_Bucket, ...]:
            out: list[_Bucket] = []
            for b in raw[side]["buckets"]:
                lo, hi = b["bucket"]
                if b["n"] == 0 or b.get("mean_bps") is None:
                    mkt = Decimal(0)
                else:
                    mkt = Decimal(str(b["mean_bps"]))
                lo_d = Decimal(str(lo))
                hi_d = Decimal("Infinity") if hi == math.inf or str(hi).lower() == "infinity" \
                       else Decimal(str(hi))
                out.append(_Bucket(lo=lo_d, hi=hi_d, markout_bps=max(Decimal(0), mkt)))
            return tuple(out)

        latency = f"left={raw.get('left_latency_ms')}ms,right={raw.get('right_latency_ms')}ms"
        return cls(
            direction_A=_parse("direction_A"),
            direction_B=_parse("direction_B"),
            latency_label=latency,
        )

    def markout_bps(self, direction_a: bool, raw_edge_bps: Decimal) -> Decimal:
        """Look up the expected adverse markout in bps for this direction.

        `direction_a=True` for Direction.A, False for Direction.B.
        Edge buckets are half-open [lo, hi). Open-ended last bucket catches
        anything not in earlier buckets. Returns 0 if disabled / no match
        (defensive — the offline tooling produces a complete bucketing).
        """
        buckets = self.direction_A if direction_a else self.direction_B
        for b in buckets:
            if b.lo <= raw_edge_bps < b.hi:
                return b.markout_bps
        return Decimal(0)
