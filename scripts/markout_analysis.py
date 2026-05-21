"""Offline markout analysis for taker-taker captures.

For each historical tick with edge_A>0 (or edge_B>0), look τ_left ms / τ_right
ms ahead per leg (matching live latency) and compute the realized slippage on
the leg's fill price vs the decision-time fill price.

Output: per (pair, direction, edge_bucket, latency_profile) the mean expected
adverse markout in bps, persisted as JSON.

Usage:
    python scripts/markout_analysis.py \\
        --data logs/spread_WTI_20260520T054328Z \\
        --left-latency-ms 350 --right-latency-ms 50 \\
        --left-venue lighter --right-venue aster \\
        --out /tmp/markout_wti.json

The output JSON has structure:
    {
      "venues": ["lighter", "aster"],
      "left_latency_ms": 350, "right_latency_ms": 50,
      "buckets_bps": [0.0, 1.0, 2.0, 5.0, 10.0, 1e9],
      "n_ticks_total": ...,
      "direction_A": {
        "n_ticks":   [...],     # tick count per bucket
        "adverse_bps_mean": [...],
        "adverse_bps_median": [...],
        "adverse_bps_p75": [...],
      },
      "direction_B": { ... }
    }
"""

from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds


_BUCKETS = [0.0, 1.0, 2.0, 5.0, 10.0, math.inf]


_LOAD_COLS = [
    "ts_ms",
    "vwap_left_sell", "vwap_left_buy",
    "vwap_right_sell", "vwap_right_buy",
    "mid_left", "mid_right",
    "edge_A_bps", "edge_B_bps",
    "gates_passed",
]


def _load_from_paths(paths: list[Path]) -> dict[str, np.ndarray]:
    """Load a pre-filtered list of parquet files. Used by the rolling-window
    builder (`scripts/build_markout_table.py`) which selects partitions by
    date before calling. Unreadable parquets are skipped with a warning
    (recorder leaves footer-less files on mid-rotation kill)."""
    import pyarrow.parquet as pq
    frames = []
    for p in sorted(paths):
        try:
            frames.append(pq.read_table(p, columns=_LOAD_COLS).to_pandas())
        except Exception as e:
            print(f"skip {p}: {e}")
    if not frames:
        raise RuntimeError(f"no readable parquet in {len(paths)} input paths")
    import pandas as pd
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("ts_ms").reset_index(drop=True)
    return _df_to_arrays(df)


def _load(data_root: Path) -> dict[str, np.ndarray]:
    """Read all parquet partitions under `data_root` sorted by ts_ms.

    Convenience for the CLI that takes a single root dir; the builder script
    uses `_load_from_paths` directly with a date-filtered file list.
    """
    return _load_from_paths(list(data_root.rglob("*.parquet")))


def _df_to_arrays(df) -> dict[str, np.ndarray]:
    """Common ndarray-column build (shared between _load and _load_from_paths).

    We convert string-decimals to float for the markout math — this is OK for
    bps-scale slippage estimation (microscopic precision loss). Strategy code
    keeps Decimal.
    """
    cols = _LOAD_COLS

    out: dict[str, np.ndarray] = {"ts_ms": df["ts_ms"].astype(np.int64).to_numpy()}

    def _to_float(s: str) -> float:
        if s in ("None", "nan", "NaN", ""):
            return float("nan")
        try:
            return float(Decimal(s))
        except Exception:
            return float("nan")

    for c in cols[1:]:
        if c == "gates_passed":
            out[c] = df[c].to_numpy().astype(bool)
        else:
            out[c] = df[c].astype(str).map(_to_float).to_numpy()
    return out


def _arrival_index(ts_ms: np.ndarray, latency_ms: int) -> np.ndarray:
    """For each row i, the smallest index j>=i s.t. ts_ms[j] >= ts_ms[i]+latency.

    Returns -1 when no such j exists (capture ends before τ elapses).
    """
    target = ts_ms + latency_ms
    j = np.searchsorted(ts_ms, target, side="left")
    j[j >= len(ts_ms)] = -1
    return j


def _bucket(edge_bps: float) -> int:
    for i, hi in enumerate(_BUCKETS[1:]):
        if edge_bps <= hi:
            return i
    return len(_BUCKETS) - 2


def analyze(
    data_root: Path,
    left_latency_ms: int,
    right_latency_ms: int,
) -> dict[str, object]:
    d = _load(data_root)
    n = len(d["ts_ms"])

    j_left  = _arrival_index(d["ts_ms"], left_latency_ms)
    j_right = _arrival_index(d["ts_ms"], right_latency_ms)
    finite = (
        np.isfinite(d["vwap_left_sell"]) & np.isfinite(d["vwap_left_buy"]) &
        np.isfinite(d["vwap_right_sell"]) & np.isfinite(d["vwap_right_buy"]) &
        np.isfinite(d["edge_A_bps"]) & np.isfinite(d["edge_B_bps"])
    )
    valid = (j_left >= 0) & (j_right >= 0) & d["gates_passed"] & finite

    mid_ref = (d["mid_left"] + d["mid_right"]) / 2.0

    # Slippage per leg, in PRICE units (signed for the bot's pnl direction).
    # Direction A: sell left (Δleft adverse if vwap_left_sell drops),
    #              buy right (Δright adverse if vwap_right_buy rises).
    # adverse_A_price = (vwap_right_buy(t+τ_right) - vwap_right_buy(t))
    #                 - (vwap_left_sell(t+τ_left)  - vwap_left_sell(t))
    j_l = np.where(valid, j_left, 0)
    j_r = np.where(valid, j_right, 0)
    adverse_A_price = (
        (d["vwap_right_buy"][j_r] - d["vwap_right_buy"]) -
        (d["vwap_left_sell"][j_l] - d["vwap_left_sell"])
    )
    # Direction B: sell right (adverse if vwap_right_sell drops),
    #              buy left  (adverse if vwap_left_buy rises).
    adverse_B_price = (
        (d["vwap_left_buy"][j_l] - d["vwap_left_buy"]) -
        (d["vwap_right_sell"][j_r] - d["vwap_right_sell"])
    )

    # Normalise to bps of ref_mid so it's comparable to edge_*_bps.
    adverse_A_bps = adverse_A_price / mid_ref * 1e4
    adverse_B_bps = adverse_B_price / mid_ref * 1e4

    out: dict[str, object] = {
        "left_latency_ms": left_latency_ms,
        "right_latency_ms": right_latency_ms,
        "buckets_bps": _BUCKETS,
        "n_rows_total": int(n),
        "n_rows_valid": int(valid.sum()),
    }

    for direction, edge_col, adverse in (
        ("direction_A", "edge_A_bps", adverse_A_bps),
        ("direction_B", "edge_B_bps", adverse_B_bps),
    ):
        edge = d[edge_col]
        mask = valid & np.isfinite(adverse) & (edge > 0)
        n_dir = int(mask.sum())
        if n_dir == 0:
            out[direction] = {"n_ticks_total": 0, "buckets": []}
            continue

        edges = edge[mask]
        advs  = adverse[mask]
        b = np.array([_bucket(e) for e in edges])

        bucket_rows: list[dict[str, object]] = []
        for i in range(len(_BUCKETS) - 1):
            sel = (b == i)
            count = int(sel.sum())
            if count == 0:
                bucket_rows.append({
                    "bucket": [_BUCKETS[i], _BUCKETS[i+1]],
                    "n": 0,
                    "mean_bps": None, "median_bps": None,
                    "p75_bps": None,  "p25_bps": None,
                })
                continue
            ad = advs[sel]
            bucket_rows.append({
                "bucket": [_BUCKETS[i], _BUCKETS[i+1]],
                "n": count,
                "mean_bps":   float(np.mean(ad)),
                "median_bps": float(np.median(ad)),
                "p75_bps":    float(np.percentile(ad, 75)),
                "p25_bps":    float(np.percentile(ad, 25)),
            })
        out[direction] = {"n_ticks_total": n_dir, "buckets": bucket_rows}

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Offline markout analysis.")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--left-latency-ms", required=True, type=int)
    p.add_argument("--right-latency-ms", required=True, type=int)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()

    res = analyze(a.data, a.left_latency_ms, a.right_latency_ms)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
