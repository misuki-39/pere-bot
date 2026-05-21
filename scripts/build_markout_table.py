"""Build a markout lookup table from rolling captured BBO data.

The live `taker_taker` strategy reads a JSON markout table at startup
(`configs/markout/<pair>.json`). This script rebuilds that table from the
most recent N days of `spread_<pair>_<ts>/date=YYYY-MM-DD/HH.parquet`
captures.

Run cadence (recommended): daily at ~03:00 UTC during a low-volume window.
The operator schedules this externally (cron / systemd timer); the script
itself does no scheduling.

Output is written atomically (`<out>.tmp` then `os.replace`) so the live bot
never sees a half-written file on startup.

Exit non-zero if any direction has every bucket below `MIN_N` samples —
the operator's cron alarm should treat that as "do not deploy".

Usage:
    python scripts/build_markout_table.py \\
        --bbo-root logs/spread_WTI_20260520T054328Z \\
        --left-latency-ms 350 --right-latency-ms 50 \\
        --lookback-days 7 \\
        --out configs/markout/wti_lighter_aster.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

# scripts/ is not a package; add its parent to sys.path so we can import the
# sibling module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from markout_analysis import _df_to_arrays, _LOAD_COLS  # noqa: E402

# Minimum sample count per direction across all buckets for the table to be
# considered "viable". Below this we refuse to write — operator must
# investigate (probably the capture broke).
_MIN_TOTAL_TICKS_PER_DIRECTION = 200

# Hive-partition date format used by the BBO recorder.
_DATE_PARTITION_RE = re.compile(r"date=(\d{4}-\d{2}-\d{2})")


def _select_parquets_by_date(bbo_root: Path, lookback_days: int) -> list[Path]:
    """Return all `<bbo_root>/date=YYYY-MM-DD/*.parquet` whose partition
    date is within `lookback_days` of today (UTC).

    Raises if `bbo_root` doesn't exist or contains no qualifying partitions.
    """
    if not bbo_root.exists():
        raise FileNotFoundError(f"bbo-root not found: {bbo_root}")
    today_utc = dt.datetime.now(dt.UTC).date()
    cutoff = today_utc - dt.timedelta(days=lookback_days)
    selected: list[Path] = []
    for date_dir in sorted(bbo_root.glob("date=*")):
        m = _DATE_PARTITION_RE.match(date_dir.name)
        if m is None:
            continue
        try:
            d = dt.date.fromisoformat(m.group(1))
        except ValueError:
            continue
        if d < cutoff:
            continue
        selected.extend(sorted(date_dir.glob("*.parquet")))
    if not selected:
        raise FileNotFoundError(
            f"no parquets under {bbo_root} within last {lookback_days} day(s) "
            f"(cutoff {cutoff})"
        )
    return selected


def _analyze_paths(paths: list[Path], left_lat_ms: int, right_lat_ms: int) -> dict:
    """Direct re-implementation of `markout_analysis.analyze` but operating
    on a pre-filtered list of paths instead of a root directory.

    We don't reuse `analyze()` directly because that function takes
    `data_root: Path` and rglobs internally — we want the date-filtered set.
    """
    import numpy as np

    import pyarrow.parquet as pq
    frames = []
    for p in sorted(paths):
        try:
            frames.append(pq.read_table(p, columns=_LOAD_COLS).to_pandas())
        except Exception as e:
            print(f"skip {p}: {e}", file=sys.stderr)
    if not frames:
        raise RuntimeError(f"no readable parquet in {len(paths)} inputs")
    import pandas as pd
    df = pd.concat(frames, ignore_index=True).sort_values("ts_ms").reset_index(drop=True)
    d = _df_to_arrays(df)

    # The analysis logic below is intentionally duplicated rather than
    # imported from markout_analysis.analyze() because that function couples
    # data loading with analysis. Refactor opportunity: extract the
    # numeric kernel; for now this duplication is small and direct.
    from markout_analysis import _arrival_index, _bucket, _BUCKETS  # noqa: E402

    n = len(d["ts_ms"])
    j_left = _arrival_index(d["ts_ms"], left_lat_ms)
    j_right = _arrival_index(d["ts_ms"], right_lat_ms)
    finite = (
        np.isfinite(d["vwap_left_sell"]) & np.isfinite(d["vwap_left_buy"]) &
        np.isfinite(d["vwap_right_sell"]) & np.isfinite(d["vwap_right_buy"]) &
        np.isfinite(d["edge_A_bps"]) & np.isfinite(d["edge_B_bps"])
    )
    valid = (j_left >= 0) & (j_right >= 0) & d["gates_passed"] & finite
    mid_ref = (d["mid_left"] + d["mid_right"]) / 2.0
    j_l = np.where(valid, j_left, 0)
    j_r = np.where(valid, j_right, 0)
    adverse_A_price = (
        (d["vwap_right_buy"][j_r] - d["vwap_right_buy"]) -
        (d["vwap_left_sell"][j_l] - d["vwap_left_sell"])
    )
    adverse_B_price = (
        (d["vwap_left_buy"][j_l] - d["vwap_left_buy"]) -
        (d["vwap_right_sell"][j_r] - d["vwap_right_sell"])
    )
    adverse_A_bps = adverse_A_price / mid_ref * 1e4
    adverse_B_bps = adverse_B_price / mid_ref * 1e4

    out: dict[str, object] = {
        "left_latency_ms": left_lat_ms,
        "right_latency_ms": right_lat_ms,
        "buckets_bps": _BUCKETS,
        "n_rows_total": int(n),
        "n_rows_valid": int(valid.sum()),
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
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
        advs = adverse[mask]
        b = np.array([_bucket(e) for e in edges])
        bucket_rows = []
        for i in range(len(_BUCKETS) - 1):
            sel = (b == i)
            count = int(sel.sum())
            if count == 0:
                bucket_rows.append({
                    "bucket": [_BUCKETS[i], _BUCKETS[i + 1]],
                    "n": 0, "mean_bps": None, "median_bps": None,
                    "p75_bps": None, "p25_bps": None,
                })
                continue
            ad = advs[sel]
            bucket_rows.append({
                "bucket": [_BUCKETS[i], _BUCKETS[i + 1]],
                "n": count,
                "mean_bps": float(np.mean(ad)),
                "median_bps": float(np.median(ad)),
                "p75_bps": float(np.percentile(ad, 75)),
                "p25_bps": float(np.percentile(ad, 25)),
            })
        out[direction] = {"n_ticks_total": n_dir, "buckets": bucket_rows}
    return out


def _validate(result: dict) -> list[str]:
    """Return a list of human-readable issues (empty = OK)."""
    issues = []
    for direction in ("direction_A", "direction_B"):
        n = result[direction]["n_ticks_total"]
        if n < _MIN_TOTAL_TICKS_PER_DIRECTION:
            issues.append(
                f"{direction}: only {n} positive-edge ticks "
                f"(threshold {_MIN_TOTAL_TICKS_PER_DIRECTION}); "
                f"capture window too short or no signal"
            )
    return issues


def _atomic_write_json(result: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, default=str))
    os.replace(tmp, out)


def _summary_line(result: dict, paths_n: int, out: Path) -> str:
    a = result["direction_A"]
    b = result["direction_B"]
    bA = ",".join(
        f"{int(r['bucket'][0])}-{int(r['bucket'][1]) if r['bucket'][1] < 9999 else '+'}:n={r['n']}/μ={r['mean_bps']:+.2f}"
        if r['n'] > 0 else f"{int(r['bucket'][0])}-:n=0"
        for r in a["buckets"]
    )
    bB = ",".join(
        f"{int(r['bucket'][0])}-{int(r['bucket'][1]) if r['bucket'][1] < 9999 else '+'}:n={r['n']}/μ={r['mean_bps']:+.2f}"
        if r['n'] > 0 else f"{int(r['bucket'][0])}-:n=0"
        for r in b["buckets"]
    )
    return (
        f"markout-build OK: files={paths_n} rows={result['n_rows_valid']}/"
        f"{result['n_rows_total']} A.n={a['n_ticks_total']} B.n={b['n_ticks_total']} "
        f"latency=L{result['left_latency_ms']}/R{result['right_latency_ms']} "
        f"out={out} | A[{bA}] B[{bB}]"
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build a markout lookup table from rolling BBO captures."
    )
    p.add_argument("--bbo-root", required=True, type=Path,
                   help="Hive-partitioned capture root, e.g. logs/spread_WTI_<ts>")
    p.add_argument("--left-latency-ms", required=True, type=int)
    p.add_argument("--right-latency-ms", required=True, type=int)
    p.add_argument("--lookback-days", type=int, default=7,
                   help="Use only date partitions within last N days (default 7)")
    p.add_argument("--out", required=True, type=Path,
                   help="Destination JSON path (e.g. configs/markout/wti.json)")
    a = p.parse_args()

    paths = _select_parquets_by_date(a.bbo_root, a.lookback_days)
    result = _analyze_paths(paths, a.left_latency_ms, a.right_latency_ms)
    issues = _validate(result)
    if issues:
        for msg in issues:
            print(f"markout-build VALIDATION FAILED: {msg}", file=sys.stderr)
        return 2
    _atomic_write_json(result, a.out)
    print(_summary_line(result, len(paths), a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
