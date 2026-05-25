"""Per-leg + decision-side latency stats for a live taker_taker run.

Decomposition (all milliseconds, all from real clocks):

  decision_ts ──lat_decision_send──▶ send_ts ──latency_ms──▶ fill_ts
                  (sync compute)        (gather submit + venue fill)

`send_ts_ms` is shared by both legs (single local timestamp captured just
before `asyncio.gather` fires them in parallel). So inter-leg skew in fill
time, |fill_a − fill_l|, is the *naked exposure window* — the time one leg
sits filled while the other is still in flight.

This script reports per-venue and joint distributions plus the gather
parallelism check (is decision→send roughly zero?).

Run:  python scripts/live_latency.py logs/live/decisions_*.csv logs/live/legs_*.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _pct(s: pd.Series, qs=(0.5, 0.9, 0.95, 0.99)) -> str:
    if len(s) == 0:
        return "(empty)"
    q = s.quantile(list(qs))
    return (
        f"n={len(s):>4}  mean={s.mean():7.1f}  "
        f"p50={q.iloc[0]:6.1f}  p90={q.iloc[1]:6.1f}  "
        f"p95={q.iloc[2]:6.1f}  p99={q.iloc[3]:6.1f}  "
        f"min={s.min():5.0f}  max={s.max():6.0f}"
    )


def main(dec_path: Path, legs_path: Path) -> None:
    dec = pd.read_csv(dec_path)
    legs = pd.read_csv(legs_path)

    entry = legs[(legs["kind"] == "entry") & (legs["success"].astype(str).str.lower() == "true")].copy()
    entry["latency_ms"] = pd.to_numeric(entry["latency_ms"], errors="coerce")
    entry["fill_ts_ms"] = pd.to_numeric(entry["fill_ts_ms"], errors="coerce")
    entry["ts_ms"] = pd.to_numeric(entry["ts_ms"], errors="coerce")

    print(f"=== Run: {dec_path.name} ===")
    print(f"  decisions={len(dec)}  successful-entry-legs={len(entry)}")

    # 1) decision → send (gather setup cost; should be ~0 with parallel submit).
    dec_lat = pd.to_numeric(dec.get("lat_decision_send_ms"), errors="coerce").dropna()
    print("\n=== decision_ts → send_ts (sync compute before gather) ===")
    print("  " + _pct(dec_lat))

    # 2) send → fill, per venue (the “order submit latency” you care about).
    print("\n=== send_ts → fill_ts  (per venue, entry legs) ===")
    for v, g in entry.groupby("exchange"):
        print(f"  {v:<8} " + _pct(g["latency_ms"].dropna()))
    print("  " + "ALL".ljust(8) + " " + _pct(entry["latency_ms"].dropna()))

    # 3) inter-leg skew = |fill_a − fill_l|.  Naked exposure window.
    piv_fill = entry.pivot_table(index="decision_id", columns="exchange",
                                 values="fill_ts_ms", aggfunc="first")
    if {"aster", "lighter"}.issubset(piv_fill.columns):
        piv_fill = piv_fill.dropna()
        skew = (piv_fill["aster"] - piv_fill["lighter"]).abs()
        signed = piv_fill["aster"] - piv_fill["lighter"]   # +ve = aster later
        first = np.where(signed < 0, "aster", "lighter")
        u, c = np.unique(first, return_counts=True)
        print("\n=== inter-leg fill skew  (naked exposure window) ===")
        print("  |skew|   " + _pct(skew))
        # signed view to confirm which venue lags systematically
        ssig = pd.Series(signed.values)
        q = ssig.quantile([.05, .5, .95])
        print(f"  signed   p05={q.iloc[0]:+7.1f}  p50={q.iloc[1]:+7.1f}  p95={q.iloc[2]:+7.1f}    "
              f"(positive = aster filled AFTER lighter)")
        print(f"  filled first → {dict(zip(u, c, strict=True))}")

        # share of decisions where skew is *materially* small (gather pays off).
        for thr in (50, 100, 200, 400):
            share = (skew <= thr).mean() * 100
            print(f"  pairs with |skew| ≤ {thr:>4} ms : {share:5.1f}%")

    # 4) total wall-clock per decision: decision_ts → max(fill_a, fill_l).
    if {"aster", "lighter"}.issubset(piv_fill.columns):
        dec_idx = dec.set_index("decision_id")
        wall = piv_fill.max(axis=1) - dec_idx.loc[piv_fill.index, "ts_ms"]
        print("\n=== decision_ts → both-legs-filled (full opportunity cost) ===")
        print("  " + _pct(wall))

    # 5) per-venue latency over time — drift / regime check.
    entry["minute_bucket"] = (entry["ts_ms"] // 60_000) * 60_000
    by_hour = entry.assign(hour=pd.to_datetime(entry["ts_ms"], unit="ms", utc=True).dt.floor("h"))
    print("\n=== per-venue median latency by hour ===")
    h = by_hour.pivot_table(index="hour", columns="exchange", values="latency_ms", aggfunc="median")
    n = by_hour.pivot_table(index="hour", columns="exchange", values="latency_ms", aggfunc="count")
    if not h.empty:
        joined = h.join(n, lsuffix="_p50", rsuffix="_n")
        print(joined.to_string(float_format=lambda x: f"{x:6.1f}"))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: live_latency.py decisions_*.csv legs_*.csv")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
