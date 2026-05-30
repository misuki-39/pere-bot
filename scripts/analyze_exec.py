"""Analyze a taker_taker run's execution telemetry (SQLite recorder).

Usage: analyze_exec.py <live.db> [run_id]      (default run_id = latest)

Answers the live-run questions: decision→send + send→fill latency, per-leg
realized-vs-expected slippage (signed so + = adverse), fill / partial rates,
and how often opportunities evaporate at a gate (the adverse-selection
precursor that paper can never show). Reads the three-table model written by
core.record_sink: rejections (declined) + trades (fired header) + legs (detail).
"""
from __future__ import annotations

import sys

import _db
import numpy as np
import pandas as pd

run = _db.load(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
trades, legs, rejections = run.trades, run.legs, run.rejections
n_dec = len(trades) + len(rejections)
print(f"run={run.run_id}  evaluated_ticks={n_dec}  fired={len(trades)}  "
      f"rejected={len(rejections)}  legs={len(legs)}")

print("\n=== outcome breakdown (every row = a tick we wanted to act on) ===")
counts = {"FIRED": len(trades), **rejections["outcome"].value_counts().to_dict()}
for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<16} {v:>6}  ({v / max(n_dec, 1) * 100:4.1f}%)")
if len(rejections):
    print("\n  abort/block reasons:")
    for k, v in rejections["abort_reason"].value_counts().items():
        print(f"    {v:>6}  {k}")

print("\n=== decision→send latency (ms), FIRED only ===")
s = pd.to_numeric(trades["lat_decision_send_ms"], errors="coerce").dropna()
if len(s):
    q = s.quantile([.5, .9, .95, .99])
    print(f"  lat_decision_send_ms   n={len(s):>5} mean={s.mean():7.1f} "
          f"p50={q.iloc[0]:7.1f} p90={q.iloc[1]:7.1f} "
          f"p95={q.iloc[2]:7.1f} p99={q.iloc[3]:7.1f} max={s.max():7.1f}")

entry = legs[legs["kind"] == "entry"].copy()
unwind = legs[legs["kind"] == "unwind"].copy()

print("\n=== fill quality (entry legs) ===")
entry["filled_qty"] = pd.to_numeric(entry["filled_qty"], errors="coerce")
entry["requested_qty"] = pd.to_numeric(entry["requested_qty"], errors="coerce")
ok = entry["success"] == 1
print(f"  entry legs={len(entry)}  success={ok.mean() * 100:.1f}%  "
      f"failed={(~ok).sum()}  "
      f"partial={(entry['filled_qty'] < entry['requested_qty']).sum()}")
fp = entry.groupby("decision_id")["success"].sum()
print(f"  fired: both-legs-filled={(fp == 2).sum()}  "
      f"one-leg-only(partial)={(fp == 1).sum()}")

print("\n=== send→fill latency + inter-leg skew (the naked-exposure window) ===")
# Latency = fill_ts_ms − send_ts_ms (mixed clock: venue match vs local send;
# NTP skew is ms-scale, well below the interesting signal here).
entry["fill_ts_ms"] = pd.to_numeric(entry["fill_ts_ms"], errors="coerce")
entry["send_ts_ms"] = pd.to_numeric(entry["send_ts_ms"], errors="coerce")
entry["_latency_ms"] = entry["fill_ts_ms"] - entry["send_ts_ms"]
for venue, g in entry.groupby("venue"):
    lat = g["_latency_ms"].dropna()
    if len(lat):
        q = lat.quantile([.5, .95, .99])
        print(f"  {venue:<8} n={len(lat):>5} mean={lat.mean():7.1f} "
              f"p50={q.iloc[0]:7.1f} p95={q.iloc[1]:7.1f} p99={q.iloc[2]:7.1f} "
              f"max={lat.max():7.1f}")
lat = entry.pivot_table(index="decision_id", columns="venue",
                        values="_latency_ms", aggfunc="first")
if {"aster", "lighter"}.issubset(lat.columns):
    lat = lat.dropna(subset=["aster", "lighter"])
    skew = (lat["aster"] - lat["lighter"]).abs()
    led = np.where(lat["aster"] < lat["lighter"], "aster", "lighter")
    q = skew.quantile([.5, .95, .99])
    print(f"  |skew| n={len(skew)} mean={skew.mean():.1f}ms "
          f"p50={q.iloc[0]:.1f} p95={q.iloc[1]:.1f} p99={q.iloc[2]:.1f} "
          f"max={skew.max():.1f}ms")
    u, c = np.unique(led, return_counts=True)
    print(f"  leg that filled first: {dict(zip(u, c, strict=True))}")
else:
    print("  (need both venues' fill timestamps — paper or single-venue run)")

print("\n=== unwind legs (partial-fill flattens — the tail cost) ===")
if len(unwind):
    unwind["expected_price"] = pd.to_numeric(unwind["expected_price"], errors="coerce")
    unwind["realized_price"] = pd.to_numeric(unwind["realized_price"], errors="coerce")
    u = unwind.dropna(subset=["expected_price", "realized_price"])
    rt = (u["realized_price"] - u["expected_price"]).abs() / u["expected_price"] * 1e4
    print(f"  count={len(unwind)}  unwind-success={(u['success'] == 1).mean() * 100:.0f}%"
          f"  round-trip cost bps: mean={rt.mean():.2f} max={rt.max():.2f}")
else:
    print("  none (no partial-fill flattens recorded)")

print("\n=== slippage bps (signed: + = worse than expected), entry fills ===")
f = entry[ok].copy()
f["expected_price"] = pd.to_numeric(f["expected_price"], errors="coerce")
f["realized_price"] = pd.to_numeric(f["realized_price"], errors="coerce")
f = f.dropna(subset=["expected_price", "realized_price"])
sign = np.where(f["side"] == "buy", 1.0, -1.0)
f["slip_bps"] = sign * (f["realized_price"] - f["expected_price"]) / f["expected_price"] * 1e4
for venue, g in f.groupby("venue"):
    q = g["slip_bps"].quantile([.5, .95, .99])
    print(f"  {venue:<8} n={len(g):>5} mean={g.slip_bps.mean():+7.3f} "
          f"p50={q.iloc[0]:+7.3f} p95={q.iloc[1]:+7.3f} p99={q.iloc[2]:+7.3f} "
          f"max={g.slip_bps.max():+7.3f}")
if len(f):
    print(f"  ALL      n={len(f):>5} mean={f.slip_bps.mean():+7.3f}  "
          f"(positive = adverse selection + latency cost vs the decision VWAP)")
