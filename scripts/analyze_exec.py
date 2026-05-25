"""Analyze a taker_taker run's execution telemetry.

Usage: analyze_exec.py decisions_*.csv legs_*.csv

Answers the live-run questions: decision/send/result latency distribution,
per-leg realized-vs-expected slippage (signed so + = adverse), fill / partial
rates, and how often opportunities evaporate at a gate (the adverse-selection
precursor that paper can never show).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

dec = pd.read_csv(sys.argv[1])
legs = pd.read_csv(sys.argv[2])

print(f"decisions={len(dec)}  legs={len(legs)}")
print("\n=== outcome breakdown (every row = a tick we wanted to act on) ===")
oc = dec["outcome"].value_counts()
for k, v in oc.items():
    print(f"  {k:<16} {v:>6}  ({v/len(dec)*100:4.1f}%)")
fired = dec[dec.outcome == "FIRED"]
if "abort_reason" in dec:
    ab = dec[dec.outcome.str.startswith("ABORT") | (dec.outcome == "BLOCKED_RISK")]
    if len(ab):
        print("\n  abort/block reasons:")
        for k, v in ab["abort_reason"].value_counts().items():
            print(f"    {v:>6}  {k}")

print("\n=== latency (ms), FIRED only ===")
for col in ("lat_decision_send_ms", "lat_send_result_ms", "lat_total_ms"):
    s = pd.to_numeric(fired[col], errors="coerce").dropna()
    if len(s):
        q = s.quantile([.5, .9, .95, .99])
        print(f"  {col:<22} n={len(s):>5} mean={s.mean():7.1f} "
              f"p50={q.iloc[0]:7.1f} p90={q.iloc[1]:7.1f} "
              f"p95={q.iloc[2]:7.1f} p99={q.iloc[3]:7.1f} max={s.max():7.1f}")

if "kind" not in legs:
    legs["kind"] = "entry"
entry = legs[legs["kind"] == "entry"]
unwind = legs[legs["kind"] == "unwind"]

print("\n=== fill quality (entry legs) ===")
entry = entry.copy()
entry["filled_qty"] = pd.to_numeric(entry["filled_qty"], errors="coerce")
entry["requested_qty"] = pd.to_numeric(entry["requested_qty"], errors="coerce")
ok = entry["success"].astype(str).str.lower().eq("true")
print(f"  entry legs={len(entry)}  success={ok.mean()*100:.1f}%  "
      f"failed={(~ok).sum()}  "
      f"partial={(entry['filled_qty'] < entry['requested_qty']).sum()}")
fp = entry.groupby("decision_id")["success"].apply(
    lambda s: s.astype(str).str.lower().eq("true").sum())
print(f"  fired: both-legs-filled={(fp == 2).sum()}  "
      f"one-leg-only(partial)={(fp == 1).sum()}")

print("\n=== inter-leg fill skew (the naked-exposure window) ===")
# Latency = fill_ts_ms − send_ts_ms (mixed clock: venue match vs local send;
# NTP skew is ms-scale, well below the interesting signal here).
entry["fill_ts_ms"] = pd.to_numeric(entry["fill_ts_ms"], errors="coerce")
entry["send_ts_ms"] = pd.to_numeric(entry["send_ts_ms"], errors="coerce")
entry["_latency_ms"] = entry["fill_ts_ms"] - entry["send_ts_ms"]
lat = entry.pivot_table(index="decision_id", columns="venue",
                        values="_latency_ms", aggfunc="first")
if {"aster", "lighter"}.issubset(lat.columns):
    lat = lat.dropna(subset=["aster", "lighter"])
    skew = (lat["aster"] - lat["lighter"]).abs()
    led = np.where(lat["aster"] < lat["lighter"], "aster", "lighter")
    q = skew.quantile([.5, .95, .99])
    print(f"  n={len(skew)}  |skew| mean={skew.mean():.1f}ms "
          f"p50={q.iloc[0]:.1f} p95={q.iloc[1]:.1f} p99={q.iloc[2]:.1f} "
          f"max={skew.max():.1f}ms")
    u, c = np.unique(led, return_counts=True)
    print(f"  leg that filled first: {dict(zip(u, c, strict=True))}")
else:
    print("  (need both venues' fill timestamps — paper or single-venue run)")

print("\n=== unwind legs (partial-fill flattens — the tail cost) ===")
if len(unwind):
    u = unwind.copy()
    u["expected_price"] = pd.to_numeric(u["expected_price"], errors="coerce")
    u["realized_price"] = pd.to_numeric(u["realized_price"], errors="coerce")
    u = u.dropna(subset=["expected_price", "realized_price"])
    # unwind reverses a stranded leg: cost = |reverse fill − cost basis|
    rt = (u["realized_price"] - u["expected_price"]).abs() / u["expected_price"] * 1e4
    print(f"  count={len(unwind)}  unwind-success={u['success'].astype(str).str.lower().eq('true').mean()*100:.0f}%"
          f"  round-trip cost bps: mean={rt.mean():.2f} max={rt.max():.2f}")
else:
    print("  none (no partial fills) — good, but paper never produces these")

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
print(f"  ALL      n={len(f):>5} mean={f.slip_bps.mean():+7.3f}  "
      f"(positive = adverse selection + latency cost vs the decision VWAP)")

print("\n  (per-decision realized-vs-expected spread capture is derivable by "
      "joining the two entry legs on decision_id — left to the strategy review.)")
