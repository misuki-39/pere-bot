"""Realized + open PnL for a live taker_taker run.

Used because the bot writing these CSVs predates the per-trade-PnL commit
(8a6c941) — the `fee` column is empty and there is no PnL log to read back.

Inputs:  decisions_*.csv + legs_*.csv from logs/live/
Outputs: per-decision cash flow, cumulative realized, mark-to-market on the
unwound-net open position, time-bucketed series, and a fees sensitivity table.

Cash-flow convention (matches src/perp_arb/core/pnl.py): sell brings cash in
(+price·qty), buy sends cash out (−price·qty). For a mean-reverting two-leg
arb, sum of per-pair cash flows == realized PnL when net qty per venue = 0;
any residual venue exposure is marked at the last observed mid.

Run:  uv run python scripts/live_pnl.py logs/live/decisions_*.csv logs/live/legs_*.csv
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

# Round-trip fee bps from configs/taker_taker_wti.yaml (split per leg).
FEES_BPS_RT = Decimal("1.0")
FEE_BPS_PER_LEG = FEES_BPS_RT / Decimal(2)


def _to_dec(x: object) -> Decimal:
    return Decimal(str(x))


def compute(dec_path: Path, legs_path: Path) -> None:
    dec = pd.read_csv(dec_path)
    legs = pd.read_csv(legs_path)

    # Only successful entry fills.
    legs = legs[(legs["kind"] == "entry") & (legs["success"].astype(str).str.lower() == "true")].copy()
    legs["filled_qty"] = legs["filled_qty"].apply(_to_dec)
    legs["realized_price"] = legs["realized_price"].apply(_to_dec)

    # Per-leg fee (price * qty * bps / 1e4).
    legs["fee"] = legs.apply(
        lambda r: r["realized_price"] * r["filled_qty"] * FEE_BPS_PER_LEG / Decimal("10000"),
        axis=1,
    )
    # Signed cash flow: sell + , buy -
    legs["cash"] = legs.apply(
        lambda r: (Decimal("1") if r["side"] == "sell" else Decimal("-1"))
        * r["realized_price"] * r["filled_qty"],
        axis=1,
    )

    # Pair legs by decision_id. Keep only decisions where both legs filled.
    g = legs.groupby("decision_id")
    paired_ids = [d for d, sub in g if len(sub) == 2]
    paired = legs[legs["decision_id"].isin(paired_ids)].copy()

    # Per-decision summary.
    rows = []
    for did, sub in paired.groupby("decision_id"):
        row = {
            "decision_id": did,
            "ts_ms": int(sub["ts_ms"].iloc[0]),
            "cash": sum(sub["cash"], Decimal("0")),
            "fee": sum(sub["fee"], Decimal("0")),
        }
        for _, leg in sub.iterrows():
            row[f"{leg['exchange']}_side"] = leg["side"]
            row[f"{leg['exchange']}_px"] = leg["realized_price"]
            row[f"{leg['exchange']}_qty"] = leg["filled_qty"]
        row["net_cash"] = row["cash"] - row["fee"]
        rows.append(row)
    pairs = pd.DataFrame(rows).sort_values("ts_ms").reset_index(drop=True)

    # Cumulative realized (net of fees), and gross-of-fees for sensitivity.
    pairs["cum_gross"] = pairs["cash"].cumsum().apply(float)
    pairs["cum_net"] = pairs["net_cash"].cumsum().apply(float)
    pairs["cum_fee"] = pairs["fee"].cumsum().apply(float)

    # Open position per venue (sum of signed filled qty: buy=+, sell=−).
    leg_sign = legs.apply(
        lambda r: (Decimal("1") if r["side"] == "buy" else Decimal("-1")) * r["filled_qty"],
        axis=1,
    )
    legs["signed_qty"] = leg_sign
    open_pos = legs.groupby("exchange")["signed_qty"].apply(lambda s: sum(s, Decimal("0")))

    # Mark-to-market on the open position using last decision mids.
    last_dec = dec.iloc[-1]
    mids = {"aster": Decimal(str(last_dec["mid_right"])), "lighter": Decimal(str(last_dec["mid_left"]))}
    # cost basis per venue: signed cash already captures avg entry; the mark = pos * mid
    # PnL_open = sum over venues of (signed_qty * mid)  +  current sum_cash
    # because: open_value = signed_qty * mid; entry cash flow already booked.
    # Realized cumulative cash flow already includes the open legs' cash;
    # the mark closes them at current mid.
    mark_cash = sum(open_pos[v] * mids[v] for v in open_pos.index)  # noqa
    # Closing the open position requires reversing: open buy → sell, open sell → buy.
    # Reverse cash flow = -signed_qty * mid_close - close_fee.
    # So PnL = realized_sum_cash + (close cash) - close fee
    close_cash = -mark_cash  # reverse cash flow at mid
    close_fee = sum(abs(open_pos[v]) * mids[v] * FEE_BPS_PER_LEG / Decimal("10000") for v in open_pos.index)

    total_gross_cash = sum(pairs["cash"].tolist(), Decimal("0"))
    total_fees_entry = sum(pairs["fee"].tolist(), Decimal("0"))
    total_realized_after_mark = total_gross_cash + close_cash - close_fee - total_fees_entry

    # Time spans.
    t0_ms = int(pairs["ts_ms"].iloc[0])
    t1_ms = int(pairs["ts_ms"].iloc[-1])
    span_h = (t1_ms - t0_ms) / 3600_000

    # Direction breakdown.
    dec_idx = dec.set_index("decision_id")
    pairs["direction"] = pairs["decision_id"].map(dec_idx["direction"])
    by_dir = pairs.groupby("direction").agg(
        n=("cash", "size"),
        gross_cash=("cash", lambda s: float(sum(s, Decimal("0")))),
        fees=("fee", lambda s: float(sum(s, Decimal("0")))),
    )
    by_dir["net_after_fee"] = by_dir["gross_cash"] - by_dir["fees"]

    # Hourly cumulative for the “when did pnl happen” question.
    pairs["hour"] = pd.to_datetime(pairs["ts_ms"], unit="ms", utc=True).dt.floor("h")
    hourly = pairs.groupby("hour").agg(
        n=("cash", "size"),
        gross_cash=("cash", lambda s: float(sum(s, Decimal("0")))),
        fees=("fee", lambda s: float(sum(s, Decimal("0")))),
    )
    hourly["net"] = hourly["gross_cash"] - hourly["fees"]
    hourly["cum_net"] = hourly["net"].cumsum()

    # ---- output ----
    print(f"=== Run summary ({dec_path.name}) ===")
    print(f"  decisions={len(dec)}  fired-and-both-filled-pairs={len(pairs)}")
    print(f"  span:  {pd.to_datetime(t0_ms, unit='ms', utc=True)}"
          f"  →  {pd.to_datetime(t1_ms, unit='ms', utc=True)}   ({span_h:.2f} h)")
    print(f"  fees assumed: {float(FEES_BPS_RT)} bps round-trip "
          f"({float(FEE_BPS_PER_LEG)} bps per leg)")

    print("\n=== Direction counts ===")
    print(by_dir.to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\n=== Per-venue open position at run end ===")
    for v, qty in open_pos.items():
        print(f"  {v:<8} net signed qty = {float(qty):+.4f}   last mid = {float(mids[v]):.4f}")
    print(f"  reverse-at-mid cash:      {float(close_cash):+.4f}")
    print(f"  reverse-at-mid fee:       {float(close_fee):.4f}")

    print("\n=== PnL ===")
    print(f"  gross cash from entries:        {float(total_gross_cash):+.4f} USD")
    print(f"  − entry fees:                   {float(total_fees_entry):.4f}")
    print(f"  + close-at-mid cash:            {float(close_cash):+.4f}")
    print(f"  − close-at-mid fees:            {float(close_fee):.4f}")
    print(f"  = total PnL (mark-to-mid):      {float(total_realized_after_mark):+.4f} USD")
    print(f"      annualized run-rate:        ${float(total_realized_after_mark) / max(span_h, 1e-9) * 24:+.2f} / day")
    print(f"      per-pair avg net:           ${float(total_realized_after_mark) / len(pairs):+.4f}")

    print("\n=== Sensitivity to fee assumption (round-trip bps) ===")
    print("  bps   total_pnl  $/day")
    for bps in (0, 0.5, 1.0, 1.5, 2.0, 3.0):
        bps_d = Decimal(str(bps))
        per_leg = bps_d / Decimal(2)
        # recompute fees scaled from gross notional
        notional = sum(
            (legs["realized_price"] * legs["filled_qty"]).tolist(), Decimal("0")
        )
        # close-leg notional ~ |open_pos|·mid summed
        close_notional = sum(abs(open_pos[v]) * mids[v] for v in open_pos.index)
        total_fee = (notional + close_notional) * per_leg / Decimal("10000")
        pnl = total_gross_cash + close_cash - total_fee
        print(f"  {bps:>4.1f}  {float(pnl):+9.4f}  {float(pnl)/max(span_h,1e-9)*24:+8.2f}")

    print("\n=== Per-direction PnL (gross cash − entry fees, no mark) ===")
    for d, row in by_dir.iterrows():
        avg = row["net_after_fee"] / row["n"]
        print(f"  {d}  n={int(row['n']):>4}  net=${row['net_after_fee']:+8.4f}  avg=${avg:+.5f}")

    print("\n=== Hourly PnL ===")
    print(hourly.to_string(float_format=lambda x: f"{x:+.4f}"))

    # Persist the per-pair table next to inputs for further drill-down.
    out_csv = dec_path.parent / f"pnl_pairs_{dec_path.stem.split('_', 2)[-1]}.csv"
    pairs.assign(
        cash=pairs["cash"].apply(float),
        fee=pairs["fee"].apply(float),
        net_cash=pairs["net_cash"].apply(float),
    ).to_csv(out_csv, index=False)
    print(f"\nWrote per-pair table → {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: live_pnl.py decisions_*.csv legs_*.csv")
    compute(Path(sys.argv[1]), Path(sys.argv[2]))
