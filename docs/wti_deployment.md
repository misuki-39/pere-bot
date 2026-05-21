# WTI taker-taker deployment checklist

Live deployment of `taker_taker` on the **WTI lighter/aster** pair with the
Wave-1 optimisations enabled (markout + same-side throttle + in-flight cap;
no inventory_skew).

Config: [`configs/taker_taker_wti.yaml`](../configs/taker_taker_wti.yaml).
Validated by backtest sweep on 2026-05-20 capture: 911 trips / +$8.28 PnL /
+0.88 bps per trip / max\|pos\|=1. OOS train/test holds ~85 % of the lift —
see memory `wti-aster-lighter-viable`.

---

## One-time setup (per host, post-clone)

```bash
mkdir -p configs/markout
# `configs/markout/*.json` is gitignored — rebuild locally.
```

If this is a fresh capture host with no historical BBO yet, first run
spread_monitor for **≥ 24 h** to accumulate enough data for a viable
markout table:

```bash
runbot --config configs/spread_monitor_lighter_aster_wti.yaml --mode paper
```

The capture lands at `logs/spread_WTI_<ts>/date=YYYY-MM-DD/HH.parquet`.

---

## Step 1 — Build the markout table

```bash
python scripts/build_markout_table.py \
  --bbo-root logs/spread_WTI_<ts> \
  --left-latency-ms 350 --right-latency-ms 50 \
  --lookback-days 7 \
  --out configs/markout/wti_lighter_aster.json
```

- `--left-latency-ms 350` = lighter (the slow leg in this pair)
- `--right-latency-ms 50` = aster
- The script does an atomic `os.replace`, so it is safe to re-run while
  the live bot is running.
- Exit code ≠ 0 means a direction has < 200 positive-edge ticks in the
  window — investigate the capture before deploying.

Expected `stdout` summary line:

```
markout-build OK: files=N rows=M/M' A.n=... B.n=... latency=L350/R50 out=...
  A[0-1:n=.../μ=+0.4x, 1-2:n=.../μ=+0.7x, ...]
  B[0-1:n=.../μ=+0.3x, ...]
```

The bucket means should sit broadly in the **+0.3 to +1.5 bps** range for
small buckets — that's the expected adverse-selection magnitude on this
pair at 350 ms.

---

## Step 2 — Paper mode validation (24–48 h)

```bash
runbot --config configs/taker_taker_wti.yaml --mode paper
```

**At startup, three INFO lines must appear (one per optimisation):**

```
markout enabled: left=350ms,right=50ms
same-side throttle enabled: bump=2 bps halflife=3.0s
in-flight cap enabled K=1 (note: structural no-op in live due to
  _evaluating serialization)
```

If any line is missing → the YAML didn't wire that knob; fix before going live.

**While running, monitor the `[FILLED]` logs:**

- `pos_a` / `pos_l` should oscillate; never reach `max_qty=10` (or `−10`)
  for more than a few minutes
- Trip count over 24 h should be in the 1k–5k range (matches backtest
  ~1.8k trips on 20 h sample)
- `decisions_*.csv` outcome distribution: `FIRED` ≈ 50 % of all
  positive-edge ticks (the rest are BLOCKED_RISK from markout filter, not
  from the cap)

**Compute realised markout offline** (sanity-check the table predictions):

```bash
# Compare each leg's realized_price vs expected_price, group by direction.
# Should be within ±1 bps of the JSON table's per-bucket mean for that
# direction. If realized markout > predicted + 1 bps for > 24 h →
# rebuild table (regime drift).
```

---

## Step 3 — Small live (3 days, qty=0.2 / max_qty=2)

Copy and downscale the config — do **not** edit the committed YAML:

```bash
cp configs/taker_taker_wti.yaml configs/taker_taker_wti_small.yaml
sed -i 's/^qty: 1$/qty: 0.2/' configs/taker_taker_wti_small.yaml
sed -i 's/^max_qty: 10$/max_qty: 2/' configs/taker_taker_wti_small.yaml
runbot --config configs/taker_taker_wti_small.yaml --mode live
```

Required env vars (see `.env.example`): `ASTER_USER`, `ASTER_SIGNER`,
`ASTER_SIGNER_PRIVKEY`, `LIGHTER_API_KEY_PRIVATE_KEY`.

**Daily checks during small-live:**
- Realised PnL trajectory shape matches the backtest (positive drift, max
  daily drawdown ≪ `daily_loss_cap_usd`)
- No partial-fill events that the RiskManager isn't catching
- Markout drift: realized vs predicted within ±1 bps

**Abort triggers** (halt + rebuild + rerun paper before resuming):
- Consecutive 3 days of negative realized PnL
- Realised markout > predicted + 1 bps sustained > 24 h
- Final position drifts to max_qty without unwind within 1 h

---

## Step 4 — Full live (qty=1 / max_qty=10)

```bash
runbot --config configs/taker_taker_wti.yaml --mode live
```

Continue daily monitoring (PnL, markout drift, latency drift).

---

## Step 5 — Continuous: rebuild markout table

Schedule **externally** (cron / systemd timer / Airflow — operator's choice).
Recommended cadence: **daily at 03:00 UTC** (low-volume window).

Example cron entry:
```
0 3 * * *  cd /path/to/perp_arbitrage && \
  python scripts/build_markout_table.py \
    --bbo-root logs/spread_WTI_<ts> \
    --left-latency-ms 350 --right-latency-ms 50 \
    --lookback-days 7 \
    --out configs/markout/wti_lighter_aster.json \
    >> logs/markout-build.log 2>&1
```

Or a systemd timer (`markout-rebuild.timer` + `markout-rebuild.service`).

The live bot reloads the table only on **restart**, so a fresh table takes
effect at the next restart. If you want hot-reload, that requires code
changes (not in scope for v1).

---

## Rollback

If the optimisations cause issues, **revert to baseline** without redeploy
by editing the YAML's `optimisations:` block to all-off (and restarting):

```yaml
optimisations:
  markout_table_path: null
  throttle_bump_bps: 0
  in_flight_cap_per_direction: 0
```

This restores the pre-Wave-1 behaviour byte-for-byte (verified by
backtest regression on 2026-05-21).

---

## Known gotchas

1. **In-flight cap is structurally a no-op in live.** The `_evaluating` gate
   in `taker_taker._schedule_eval` already serialises evaluation — at most
   one decision can be in flight at a time. The cap is wired for parity
   with the backtest path and as a forward-safety belt; it will never log
   `in-flight cap reached` under the current event-loop design. Expect 0
   occurrences in `decisions_*.csv`; if you see non-zero, investigate.

2. **Throttle decay uses `now_ms()` (wall-clock).** Backtest uses
   `snap.ts_ms` (capture time). For halflife = 3 s this is fine; sub-second
   halflives would be unsafe.

3. **Markout table is latency-profile-specific.** It was built for
   `lighter=350 ms, aster=50 ms`. If real-world latency drifts > 50 ms,
   the table mis-predicts. Monitor `legs_*.csv:latency_ms` per venue
   weekly.

4. **First-time deployment needs a capture first.** Markout table requires
   ≥ 24 h of spread_monitor capture before the first build. Do not point
   the YAML at a non-existent path — the Pydantic validator will refuse
   to start.

5. **`inventory_skew_bps` is NOT exposed.** The backtest sweep showed it
   is alone-bad / combined-good with markout. Deliberately omitted from
   `OptimisationsCfg` until rolling markout calibration is stable. To
   add later: extend `OptimisationsCfg` in `core/config.py` and thread
   through `taker_taker.py`'s `AssessParams` construction.

---

## Reference numbers (Wave-1 backtest, 20 h on 2026-05-20 WTI capture)

| variant | trips | $/day | bps/trip | max\|pos\| |
|---|---:|---:|---:|---:|
| baseline (no opts) | 4097 | +$3.95 | +0.08 | 10 |
| Wave-1 (this config) | **911** | **+$9.93** | **+0.88** | **1** |
| Wave-1 + inv_skew | 1445 | +$14.92 | +0.84 | 2 |

OOS (train markout on first 10 h, backtest second 10 h): **+$9.93/day → +$12.62/day**
(invsewn variant). The non-skew config preserves ~85 % of the in-sample
PnL OOS, vs markout-only which only preserves 16 %.

Source: `memory/wti-aster-lighter-viable.md`.
