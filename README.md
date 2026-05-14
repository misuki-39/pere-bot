# perp-arb

Cross-venue perp arbitrage bot. First strategy: taker-taker between **Aster** and **Lighter** on ETH perps.

## Quick start

```bash
# 1. Install deps (uv handles the venv)
uv sync

# 2. Configure secrets
cp .env.example .env
# edit .env with your Aster + Lighter keys

# 3. (optional) Pure observability — runs the spread_monitor strategy, writes per-tick spread CSV
uv run runbot --config configs/spread_monitor_eth.yaml      # if you create one
# or run the same taker_taker config in paper mode (full decision math, no orders)
uv run runbot --config configs/taker_taker_eth.yaml --mode paper

# 4. Live (after paper looks correct)
uv run runbot --config configs/taker_taker_eth.yaml --mode live
```

## Architecture

```
src/perp_arb/
├── core/         # types, BaseExchange, config, logging
├── exchanges/
│   ├── aster/    # V3 ECDSA signer + REST + WS
│   └── lighter/  # wraps official lighter SDK
├── strategy/     # spread_monitor, taker_taker
├── risk/         # kill switch, position cap, loss cap
└── utils/        # retry, precision, time
```

## Modes

- `paper` — full decision math; orders are no-ops that log the would-be fill and update an in-memory synthetic position.
- `live`  — real orders.

For pure observability (per-tick spread CSV, no decision math) use the `spread_monitor` *strategy* rather than a separate mode. Credentials are only required for `live`; `paper` runs against public WS streams with placeholder keys.

Always start with `paper`, collect at least a few hours of data, then `live` with `qty` set to venue minimum.

## Tests

```bash
uv run pytest -q          # unit tests (signer goldens, entry math, precision)
uv run ruff check
uv run mypy src/
```
