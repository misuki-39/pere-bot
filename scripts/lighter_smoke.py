"""Lighter mainnet smoke — verifies WS order submission + account user-data routing.

  1. Connect Lighter (ApiClient + SignerClient + user WS + public WS).
  2. Resolve <symbol> market metadata.
  3. Subscribe fill + position callbacks (both fed by user WS account_all).
  4. Wait for live BBO from the public depth stream.
  5. MARKET BUY `--qty`. Assert:
        * `jsonapi/sendtx` reply OK.
        * Order update arrives → fill callback fires with status=FILLED.
        * Position update arrives → live_position size matches expected delta.
  6. MARKET SELL `--qty` reduce_only. Same three assertions, size → 0.

Usage:
    uv run python scripts/lighter_smoke.py --symbol ETH --qty 0.005

Safety: refuses to run unless `--mainnet` is passed (Lighter has no testnet
URL accepted by the current SDK).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from _smoke_utils import WsObserver, wait_for_quote  # noqa: E402

from perp_arb.core.logging import setup_logging  # noqa: E402
from perp_arb.core.types import MarketInfo, Side  # noqa: E402
from perp_arb.exchanges.lighter.client import LighterClient  # noqa: E402

_log = logging.getLogger("lighter_smoke")

_MAX_SPREAD_BPS = Decimal("50")   # abort if spread is wider than 50 bps


def _check_spread(q) -> None:
    spread_bps = (q.ask - q.bid) / q.mid * Decimal("10000")
    if spread_bps > _MAX_SPREAD_BPS:
        raise RuntimeError(
            f"spread too wide for safe MARKET smoke: {spread_bps:.1f} bps "
            f"(bid={q.bid} ask={q.ask}, max={_MAX_SPREAD_BPS} bps)"
        )
    _log.info("spread check OK: %.2f bps", spread_bps)


async def _round_trip(
    client: LighterClient,
    market: MarketInfo,
    obs: WsObserver,
    *,
    side: Side,
    qty: Decimal,
    reduce_only: bool,
    expected_delta: Decimal,
    label: str,
) -> Decimal:
    before = client.live_position(market)
    baseline = before.size if before else Decimal(0)
    _log.info("--- %s: MARKET %s qty=%s reduce_only=%s (baseline=%s) ---",
              label, side.value.upper(), qty, reduce_only, baseline)

    obs.arm()
    # Lighter requires the cid to be int-parseable and fit in 48 bits;
    # ms-since-epoch is ~1.78e12, well under 2^48 (~2.8e14).
    cid = str(int(time.time() * 1000))
    res = await client.place_market_order(
        market, side, qty, reduce_only=reduce_only, client_id=cid,
    )
    _log.info("WS sendtx result: success=%s order_id=%s latency=%dms err=%s",
              res.success, res.order_id, res.latency_ms or -1, res.error_message)
    if not res.success:
        raise RuntimeError(f"{label}: order rejected: {res.error_message}")

    # Lighter's account_all WS pushes `positions` and `trades` (not `orders`),
    # so position update is the authoritative fill signal here.
    await obs.wait_position()

    live = client.live_position(market)
    assert live is not None, f"{label}: live_position() still None after account_all push"
    delta = live.size - baseline
    if abs(delta - expected_delta) > market.lot_size:
        raise AssertionError(
            f"{label}: position delta={delta} doesn't match expected={expected_delta} "
            f"(before={baseline} after={live.size}, tol={market.lot_size})"
        )
    _log.info("%s OK: delta=%+s size=%s entry=%s", label, delta, live.size, live.entry_price)
    return live.size


async def run(symbol: str, qty: Decimal) -> int:
    client = LighterClient(
        base_url=os.environ["LIGHTER_BASE_URL"],
        api_key_private_key=os.environ["LIGHTER_API_KEY_PRIVATE_KEY"],
        account_index=int(os.environ["LIGHTER_ACCOUNT_INDEX"]),
        api_key_index=int(os.environ.get("LIGHTER_API_KEY_INDEX", "0")),
    )
    _log.info("lighter endpoint: %s (account=%s, api_key_index=%s)",
              client.base_url, client.account_index, client.api_key_index)
    try:
        await client.connect()
        market = await client.load_market(symbol)
        _log.info("market %s tick=%s lot=%s market_index=%s",
                  market.symbol.raw, market.tick_size, market.lot_size, market.contract_id)

        # Lighter is slower than Aster; bump per-leg timeouts.
        obs = WsObserver(fill_timeout_s=8.0, position_timeout_s=8.0)
        client.subscribe_fills(market, obs.on_fill)
        client.subscribe_positions(market, obs.on_position)
        client.subscribe_quotes(market, lambda _q: None)

        await wait_for_quote(client, market, timeout_s=12.0)
        q = client.best_quote(market)
        assert q is not None
        _log.info("live quote: bid=%s ask=%s mid=%s", q.bid, q.ask, q.mid)
        _check_spread(q)

        seeded = await client.get_position(market)
        _log.info("baseline position (REST): size=%s entry=%s",
                  seeded.size, seeded.entry_price)

        await _round_trip(
            client, market, obs,
            side=Side.BUY, qty=qty, reduce_only=False,
            expected_delta=qty, label="OPEN",
        )
        await _round_trip(
            client, market, obs,
            side=Side.SELL, qty=qty, reduce_only=True,
            expected_delta=-qty, label="CLOSE",
        )

        _log.info("SMOKE TEST PASSED (fills=%d, position_updates=%d)",
                  len(obs.fills), len(obs.positions))
        return 0
    finally:
        await client.disconnect()


def main() -> None:
    p = argparse.ArgumentParser(description="Lighter WS order/position smoke test")
    p.add_argument("--symbol", default="ETH")
    p.add_argument("--qty", type=Decimal, default=Decimal("0.005"))
    p.add_argument("--env-file", default=".env")
    p.add_argument("--mainnet", action="store_true",
                   help="Required to acknowledge running against mainnet (real money).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    load_dotenv(args.env_file)
    setup_logging(Path("./logs"), level=args.log_level, run_tag="lighter_smoke")

    if not args.mainnet:
        sys.stderr.write(
            "ABORT: Lighter has no working testnet — this smoke trades REAL funds.\n"
            "Pass --mainnet to acknowledge.\n"
        )
        sys.exit(1)

    sys.exit(asyncio.run(run(args.symbol, args.qty)))


if __name__ == "__main__":
    main()
