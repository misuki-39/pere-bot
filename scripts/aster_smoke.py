"""Aster testnet smoke test — verifies order placement AND user-data WS routing.

  1. Connect Aster (REST + listenKey user stream + public depth).
  2. Resolve <symbol> market metadata.
  3. Subscribe fill + position callbacks (both fed by the user-data WS).
  4. Wait for live BBO from the public depth stream.
  5. MARKET BUY `--qty`. Assert:
        * `ORDER_TRADE_UPDATE` arrives → fill callback fires with status=FILLED.
        * `ACCOUNT_UPDATE` arrives → position callback fires with size ≈ qty.
        * `client.live_position(market)` matches.
  6. MARKET SELL `--qty` reduce_only. Same three assertions, size → 0.

Usage:
    uv run python scripts/aster_smoke.py --symbol BTCUSDT --qty 0.001

Safety: refuses to run when ASTER_REST_URL does not contain "testnet"
unless --mainnet is passed explicitly.
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
from perp_arb.exchanges.aster.client import AsterClient  # noqa: E402
from perp_arb.exchanges.aster.signer import AsterSigner  # noqa: E402

_log = logging.getLogger("aster_smoke")


async def _round_trip(
    client: AsterClient,
    market: MarketInfo,
    obs: WsObserver,
    *,
    side: Side,
    qty: Decimal,
    reduce_only: bool,
    expected_delta: Decimal,
    label: str,
) -> Decimal:
    """Place one order, verify both fill + ACCOUNT_UPDATE arrive, return the new size."""
    before = client.live_position(market)
    baseline = before.size if before else Decimal(0)
    _log.info("--- %s: MARKET %s qty=%s reduce_only=%s (baseline=%s) ---",
              label, side.value.upper(), qty, reduce_only, baseline)

    obs.arm()
    cid = f"smoke-{int(time.time() * 1000)}"
    res = await client.place_market_order(
        market, side, qty, reduce_only=reduce_only, client_id=cid,
    )
    _log.info("REST result: success=%s client_id=%s status=%s err=%s",
              res.success, res.client_id, res.status.value, res.error_message)
    if not res.success:
        raise RuntimeError(f"{label}: order rejected: {res.error_message}")

    await obs.wait_fill()
    await obs.wait_position()

    live = client.live_position(market)
    assert live is not None, f"{label}: live_position() still None after ACCOUNT_UPDATE"
    delta = live.size - baseline
    if delta != expected_delta:
        raise AssertionError(
            f"{label}: position delta={delta} doesn't match expected={expected_delta} "
            f"(before={baseline} after={live.size})"
        )
    _log.info("%s OK: delta=%+s size=%s entry=%s", label, delta, live.size, live.entry_price)
    return live.size


async def run(symbol: str, qty: Decimal) -> int:
    signer = AsterSigner(
        user=os.environ["ASTER_USER"],
        signer=os.environ["ASTER_SIGNER"],
        signer_privkey=os.environ["ASTER_SIGNER_PRIVKEY"],
        chain_id=int(os.environ.get("ASTER_CHAIN_ID", "1666")),
    )
    _log.info("aster endpoint: %s (chain_id=%d)",
              os.environ["ASTER_REST_URL"], signer.chain_id)

    client = AsterClient(
        rest_url=os.environ["ASTER_REST_URL"],
        ws_url=os.environ["ASTER_WS_URL"],
        signer=signer,
    )
    try:
        await client.connect()
        market = await client.load_market(symbol)
        _log.info("market %s tick=%s lot=%s min_qty=%s",
                  market.symbol.raw, market.tick_size, market.lot_size, market.min_qty)

        # Wire WS observers before placing any order so we don't race the events.
        obs = WsObserver()
        client.subscribe_fills(market, obs.on_fill)
        client.subscribe_positions(market, obs.on_position)
        client.subscribe_quotes(market, lambda _q: None)   # kicks off public depth WS

        await wait_for_quote(client, market)
        q = client.best_quote(market)
        assert q is not None
        _log.info("live quote: bid=%s ask=%s mid=%s", q.bid, q.ask, q.mid)

        # Seed the live_position cache from REST so OPEN's delta check is accurate
        # even before any ACCOUNT_UPDATE has fired.
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
    p = argparse.ArgumentParser(description="Aster order/position smoke test")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--qty", type=Decimal, default=Decimal("0.001"))
    p.add_argument("--env-file", default=".env")
    p.add_argument("--mainnet", action="store_true",
                   help="Allow running against a non-testnet endpoint (DANGER).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    load_dotenv(args.env_file)
    setup_logging(Path("./logs"), level=args.log_level, run_tag="aster_smoke")

    if "testnet" not in os.environ.get("ASTER_REST_URL", "") and not args.mainnet:
        sys.stderr.write(
            f"ABORT: ASTER_REST_URL={os.environ.get('ASTER_REST_URL')!r} "
            "doesn't look like testnet. Pass --mainnet to override.\n"
        )
        sys.exit(1)

    sys.exit(asyncio.run(run(args.symbol, args.qty)))


if __name__ == "__main__":
    main()
