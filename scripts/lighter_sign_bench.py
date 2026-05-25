"""Lighter sign-only micro-bench.

Mirrors everything `LighterClient.place_market_order` does up to (but NOT
including) `await self._user_ws.send_tx(...)`. Reports the latency
distribution of:

  * `sign_only`  — just `SignerClient.sign_create_order(...)` (the ctypes
                   → lighter-go BabyJubJub signature path).
  * `pre_send`   — the full prep block in `place_market_order` (meta
                   lookup, Decimal math, sign). Confirms whether the
                   non-sign prep is negligible.

The point: isolate signing cost from WS round-trip cost, so you can
decide whether moving `sign_create_order` off the event loop (e.g. via
`asyncio.to_thread`) is worth the work.

Usage:
    uv run python scripts/lighter_sign_bench.py --symbol ETH --qty 0.005 --iters 500

Reads the same env as `lighter_smoke.py`:
    LIGHTER_BASE_URL, LIGHTER_API_KEY_PRIVATE_KEY,
    LIGHTER_ACCOUNT_INDEX, LIGHTER_API_KEY_INDEX (optional, default 0)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import statistics
import sys
import time
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from perp_arb.core.types import MarketInfo, Side  # noqa: E402
from perp_arb.exchanges.lighter.client import (  # noqa: E402
    LighterClient,
    _worst_acceptable_price,
)

_log = logging.getLogger("lighter_sign_bench")


async def _wait_for_quote(
    client: LighterClient, market: MarketInfo, timeout_s: float = 20.0,
) -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if client.best_quote(market) is not None:
            return
        await asyncio.sleep(0.1)
    raise RuntimeError("no quote received — public depth WS not connected?")


def _pct(samples: list[float]) -> str:
    s = sorted(samples)
    n = len(s)

    def q(p: float) -> float:
        return s[min(n - 1, int(n * p))]

    return (
        f"n={n}  min={s[0]:6.2f}  p50={q(.5):6.2f}  "
        f"p90={q(.9):6.2f}  p99={q(.99):6.2f}  max={s[-1]:6.2f}  "
        f"mean={statistics.mean(s):6.2f}  stdev={statistics.stdev(s):5.2f}  (ms)"
    )


async def run(symbol: str, qty: Decimal, iters: int, side: Side) -> int:
    client = LighterClient(
        base_url=os.environ["LIGHTER_BASE_URL"],
        api_key_private_key=os.environ["LIGHTER_API_KEY_PRIVATE_KEY"],
        account_index=int(os.environ["LIGHTER_ACCOUNT_INDEX"]),
        api_key_index=int(os.environ.get("LIGHTER_API_KEY_INDEX", "0")),
    )
    try:
        await client.connect()
        market = await client.load_market(symbol)

        # Need a real BBO so `_worst_acceptable_price` doesn't hit the
        # extreme-bound fallback (signs would still work, but price_int
        # would be unrealistic and the comparison vs production wouldn't
        # hold). Spin up public WS and wait.
        client._ensure_public_ws(market)
        await _wait_for_quote(client, market, timeout_s=20)

        meta = client._meta_by_symbol[symbol]
        signer = client._signer
        assert signer is not None

        is_ask = side is Side.SELL
        q = client.best_quote(market)
        _log.info("symbol=%s side=%s qty=%s mid=%s", symbol, side.value, qty, q.mid if q else "?")

        # Warmup — first few signs can be slow (lib load, page faults).
        for _ in range(10):
            signer.sign_create_order(
                market_index=meta.market_index,
                client_order_index=int(time.time() * 1e6) % (1 << 48),
                base_amount=int(qty * meta.base_multiplier),
                price=int(_worst_acceptable_price(q, is_ask) * meta.price_multiplier),
                is_ask=int(is_ask),
                order_type=signer.ORDER_TYPE_MARKET,
                time_in_force=signer.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
                reduce_only=0,
                order_expiry=signer.DEFAULT_IOC_EXPIRY,
                api_key_index=client.api_key_index,
            )
        _log.info("warmup done, benchmarking %d iterations...", iters)

        sign_only_ms: list[float] = []
        pre_send_ms: list[float] = []
        base_t = int(time.time() * 1e6)

        for i in range(iters):
            # Re-read quote each iter (in production, prep uses fresh BBO).
            q = client.best_quote(market)

            t_pre = time.perf_counter()
            # --- begin: identical to place_market_order prep block ---
            meta_ = client._meta_by_symbol[market.symbol.raw]
            worst_price = _worst_acceptable_price(q, is_ask)
            avg_price_int = int(worst_price * meta_.price_multiplier)
            base_amount = int(qty * meta_.base_multiplier)
            coi = (base_t + i) % (1 << 48)
            # --- end prep ---

            t_sign = time.perf_counter()
            tx_type, tx_info, _h, err = signer.sign_create_order(
                market_index=meta_.market_index,
                client_order_index=coi,
                base_amount=base_amount,
                price=avg_price_int,
                is_ask=int(is_ask),
                order_type=signer.ORDER_TYPE_MARKET,
                time_in_force=signer.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
                reduce_only=0,
                order_expiry=signer.DEFAULT_IOC_EXPIRY,
                api_key_index=client.api_key_index,
            )
            t_done = time.perf_counter()

            if err is not None:
                _log.error("sign #%d failed: %s", i, err)
                return 1
            assert tx_type is not None and tx_info is not None

            sign_only_ms.append((t_done - t_sign) * 1000)
            pre_send_ms.append((t_done - t_pre) * 1000)

            # Yield occasionally so the event loop can service WS heartbeats.
            if i % 50 == 49:
                await asyncio.sleep(0)

        print()
        print("=== lighter sign latency (event loop, single-threaded) ===")
        print("  sign_only : " + _pct(sign_only_ms))
        print("  pre_send  : " + _pct(pre_send_ms))
        prep_overhead = statistics.mean(pre_send_ms) - statistics.mean(sign_only_ms)
        print(f"  prep overhead (pre_send − sign_only mean): {prep_overhead:.3f} ms")
        return 0
    finally:
        await client.disconnect()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ETH")
    parser.add_argument("--qty", type=Decimal, default=Decimal("0.005"))
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--side", choices=["buy", "sell"], default="buy")
    parser.add_argument("--env-file", default=str(Path(__file__).parent.parent / ".env"))
    args = parser.parse_args()
    load_dotenv(args.env_file)
    return asyncio.run(run(
        args.symbol, args.qty, args.iters,
        Side.BUY if args.side == "buy" else Side.SELL,
    ))


if __name__ == "__main__":
    sys.exit(main())
