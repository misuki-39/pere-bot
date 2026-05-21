"""Probe Lighter user WS — subscribe `account_market/<market>/<acct>` on a
single connection, pretty-print every payload that arrives. Use to
confirm the actual wire shape (orders/trades/positions all bundled per
market) before wiring it into the strategy code.

Usage:
    python scripts/lighter_ws_probe.py --symbol WTI [--seconds 120]

Env (same names as lighter_smoke.py):
    LIGHTER_BASE_URL
    LIGHTER_API_KEY_PRIVATE_KEY
    LIGHTER_ACCOUNT_INDEX
    LIGHTER_API_KEY_INDEX        (optional, default 0)

To see actual order/trade payloads you need to either:
  (a) already have open orders / fresh fills on the account, OR
  (b) place a tiny test order from a separate terminal (e.g.
      `scripts/lighter_smoke.py --symbol WTI --qty <lot_size>`)
  during the probe window. Subscriptions alone only give snapshots of
  whatever state exists.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import UTC, datetime

from dotenv import load_dotenv

# Make `perp_arb` importable when running from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import lighter  # noqa: E402

from perp_arb.exchanges.lighter.ws import LighterUserWs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("ws-probe")


def _ts() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S.%f")[:-3]


def _dump(label: str, data: dict) -> None:
    """Print one ws frame. Truncate giant `positions` snapshots so the
    interesting bits (`orders`, `trades`) aren't lost in the scroll."""
    pretty = json.dumps(data, indent=2, sort_keys=False, default=str)
    if len(pretty) > 4000:
        pretty = pretty[:4000] + f"\n... [truncated, total {len(pretty)} chars]"
    print(f"\n=== {_ts()}  {label} ===")
    print(pretty)
    sys.stdout.flush()


async def resolve_market_index(base_url: str, symbol: str) -> int:
    """Cheap REST hit to map symbol → market_index. Avoids needing a
    full LighterClient + signer up just to read the OrderBooks table."""
    async with lighter.ApiClient(configuration=lighter.Configuration(host=base_url)) as api:
        books = await lighter.OrderApi(api).order_books()
    for m in books.order_books:
        if m.symbol == symbol:
            return int(m.market_id)
    raise SystemExit(f"symbol {symbol!r} not found on lighter")


async def main(symbol: str, duration_s: int) -> None:
    load_dotenv()
    base_url = os.environ["LIGHTER_BASE_URL"]
    account_index = int(os.environ["LIGHTER_ACCOUNT_INDEX"])
    api_key_index = int(os.environ.get("LIGHTER_API_KEY_INDEX", "0"))
    api_key_private_key = os.environ["LIGHTER_API_KEY_PRIVATE_KEY"]

    market_index = await resolve_market_index(base_url, symbol)
    _log.info("%s → market_index=%d", symbol, market_index)

    signer = lighter.SignerClient(
        url=base_url, account_index=account_index,
        api_private_keys={api_key_index: api_key_private_key},
    )
    if (err := signer.check_client()) is not None:
        raise SystemExit(f"signer check_client failed: {err}")

    def auth_token_factory() -> str:
        token, err = signer.create_auth_token_with_expiry(api_key_index=api_key_index)
        if err:
            raise RuntimeError(f"create_auth_token failed: {err}")
        return token

    ws = LighterUserWs(
        base_url=base_url, account_index=account_index,
        auth_token_factory=auth_token_factory,
        subscribe_account_all=False,   # probe only wants account_market
    )
    ws.add_market_callback(lambda d: _dump("account_market", d))

    await ws.start()
    await ws.subscribe_account_market(market_index)
    _log.info("subscribed: account_market/%d/%d — listening %ds",
              market_index, account_index, duration_s)

    try:
        await asyncio.sleep(duration_s)
    finally:
        _log.info("shutting down")
        await ws.stop()
        await signer.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, help="lighter symbol e.g. WTI / ETH")
    ap.add_argument("--seconds", type=int, default=12000,
                    help="listen window in seconds (default 120)")
    args = ap.parse_args()
    asyncio.run(main(args.symbol, args.seconds))
