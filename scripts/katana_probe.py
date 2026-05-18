"""Pre-flight probe for the Katana Perps v1 public API.

Run this BEFORE finalizing the KatanaPublicWs parser. It pins down:
  * the working mainnet WS URL,
  * the l2_orderbook subscribe ack + snapshot vs diff message shapes,
  * field names for bid/ask price/size and the monotonic `sequence`,
  * app-level ping behavior,
  * GET /v1/markets (tickSize/stepSize) and GET /v1/orderbook shapes.

Usage:
    uv run python scripts/katana_probe.py [--market ETH-USD] [--n 12]
    KATANA_BASE_URL / KATANA_WS_URL env vars override the defaults.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import aiohttp
import websockets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from perp_arb.utils.proxy import get_proxy_url  # noqa: E402

DEFAULT_REST = "https://api-perps.katana.network"
DEFAULT_WS = "wss://websocket-perps.katana.network/v1"


def _pp(label: str, obj: object) -> None:
    print(f"\n===== {label} =====")
    print(json.dumps(obj, indent=2, default=str)[:4000])


async def probe_rest(base_url: str, market: str) -> None:
    async with aiohttp.ClientSession(trust_env=True) as s:
        try:
            async with s.get(f"{base_url}/v1/markets", timeout=aiohttp.ClientTimeout(total=15)) as r:
                body = await r.json(content_type=None)
            _pp(f"GET /v1/markets (status {r.status})", body)
            rows = body if isinstance(body, list) else body.get("data") or body.get("markets") or body
            if isinstance(rows, list):
                hit = [m for m in rows if str(m.get("market")) == market]
                _pp(f"market == {market}", hit or "NOT FOUND — inspect the list above")
        except Exception as e:  # noqa: BLE001
            print(f"/v1/markets failed: {e!r}")

        try:
            url = f"{base_url}/v1/orderbook?market={market}&level=2"
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                body = await r.json(content_type=None)
            _pp(f"GET /v1/orderbook?market={market}&level=2 (status {r.status})", body)
        except Exception as e:  # noqa: BLE001
            print(f"/v1/orderbook failed: {e!r}")


async def probe_ws(ws_url: str, market: str, n: int) -> None:
    sub = {"type": "subscribe", "channels": ["l2_orderbook"], "market": market}
    print(f"\n===== WS connect {ws_url} =====")
    try:
        async with websockets.connect(ws_url, ping_interval=20, proxy=get_proxy_url()) as ws:
            await ws.send(json.dumps(sub))
            print(f"sent subscribe: {sub}")
            for i in range(n):
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
                try:
                    msg = json.loads(raw)
                except Exception:  # noqa: BLE001
                    print(f"[{i}] non-JSON frame: {raw!r}")
                    continue
                t = msg.get("type") if isinstance(msg, dict) else type(msg).__name__
                seq = msg.get("sequence") if isinstance(msg, dict) else None
                print(f"[{i}] type={t} sequence={seq} keys={list(msg)[:12] if isinstance(msg, dict) else msg}")
                if i < 3 or t not in ("l2_orderbook", "l2orderbook", "update"):
                    _pp(f"[{i}] full frame", msg)
                if isinstance(msg, dict) and t in ("ping",):
                    await ws.send(json.dumps({"type": "pong"}))
                    print(f"[{i}] -> sent pong")
    except Exception as e:  # noqa: BLE001
        print(f"WS probe failed on {ws_url}: {e!r}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="ETH-USD")
    ap.add_argument("--n", type=int, default=12, help="WS frames to capture")
    args = ap.parse_args()

    base_url = os.getenv("KATANA_BASE_URL", DEFAULT_REST).rstrip("/")
    ws_url = os.getenv("KATANA_WS_URL", DEFAULT_WS)

    await probe_rest(base_url, args.market)
    await probe_ws(ws_url, args.market, args.n)


if __name__ == "__main__":
    asyncio.run(main())
