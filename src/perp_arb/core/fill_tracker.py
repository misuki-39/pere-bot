"""Per-`client_id` fill aggregator owned by exchange drivers.

Drivers fan WS fill events into `on_event`. Callers (the execution layer)
pre-register a `client_id` before submitting the order, then `await
await_terminal(cid, qty, timeout_s)` to receive the aggregated
`LegOutcome` once the venue settles the parent order — or whatever
landed by the timeout.

Pre-registration is a deliberate whitelist: events for unknown cids are
silently dropped, so a stale order from a previous session or another
client on the same account cannot pollute current waits.

Race-safety invariant: `register(cid)` must happen in the same synchronous
stretch as the order submit — there is no `await` between them, so a fast
fill (fills can land before the place ack returns on some venues) cannot
arrive before its slot exists. Callers who break this invariant will
silently drop fast fills.

The tracker only populates the *fill-side* fields of LegOutcome
(filled_qty, _weighted_price_sum, last_ts_ms, last_status, total_fee);
ack-side fields stay at defaults. The driver overlays ack-side fields
separately via `submit_and_await`.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from .types import FillDelta, LegOutcome, OrderSnapshot


class _PerCidFillTracker:
    """Single per-driver tracker. Cheap — no background tasks, no sockets."""

    def __init__(self) -> None:
        self._events: dict[str, asyncio.Event] = {}
        self._fills: dict[str, LegOutcome] = {}

    def register(self, client_id: str) -> None:
        """Whitelist `client_id` so subsequent `on_event(ev)` matching this
        cid will accumulate. MUST be called before the order submit.

        Idempotent: re-registering an active cid is a no-op (preserves any
        already-accumulated state). This matters because driver-internal
        retries could re-enter the path."""
        if client_id not in self._events:
            self._events[client_id] = asyncio.Event()

    def on_event(self, event: OrderSnapshot | FillDelta) -> None:
        """Driver-side WS handler entry point. Drops events for unregistered
        cids (stale orders, other sessions on the same account)."""
        cid = event.client_id
        if not cid or cid not in self._events:
            return
        self._fills.setdefault(cid, LegOutcome()).add(event)
        self._events[cid].set()

    async def await_terminal(
        self,
        client_id: str,
        requested_qty: Decimal,
        timeout_s: float,
    ) -> LegOutcome | None:
        """Wait up to `timeout_s` for the fill aggregate to reach
        `requested_qty` or a terminal status. Returns whatever landed
        (None if no events at all)."""
        ev = self._events.get(client_id)
        if ev is None:
            return None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while (remaining := deadline - loop.time()) > 0:
            try:
                await asyncio.wait_for(ev.wait(), timeout=remaining)
            except TimeoutError:
                break
            fill = self._fills.get(client_id)
            if fill and fill.is_complete(requested_qty):
                return fill
            ev.clear()
        return self._fills.get(client_id)

    def release(self, client_id: str) -> None:
        """Drop the slot. Idempotent."""
        self._events.pop(client_id, None)
        self._fills.pop(client_id, None)
