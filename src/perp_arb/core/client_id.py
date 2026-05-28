"""Composable client-id minting policy used by every venue driver.

The executor calls `exchange.client_id_generator.next(side=...)` to obtain a
cid for each leg of a fire. Default policy is a monotonic, epoch-seeded
counter — same shape every venue used historically when the executor owned
the counter itself.

Venues that pre-stage cids (e.g., a pre-signed pool that bakes the
`client_order_index` into the signature) plug in their own generator at
runtime; the executor stays venue-agnostic.
"""

from __future__ import annotations

import time
from typing import Protocol

from .types import Side


class ClientIdGenerator(Protocol):
    def next(self, *, side: Side) -> str: ...


class CounterClientIdGenerator:
    """Monotonic int counter, stringified. Seed defaults to epoch seconds —
    keeps cids comfortably inside Lighter's `2^48 - 10` cap and ensures
    cross-session monotonicity without external coordination."""

    def __init__(self, seed: int | None = None) -> None:
        self._counter = seed if seed is not None else int(time.time())

    def next(self, *, side: Side) -> str:
        # Return-then-increment so the FIRST cid emitted equals the seed
        # (preserves the bit-for-bit cid shape of the legacy executor counter).
        cid = str(self._counter)
        self._counter += 1
        return cid
