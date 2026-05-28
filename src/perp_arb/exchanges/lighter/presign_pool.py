"""Pre-signed order pool for Lighter — removes `sign_create_order` from the
order-firing hot path.

At any moment the pool holds two fresh signed market-IOC payloads (one BUY,
one SELL), BOTH baked with the same currently-expected nonce. When a fire
happens, `place_market_order` consumes the matching slot and immediately
ships its `tx_info` over WS; the other side becomes stale (its nonce was
just consumed) and is invalidated. A background task re-signs both slots
with the new nonce.

Empirical validation (bench): single api_key + same-nonce dual-side
pre-sign succeeds — the server only consumes a nonce when a tx is actually
sent. See scripts/bench_lighter_sign.py and docs/wti_deployment.md.

The pool refreshes on three independent triggers:
  - Post-fire: invalidate-and-resign once `place_market_order` succeeds
  - Time: per-slot age >= `refresh_interval_s` (defaults to 4 min, well
    inside the 5-min observed shelf life)
  - Drift: live quotes drop the remaining slippage buffer below
    `drift_threshold_bps` (worst price is signed at ±5% of mid, i.e.
    500 bps; default trigger of 200 bps means mid drifted ~300 bps)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, cast

from ...core.client_id import ClientIdGenerator
from ...core.types import Quote, Side

_log = logging.getLogger(__name__)

# `_refresh_loop` polls slot age every `_POLL_INTERVAL_S` seconds. Doesn't
# need to be tight — drift refresh handles the urgent path; the time trigger
# is just a defence against staying on a stale signature for too long.
_POLL_INTERVAL_S = 10.0


@dataclass(slots=True)
class _Slot:
    coi: int
    nonce: int
    side: Side
    tx_type: int
    tx_info: str
    worst_price: Decimal     # baked into the signature; drift math anchor
    signed_mid: Decimal      # diagnostic only
    signed_at_monotonic: float


class LighterPreSignedPool:
    def __init__(
        self,
        *,
        signer: Any,                            # lighter.SignerClient
        market_index: int,
        price_multiplier: int,
        base_multiplier: int,
        api_key_index: int,
        qty: Decimal,
        get_quote: Callable[[], Quote | None],
        nonce_borrower: Callable[[], Awaitable[int]],
        refresh_interval_s: float,
        drift_threshold_bps: Decimal,
        coi_seed: int | None = None,
    ) -> None:
        self._signer = signer
        self._market_index = market_index
        self._price_mult = price_multiplier
        self._base_mult = base_multiplier
        self._api_key_index = api_key_index
        self._qty_units = int(qty * base_multiplier)
        self._get_quote = get_quote
        self._nonce_borrower = nonce_borrower
        self._refresh_interval_s = refresh_interval_s
        self._drift_threshold_bps = drift_threshold_bps

        self._slots: dict[Side, _Slot | None] = {Side.BUY: None, Side.SELL: None}
        self._pool_lock = asyncio.Lock()
        self._stopping = asyncio.Event()
        self._refresh_task: asyncio.Task[None] | None = None
        self._post_fire_task: asyncio.Task[None] | None = None
        self._drift_task: asyncio.Task[None] | None = None
        # Seed disjoint from CounterClientIdGenerator's epoch-second space.
        # `time.time_ns() // 1_000_000` is epoch-ms; mod 2^40 keeps it
        # comfortably inside Lighter's 2^48-10 coi cap with room to grow.
        self._coi_counter = (
            coi_seed if coi_seed is not None
            else (time.time_ns() // 1_000_000) % (1 << 40)
        )

    # ---- lifecycle ----

    async def start(self) -> None:
        """Sign the initial BUY+SELL pair and spawn the time-refresh loop.

        Tolerates the no-quote-yet case: if `get_quote()` returns None,
        the initial sign is skipped and the refresh loop will retry.
        Callers (place_market_order) must handle empty slots via the cold-
        sign fallback.
        """
        await self._refresh_both("startup")
        self._refresh_task = asyncio.create_task(
            self._refresh_loop(), name="lighter-presign-refresh",
        )

    async def stop(self) -> None:
        self._stopping.set()
        for t in (self._refresh_task, self._post_fire_task, self._drift_task):
            if t is not None and not t.done():
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
        self._refresh_task = None
        self._post_fire_task = None
        self._drift_task = None
        async with self._pool_lock:
            self._slots[Side.BUY] = None
            self._slots[Side.SELL] = None

    # ---- consumer-facing API ----

    def peek_coi(self, side: Side) -> int | None:
        """Returns the staged coi for `side`, or None if empty.

        Read-only; safe to call from sync code (e.g.,
        `PreSignedPoolClientIdGenerator.next`).
        """
        slot = self._slots[side]
        return slot.coi if slot is not None else None

    def pop_for_fire(self, side: Side, client_id: str) -> _Slot | None:
        """Atomic: validate cid match, pop the slot, invalidate the sibling,
        schedule background re-sign. Returns None on mismatch (caller falls
        back to cold sign).

        Synchronous — must run in `place_market_order`'s tight pre-lock
        window so no fill event can interleave. The actual re-sign happens
        off-stack in the post-fire task.
        """
        try:
            cid_int = int(client_id)
        except (TypeError, ValueError):
            return None
        slot = self._slots[side]
        if slot is None or slot.coi != cid_int:
            return None
        # Consume + invalidate sibling (its nonce is about to be eaten by
        # the fire we're about to make).
        self._slots[side] = None
        self._slots[side.opposite] = None
        if self._post_fire_task is None or self._post_fire_task.done():
            self._post_fire_task = asyncio.create_task(
                self._refresh_both("post-fire"),
                name="lighter-presign-post-fire",
            )
        return slot

    def on_quote(self, quote: Quote) -> None:
        """Sync quote callback. Checks remaining buffer per side; if any
        side's buffer dipped below `drift_threshold_bps`, kicks off a
        background re-sign.

        In-flight guard prevents a quote storm from spawning duplicates;
        only one drift task at a time.
        """
        if self._drift_task is not None and not self._drift_task.done():
            return
        breached = False
        for side in (Side.BUY, Side.SELL):
            slot = self._slots[side]
            if slot is None:
                continue
            remaining_bps = _remaining_buffer_bps(side, slot.worst_price, quote.mid)
            if remaining_bps < self._drift_threshold_bps:
                breached = True
                break
        if breached:
            self._drift_task = asyncio.create_task(
                self._refresh_both("drift"),
                name="lighter-presign-drift",
            )

    # ---- internals ----

    async def _refresh_loop(self) -> None:
        """Polls slot age every _POLL_INTERVAL_S; triggers refresh when any
        slot is older than refresh_interval_s OR currently empty."""
        try:
            while not self._stopping.is_set():
                try:
                    await asyncio.wait_for(
                        self._stopping.wait(), timeout=_POLL_INTERVAL_S,
                    )
                    return  # stop signalled
                except TimeoutError:
                    pass
                now_mono = time.monotonic()
                needs_refresh = False
                for side in (Side.BUY, Side.SELL):
                    slot = self._slots[side]
                    if slot is None:
                        needs_refresh = True
                        break
                    if now_mono - slot.signed_at_monotonic >= self._refresh_interval_s:
                        needs_refresh = True
                        break
                if needs_refresh:
                    await self._refresh_both("time")
        except asyncio.CancelledError:
            pass

    async def _refresh_both(self, reason: str) -> None:
        """Re-sign both BUY and SELL with the current expected nonce.

        Acquires _pool_lock to serialise against concurrent refreshes;
        nonce is borrowed via the injected callable (which takes the
        client's _nonce_lock briefly).
        """
        async with self._pool_lock:
            quote = self._get_quote()
            if quote is None:
                _log.warning(
                    "lighter-presign-pool: no quote yet, skip refresh (%s)", reason,
                )
                return
            try:
                nonce = await self._nonce_borrower()
            except Exception as e:  # noqa: BLE001
                _log.warning(
                    "lighter-presign-pool: nonce borrow failed (%s): %s", reason, e,
                )
                return
            mid = quote.mid
            tick = Decimal(1) / Decimal(self._price_mult)
            buy_worst = (mid * Decimal("1.05")).quantize(tick)
            sell_worst = (mid * Decimal("0.95")).quantize(tick)

            buy_slot = self._sign_slot(Side.BUY, nonce, mid, buy_worst)
            sell_slot = self._sign_slot(Side.SELL, nonce, mid, sell_worst)
            if buy_slot is None or sell_slot is None:
                # One leg failed to sign — keep both slots empty rather
                # than leave a mixed pair. Cold-sign covers fires until
                # the next refresh attempt.
                self._slots[Side.BUY] = None
                self._slots[Side.SELL] = None
                _log.warning(
                    "lighter-presign-pool: partial sign on refresh (%s) — slots cleared",
                    reason,
                )
                return
            self._slots[Side.BUY] = buy_slot
            self._slots[Side.SELL] = sell_slot
            _log.debug(
                "lighter-presign-pool: refreshed (%s) nonce=%s buy_coi=%s sell_coi=%s mid=%s",
                reason, nonce, buy_slot.coi, sell_slot.coi, mid,
            )

    def _sign_slot(
        self, side: Side, nonce: int, mid: Decimal, worst_price: Decimal,
    ) -> _Slot | None:
        self._coi_counter += 1
        coi = self._coi_counter
        worst_int = int(worst_price * self._price_mult)
        is_ask = 1 if side is Side.SELL else 0
        tx_type, tx_info, _tx_hash, err = self._signer.sign_create_order(
            market_index=self._market_index,
            client_order_index=coi,
            base_amount=self._qty_units,
            price=worst_int,
            is_ask=is_ask,
            order_type=self._signer.ORDER_TYPE_MARKET,
            time_in_force=self._signer.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
            reduce_only=0,
            order_expiry=self._signer.DEFAULT_IOC_EXPIRY,
            nonce=nonce,
            api_key_index=self._api_key_index,
        )
        if err is not None:
            _log.warning(
                "lighter-presign-pool: sign error side=%s coi=%s err=%s",
                side, coi, err,
            )
            return None
        return _Slot(
            coi=coi,
            nonce=nonce,
            side=side,
            tx_type=int(cast(int, tx_type)),
            tx_info=str(tx_info),
            worst_price=worst_price,
            signed_mid=mid,
            signed_at_monotonic=time.monotonic(),
        )


def _remaining_buffer_bps(
    side: Side, worst_price: Decimal, current_mid: Decimal,
) -> Decimal:
    """How many bps of slippage budget remain between current mid and the
    signed worst price.

    BUY worst is ABOVE mid (we'd pay up to `worst`); as mid rises, the gap
    shrinks. SELL worst is BELOW mid; as mid falls, the gap shrinks. The
    pool watches whichever side's buffer is closing.
    """
    if current_mid <= 0:
        return Decimal(0)
    if side is Side.BUY:
        return (worst_price - current_mid) / current_mid * Decimal(10000)
    return (current_mid - worst_price) / current_mid * Decimal(10000)


class PreSignedPoolClientIdGenerator:
    """Returns the pool's pre-staged coi; falls back to the wrapped counter
    generator when the pool's slot for that side is empty (startup before
    first refresh / mid-refresh window / sign error)."""

    def __init__(
        self,
        pool: LighterPreSignedPool,
        fallback: ClientIdGenerator,
    ) -> None:
        self._pool = pool
        self._fallback = fallback

    def next(self, *, side: Side) -> str:
        coi = self._pool.peek_coi(side)
        if coi is not None:
            return str(coi)
        return self._fallback.next(side=side)
