"""State-machine tests for `LighterPreSignedPool`.

The pool's correctness is independent of the real Lighter signer / WS — we
stub them out and exercise the lifecycle:
  - initial double-sign on start
  - post-fire invalidate + re-sign
  - drift-triggered refresh (with quote-storm in-flight guard)
  - time-triggered refresh
  - sign-error recovery
  - clean stop/start cycle
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal

import pytest

from perp_arb.core.client_id import CounterClientIdGenerator
from perp_arb.core.types import Quote, Side, Symbol
from perp_arb.exchanges.lighter.presign_pool import (
    LighterPreSignedPool,
    PreSignedPoolClientIdGenerator,
)

# ---------- fakes ----------


@dataclass
class _SignCall:
    market_index: int
    client_order_index: int
    base_amount: int
    price: int
    is_ask: int
    nonce: int
    api_key_index: int


class _FakeSigner:
    """Captures sign_create_order args and returns a deterministic payload
    encoded so tests can recover (nonce, coi) from the tx_info string."""

    # Constants mirrored from lighter.SignerClient that the pool reads.
    ORDER_TYPE_MARKET = 1
    ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL = 0
    DEFAULT_IOC_EXPIRY = 0

    def __init__(self, *, sign_err: str | None = None) -> None:
        self.calls: list[_SignCall] = []
        self.sign_err = sign_err  # if set, every sign returns this error

    def sign_create_order(
        self,
        *,
        market_index: int,
        client_order_index: int,
        base_amount: int,
        price: int,
        is_ask: int,
        order_type: int,
        time_in_force: int,
        reduce_only: int,
        order_expiry: int,
        nonce: int,
        api_key_index: int,
    ) -> tuple[int | None, str | None, None, str | None]:
        self.calls.append(_SignCall(
            market_index=market_index,
            client_order_index=client_order_index,
            base_amount=base_amount,
            price=price,
            is_ask=is_ask,
            nonce=nonce,
            api_key_index=api_key_index,
        ))
        if self.sign_err is not None:
            return None, None, None, self.sign_err
        # Encode (nonce, coi) in tx_info so tests can verify which sign
        # produced which slot.
        return 99, f'{{"nonce":{nonce},"coi":{client_order_index}}}', None, None


def _make_quote(mid_str: str) -> Quote:
    mid = Decimal(mid_str)
    bid = mid - Decimal("0.1")
    ask = mid + Decimal("0.1")
    return Quote(
        symbol=Symbol(exchange="lighter", raw="ETH", base="ETH", quote="USD"),
        bid=bid,
        bid_size=Decimal("10"),
        ask=ask,
        ask_size=Decimal("10"),
        ts_ms=0,
    )


@dataclass
class _Env:
    signer: _FakeSigner
    nonce_value: int = 1000
    quote: Quote | None = field(default_factory=lambda: _make_quote("2000"))
    borrow_calls: int = 0

    async def borrow_nonce(self) -> int:
        self.borrow_calls += 1
        return self.nonce_value


def _make_pool(
    env: _Env,
    *,
    refresh_interval_s: float = 240.0,
    drift_threshold_bps: Decimal = Decimal("200"),
    coi_seed: int = 100,
) -> LighterPreSignedPool:
    return LighterPreSignedPool(
        signer=env.signer,
        market_index=0,
        price_multiplier=100,
        base_multiplier=10000,
        api_key_index=0,
        qty=Decimal("0.01"),
        get_quote=lambda: env.quote,
        nonce_borrower=env.borrow_nonce,
        refresh_interval_s=refresh_interval_s,
        drift_threshold_bps=drift_threshold_bps,
        coi_seed=coi_seed,
    )


# ---------- tests ----------


@pytest.mark.asyncio
async def test_start_signs_both_sides_with_same_nonce() -> None:
    env = _Env(signer=_FakeSigner())
    pool = _make_pool(env)
    await pool.start()
    try:
        buy_coi = pool.peek_coi(Side.BUY)
        sell_coi = pool.peek_coi(Side.SELL)
        assert buy_coi is not None and sell_coi is not None
        assert buy_coi != sell_coi
        # Both signed with nonce 1000.
        nonces = {c.nonce for c in env.signer.calls}
        assert nonces == {1000}
        assert len(env.signer.calls) == 2
        # Ask sides correctly tagged.
        ask_set = {c.is_ask for c in env.signer.calls}
        assert ask_set == {0, 1}
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_pop_for_fire_matching_consumes_and_invalidates_sibling() -> None:
    env = _Env(signer=_FakeSigner())
    pool = _make_pool(env)
    await pool.start()
    try:
        buy_coi = pool.peek_coi(Side.BUY)
        assert buy_coi is not None
        slot = pool.pop_for_fire(Side.BUY, str(buy_coi))
        assert slot is not None
        assert slot.side is Side.BUY
        assert slot.coi == buy_coi
        # Both sides now empty (sibling invalidated; nonce was about to
        # be consumed).
        assert pool.peek_coi(Side.BUY) is None
        assert pool.peek_coi(Side.SELL) is None

        # Bump nonce to simulate place_market_order's success path
        # advancing _next_nonce, then await post-fire re-sign task.
        env.nonce_value = 1001
        # Wait for the scheduled re-sign to complete.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if pool.peek_coi(Side.BUY) is not None and pool.peek_coi(Side.SELL) is not None:
                break
        new_buy = pool.peek_coi(Side.BUY)
        new_sell = pool.peek_coi(Side.SELL)
        assert new_buy is not None and new_sell is not None
        # Fresh COIs from the counter.
        assert new_buy != buy_coi
        assert new_sell != buy_coi
        # Signed with the new nonce.
        post_fire_nonces = [c.nonce for c in env.signer.calls[2:]]
        assert post_fire_nonces == [1001, 1001]
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_pop_for_fire_mismatch_returns_none() -> None:
    env = _Env(signer=_FakeSigner())
    pool = _make_pool(env)
    await pool.start()
    try:
        slot = pool.pop_for_fire(Side.BUY, "999999999")
        assert slot is None
        # Slots untouched.
        assert pool.peek_coi(Side.BUY) is not None
        assert pool.peek_coi(Side.SELL) is not None
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_pop_for_fire_non_int_client_id_returns_none() -> None:
    env = _Env(signer=_FakeSigner())
    pool = _make_pool(env)
    await pool.start()
    try:
        slot = pool.pop_for_fire(Side.BUY, "not-an-int")
        assert slot is None
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_drift_breach_triggers_refresh() -> None:
    env = _Env(signer=_FakeSigner())
    # Sign at mid=2000 (worst_buy = 2100, worst_sell = 1900). 50 bps
    # threshold so a small move trips it.
    pool = _make_pool(env, drift_threshold_bps=Decimal("50"))
    await pool.start()
    try:
        sign_count_before = len(env.signer.calls)
        # Push mid up so remaining BUY buffer (worst_buy - mid)/mid shrinks.
        # worst_buy = 2100; if mid = 2099, buffer = 1/2099 ≈ 4.8 bps < 50.
        env.quote = _make_quote("2099")
        env.nonce_value = 1001  # post-fire-style advance to confirm new nonce
        pool.on_quote(env.quote)
        # Wait for the scheduled drift task.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if len(env.signer.calls) > sign_count_before + 2:
                break
        assert len(env.signer.calls) >= sign_count_before + 2
        post_drift_nonces = [c.nonce for c in env.signer.calls[-2:]]
        assert post_drift_nonces == [1001, 1001]
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_drift_storm_only_one_refresh() -> None:
    env = _Env(signer=_FakeSigner())
    pool = _make_pool(env, drift_threshold_bps=Decimal("50"))
    await pool.start()
    try:
        sign_count_before = len(env.signer.calls)
        env.quote = _make_quote("2099")
        # Hammer on_quote 20 times in a row.
        for _ in range(20):
            pool.on_quote(env.quote)
        # Allow the single refresh to finish.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if len(env.signer.calls) > sign_count_before + 2:
                break
        # Exactly one extra refresh, not 20.
        assert len(env.signer.calls) == sign_count_before + 2
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_drift_within_threshold_no_refresh() -> None:
    env = _Env(signer=_FakeSigner())
    pool = _make_pool(env, drift_threshold_bps=Decimal("50"))
    await pool.start()
    try:
        sign_count_before = len(env.signer.calls)
        # Tiny move — buffer still ~500 bps, well above 50.
        env.quote = _make_quote("2001")
        pool.on_quote(env.quote)
        await asyncio.sleep(0.05)
        assert len(env.signer.calls) == sign_count_before
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_sign_error_leaves_slots_empty() -> None:
    """If sign fails during refresh, both slots stay None — partial pairs
    are unsafe (the unfired side would be stale on the next nonce)."""
    env = _Env(signer=_FakeSigner(sign_err="boom"))
    pool = _make_pool(env)
    await pool.start()
    try:
        assert pool.peek_coi(Side.BUY) is None
        assert pool.peek_coi(Side.SELL) is None
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_no_quote_yet_skips_refresh() -> None:
    env = _Env(signer=_FakeSigner(), quote=None)
    pool = _make_pool(env)
    await pool.start()
    try:
        assert pool.peek_coi(Side.BUY) is None
        assert pool.peek_coi(Side.SELL) is None
        assert env.signer.calls == []
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_stop_then_start_cycles_cleanly() -> None:
    env = _Env(signer=_FakeSigner())
    pool = _make_pool(env)
    await pool.start()
    assert pool.peek_coi(Side.BUY) is not None
    await pool.stop()
    assert pool.peek_coi(Side.BUY) is None
    # Re-start should re-sign.
    sign_count_before = len(env.signer.calls)
    await pool.start()
    try:
        assert pool.peek_coi(Side.BUY) is not None
        assert len(env.signer.calls) == sign_count_before + 2
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_generator_uses_pool_then_fallback() -> None:
    env = _Env(signer=_FakeSigner())
    pool = _make_pool(env)
    fallback = CounterClientIdGenerator(seed=5000)
    gen = PreSignedPoolClientIdGenerator(pool=pool, fallback=fallback)

    # Before start: pool is empty, generator falls through to counter.
    cid = gen.next(side=Side.BUY)
    assert cid == "5000"
    cid = gen.next(side=Side.SELL)
    assert cid == "5001"

    await pool.start()
    try:
        # After start: generator returns pool's staged COIs.
        buy_coi = pool.peek_coi(Side.BUY)
        sell_coi = pool.peek_coi(Side.SELL)
        assert buy_coi is not None and sell_coi is not None
        assert gen.next(side=Side.BUY) == str(buy_coi)
        assert gen.next(side=Side.SELL) == str(sell_coi)
        # peek doesn't consume, so repeated next() returns same coi.
        assert gen.next(side=Side.BUY) == str(buy_coi)
    finally:
        await pool.stop()
