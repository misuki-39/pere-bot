"""Pure decision math for taker-taker arbitrage.

Shared by the live `TakerTakerArbitrage` strategy (which composes EWMA, risk,
asyncio firing around this) and the backtest `TakerTakerBT` strategy. The
function never reads or mutates global state — caller supplies the EWMA bias
and the current synthetic position; this function only does math + builds a
`Decision`.

Convention: "left" = monitor_pair[0], "right" = monitor_pair[1]. In the
existing live setup that's aster=left, lighter=right; in the WTI capture it's
lighter=left, aster=right. The pure function does not care; it just labels.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import NamedTuple

from ..core.exec_record import Decision, Direction, Outcome
from ..core.types import OrderBook, Quote, Side
from ..utils.precision import BPS, vwap_fill
from .markout import MarkoutTable


class _Vwaps(NamedTuple):
    left_sell: Decimal
    left_buy: Decimal
    right_sell: Decimal
    right_buy: Decimal


@dataclass(frozen=True, slots=True)
class AssessParams:
    """Static decision parameters; build once per strategy instance.

    Optional tuning knobs (default-off = same behaviour as v0):
      markout: per-direction adverse-selection table, subtracted from raw
               edge before threshold check. `MarkoutTable.disabled()` is the
               no-op default. See `strategy/markout.py`.
      inventory_skew_bps: κ in the AS-style threshold widener. Per unit of
               |position|/max_qty, raise the entry threshold by κ bps when
               the trade GROWS |position|, lower it by κ bps when it FLATTENS.
               κ=0 = current binary max_qty gate only.
    """
    qty: Decimal
    max_levels: int
    fees_bps: Decimal
    min_profit_bps: Decimal
    max_slippage_bps: Decimal
    max_stale_ms: int
    max_qty: Decimal
    markout: MarkoutTable = MarkoutTable.disabled()
    inventory_skew_bps: Decimal = Decimal(0)


@dataclass(frozen=True, slots=True)
class AssessInputs:
    """Per-tick inputs. Caller is responsible for updating the EWMA model and
    passing the resulting `bias` + warm flag; we never touch EWMA state here.

    Optional same-direction throttle bumps (default 0 = throttle off). When the
    caller maintains a per-direction TimeEwma of "recently fired" bumps,
    `bump_a_bps` and `bump_b_bps` add directly to that direction's threshold;
    the pure function does not own the EWMA state.
    """
    now_ms: int
    left_book: OrderBook
    right_book: OrderBook
    left_quote: Quote
    right_quote: Quote
    bias: Decimal
    is_warm: bool
    position_left: Decimal
    position_right: Decimal
    bump_a_bps: Decimal = Decimal(0)
    bump_b_bps: Decimal = Decimal(0)


def left_side(direction: Direction) -> Side:
    """The side the left leg takes given the direction."""
    return Side.SELL if direction is Direction.A else Side.BUY


def right_side(direction: Direction) -> Side:
    """The side the right leg takes given the direction."""
    return Side.BUY if direction is Direction.A else Side.SELL


def assess_taker_taker(p: AssessParams, x: AssessInputs) -> Decision | None:
    """Returns a Decision (outcome terminal) or None when the tick isn't worth
    recording (warmup, or no positive edge). Pure; never raises for ordinary
    market states."""
    mid_left = x.left_quote.mid
    mid_right = x.right_quote.mid

    def new(
        outcome: Outcome, reason: str | None = None, *,
        bias: Decimal = Decimal(0), edge_bps: Decimal = Decimal(0),
        direction: Direction | None = None, vwaps: _Vwaps | None = None,
    ) -> Decision:
        v = vwaps or _Vwaps(Decimal(0), Decimal(0), Decimal(0), Decimal(0))
        return Decision(
            decision_id=f"d-{uuid.uuid4().hex[:10]}",
            ts_ms=x.now_ms,
            mid_left=mid_left, mid_right=mid_right,
            left_quote_ts_ms=x.left_quote.ts_ms,
            right_quote_ts_ms=x.right_quote.ts_ms,
            bias=bias,
            vwap_left_sell=v.left_sell, vwap_left_buy=v.left_buy,
            vwap_right_sell=v.right_sell, vwap_right_buy=v.right_buy,
            edge_bps=edge_bps, direction=direction,
            outcome=outcome, abort_reason=reason,
        )

    if (x.now_ms - max(x.left_quote.ts_ms, x.right_quote.ts_ms)) > p.max_stale_ms:
        return new(Outcome.ABORT_STALE, "quote older than max_stale_ms")

    if not x.is_warm:
        return None

    qty = p.qty
    vls, _ = vwap_fill(x.left_book.bids,  qty, max_levels=p.max_levels)
    vlb, _ = vwap_fill(x.left_book.asks,  qty, max_levels=p.max_levels)
    vrs, _ = vwap_fill(x.right_book.bids, qty, max_levels=p.max_levels)
    vrb, _ = vwap_fill(x.right_book.asks, qty, max_levels=p.max_levels)
    if vls is None or vlb is None or vrs is None or vrb is None:
        return new(Outcome.ABORT_NO_DEPTH,
                   "qty does not fill within max_levels", bias=x.bias)
    vwap_left_sell, vwap_left_buy, vwap_right_sell, vwap_right_buy = vls, vlb, vrs, vrb

    vw = _Vwaps(vwap_left_sell, vwap_left_buy, vwap_right_sell, vwap_right_buy)

    slip = p.max_slippage_bps / BPS
    if (abs((vwap_left_sell  - mid_left)  / mid_left)  > slip
            or abs((vwap_left_buy   - mid_left)  / mid_left)  > slip
            or abs((vwap_right_sell - mid_right) / mid_right) > slip
            or abs((vwap_right_buy  - mid_right) / mid_right) > slip):
        return new(Outcome.ABORT_SLIPPAGE, "vwap-mid exceeds max_slippage_bps",
                   bias=x.bias, vwaps=vw)

    ref_mid = (mid_left + mid_right) / Decimal(2)

    # Raw bias-adjusted edge in PRICE units (positive = arb in that direction).
    raw_edge_A = (vwap_left_sell  - vwap_right_buy) - x.bias
    raw_edge_B = (vwap_right_sell - vwap_left_buy)  + x.bias

    # Threshold contributors. All in bps; converted to price units below.
    fee_bps = p.fees_bps + p.min_profit_bps
    raw_edge_A_bps = raw_edge_A / ref_mid * BPS
    raw_edge_B_bps = raw_edge_B / ref_mid * BPS
    markout_A_bps = p.markout.markout_bps(direction_a=True,  raw_edge_bps=raw_edge_A_bps)
    markout_B_bps = p.markout.markout_bps(direction_a=False, raw_edge_bps=raw_edge_B_bps)
    skew_A_bps = _inventory_skew_bps(
        p.inventory_skew_bps, x.position_left, left_side(Direction.A).sign, p.max_qty)
    skew_B_bps = _inventory_skew_bps(
        p.inventory_skew_bps, x.position_left, left_side(Direction.B).sign, p.max_qty)

    total_thresh_A_bps = fee_bps + markout_A_bps + skew_A_bps + x.bump_a_bps
    total_thresh_B_bps = fee_bps + markout_B_bps + skew_B_bps + x.bump_b_bps

    threshold_A = ref_mid * total_thresh_A_bps / BPS
    threshold_B = ref_mid * total_thresh_B_bps / BPS
    edge_A = raw_edge_A - threshold_A
    edge_B = raw_edge_B - threshold_B

    if edge_A <= 0 and edge_B <= 0:
        return None

    direction = Direction.A if edge_A >= edge_B else Direction.B
    edge_bps = max(edge_A, edge_B) / ref_mid * BPS

    post_left  = x.position_left  + qty * Decimal(left_side(direction).sign)
    post_right = x.position_right + qty * Decimal(right_side(direction).sign)
    if max(abs(post_left), abs(post_right)) > p.max_qty:
        return new(Outcome.BLOCKED_RISK,
                   f"post-trade abs position {max(abs(post_left), abs(post_right))} > max_qty {p.max_qty}",
                   bias=x.bias, edge_bps=edge_bps, direction=direction, vwaps=vw)

    # NOTE: caller marks Phase.DECISION — live uses `mark()` (mono_ms), backtest
    # uses `mark_at(snap.ts_ms)`. Keeping it out of the pure fn avoids clock-source
    # coupling.
    return new(Outcome.FIRED, bias=x.bias, edge_bps=edge_bps,
               direction=direction, vwaps=vw)


def _inventory_skew_bps(
    kappa_bps: Decimal,
    position_left: Decimal,
    delta_sign: int,
    max_qty: Decimal,
) -> Decimal:
    """Avellaneda-Stoikov-shape inventory skew.

    Returns a bps shift to ADD to the entry threshold (positive = harder to
    fire, negative = easier). The shift is proportional to current
    |position|/max_qty and signed by whether the trade grows or shrinks
    |position|:

      skew = kappa_bps * (position_left * delta_sign) / max_qty

    With `delta_sign = left_side(direction).sign` (sell=−1, buy=+1):
      - If `position_left` and `delta_sign` AGREE in sign  → growing |pos|
        → positive skew (raise threshold; require stronger edge to add more).
      - If they DISAGREE                                   → shrinking |pos|
        → negative skew (lower threshold; reward flattening).
      - At `|position_left| = max_qty`, |skew| = kappa_bps (full strength).

    κ=0 disables the skew (same as binary max_qty gate downstream).
    """
    if kappa_bps == 0 or max_qty == 0:
        return Decimal(0)
    return kappa_bps * (position_left * Decimal(delta_sign)) / max_qty
