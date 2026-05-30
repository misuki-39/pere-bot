"""Pure decision core for the spread-reversion arbitrage strategy.

This is more than a bare signal: it owns both the *edge signal* (given
depth-aware fill prices and the EWMA bias, is the bias-adjusted spread beyond
the entry threshold?) AND the *inventory-management policy* layered on top —
the Avellaneda-Stoikov threshold skew and the position cap. Those are strategy
decisions, not pure signal; they live here so the live and backtest strategies
share one implementation and stay identical by construction.

It is generic to a spread-reversion bet — it never computes fills and so makes
no assumption about execution style; the caller supplies the fill prices (see
`taker_fill_model.compute_taker_fills`).

Shared by the live `TakerTakerArbitrage` strategy (which composes EWMA, the
operational `RiskManager`, asyncio firing around this) and the backtest
`TakerTakerBT` strategy. The function never reads or mutates global state —
caller supplies the EWMA bias, the fill prices, and the current position; this
function only does math + builds a `Decision`.

Convention: "left" = monitor_pair[0], "right" = monitor_pair[1]. In the
existing live setup that's aster=left, lighter=right; in the WTI capture it's
lighter=left, aster=right. The pure function does not care; it just labels.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal

from ..core.recording.decision import Decision, Direction, Verdict
from ..core.types import Quote, Side
from ..utils.precision import BPS
from .taker_fill_model import FillAbort, TakerFills


@dataclass(frozen=True, slots=True)
class AssessParams:
    """Static decision parameters; build once per strategy instance.

    Optional tuning knobs (default-off = same behaviour as v0):
      inventory_skew_bps: κ in the AS-style threshold widener. Per unit of
               |position|/max_qty, raise the entry threshold by κ bps when
               the trade GROWS |position|, lower it by κ bps when it FLATTENS.
               κ=0 = current binary max_qty gate only.
      inventory_skew_close_bps: optional separate κ for the FLATTEN side. None
               (default) = symmetric (use `inventory_skew_bps` for both sides).
               Set to 0 to disable exit-easing while keeping entry-tightening.
    """
    qty: Decimal
    fees_bps: Decimal
    min_profit_bps: Decimal
    max_stale_ms: int
    max_qty: Decimal
    inventory_skew_bps: Decimal = Decimal(0)
    inventory_skew_close_bps: Decimal | None = None


@dataclass(frozen=True, slots=True)
class AssessInputs:
    """Per-tick inputs. Caller is responsible for updating the EWMA model and
    passing the resulting `bias` + warm flag; we never touch EWMA state here.
    Caller is likewise responsible for pricing the fills (`compute_taker_fills`)
    and passing the result — a `TakerFills` or a `FillAbort`.

    Optional same-direction throttle bumps (default 0 = throttle off). When the
    caller maintains a per-direction TimeEwma of "recently fired" bumps,
    `bump_a_bps` and `bump_b_bps` add directly to that direction's threshold;
    the pure function does not own the EWMA state.

    `position` is the single offsetting position, keyed off the left leg: the
    two legs are equal-magnitude / opposite-sign by construction, so the right
    leg's position is just `-position` and need not be passed separately.
    """
    now_ms: int
    left_quote: Quote
    right_quote: Quote
    fills: TakerFills | FillAbort
    bias: Decimal
    is_warm: bool
    position: Decimal
    bump_a_bps: Decimal = Decimal(0)
    bump_b_bps: Decimal = Decimal(0)


def left_side(direction: Direction) -> Side:
    """The side the left leg takes given the direction (execution layer)."""
    return Side.SELL if direction.sign < 0 else Side.BUY


def right_side(direction: Direction) -> Side:
    """Right leg always hedges the left (offsetting taker-taker pair)."""
    return left_side(direction).opposite


def assess_reversion(p: AssessParams, x: AssessInputs) -> Decision | None:
    """Evaluate one tick. Returns a `Decision` when the tick is worth surfacing
    — a fire, or a noteworthy abort (stale quote / no depth) — or `None` for an
    ordinary non-event (pre-warmup, no positive edge, or the position cap doing
    its job). Pure; never raises for ordinary market states."""
    mid_left = x.left_quote.mid
    mid_right = x.right_quote.mid

    def new(
        outcome: Verdict, reason: str | None = None, *,
        bias: Decimal = Decimal(0), edge_bps: Decimal = Decimal(0),
        direction: Direction | None = None, vwaps: TakerFills | None = None,
    ) -> Decision:
        v = vwaps or TakerFills(Decimal(0), Decimal(0), Decimal(0), Decimal(0))
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
        return new(Verdict.ABORT_STALE, "quote older than max_stale_ms")

    if not x.is_warm:
        return None

    # The caller-supplied taker fill model may have declined to price the
    # tick (book too thin for qty within max_levels). Surface as ABORT_NO_DEPTH
    # — done after the stale / warmup gates so outcome precedence is unchanged.
    if isinstance(x.fills, FillAbort):
        return new(Verdict.ABORT_NO_DEPTH,
                   "qty does not fill within max_levels", bias=x.bias)
    vw = x.fills
    vwap_left_sell, vwap_left_buy = vw.left_sell, vw.left_buy
    vwap_right_sell, vwap_right_buy = vw.right_sell, vw.right_buy

    ref_mid = (mid_left + mid_right) / Decimal(2)

    # Raw bias-adjusted edge in PRICE units (positive = arb in that direction).
    raw_edge_A = (vwap_left_sell  - vwap_right_buy) - x.bias
    raw_edge_B = (vwap_right_sell - vwap_left_buy)  + x.bias

    # Threshold contributors. All in bps; converted to price units below.
    fee_bps = p.fees_bps + p.min_profit_bps
    kappa_close = (p.inventory_skew_close_bps
                   if p.inventory_skew_close_bps is not None
                   else p.inventory_skew_bps)
    skew_A_bps = _inventory_skew_bps(
        p.inventory_skew_bps, kappa_close, x.position,
        Direction.A.sign, p.max_qty)
    skew_B_bps = _inventory_skew_bps(
        p.inventory_skew_bps, kappa_close, x.position,
        Direction.B.sign, p.max_qty)

    total_thresh_A_bps = fee_bps + skew_A_bps + x.bump_a_bps
    total_thresh_B_bps = fee_bps + skew_B_bps + x.bump_b_bps

    threshold_A = ref_mid * total_thresh_A_bps / BPS
    threshold_B = ref_mid * total_thresh_B_bps / BPS
    edge_A = raw_edge_A - threshold_A
    edge_B = raw_edge_B - threshold_B

    if edge_A <= 0 and edge_B <= 0:
        return None

    direction = Direction.A if edge_A >= edge_B else Direction.B
    edge_bps = max(edge_A, edge_B) / ref_mid * BPS

    if _exceeds_position_cap(x.position, direction, p.qty, p.max_qty):
        # Cap doing its job — drop the tick (don't flood the log with identical
        # "over cap" rows while we wait for a reverse-direction signal). Not a
        # risk event; reverse/flatten fires pass the same check and proceed.
        return None

    # NOTE: caller marks Phase.DECISION — live uses `mark()` (mono_ms), backtest
    # uses `mark_at(snap.ts_ms)`. Keeping it out of the pure fn avoids clock-source
    # coupling.
    return new(Verdict.FIRED, bias=x.bias, edge_bps=edge_bps,
               direction=direction, vwaps=vw)


def _exceeds_position_cap(
    position: Decimal, direction: Direction, qty: Decimal, max_qty: Decimal,
) -> bool:
    """Would firing `direction` push |position| past the cap? Post-trade-aware:
    reverse/flatten trades shrink |pos| and pass even when already at the cap.
    The cap is strategy policy (a position limit), not a risk event."""
    post = position + qty * Decimal(direction.sign)
    return abs(post) > max_qty


def _inventory_skew_bps(
    kappa_open_bps: Decimal,
    kappa_close_bps: Decimal,
    position: Decimal,
    delta_sign: int,
    max_qty: Decimal,
) -> Decimal:
    """Avellaneda-Stoikov-shape inventory skew.

    Returns a bps shift to ADD to the entry threshold (positive = harder to
    fire, negative = easier). The shift is proportional to current
    |position|/max_qty and signed by whether the trade grows or shrinks
    |position|:

      growing = position * delta_sign / max_qty
      skew    = kappa_open  * growing   if growing > 0   (raise threshold)
              = kappa_close * growing   if growing < 0   (lower threshold; growing<0)

    With `delta_sign = direction.sign` (sell=−1, buy=+1):
      - If `position` and `delta_sign` AGREE in sign  → growing |pos|
        → positive skew (raise threshold; require stronger edge to add more).
      - If they DISAGREE                               → shrinking |pos|
        → negative skew (lower threshold; reward flattening).
      - At `|position| = max_qty`, |skew| = the relevant κ (full strength).

    Asymmetric usage: pass kappa_close_bps=0 to disable exit-easing while
    keeping entry-tightening (the "raise threshold for adding" intuition).
    Symmetric is recovered by kappa_close_bps == kappa_open_bps.
    """
    if max_qty == 0:
        return Decimal(0)
    growing = position * Decimal(delta_sign) / max_qty
    if growing > 0:
        return kappa_open_bps * growing
    if growing < 0:
        return kappa_close_bps * growing
    return Decimal(0)
