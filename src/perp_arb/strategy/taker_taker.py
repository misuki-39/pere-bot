"""TakerTakerArbitrage — depth-aware VWAP edge vs EWMA bias on mid-mid.

Generic two-venue strategy: refers to its two venues only as `leg_a` /
`leg_b`. The actual venues are bound at the config level (PairCfg) and
the factory wires the `exchanges` / `markets` dicts under those leg
labels. This file does not name any specific venue.

Layering: this module is the *signal* layer. It produces a `Decision`
per tick (via `assess_taker_taker`) and hands fired decisions to
`TwoLegExecutor`, which owns cid generation, the two-leg gather, WS
fill tracking, partial-failure unwind, and paper-mode synth. The
strategy only feeds `TradeReport` back into position / risk /
throttle / heartbeat state — it never sees place acks or WS fills.

Modes (passed through to the executor at construction time):
  * `paper` — synthetic fills from current book VWAP; no venue submits.
  * `live`  — real submits on both venues, gathered concurrently.

What it bets on:
  An EWMA of (mid_leg_a - mid_leg_b) tracks the long-run inter-venue
  *bias*. The bias itself can be structurally non-zero (different oracles,
  funding rates, USDT-vs-USDC quote currencies). What matters is that the
  current spread oscillates around the bias. The strategy fires when the
  current depth-aware spread deviates from the EWMA bias by more than
  `fees_bps + min_profit_bps`, betting on reversion to bias.

Exit policy is implicit: a reverse-direction entry IS the exit. Direction B
fires with the same `qty` exactly cancel a direction-A position (and vice
versa). The risk manager caps absolute exposure via `max_qty`, so positions
stack up to the cap and unwind naturally when the spread reverts toward bias.

Failure modes (when max-qty position gets stuck):
  * Spread *trends* away from bias without reverting (regime change faster
    than the EWMA can chase) → one-direction entries keep stacking.
  * Spread variance is too low — never exceeds threshold → no signal.
  * `bias_halflife_s` mistuned: too short eats the ~2 s reversion signal
    (center chases its own residual to zero); too long lags the slow
    intraday center wander.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from ..core.config import RunMode
from ..core.exec_record import (
    Decision,
    Direction,
    ExecutionRecorder,
    Outcome,
    Phase,
)
from ..core.executor import LegIntent, TwoLegExecutor
from ..risk.manager import RiskManager
from ..utils.time import now_ms
from .base import BaseStrategy, SpreadModel, TimeEwma
from .markout import MarkoutTable
from .taker_taker_core import (
    AssessInputs,
    AssessParams,
    assess_taker_taker,
    left_side,
    right_side,
)

_log = logging.getLogger(__name__)


@dataclass
class SyntheticPosition:
    """Bot-side position tracker (signed). For paper + live both."""

    leg_a: Decimal = Decimal("0")
    leg_b: Decimal = Decimal("0")
    realised_pnl: Decimal = Decimal("0")


class TakerTakerArbitrage(BaseStrategy):
    name = "taker_taker"

    def __init__(self, cfg, exchanges, markets) -> None:
        super().__init__(cfg, exchanges, markets)
        s = cfg.strategy
        self._spread = SpreadModel(
            center_half_life_s=s.bias_halflife_s,
            scale_half_life_s=s.scale_halflife_s,
            warmup_s=s.warmup_seconds,
        )
        self._recorder: ExecutionRecorder | None = None
        self._evaluating = False
        self._position = SyntheticPosition()
        self._risk = RiskManager(s.risk, max_qty=s.max_qty)
        self._last_heartbeat_ms = 0
        self._heartbeat_interval_ms = 60_000  # liveness only; trades go to CSV
        self._risk_blocked = False
        # parameters consumed by the pure decision function.
        # Wave-1 optimisations (markout, throttle, cap) come from
        # `s.optimisations` — defaults are all "off" so legacy configs that
        # omit the block keep their pre-Wave-1 behaviour.
        opt = s.optimisations
        markout = (
            MarkoutTable.from_json(opt.markout_table_path)
            if opt.markout_table_path is not None
            else MarkoutTable.disabled()
        )
        self._params = AssessParams(
            qty=Decimal(str(s.qty)),
            max_levels=s.max_levels,
            fees_bps=Decimal(str(s.fees_bps)),
            min_profit_bps=Decimal(str(s.min_profit_bps)),
            max_slippage_bps=Decimal(str(s.max_slippage_bps)),
            max_stale_ms=s.max_stale_ms,
            max_qty=Decimal(str(s.max_qty)),
            markout=markout,
            # inventory_skew_bps intentionally left at default 0; not
            # exposed in OptimisationsCfg until rolling markout calibration
            # is stable.
        )

        # Same-direction throttle: each FILLED on direction X bumps that
        # direction's threshold by `throttle_bump_bps`; the bump decays
        # back toward 0 with `throttle_halflife_s` half-life. Δ=0 disables.
        self._throttle_enabled = opt.throttle_bump_bps > 0
        self._throttle_bump_bps = Decimal(str(opt.throttle_bump_bps))
        self._bump_a = TimeEwma(half_life_s=opt.throttle_halflife_s)
        self._bump_b = TimeEwma(half_life_s=opt.throttle_halflife_s)

        # Per-direction in-flight cap. K=0 disables. K=1 = at most one
        # outstanding entry of that direction at a time.
        # NOTE: live's `_evaluating` gate in `_schedule_eval` already
        # serializes evaluation, so `_inflight_dir` is in practice always
        # empty by the time `_assess` runs. The cap is wired for parity
        # with the backtest path and as a forward-safety belt if the gate
        # is ever relaxed.
        self._inflight_cap = int(opt.in_flight_cap_per_direction)
        self._inflight_dir: dict[str, Direction] = {}

        # Execution is delegated: two-leg gather, WS fill tracking,
        # partial-failure unwind, paper synth all live in the executor +
        # driver layers. Strategy resolves Direction→sides itself and
        # hands the executor concrete LegIntents — the executor never
        # sees strategy-internal concepts.
        self._executor = TwoLegExecutor(
            exchanges, markets,
            is_paper=(s.mode is RunMode.PAPER),
            max_levels=s.max_levels,
        )

        if markout.direction_A or markout.direction_B:
            _log.info(
                "markout enabled: %s",
                markout.latency_label or "(unlabelled)",
            )
        if self._throttle_enabled:
            _log.info(
                "same-side throttle enabled: bump=%s bps halflife=%ss",
                self._throttle_bump_bps, opt.throttle_halflife_s,
            )
        if self._inflight_cap > 0:
            _log.info(
                "in-flight cap enabled K=%d (note: structural no-op in "
                "live due to _evaluating serialization)",
                self._inflight_cap,
            )

    async def run(self) -> None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self._recorder = ExecutionRecorder(self.cfg.runtime.log_dir, ts)
        _log.info("taker_taker mode=%s recording to %s",
                  self.cfg.strategy.mode, self.cfg.runtime.log_dir)

        self._leg_a().subscribe_book(self._leg_a_market(), lambda _b: self._schedule_eval())
        self._leg_b().subscribe_book(self._leg_b_market(), lambda _b: self._schedule_eval())

        try:
            await self._stop.wait()
        finally:
            if self._recorder:
                self._recorder.close()

    # ---- evaluation ----

    def _schedule_eval(self) -> None:
        if self._evaluating:
            return
        self._evaluating = True
        asyncio.create_task(self._evaluate_once(), name="taker_taker-eval")

    async def _evaluate_once(self) -> None:
        try:
            await self._evaluate()
        finally:
            self._evaluating = False

    async def _evaluate(self) -> None:
        a_book = self._leg_a().order_book(self._leg_a_market())
        b_book = self._leg_b().order_book(self._leg_b_market())
        a_q = self._leg_a().best_quote(self._leg_a_market())
        b_q = self._leg_b().best_quote(self._leg_b_market())
        if a_book is None or b_book is None or a_q is None or b_q is None:
            return

        d = self._assess(a_book, b_book, a_q, b_q)
        if d is None:
            return
        try:
            if d.outcome is Outcome.FIRED:
                await self._fire(d)
        finally:
            if self._recorder:
                self._recorder.emit(d)

    def _assess(self, a_book, b_book, a_q, b_q) -> Decision | None:
        """Thin wrapper around the pure `assess_taker_taker` decision math.

        Live-only responsibilities kept here: update the EWMA, run the
        operational `RiskManager` gates (halted / consecutive-failures / daily
        PnL — the pure function only checks the structural `max_qty` cap), and
        emit the throttled heartbeat log."""
        now = now_ms()
        bias = self._spread.update(a_q.mid - b_q.mid, now).center

        # Same-side throttle bumps: decay to "now" before reading. Calling
        # `update` with the current value advances the internal timestamp
        # without altering the value, so a follow-up `update(0, now+dt)` in
        # a later tick decays correctly. We pass 0 here as the new sample
        # to push the EWMA toward 0 over the halflife.
        bump_a = Decimal(0)
        bump_b = Decimal(0)
        if self._throttle_enabled and self._bump_a.value is not None:
            bump_a = self._bump_a.update(Decimal(0), now)
            bump_b = self._bump_b.update(Decimal(0), now)

        d = assess_taker_taker(self._params, AssessInputs(
            now_ms=now,
            left_book=a_book, right_book=b_book,
            left_quote=a_q, right_quote=b_q,
            bias=bias, is_warm=self._spread.is_warm,
            position_left=self._position.leg_a,
            position_right=self._position.leg_b,
            bump_a_bps=bump_a,
            bump_b_bps=bump_b,
        ))

        if d is not None and d.outcome is Outcome.FIRED:
            assert d.direction is not None

            # In-flight cap: block if K same-direction entries are already
            # pending. Live's `_evaluating` gate already serializes
            # evaluation so this is in practice unreachable, but it
            # mirrors the backtest contract and protects against future
            # async relaxation. Records BLOCKED_RISK with a distinct
            # `abort_reason` so the cap can be distinguished from
            # RiskManager blocks in logs.
            if self._inflight_cap > 0:
                same_dir = sum(
                    1 for x in self._inflight_dir.values() if x is d.direction
                )
                if same_dir >= self._inflight_cap:
                    d.outcome = Outcome.BLOCKED_RISK
                    d.abort_reason = (
                        f"in-flight cap {self._inflight_cap} reached for "
                        f"direction {d.direction.value}"
                    )

        # The RiskManager gate runs on the (possibly already-blocked-by-cap)
        # decision. If the cap already rewrote to BLOCKED_RISK we still let
        # RiskManager observe the FIRED-intent state via the position math —
        # but we must not overwrite the cap's reason. Guard with the outcome.
        if d is not None and d.outcome is Outcome.FIRED:
            d.timeline.mark(Phase.DECISION)
            qty = self.cfg.strategy.qty
            post_a = self._position.leg_a + qty * Decimal(left_side(d.direction).sign)
            post_b = self._position.leg_b + qty * Decimal(right_side(d.direction).sign)
            ok, reason = self._risk.can_trade(
                post_trade_abs_position=max(abs(post_a), abs(post_b)),
            )
            if not ok:
                if not self._risk_blocked:
                    self._risk_blocked = True
                    _log.info("entry blocked by risk: %s (suppressing until cleared)", reason)
                d.outcome = Outcome.BLOCKED_RISK
                d.abort_reason = reason
            elif self._risk_blocked:
                self._risk_blocked = False
                _log.info("risk block cleared — resuming entries")

        self._maybe_heartbeat(now, a_q.mid, b_q.mid, bias, d)
        return d

    def _maybe_heartbeat(
        self,
        now: int,
        mid_left: Decimal, mid_right: Decimal, bias: Decimal,
        d: Decision | None,
    ) -> None:
        """Throttled INFO log so paper runs are observable when nothing fires.

        When `d` has VWAPs (any post-warmup non-stale tick), log both raw
        edge directions in bps — operators rely on seeing whether the *other*
        direction is approaching threshold even when this one isn't firing.
        """
        if now - self._last_heartbeat_ms < self._heartbeat_interval_ms:
            return
        self._last_heartbeat_ms = now
        if d is not None and d.vwap_left_sell > 0:
            ref_mid = (mid_left + mid_right) / Decimal(2)
            edge_A = (d.vwap_left_sell  - d.vwap_right_buy) - bias
            edge_B = (d.vwap_right_sell - d.vwap_left_buy)  + bias
            edge_str = (f"edge_A={edge_A / ref_mid * Decimal(10_000):+.2f}bps "
                        f"edge_B={edge_B / ref_mid * Decimal(10_000):+.2f}bps")
        else:
            edge_str = "no-edge"
        _log.info(
            "tick: mid_a=%.2f mid_b=%.2f bias=%.4f %s pos_a=%s pos_b=%s",
            mid_left, mid_right, bias, edge_str,
            self._position.leg_a, self._position.leg_b,
        )

    # ---- order firing ----

    async def _fire(self, d: Decision) -> None:
        """Resolve Direction → venue-side intents, delegate execution,
        then apply the TradeReport to position / risk / throttle.

        Strategy owns the Direction→Side resolution because Direction is
        a strategy concept; the executor only sees concrete `Side`s.
        Strategy also owns position / risk / throttle updates because
        they feed back into the next `_assess`."""
        assert d.direction is not None
        qty = self.cfg.strategy.qty
        a_side, b_side = left_side(d.direction), right_side(d.direction)
        if d.direction is Direction.A:
            a_exp, b_exp = d.vwap_left_sell, d.vwap_right_buy
        else:
            a_exp, b_exp = d.vwap_left_buy, d.vwap_right_sell

        _log.info("[%s] FIRE qty=%s leg_a=%s leg_b=%s mode=%s",
                  d.decision_id, qty, a_side, b_side, self.cfg.strategy.mode)

        if self._inflight_cap > 0:
            self._inflight_dir[d.decision_id] = d.direction
        try:
            report = await self._executor.execute(
                trade_id=d.decision_id,
                legs=(
                    LegIntent(venue="leg_a", side=a_side, expected_price=a_exp),
                    LegIntent(venue="leg_b", side=b_side, expected_price=b_exp),
                ),
                qty=qty,
                timeline=d.timeline,
            )
            d.legs = report.legs
            d.send_ts_ms = report.send_ts_ms

            if report.success:
                self._position.leg_a += qty * Decimal(a_side.sign)
                self._position.leg_b += qty * Decimal(b_side.sign)
                self._risk.record_success(leg_latency_ms=report.latency_ms or 0)

                # Seed throttle bump only on confirmed FILLED — a partial /
                # both-fail consumed no edge.
                if self._throttle_enabled:
                    target = self._bump_a if d.direction is Direction.A else self._bump_b
                    target.bump(self._throttle_bump_bps, now_ms())

                _log.info(
                    "[%s] pos_a=%s pos_b=%s",
                    d.decision_id,
                    self._position.leg_a, self._position.leg_b,
                )
            else:
                self._risk.record_failure(report.failure_reason or "unknown")
        finally:
            if self._inflight_cap > 0:
                self._inflight_dir.pop(d.decision_id, None)
