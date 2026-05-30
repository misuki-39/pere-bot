"""TakerTakerArbitrage — depth-aware VWAP edge vs EWMA bias on mid-mid.

Generic two-venue strategy: refers to its two venues only as `leg_a` /
`leg_b`. The actual venues are bound at the config level (PairCfg) and
the factory wires the `exchanges` / `markets` dicts under those leg
labels. This file does not name any specific venue.

Layering: this module is the *signal* layer. It produces a `Decision`
per tick (via `compute_taker_fills` + `assess_reversion`) and hands fired
decisions to
`TwoLegExecutor`, which owns cid generation, the two-leg gather, WS
fill tracking, partial-failure unwind, and paper-mode synth. The
strategy only feeds `ExecutionResult` back into position / risk /
heartbeat state — it never sees place acks or WS fills.

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

from ..core.executor import ExecutionResult, LegIntent, TwoLegExecutor
from ..core.pnl import pair_pnl_from_legs
from ..core.recording.decision import (
    Decision,
    Direction,
    Phase,
    Verdict,
)
from ..core.recording.sqlite_recorder import SqliteRecorder
from ..core.types import LegKind, LegOutcome, MarketInfo, Quote, Side
from ..risk.manager import RiskManager
from ..utils.time import now_ms
from .base import BaseStrategy, SpreadModel
from .persistence_gate import PersistenceGate, PersistenceParams
from .reversion_signal import (
    AssessInputs,
    AssessParams,
    assess_reversion,
    left_side,
    right_side,
)
from .taker_fill_model import TakerFillParams, compute_taker_fills

_log = logging.getLogger(__name__)


@dataclass
class SyntheticPosition:
    """Realised cash-flow PnL accumulator.

    Per-venue position is read from `exchange.live_position()` (WS-fed
    snapshot of venue truth) plus a small `_pending_*` overlay that
    bridges the gap between order-ack-success and the ACCOUNT_UPDATE
    landing. WS is the single source of truth for size; this class only
    carries cumulative realised PnL.
    """

    realised_pnl: Decimal = Decimal("0")


class TakerTakerArbitrage(BaseStrategy):
    name = "taker_taker"

    def __init__(self, cfg, exchanges, markets, session) -> None:
        super().__init__(cfg, exchanges, markets, session)
        s = cfg.strategy
        self._spread = SpreadModel(
            center_half_life_s=s.bias_halflife_s,
            scale_half_life_s=s.scale_halflife_s,
            warmup_s=s.warmup_seconds,
        )
        self._recorder: SqliteRecorder | None = None
        self._evaluating = False
        self._position = SyntheticPosition()
        # Provisional overlay: predicted signed delta NOT YET pushed by
        # the WS account stream. Set on a successful `_fire` as
        # `delta - (post - pre)` so any portion already absorbed during
        # the executor await (e.g. ACCOUNT_UPDATE landing during the
        # aster 400/timeout queryOrder recovery roundtrip) is NOT
        # double-counted on top of `live_position()`. Cleared on the
        # next venue position event (subscribed in `run()`). The
        # `_evaluating` gate ensures only one trade is in flight at a
        # time, so each overlay slot only ever carries one trade's worth.
        # Live mode only — paper has no WS account stream.
        self._pending_a: Decimal = Decimal("0")
        self._pending_b: Decimal = Decimal("0")
        # Paper mode synthetic ledger. Paper has no WS account stream and
        # no real venue inventory, so the strategy keeps its own
        # accumulator off successful paper fills. Unused in live.
        self._paper_pos_a: Decimal = Decimal("0")
        self._paper_pos_b: Decimal = Decimal("0")
        # Reconcile-after-failure state. When `_fire` fails the strategy
        # runs sync→balance→cooldown (see plan agile-waddling-beacon). If
        # the sync or the balance step itself fails, `_reconcile_pending`
        # is set True and `_reconcile_target` carries the pre-fire snapshot
        # so the next `_evaluate` (after cooldown expiry) can retry
        # before any new entry is allowed.
        self._reconcile_pending: bool = False
        self._reconcile_target: tuple[Decimal, Decimal] | None = None
        self._risk = RiskManager(s.risk)
        self._last_heartbeat_ms = 0
        self._heartbeat_interval_ms = 300_000  # 5 min; liveness only, trades go to the recorder
        self._risk_blocked = False
        # parameters consumed by the pure decision function.
        # Wave-1 optimisations (cap) come from `s.optimisations` — defaults
        # are all "off" so legacy configs that omit the block keep their
        # pre-Wave-1 behaviour.
        opt = s.optimisations
        self._fill_params = TakerFillParams(
            qty=Decimal(str(s.qty)),
            max_levels=s.max_levels,
        )
        self._params = AssessParams(
            qty=Decimal(str(s.qty)),
            fees_bps=Decimal(str(s.fees_bps)),
            min_profit_bps=Decimal(str(s.min_profit_bps)),
            max_stale_ms=s.max_stale_ms,
            max_qty=Decimal(str(s.max_qty)),
            inventory_skew_bps=Decimal(str(opt.inventory_skew_bps)),
            inventory_skew_close_bps=(
                Decimal(str(opt.inventory_skew_close_bps))
                if opt.inventory_skew_close_bps is not None
                else None
            ),
        )

        # Per-direction in-flight cap. K=0 disables. K=1 = at most one
        # outstanding entry of that direction at a time.
        # NOTE: live's `_evaluating` gate in `_schedule_eval` already
        # serializes evaluation, so `_inflight_dir` is in practice always
        # empty by the time `_assess` runs. The cap is wired for parity
        # with the backtest path and as a forward-safety belt if the gate
        # is ever relaxed.
        self._inflight_cap = int(opt.in_flight_cap_per_direction)
        self._inflight_dir: dict[str, Direction] = {}

        # Edge-persistence confirmation gate (search 2026-05-22, "Strategy 3").
        # A temporal filter between signal and execution: a FIRED decision is
        # suppressed until its edge has survived `t_confirm_ms` across
        # `n_confirm` venue updates with no adverse mid-drift. Disabled by
        # default → identity pass-through.
        pc = opt.persistence_confirm
        self._gate = PersistenceGate(PersistenceParams(
            enabled=pc.enabled,
            t_confirm_ms=pc.t_confirm_ms,
            n_confirm=pc.n_confirm,
            drift_max_bps=Decimal(str(pc.drift_max_bps)),
        ))
        # Per-venue last-seen quote ts, for the gate's per-tick update count.
        self._prev_a_ts: int | None = None
        self._prev_b_ts: int | None = None

        # Execution is delegated: two-leg gather, WS fill tracking,
        # partial-failure unwind, paper synth all live in the executor +
        # driver layers. Strategy resolves Direction→sides itself and
        # hands the executor concrete LegIntents — the executor never
        # sees strategy-internal concepts.
        self._executor = TwoLegExecutor(
            exchanges, markets,
            is_paper=session.is_paper,
            max_levels=s.max_levels,
        )

        if self._inflight_cap > 0:
            _log.info(
                "in-flight cap enabled K=%d (note: structural no-op in "
                "live due to _evaluating serialization)",
                self._inflight_cap,
            )
        if pc.enabled:
            _log.info(
                "persistence-confirm enabled: t_confirm=%dms n_confirm=%d "
                "drift_max=%s bps",
                pc.t_confirm_ms, pc.n_confirm, pc.drift_max_bps,
            )
        if opt.inventory_skew_bps > 0 or (
            opt.inventory_skew_close_bps is not None
            and opt.inventory_skew_close_bps > 0
        ):
            close_repr = (
                f"{opt.inventory_skew_close_bps}"
                if opt.inventory_skew_close_bps is not None
                else f"{opt.inventory_skew_bps} (symmetric)"
            )
            _log.info(
                "inventory skew enabled: κ_open=%s bps κ_close=%s bps",
                opt.inventory_skew_bps, close_repr,
            )

    async def run(self) -> None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        mode_label = "paper" if self.session.is_paper else "live"
        self._recorder = SqliteRecorder(
            ts,
            strategy_id="taker_taker",
            mode=mode_label,
            config_json=self.cfg.strategy.model_dump_json(),
            turso=self.cfg.runtime.turso,
        )
        await self._recorder.start()
        _log.info("taker_taker mode=%s recording to %s",
                  mode_label, self.cfg.runtime.turso.db_path)

        # Seed venue position truth before any FIRE evaluates. After a
        # restart with real inventory, this is the only way max_qty stays
        # meaningful on the first tick (before WS pushes an ACCOUNT_UPDATE).
        # `snapshot_position` calls `get_position`, which seeds the driver's
        # `live_position()` cache via setdefault — so subsequent reads of
        # `live_position()` return this value until WS overwrites it.
        # Paper returns 0; the paper synthetic ledger picks up from there.
        seed_a, seed_b = await asyncio.gather(
            self.session.snapshot_position(self._leg_a(), self._leg_a_market()),
            self.session.snapshot_position(self._leg_b(), self._leg_b_market()),
        )
        self._paper_pos_a, self._paper_pos_b = seed_a, seed_b
        _log.info("seeded position: leg_a=%s leg_b=%s", seed_a, seed_b)

        # Subscribe to venue position pushes so the overlay clears the
        # moment WS catches up to a fired trade. The Position callback's
        # payload is irrelevant — its arrival is the signal. We clear both
        # eagerly; over-clearing is safe because only one trade is in
        # flight at a time (gated by `_evaluating`).
        if not self.session.is_paper:
            self._leg_a().subscribe_positions(
                self._leg_a_market(), lambda _p: self._clear_pending_a(),
            )
            self._leg_b().subscribe_positions(
                self._leg_b_market(), lambda _p: self._clear_pending_b(),
            )

        self._leg_a().subscribe_book(self._leg_a_market(), lambda _b: self._schedule_eval())
        self._leg_b().subscribe_book(self._leg_b_market(), lambda _b: self._schedule_eval())

        try:
            await self._stop.wait()
        finally:
            if self._recorder:
                await self._recorder.aclose()

    # ---- position view (WS-truth + pending overlay) ----

    def _live_size_a(self) -> Decimal:
        """Last WS-pushed signed size on leg A, 0 if no event yet."""
        live = self._leg_a().live_position(self._leg_a_market())
        return live.size if live is not None else Decimal("0")

    def _live_size_b(self) -> Decimal:
        live = self._leg_b().live_position(self._leg_b_market())
        return live.size if live is not None else Decimal("0")

    def _pos_a(self) -> Decimal:
        """Effective signed position on leg A.

        Live: WS-fed `live_position()` + provisional pending overlay.
        Paper: dedicated synthetic ledger (no WS account stream).
        """
        if self.session.is_paper:
            return self._paper_pos_a
        return self._live_size_a() + self._pending_a

    def _pos_b(self) -> Decimal:
        if self.session.is_paper:
            return self._paper_pos_b
        return self._live_size_b() + self._pending_b

    def _clear_pending_a(self) -> None:
        self._pending_a = Decimal("0")

    def _clear_pending_b(self) -> None:
        self._pending_b = Decimal("0")

    # ---- reconcile after failure (sync → balance → cooldown) ----

    async def _reconcile_after_failure(
        self, target_a: Decimal, target_b: Decimal,
    ) -> None:
        """Recover from a partial / total leg failure.

        Flow: REST-snapshot both legs; for any leg whose snap differs
        from the pre-fire target by at least a dust threshold, place a
        reduce-only market order on that leg to flatten. On success,
        zero the WS pending overlay (REST truth is now authoritative
        until WS catches up).

        Contingency (user-specified flow): if the REST snapshot itself
        raises, or any rebalance market order fails, **short-circuit
        immediately** — set `_reconcile_pending=True` and save the
        target. The next `_evaluate` after cooldown expiry will retry.
        No in-cycle retries; no recursion. `record_failure` (called by
        `_fire`) arms the cooldown either way.

        Paper mode: REST returns 0 for both legs and no real venue
        order can be placed; treat as a no-op.
        """
        if self.session.is_paper:
            self._reconcile_pending = False
            self._reconcile_target = None
            return

        try:
            snap_a, snap_b = await asyncio.gather(
                self.session.snapshot_position(self._leg_a(), self._leg_a_market()),
                self.session.snapshot_position(self._leg_b(), self._leg_b_market()),
            )
        except Exception as e:  # noqa: BLE001
            _log.error(
                "reconcile: REST snapshot failed (%s) — deferring to next "
                "cycle after cooldown expires",
                e,
            )
            self._reconcile_pending = True
            self._reconcile_target = (target_a, target_b)
            return

        diff_a = target_a - snap_a
        diff_b = target_b - snap_b
        _log.warning(
            "reconcile: target=(%s, %s) snap=(%s, %s) diff=(%s, %s)",
            target_a, target_b, snap_a, snap_b, diff_a, diff_b,
        )

        ok_a = await self._rebalance_one_leg(
            self._leg_a(), self._leg_a_market(), diff_a,
        )
        ok_b = await self._rebalance_one_leg(
            self._leg_b(), self._leg_b_market(), diff_b,
        )

        if ok_a and ok_b:
            # Both legs reconciled (or already at target). REST is now the
            # truth; clear the overlay so `_pos_a/b()` reads `live_position`
            # cleanly without double-counting the just-placed reduce-only.
            self._reconcile_pending = False
            self._reconcile_target = None
            self._pending_a = Decimal("0")
            self._pending_b = Decimal("0")
            _log.info("reconcile: complete")
        else:
            self._reconcile_pending = True
            self._reconcile_target = (target_a, target_b)

    async def _rebalance_one_leg(
        self, exchange, market: MarketInfo, diff: Decimal,
    ) -> bool:
        """Flatten `|diff|` on `market` via reduce-only market. Returns
        True if the leg is settled (either already at target or the
        reduce-only fill succeeded), False if the leg still needs work.

        Dust threshold: `max(market.min_qty, market.lot_size)` — never
        submit sub-lot orders. A diff smaller than the dust threshold is
        treated as "already at target."
        """
        dust = max(market.min_qty, market.lot_size)
        if abs(diff) < dust:
            return True
        side = Side.BUY if diff > 0 else Side.SELL
        qty = abs(diff)
        # Quantize to lot_size to satisfy the venue's stepSize / amount-
        # tick filter. The dust threshold above already guarantees qty>0.
        if market.lot_size > 0:
            qty = (qty // market.lot_size) * market.lot_size
            if qty < dust:
                return True
        # Reuse the venue's monotonic cid generator so the reconcile order
        # is guaranteed unique across the run (cids are per-driver scoped;
        # an old cid could collide if we rolled our own clock-based string
        # and the bot restarts within a second). Reduce-only reconciles
        # never consume pool slots — they request from the venue's own
        # generator, which for a pool-backed lighter falls back to the
        # underlying counter when slots are empty / not addressable.
        cid = exchange.client_id_generator.next(side=side)
        _log.warning(
            "reconcile: rebalancing %s with %s %s (reduce_only)",
            exchange.name, side.value, qty,
        )
        try:
            outcome = await exchange.submit_and_await(
                market, side, qty, client_id=cid,
                timeout_s=5.0, reduce_only=True,
            )
        except Exception as e:  # noqa: BLE001
            _log.error("reconcile: %s rebalance raised: %s", exchange.name, e)
            return False
        if not outcome.success:
            _log.error(
                "reconcile: %s rebalance failed: %s",
                exchange.name, outcome.error_message,
            )
            return False
        return True

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
        # Pending reconcile from a prior failure: when cooldown has
        # expired, retry the sync→balance flow BEFORE evaluating new
        # entries. If retry still fails, `_reconcile_after_failure` re-
        # arms cooldown and we exit this tick. Either way the normal
        # signal evaluation is suppressed while the reconcile is open.
        if self._reconcile_pending:
            ok, _ = self._risk.can_trade()
            if not ok:
                return  # still inside cooldown — wait
            assert self._reconcile_target is not None
            target_a, target_b = self._reconcile_target
            await self._reconcile_after_failure(target_a, target_b)
            if self._reconcile_pending:
                # Retry didn't complete — re-arm cooldown so we wait
                # again before the next attempt. record_failure handles
                # the consecutive-failures ratchet toward halt.
                self._risk.record_failure("reconcile retry failed")
            return

        a_book = self._leg_a().order_book(self._leg_a_market())
        b_book = self._leg_b().order_book(self._leg_b_market())
        a_q = self._leg_a().best_quote(self._leg_a_market())
        b_q = self._leg_b().best_quote(self._leg_b_market())
        if a_book is None or b_book is None or a_q is None or b_q is None:
            return

        d = self._assess(a_book, b_book, a_q, b_q)
        if d is None:
            return
        result: ExecutionResult | None = None
        try:
            if d.outcome is Verdict.FIRED:
                result = await self._fire(d, a_q, b_q)
        finally:
            if self._recorder:
                self._recorder.emit(d)
                if result is not None:
                    self._recorder.emit_legs(d.decision_id, d.ts_ms, result.legs)

    def _assess(self, a_book, b_book, a_q, b_q) -> Decision | None:
        """Thin wrapper around the pure `assess_reversion` decision math.

        Live-only responsibilities kept here: update the EWMA, run the
        operational `RiskManager` gates (halted / consecutive-failures / daily
        PnL — the pure function only checks the structural `max_qty` cap), and
        emit the throttled heartbeat log."""
        now = now_ms()
        bias = self._spread.update(a_q.mid - b_q.mid, now).center

        fills = compute_taker_fills(self._fill_params, a_book, b_book)
        d = assess_reversion(self._params, AssessInputs(
            now_ms=now,
            left_quote=a_q, right_quote=b_q,
            fills=fills,
            bias=bias, is_warm=self._spread.is_warm,
            position=self._pos_a(),
        ))

        # Persistence-confirm gate: a temporal filter that suppresses a FIRED
        # decision until its edge has survived the confirmation window. A
        # no-op when disabled. Runs BEFORE the cap / risk gates so those only
        # ever see a confirmed fire.
        left_ticked = a_q.ts_ms != self._prev_a_ts
        right_ticked = b_q.ts_ms != self._prev_b_ts
        self._prev_a_ts = a_q.ts_ms
        self._prev_b_ts = b_q.ts_ms
        d = self._gate.admit(d, left_ticked=left_ticked, right_ticked=right_ticked)

        if d is not None and d.outcome is Verdict.FIRED:
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
                    d.outcome = Verdict.BLOCKED_RISK
                    d.abort_reason = (
                        f"in-flight cap {self._inflight_cap} reached for "
                        f"direction {d.direction.value}"
                    )

        # Operational risk gate: halted / consec-failures / daily-loss cap.
        # Position-cap is enforced upstream in the pure function.
        if d is not None and d.outcome is Verdict.FIRED:
            assert d.direction is not None
            d.timeline.mark(Phase.DECISION)
            ok, reason = self._risk.can_trade()
            if not ok:
                if not self._risk_blocked:
                    self._risk_blocked = True
                    _log.info("entry blocked by risk: %s (suppressing until cleared)", reason)
                d.outcome = Verdict.BLOCKED_RISK
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
            "tick: mid_a=%.2f mid_b=%.2f bias=%.4f %s pos_a=%s pos_b=%s pnl=%s",
            mid_left, mid_right, bias, edge_str,
            self._pos_a(), self._pos_b(),
            self._position.realised_pnl,
        )

    # ---- order firing ----

    async def _fire(self, d: Decision, a_q: Quote, b_q: Quote) -> ExecutionResult:
        """Resolve Direction → venue-side intents, delegate execution,
        then apply the ExecutionResult to position / risk.

        Strategy owns the Direction→Side resolution because Direction is
        a strategy concept; the executor only sees concrete `Side`s.
        Strategy also owns position / risk updates because
        they feed back into the next `_assess`. `a_q`/`b_q` are threaded in
        only to stamp the legs' decision-time per-venue context (mid / quote
        freshness / position / depth) — telemetry the executor never sees."""
        assert d.direction is not None
        qty = self.cfg.strategy.qty
        # Decision-time position the signal saw (left leg = the single position
        # that drives skew / cap), captured before the execute() await mutates
        # the WS/overlay view. A decision-level fact → recorded on the trades row.
        pos_a_before = self._pos_a()
        d.position_before = pos_a_before
        a_side, b_side = left_side(d.direction), right_side(d.direction)
        if d.direction is Direction.A:
            a_exp, b_exp = d.vwap_left_sell, d.vwap_right_buy
        else:
            a_exp, b_exp = d.vwap_left_buy, d.vwap_right_sell

        _log.info("[%s] FIRE qty=%s leg_a=%s leg_b=%s mode=%s",
                  d.decision_id, qty, a_side, b_side, self.cfg.strategy.mode)

        if self._inflight_cap > 0:
            self._inflight_dir[d.decision_id] = d.direction
        # Snapshot live_position BEFORE the executor await so we can
        # detect how much of `delta` the WS account stream already
        # absorbed during execution. Without this, a slow path (e.g. the
        # aster 400/timeout → queryOrder recovery) lets ACCOUNT_UPDATE
        # land during the await, `_clear_pending_*` zeros the overlay,
        # then we bump pending on success and double-count by `delta`
        # until the *next* trade's ACCOUNT_UPDATE arrives.
        pre_a = self._live_size_a()
        pre_b = self._live_size_b()
        try:
            result = await self._executor.execute(
                trade_id=d.decision_id,
                legs=(
                    LegIntent(venue="leg_a", side=a_side, expected_price=a_exp),
                    LegIntent(venue="leg_b", side=b_side, expected_price=b_exp),
                ),
                qty=qty,
                timeline=d.timeline,
            )
            d.failure_reason = result.failure_reason
            # Stamp each leg's per-venue context (quote freshness). result.legs
            # is ordered [leg_a, leg_b], lining up with a_q/b_q.
            if len(result.legs) == 2:
                _stamp_leg_ctx(result.legs[0], a_q)
                _stamp_leg_ctx(result.legs[1], b_q)
            entry_legs = [lg for lg in result.legs if lg.kind is LegKind.ENTRY]
            assert len(entry_legs) == 2
            d.send_ts_ms = entry_legs[0].send_ts_ms

            success = result.failure_reason is None
            if success:
                # The trade's predicted signed delta per leg.
                delta_a = qty * Decimal(a_side.sign)
                delta_b = qty * Decimal(b_side.sign)
                if self.session.is_paper:
                    # Paper has no WS account stream; the ledger IS truth.
                    self._paper_pos_a += delta_a
                    self._paper_pos_b += delta_b
                else:
                    # Live: bump pending only by the part WS hasn't pushed
                    # yet. `gap = delta - (post - pre)`. If WS fully caught
                    # up during the await, post-pre == delta and gap == 0
                    # (no overlay needed). If WS hasn't arrived yet,
                    # gap == delta and the overlay carries the prediction
                    # until `_clear_pending_*` fires.
                    gap_a = delta_a - (self._live_size_a() - pre_a)
                    gap_b = delta_b - (self._live_size_b() - pre_b)
                    self._pending_a += gap_a
                    self._pending_b += gap_b
                # Derive per-leg latency on the fly from the two timestamps
                # the recorder also persists. Mixed clock (local send vs
                # venue match), NTP-skew-sensitive at ms scale — acceptable
                # given `max_leg_latency_ms` is typically 500 ms.
                worst = 0
                for leg in entry_legs:
                    if leg.fill_ts_ms is not None and leg.send_ts_ms is not None:
                        worst = max(worst, leg.fill_ts_ms - leg.send_ts_ms)
                self._risk.record_success(leg_latency_ms=worst)

                # Cash-flow realized PnL: feeds the dormant daily-loss-cap
                # gate (RiskManager.can_trade) and surfaces in heartbeat +
                # decisions_*.csv. Mirrors backtest pnl.apply_pair so live
                # and backtest numbers reconcile by construction.
                pnl = pair_pnl_from_legs(entry_legs[0], entry_legs[1])
                if pnl is not None:
                    self._position.realised_pnl += pnl
                    self._risk.record_pnl(pnl)
                    d.realised_pnl = pnl

                _log.info(
                    "[%s] pos_a=%s pos_b=%s pnl_total=%s",
                    d.decision_id,
                    self._pos_a(), self._pos_b(),
                    self._position.realised_pnl,
                )
            else:
                # Failure path: sync → balance → cooldown.
                # `_reconcile_after_failure` does the first two; if either
                # step itself fails it sets `_reconcile_pending` so the
                # next eval after cooldown will retry. `record_failure`
                # arms the cooldown unconditionally.
                await self._reconcile_after_failure(pre_a, pre_b)
                self._risk.record_failure(result.failure_reason or "unknown")
        finally:
            if self._inflight_cap > 0:
                self._inflight_dir.pop(d.decision_id, None)
        # Hand the execution result back so the caller can record the legs
        # separately (Decision no longer carries them).
        return result


def _stamp_leg_ctx(leg: LegOutcome, q: Quote) -> None:
    """Attach a leg's per-venue decision-time context: this venue's quote
    freshness (staleness forensics). That's all the leg owns — fill quality is
    `realized − expected_price` (the book is already folded into expected_price),
    and decision-time inventory is a decision-level fact on the `trades` row."""
    leg.quote_ts_ms = q.ts_ms
