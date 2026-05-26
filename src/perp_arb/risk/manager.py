"""Risk manager — gates entries and kills the bot when limits breach."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from ..core.config import RiskCfg
from ..utils.time import now_ms

_log = logging.getLogger(__name__)


@dataclass
class RiskState:
    consecutive_failures: int = 0
    last_leg_latency_ms: int = 0
    realised_pnl_today: Decimal = Decimal("0")
    halted: bool = False
    halt_reason: str | None = None
    day_key: str = field(default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d"))
    # Soft gate: while `now_ms() < cooldown_until_ms`, `can_trade()` blocks
    # new fires with a "cooldown" reason. Set by `record_failure`. Once it
    # elapses, trading resumes naturally (no manual unhalt needed). The
    # strategy's `_reconcile_after_failure` runs BEFORE this is armed so
    # any residual is flattened before the wait begins.
    cooldown_until_ms: int = 0


class RiskManager:
    def __init__(self, cfg: RiskCfg) -> None:
        self.cfg = cfg
        self.state = RiskState()

    # ---- gates ----

    def can_trade(self) -> tuple[bool, str | None]:
        """Operational gates: halted, cooldown, consecutive failures, daily loss cap.

        Position-cap enforcement lives in the pure decision function
        (`strategy.reversion_signal.assess_reversion`), which drops the tick
        before it reaches here — so cap-hit ticks never become BLOCKED_RISK
        rows."""
        self._rollover_day()
        if self.state.halted:
            return False, f"halted: {self.state.halt_reason}"
        # Cooldown is the soft, time-bounded gate set on every failure.
        # Checked before consecutive-failures so a transient blip surfaces
        # as "cooldown 42s" instead of the cumulative-failure narrative.
        remaining_ms = self.state.cooldown_until_ms - now_ms()
        if remaining_ms > 0:
            return False, f"cooldown {remaining_ms / 1000:.1f}s"
        if self.state.consecutive_failures >= self.cfg.max_consecutive_failures:
            return False, f"too many consecutive failures ({self.state.consecutive_failures})"
        if self.state.realised_pnl_today <= -self.cfg.daily_loss_cap_usd:
            return False, f"daily loss cap breached ({self.state.realised_pnl_today})"
        return True, None

    # ---- mutation ----

    def record_success(self, *, leg_latency_ms: int) -> None:
        self.state.consecutive_failures = 0
        self.state.last_leg_latency_ms = leg_latency_ms
        if leg_latency_ms > self.cfg.max_leg_latency_ms:
            _log.warning(
                "leg latency %dms > budget %dms",
                leg_latency_ms, self.cfg.max_leg_latency_ms,
            )

    def record_failure(self, reason: str) -> None:
        self.state.consecutive_failures += 1
        # Arm cooldown BEFORE the halt branch — so the failure that finally
        # trips the halt still leaves the cooldown timestamp set, which
        # keeps observability consistent (`can_trade` log lines show
        # "cooldown" before the halt narrative takes over).
        self.state.cooldown_until_ms = now_ms() + int(self.cfg.cooldown_s * 1000)
        _log.warning(
            "risk: failure recorded (%d/%d), cooldown %ss: %s",
            self.state.consecutive_failures, self.cfg.max_consecutive_failures,
            self.cfg.cooldown_s, reason,
        )
        if self.state.consecutive_failures >= self.cfg.max_consecutive_failures:
            self.halt(f"max_consecutive_failures: {reason}")

    def record_pnl(self, realised: Decimal) -> None:
        self._rollover_day()
        self.state.realised_pnl_today += realised
        if self.state.realised_pnl_today <= -self.cfg.daily_loss_cap_usd:
            self.halt(
                f"daily_loss_cap: realised={self.state.realised_pnl_today} "
                f"<= -{self.cfg.daily_loss_cap_usd}",
            )

    def halt(self, reason: str) -> None:
        if not self.state.halted:
            _log.error("RISK HALT: %s", reason)
        self.state.halted = True
        self.state.halt_reason = reason

    def _rollover_day(self) -> None:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self.state.day_key:
            _log.info("risk: day rollover %s -> %s", self.state.day_key, today)
            self.state.day_key = today
            self.state.realised_pnl_today = Decimal("0")
            # do NOT auto-unhalt — operator must restart after a halt.
