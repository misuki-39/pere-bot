"""Risk manager — gates entries and kills the bot when limits breach."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from ..core.config import RiskCfg

_log = logging.getLogger(__name__)


@dataclass
class RiskState:
    consecutive_failures: int = 0
    last_leg_latency_ms: int = 0
    realised_pnl_today: Decimal = Decimal("0")
    halted: bool = False
    halt_reason: str | None = None
    day_key: str = field(default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d"))


class RiskManager:
    def __init__(self, cfg: RiskCfg) -> None:
        self.cfg = cfg
        self.state = RiskState()

    # ---- gates ----

    def can_trade(self) -> tuple[bool, str | None]:
        """Operational gates: halted, consecutive failures, daily loss cap.

        Position-cap enforcement lives in the pure decision function
        (`strategy.reversion_signal.assess_reversion`), which drops the tick
        before it reaches here — so cap-hit ticks never become BLOCKED_RISK
        rows."""
        self._rollover_day()
        if self.state.halted:
            return False, f"halted: {self.state.halt_reason}"
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
        _log.warning(
            "risk: failure recorded (%d/%d): %s",
            self.state.consecutive_failures, self.cfg.max_consecutive_failures, reason,
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
