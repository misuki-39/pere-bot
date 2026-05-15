"""Spread monitor — no trading. Streams BBO from both venues, logs spread + CSV.

Run this for a few hours before turning on taker_taker to understand the
real-world spread distribution for the chosen pair.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal

from ..core.logging import SPREAD_CSV_HEADER, CsvWriter
from ..core.types import OrderBook, Quote
from ..utils.precision import vwap_fill
from ..utils.time import now_ms
from .base import BaseStrategy, EwmaTracker

_log = logging.getLogger(__name__)


class SpreadMonitor(BaseStrategy):
    name = "spread_monitor"

    def __init__(self, cfg, exchanges, markets) -> None:
        super().__init__(cfg, exchanges, markets)
        self._csv: CsvWriter | None = None
        self._bias = EwmaTracker(cfg.strategy.bias_window_ticks)
        self._last_log_ms = 0

    async def run(self) -> None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        csv_path = self.cfg.runtime.log_dir / f"spread_{self.cfg.strategy.pair.base}_{ts}.csv"
        self._csv = CsvWriter(csv_path, SPREAD_CSV_HEADER)
        _log.info("spread_monitor writing %s", csv_path)

        self._aster().subscribe_book(self._aster_market(), self._on_aster_book)
        self._lighter().subscribe_book(self._lighter_market(), self._on_lighter_book)

        try:
            await self._stop.wait()
        finally:
            if self._csv:
                self._csv.close()

    # ---- callbacks: just trigger an evaluation; both books need to be present ----

    def _on_aster_book(self, _: OrderBook) -> None:
        self._evaluate()

    def _on_lighter_book(self, _: OrderBook) -> None:
        self._evaluate()

    def _evaluate(self) -> None:
        a_book = self._aster().order_book(self._aster_market())
        l_book = self._lighter().order_book(self._lighter_market())
        if a_book is None or l_book is None:
            return
        a_q: Quote | None = self._aster().best_quote(self._aster_market())
        l_q: Quote | None = self._lighter().best_quote(self._lighter_market())
        if a_q is None or l_q is None:
            return

        qty = self.cfg.strategy.qty

        mid_a = a_q.mid
        mid_l = l_q.mid
        raw_spread = mid_a - mid_l
        bias = self._bias.update(raw_spread)

        vwap_a_sell, _ = vwap_fill(a_book.bids, qty, max_levels=self.cfg.strategy.max_levels)
        vwap_a_buy,  _ = vwap_fill(a_book.asks, qty, max_levels=self.cfg.strategy.max_levels)
        vwap_l_sell, _ = vwap_fill(l_book.bids, qty, max_levels=self.cfg.strategy.max_levels)
        vwap_l_buy,  _ = vwap_fill(l_book.asks, qty, max_levels=self.cfg.strategy.max_levels)

        edge_A_bps: Decimal | None = None
        edge_B_bps: Decimal | None = None
        if vwap_a_sell is not None and vwap_l_buy is not None:
            edge_A_bps = ((vwap_a_sell - vwap_l_buy) - bias) / mid_a * Decimal(10_000)
        if vwap_l_sell is not None and vwap_a_buy is not None:
            edge_B_bps = ((vwap_l_sell - vwap_a_buy) + bias) / mid_a * Decimal(10_000)

        ts_ms = now_ms()
        gates = (vwap_a_sell is not None and vwap_l_buy is not None
                 and vwap_l_sell is not None and vwap_a_buy is not None)

        if self._csv:
            self._csv.write([
                ts_ms,
                a_q.bid, a_q.bid_size, a_q.ask, a_q.ask_size,
                l_q.bid, l_q.bid_size, l_q.ask, l_q.ask_size,
                mid_a, mid_l, raw_spread, bias,
                vwap_a_sell, vwap_a_buy, vwap_l_sell, vwap_l_buy,
                edge_A_bps, edge_B_bps, gates,
            ])

        # human-readable log throttled to once per minute; CSV has the full stream
        if ts_ms - self._last_log_ms >= 60_000:
            self._last_log_ms = ts_ms
            _log.info(
                "spread: mid_a=%s mid_l=%s spread=%s bias=%s edge_A=%s edge_B=%s gates=%s",
                _fmt(mid_a, 2), _fmt(mid_l, 2),
                _fmt(raw_spread, 4), _fmt(bias, 4),
                _fmt(edge_A_bps, 2), _fmt(edge_B_bps, 2), gates,
            )


def _fmt(v: Decimal | None, places: int) -> str:
    if v is None:
        return "?"
    q = Decimal(10) ** -places
    return str(v.quantize(q))
