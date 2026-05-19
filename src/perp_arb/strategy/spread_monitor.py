"""Spread monitor — no trading. Streams BBO from two venues, records the
tick-level spread / bias / VWAP-edge series to hourly Parquet.

Venue-agnostic: the pair is `cfg.strategy.monitor_pair` (left, right); absent it
defaults to aster↔lighter for back-compat.

Order-book maintenance lives entirely in the clients; each exposes `book_ts()`
— the wall-clock of the last update applied to a *valid, in-sync* book. The
recorder has one uniform rule: **record a row only when both legs are fresh;
otherwise skip.** A down/desynced feed applies no in-sync updates, so its
`book_ts` ages out and rows are simply not written — an outage becomes a gap in
the captured time series (and a large `gap_ms` on the recovery row), never
garbage rows. Each row also carries the two legs' source timestamps so
freshness is auditable offline.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal

import pyarrow as pa

from ..core.capture import RotatingParquetWriter
from ..core.types import OrderBook, Quote
from ..utils.precision import vwap_fill
from ..utils.time import now_ms
from .base import BaseStrategy, SpreadModel

_log = logging.getLogger(__name__)

_STR_COLS = [
    "left_bid", "left_bid_size", "left_ask", "left_ask_size",
    "right_bid", "right_bid_size", "right_ask", "right_ask_size",
    "mid_left", "mid_right", "raw_spread", "bias_ewma",
    "vwap_left_sell", "vwap_left_buy", "vwap_right_sell", "vwap_right_buy",
    "edge_A_bps", "edge_B_bps",
]

SPREAD_PARQUET_SCHEMA = pa.schema(
    [("ts_ms", pa.int64()), ("left_venue", pa.string()), ("right_venue", pa.string())]
    + [(c, pa.string()) for c in _STR_COLS]
    + [
        ("gates_passed", pa.bool_()),
        # each leg's last in-sync book update (wall-clock ms) + time since the
        # previous *recorded* row: the only completeness signals needed — an
        # outage shows as a missing time range + a large gap_ms on recovery.
        ("left_ts_ms", pa.int64()),
        ("right_ts_ms", pa.int64()),
        ("gap_ms", pa.int64()),
    ]
)

# A leg whose book hasn't applied an in-sync update within this many ms is
# treated as not fresh — the row is skipped entirely. Set well above the
# observed healthy Katana inter-update gap (~3 s max on a quiet book) so
# quiet-but-live books aren't false-skipped; genuine outages (the failure
# mode) last minutes, so the wider window still catches them.
_FRESH_MS = 8000


class SpreadMonitor(BaseStrategy):
    name = "spread_monitor"

    def __init__(self, cfg, exchanges, markets) -> None:
        super().__init__(cfg, exchanges, markets)
        self._writer: RotatingParquetWriter | None = None
        self._spread = SpreadModel(
            center_half_life_s=cfg.strategy.bias_halflife_s,
            scale_half_life_s=cfg.strategy.scale_halflife_s,
            warmup_s=0,  # monitor logs from the first tick; no warmup gating
        )
        self._last_log_ms = 0
        self._last_row_ms = 0
        self._skipped = 0  # rows skipped because a leg was not fresh

        mp = cfg.strategy.monitor_pair
        if mp is not None:
            self._left_name, self._right_name = mp[0].venue, mp[1].venue
        else:
            self._left_name, self._right_name = "aster", "lighter"

    async def run(self) -> None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        root = self.cfg.runtime.log_dir / f"spread_{self.cfg.strategy.pair.base}_{ts}"
        self._writer = RotatingParquetWriter(root, SPREAD_PARQUET_SCHEMA)
        await self._writer.start()
        _log.info(
            "spread_monitor %s↔%s writing %s",
            self._left_name, self._right_name, root,
        )

        self._venue(self._left_name).subscribe_book(
            self._market(self._left_name), self._on_book
        )
        self._venue(self._right_name).subscribe_book(
            self._market(self._right_name), self._on_book
        )

        try:
            await self._stop.wait()
        finally:
            if self._writer:
                await self._writer.close()

    # both books must be present + fresh; either side ticking triggers an eval
    def _on_book(self, _: OrderBook) -> None:
        self._evaluate()

    def _evaluate(self) -> None:
        lx, rx = self._venue(self._left_name), self._venue(self._right_name)
        lm, rm = self._market(self._left_name), self._market(self._right_name)
        l_book = lx.order_book(lm)
        r_book = rx.order_book(rm)
        if l_book is None or r_book is None:
            return
        l_q: Quote | None = lx.best_quote(lm)
        r_q: Quote | None = rx.best_quote(rm)
        if l_q is None or r_q is None:
            return

        # Uniform freshness gate: skip the row entirely if either leg's book is
        # not in-sync / not recently updated. Outage ⇒ gap in the series.
        ts_ms = now_ms()
        l_ts = lx.book_ts(lm)
        r_ts = rx.book_ts(rm)
        if (l_ts is None or r_ts is None
                or ts_ms - l_ts > _FRESH_MS or ts_ms - r_ts > _FRESH_MS):
            self._skipped += 1
            return

        qty = self.cfg.strategy.qty
        mx = self.cfg.strategy.max_levels

        mid_l = l_q.mid
        mid_r = r_q.mid
        raw_spread = mid_l - mid_r
        bias = self._spread.update(raw_spread, ts_ms).center

        vwap_l_sell, _ = vwap_fill(l_book.bids, qty, max_levels=mx)
        vwap_l_buy,  _ = vwap_fill(l_book.asks, qty, max_levels=mx)
        vwap_r_sell, _ = vwap_fill(r_book.bids, qty, max_levels=mx)
        vwap_r_buy,  _ = vwap_fill(r_book.asks, qty, max_levels=mx)

        edge_A_bps: Decimal | None = None
        edge_B_bps: Decimal | None = None
        if vwap_l_sell is not None and vwap_r_buy is not None:
            edge_A_bps = ((vwap_l_sell - vwap_r_buy) - bias) / mid_l * Decimal(10_000)
        if vwap_r_sell is not None and vwap_l_buy is not None:
            edge_B_bps = ((vwap_r_sell - vwap_l_buy) + bias) / mid_l * Decimal(10_000)

        gates = (vwap_l_sell is not None and vwap_r_buy is not None
                 and vwap_r_sell is not None and vwap_l_buy is not None)

        gap_ms = 0 if self._last_row_ms == 0 else ts_ms - self._last_row_ms
        self._last_row_ms = ts_ms

        if self._writer:
            self._writer.submit({
                "ts_ms": ts_ms,
                "left_venue": self._left_name, "right_venue": self._right_name,
                "left_bid": l_q.bid, "left_bid_size": l_q.bid_size,
                "left_ask": l_q.ask, "left_ask_size": l_q.ask_size,
                "right_bid": r_q.bid, "right_bid_size": r_q.bid_size,
                "right_ask": r_q.ask, "right_ask_size": r_q.ask_size,
                "mid_left": mid_l, "mid_right": mid_r,
                "raw_spread": raw_spread, "bias_ewma": bias,
                "vwap_left_sell": vwap_l_sell, "vwap_left_buy": vwap_l_buy,
                "vwap_right_sell": vwap_r_sell, "vwap_right_buy": vwap_r_buy,
                "edge_A_bps": edge_A_bps, "edge_B_bps": edge_B_bps,
                "gates_passed": gates,
                "left_ts_ms": l_ts, "right_ts_ms": r_ts,
                "gap_ms": gap_ms,
            })

        # human-readable log throttled to once per minute; Parquet has the full stream
        if ts_ms - self._last_log_ms >= 60_000:
            self._last_log_ms = ts_ms
            _log.info(
                "spread: mid_l=%s mid_r=%s spread=%s bias=%s edge_A=%s edge_B=%s "
                "gates=%s age_l=%dms age_r=%dms skipped=%d",
                _fmt(mid_l, 2), _fmt(mid_r, 2),
                _fmt(raw_spread, 4), _fmt(bias, 4),
                _fmt(edge_A_bps, 2), _fmt(edge_B_bps, 2), gates,
                ts_ms - l_ts, ts_ms - r_ts, self._skipped,
            )


def _fmt(v: Decimal | None, places: int) -> str:
    if v is None:
        return "?"
    q = Decimal(10) ** -places
    return str(v.quantize(q))
