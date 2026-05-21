"""Builds the per-tick MarketSnapshot the strategy sees.

The snapshot wraps the raw `BBORow` in project-native `Quote`/`OrderBook`
types so strategies can share helpers with live code (`q.mid`, etc.) without
caring that the source is a Parquet row.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from ..core.types import BookLevel, OrderBook, Quote, Symbol
from .dataset import BBORow

VenueSide = Literal["left", "right"]


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """One tick worth of joint book state.

    `ts_ms` is the bot wall-clock at capture (= sim "now"). `left_ts_ms` /
    `right_ts_ms` are the per-leg venue source timestamps — important for the
    book-at-arrival lookup that drives adverse-selection realism.
    """
    ts_ms: int
    left_venue: str
    right_venue: str
    left_ts_ms: int
    right_ts_ms: int
    left_quote: Quote
    right_quote: Quote
    left_book: OrderBook
    right_book: OrderBook
    vwap_left_sell: Decimal | None
    vwap_left_buy: Decimal | None
    vwap_right_sell: Decimal | None
    vwap_right_buy: Decimal | None
    edge_A_bps: Decimal | None
    edge_B_bps: Decimal | None
    bias_capture: Decimal
    gates_passed: bool
    gap_ms: int


def _symbol(venue: str) -> Symbol:
    return Symbol(exchange=venue, raw="", base="", quote="")


def _quote(row: BBORow, side: VenueSide) -> Quote:
    if side == "left":
        return Quote(
            symbol=_symbol(row.left_venue),
            bid=row.left_bid, bid_size=row.left_bid_size,
            ask=row.left_ask, ask_size=row.left_ask_size,
            ts_ms=row.left_ts_ms,
        )
    return Quote(
        symbol=_symbol(row.right_venue),
        bid=row.right_bid, bid_size=row.right_bid_size,
        ask=row.right_ask, ask_size=row.right_ask_size,
        ts_ms=row.right_ts_ms,
    )


def _book(row: BBORow, side: VenueSide) -> OrderBook:
    """Lift the captured top-of-book into a one-level OrderBook so strategies
    can share `vwap_fill` etc. with live code (it'll just refuse anything
    larger than the top level — which is exactly the v1 BBO-fill semantic)."""
    q = _quote(row, side)
    return OrderBook(
        symbol=q.symbol,
        bids=[BookLevel(q.bid, q.bid_size)],
        asks=[BookLevel(q.ask, q.ask_size)],
        ts_ms=q.ts_ms,
    )


def build_snapshot(row: BBORow) -> MarketSnapshot:
    return MarketSnapshot(
        ts_ms=row.ts_ms,
        left_venue=row.left_venue, right_venue=row.right_venue,
        left_ts_ms=row.left_ts_ms, right_ts_ms=row.right_ts_ms,
        left_quote=_quote(row, "left"),
        right_quote=_quote(row, "right"),
        left_book=_book(row, "left"),
        right_book=_book(row, "right"),
        vwap_left_sell=row.vwap_left_sell, vwap_left_buy=row.vwap_left_buy,
        vwap_right_sell=row.vwap_right_sell, vwap_right_buy=row.vwap_right_buy,
        edge_A_bps=row.edge_A_bps, edge_B_bps=row.edge_B_bps,
        bias_capture=row.bias_ewma,
        gates_passed=row.gates_passed, gap_ms=row.gap_ms,
    )
