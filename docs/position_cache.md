# Position cache: single-writer invariant

## Pattern A — WS owns the cache

Each `BaseExchange` adapter maintains a `_live_positions` cache fed by the
venue's user-data WebSocket (Aster `ACCOUNT_UPDATE`, Lighter `account_all`).

**Invariant**: once `connect()` returns and the user-data WS is up, the WS
handler is the **only writer** to `_live_positions`. REST calls are pure reads.

A single writer kills an entire class of cache-stomp races where an in-flight
REST snapshot taken at `t=0` overwrites a fresher WS update applied at `t=15ms`
when the REST response finally lands at `t=30ms`. With one writer, the cache
is monotone in event-time by construction — no sequence numbers or timestamps
needed.

## Bootstrap

The first call to `get_position(market)` seeds the cache **only if** no WS
event has fired yet. Implementation: `dict.setdefault`.

If a WS event arrives during the seed REST call, it wins:

```
t=0    get_position()  →  sends GET /positionRisk
t=15   WS pushes ACCOUNT_UPDATE
t=16   _handle_account_update writes cache[X] = fresh_pos     ← winner
t=30   REST response returns stale (t=0) snapshot
t=31   get_position calls setdefault(X, stale_pos)            ← no-op
```

## What this rules out

- **Periodic REST reconciliation that overwrites the cache.** If we want drift
  monitoring later, REST should *compare* against WS state and alarm on
  divergence, not replace it.
- **Calling `get_position()` to "refresh" live state mid-session.** Use
  `live_position()` — it's already the authoritative view.
- **Calling `get_position()` from inside the trading hot path.** It's a REST
  round-trip; treat it as cold-start / debug only.

## Why not sequence/timestamp gating (Pattern B)?

Monotone-replacement-by-sequence is robust against multiple writers — but we
don't have multiple writers. Pattern A is one line of code; Pattern B is
per-venue plumbing for monotone counters that not every endpoint publishes
(Aster's `ACCOUNT_UPDATE` has `E` ms; `positionRisk` has no native timestamp).

Revisit Pattern B if we ever add: periodic REST reconciliation, shared
accounts (another bot/human moving the same position), or strategies that
trust REST snapshots over WS state.
