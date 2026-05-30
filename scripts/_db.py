"""Load a live recorder run (SQLite source-of-truth, or a Turso pull) into
DataFrames. Shared by the offline analysis scripts.

The local SQLite file written by `core.record_sink.SqliteRecorder` and the
Turso mirror share the same schema, so either works as `db_path`. A run is
identified by `run_id` (the YYYYMMDDTHHMMSSZ label); default = the latest.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pandas as pd


@dataclass
class Run:
    run_id: str
    trades: pd.DataFrame      # one row per fired trade (pair-level header)
    legs: pd.DataFrame        # 2 rows per trade (per-venue detail)
    rejections: pd.DataFrame  # ticks we declined to fire


def latest_run_id(db_path: str) -> str | None:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT run_id FROM runs ORDER BY started_at_ms DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def load(db_path: str, run_id: str | None = None) -> Run:
    conn = sqlite3.connect(db_path)
    try:
        if run_id is None:
            row = conn.execute(
                "SELECT run_id FROM runs ORDER BY started_at_ms DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise SystemExit(f"no runs found in {db_path}")
            run_id = row[0]
        q = "SELECT * FROM {} WHERE run_id=? ORDER BY rowid"
        return Run(
            run_id=run_id,
            trades=pd.read_sql(q.format("trades"), conn, params=(run_id,)),
            legs=pd.read_sql(q.format("legs"), conn, params=(run_id,)),
            rejections=pd.read_sql(q.format("rejections"), conn, params=(run_id,)),
        )
    finally:
        conn.close()
