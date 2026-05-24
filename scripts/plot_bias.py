"""Plot bias_ewma (and raw_spread) from a spread_monitor capture in the browser.

Reads all hourly parquet files under <capture_dir>/date=*/HH.parquet, optionally
downsamples for responsive rendering, writes a self-contained HTML, and opens it
in the default browser. Pan/zoom/rangeslider are native plotly controls; no
server is required.

Usage:
    python scripts/plot_bias.py [capture_dir] [--out OUT_HTML] [--max-points N]
                                [--no-raw] [--no-open]

Defaults to the only spread_WTI capture under logs/ if no dir is given.
"""

from __future__ import annotations

import argparse
import glob
import sys
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq


def _load(capture_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(capture_dir / "date=*" / "*.parquet")))
    if not files:
        sys.exit(f"no parquet files under {capture_dir}")

    frames: list[pd.DataFrame] = []
    skipped: list[tuple[str, str]] = []
    cols = ["ts_ms", "bias_ewma", "raw_spread"]
    for f in files:
        try:
            frames.append(pq.read_table(f, columns=cols).to_pandas())
        except Exception as e:  # in-flight file or corrupt footer
            skipped.append((f, str(e).splitlines()[0][:80]))

    if skipped:
        print(f"[warn] skipped {len(skipped)} unreadable file(s):", file=sys.stderr)
        for f, msg in skipped:
            print(f"  {Path(f).name}: {msg}", file=sys.stderr)

    df = pd.concat(frames, ignore_index=True)
    df["bias_ewma"] = pd.to_numeric(df["bias_ewma"], errors="coerce")
    df["raw_spread"] = pd.to_numeric(df["raw_spread"], errors="coerce")
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.dropna(subset=["bias_ewma"]).sort_values("ts").reset_index(drop=True)
    return df


_BUCKET_ALIAS = {"m": "min"}  # pandas dropped 'm' (was minute) — it's month-end now


def _bucket(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    """Aggregate to fixed time buckets (e.g. '1s', '1m', '5m'). Empty buckets dropped."""
    suffix = bucket.lstrip("0123456789").lower()
    freq = bucket[: len(bucket) - len(suffix)] + _BUCKET_ALIAS.get(suffix, suffix)
    g = df.set_index("ts").resample(freq)
    out = g.agg({"bias_ewma": "mean", "raw_spread": "mean"}).dropna(how="all")
    return out.reset_index()


def _cap(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = len(df) // max_points + 1
    return df.iloc[::step].reset_index(drop=True)


def _figure(df: pd.DataFrame, show_raw: bool, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=df["ts"],
            y=df["bias_ewma"],
            mode="lines",
            name="bias_ewma",
            line=dict(color="#1f77b4", width=1.2),
            hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>bias=%{y:.6f}<extra></extra>",
        )
    )
    if show_raw:
        fig.add_trace(
            go.Scattergl(
                x=df["ts"],
                y=df["raw_spread"],
                mode="lines",
                name="raw_spread",
                line=dict(color="#bbbbbb", width=0.6),
                opacity=0.5,
                hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>raw=%{y:.6f}<extra></extra>",
            )
        )
    fig.add_hline(y=0.0, line=dict(color="#888", width=0.6, dash="dot"))
    fig.update_layout(
        title=title,
        xaxis=dict(title="time (UTC)", rangeslider=dict(visible=True), type="date"),
        yaxis=dict(title="spread"),
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    return fig


def _default_capture() -> Path:
    root = Path(__file__).resolve().parent.parent / "logs"
    matches = sorted(root.glob("spread_WTI_*"))
    if not matches:
        sys.exit(f"no spread_WTI_* capture under {root}; pass capture_dir explicitly")
    return matches[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture_dir", nargs="?", type=Path, default=None,
                   help="capture directory (default: latest logs/spread_WTI_*)")
    p.add_argument("--out", type=Path, default=None,
                   help="output HTML path (default: <capture_dir>/bias_plot.html)")
    p.add_argument("--bucket", type=str, default="1s",
                   help="time-bucket aggregation, pandas offset alias "
                        "(e.g. '1s', '500ms', '1m', '5m'); 'none' disables (default 1s)")
    p.add_argument("--max-points", type=int, default=200_000,
                   help="hard cap on plotted points after bucketing (default 200k)")
    p.add_argument("--no-raw", action="store_true", help="hide raw_spread trace")
    p.add_argument("--no-open", action="store_true", help="don't auto-open browser")
    args = p.parse_args()

    capture = args.capture_dir or _default_capture()
    capture = capture.resolve()
    print(f"[info] capture: {capture}")

    df = _load(capture)
    print(f"[info] loaded {len(df):,} rows  "
          f"({df['ts'].iloc[0]} → {df['ts'].iloc[-1]})")

    df_plot = df
    if args.bucket.lower() != "none":
        df_plot = _bucket(df, args.bucket)
        print(f"[info] bucketed to {len(df_plot):,} points @ {args.bucket} (mean)")
    before_cap = len(df_plot)
    df_plot = _cap(df_plot, args.max_points)
    if len(df_plot) < before_cap:
        print(f"[info] capped to {len(df_plot):,} points (max-points)")

    title = (f"{capture.name} — bias_ewma  "
             f"[{df['ts'].iloc[0]:%Y-%m-%d %H:%M} → "
             f"{df['ts'].iloc[-1]:%Y-%m-%d %H:%M} UTC]")
    fig = _figure(df_plot, show_raw=not args.no_raw, title=title)

    out = args.out or (capture / "bias_plot.html")
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    print(f"[info] wrote {out}")

    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
