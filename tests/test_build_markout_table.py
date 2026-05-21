"""Smoke test for `scripts/build_markout_table.py`.

Builds a tiny synthetic Hive-partitioned capture (one parquet under
date=<today>/00.parquet), runs the builder, and asserts the output JSON is
valid + loadable by `MarkoutTable.from_json`.

We don't assert specific markout numbers (the synthetic distribution is
arbitrary); we only assert structural correctness so the round-trip works.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

# The build_markout_table script depends on numpy + pandas (via the
# markout_analysis module it imports). The project's .venv has pyarrow but
# not numpy/pandas; operators run the script under their own analysis env.
# Skip these smoke tests gracefully when the deps aren't present.
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from perp_arb.strategy.markout import MarkoutTable


def _load_builder():
    """Import scripts/build_markout_table.py without installing it.

    scripts/ is gitignored and not a package; importlib via file path.
    """
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "build_markout_table.py"
    spec = importlib.util.spec_from_file_location("build_markout_table", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_markout_table"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_synthetic_capture(root: Path, n_rows: int = 5_000) -> None:
    """Hive-partitioned synthetic capture at root/date=<today>/00.parquet.

    Generates a deterministic walk so adverse markout buckets are populated
    with at least the validator's MIN threshold of samples.
    """
    today = dt.datetime.now(dt.UTC).date().isoformat()
    out_dir = root / f"date={today}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build rows: 200ms cadence, alternating "edge" and "no-edge" ticks so
    # the analyser sees both positive-edge and zero-edge regimes.
    ts0 = 1_700_000_000_000
    cols: dict[str, list[str | int | bool]] = {
        "ts_ms": [], "vwap_left_sell": [], "vwap_left_buy": [],
        "vwap_right_sell": [], "vwap_right_buy": [],
        "mid_left": [], "mid_right": [],
        "edge_A_bps": [], "edge_B_bps": [],
        "gates_passed": [],
    }
    for i in range(n_rows):
        ts = ts0 + i * 200
        # left mid drifts; right mid stable. Generates +A-edge ticks.
        left_mid = Decimal("100") + Decimal("0.001") * (i % 7)
        right_mid = Decimal("100")
        cols["ts_ms"].append(ts)
        cols["mid_left"].append(str(left_mid))
        cols["mid_right"].append(str(right_mid))
        cols["vwap_left_sell"].append(str(left_mid))
        cols["vwap_left_buy"].append(str(left_mid))
        cols["vwap_right_sell"].append(str(right_mid))
        cols["vwap_right_buy"].append(str(right_mid))
        # edge_A in bps: (left_mid - right_mid) / mid_ref * 1e4 ≈ 1*(i%7) bps
        edge_a = (left_mid - right_mid) / ((left_mid + right_mid) / 2) * Decimal("1e4")
        cols["edge_A_bps"].append(str(edge_a))
        cols["edge_B_bps"].append(str(-edge_a))
        cols["gates_passed"].append(True)

    tbl = pa.table(cols)
    pq.write_table(tbl, out_dir / "00.parquet")


def test_builder_smoke_roundtrip(tmp_path: Path) -> None:
    builder = _load_builder()
    bbo_root = tmp_path / "spread_TEST_synth"
    out = tmp_path / "out" / "markout.json"
    _make_synthetic_capture(bbo_root, n_rows=5_000)

    # The validator requires >= 200 ticks per direction; the synthetic
    # capture should comfortably exceed that.
    paths = builder._select_parquets_by_date(bbo_root, lookback_days=2)
    assert len(paths) == 1

    result = builder._analyze_paths(paths, left_lat_ms=350, right_lat_ms=50)
    assert "direction_A" in result and "direction_B" in result
    assert result["n_rows_total"] == 5_000

    issues = builder._validate(result)
    assert issues == []  # synthetic walk should fill both directions

    builder._atomic_write_json(result, out)
    assert out.exists()
    assert not out.with_suffix(out.suffix + ".tmp").exists()  # tmp cleaned

    # Reload via the strategy's JSON loader
    table = MarkoutTable.from_json(out)
    assert table.latency_label == "left=350ms,right=50ms"
    # Direction A 0-1 bps bucket: every tick has edge_A near 0-1, so the
    # bucket should have a non-trivial sample count.
    a0 = table.direction_A[0]
    assert a0.lo == Decimal("0") and a0.hi == Decimal("1")


def test_builder_rejects_empty_window(tmp_path: Path) -> None:
    """With lookback=0, no date partitions qualify → FileNotFoundError."""
    builder = _load_builder()
    bbo_root = tmp_path / "spread_TEST_synth"
    _make_synthetic_capture(bbo_root, n_rows=500)
    with pytest.raises(FileNotFoundError):
        builder._select_parquets_by_date(bbo_root, lookback_days=0)


def test_builder_validation_flags_too_few_samples(tmp_path: Path) -> None:
    """A capture with too few positive-edge ticks should fail validation."""
    builder = _load_builder()
    # Tiny capture: validator requires >=200 ticks/direction
    bbo_root = tmp_path / "spread_TEST_synth"
    _make_synthetic_capture(bbo_root, n_rows=100)
    paths = builder._select_parquets_by_date(bbo_root, lookback_days=2)
    result = builder._analyze_paths(paths, left_lat_ms=350, right_lat_ms=50)
    issues = builder._validate(result)
    assert issues, "validator should refuse a tiny capture"


def test_atomic_write_clean(tmp_path: Path) -> None:
    builder = _load_builder()
    out = tmp_path / "nested" / "dir" / "markout.json"
    builder._atomic_write_json({"hello": "world"}, out)
    assert json.loads(out.read_text()) == {"hello": "world"}
    # The tmp side-file must not survive.
    assert not out.with_suffix(out.suffix + ".tmp").exists()
