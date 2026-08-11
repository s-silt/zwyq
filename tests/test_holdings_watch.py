"""tests for scripts.holdings_watch —— EOD 账户估值端到端。

覆盖: stdout与产物一致、文件名/日期一致、shares×close、
日期不齐不写、缺行情incomplete、不改输入、重复运行确定。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.holdings_watch as hw
from ashare_gauntlet.account_state import AccountFreshnessError, AccountSchemaError


@pytest.fixture(autouse=True)
def _isolate_output_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(hw, "OUT_DIR", str(tmp_path / "data/account_state"))


def _write_cache(root: Path, dates: list[str]) -> None:
    """写入 daily/adj_factor 缓存 parquet。"""
    daily_dir = root / "data/cache/daily"
    adj_dir = root / "data/cache/adj_factor"
    daily_dir.mkdir(parents=True, exist_ok=True)
    adj_dir.mkdir(parents=True, exist_ok=True)

    for d in dates:
        daily = pd.DataFrame({
            "ts_code": ["000001.SZ", "000002.SZ"],
            "close": [12.5, 15.0],
            "low": [12.0, 14.5],
            "pct_chg": [1.2, -0.5],
        })
        daily.to_parquet(daily_dir / f"{d}.parquet", index=False)

        adj = pd.DataFrame({
            "ts_code": ["000001.SZ", "000002.SZ"],
            "adj_factor": [1.0, 1.0],
        })
        adj.to_parquet(adj_dir / f"{d}.parquet", index=False)


def _write_holdings(root: Path, **kw) -> None:
    base = {
        "as_of": "20260807",
        "cash": 50000.0,
        "positions": [
            {
                "ts_code": "000001.SZ",
                "name": "平安银行",
                "industry": "银行",
                "shares": 1000,
                "cost": 12.0,
                "stop": 10.0,
                "mv": 12500.0,
                "bucket": "长线",
            },
            {
                "ts_code": "000002.SZ",
                "name": "万科A",
                "industry": "地产",
                "shares": 500,
                "cost": 14.0,
                "stop": 12.0,
                "mv": 7500.0,
                "bucket": "短线",
            },
        ],
    }
    base.update(kw)
    path = root / "data/holdings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")


def _make_dates(n: int = 21, base: str = "20260807") -> list[str]:
    """生成连续 n 个交易日日期(简单递增,测试用)。"""
    base_int = int(base)
    result: list[str] = []
    day = base_int - n + 1
    for i in range(n):
        d = day + i
        if d % 100 <= 31 and d % 100 >= 1:
            result.append(str(d))
    return result


def test_stdout_output_matches_file(tmp_path: Path, monkeypatch, capsys):
    dates = _make_dates(21)
    _write_cache(tmp_path, dates)
    _write_holdings(tmp_path, as_of=dates[-1])
    # 重定向 CACHE 和 HOLDINGS
    monkeypatch.setattr(hw, "CACHE", str(tmp_path / "data/cache"))
    monkeypatch.setattr(hw, "HOLDINGS", str(tmp_path / "data/holdings.json"))
    # capture stdout
    hw.main([dates[-1], "--stdout-only"])
    stdout = capsys.readouterr().out.strip()
    payload = json.loads(stdout)
    assert payload["schema_version"] == "account_eod.v1"
    assert payload["as_of"] == dates[-1]
    assert payload["account_as_of"] == dates[-1]
    assert payload["source_schema"] == "legacy_unversioned"
    assert payload["account_data_status"] == "complete"
    assert payload["account_freshness"] == "aligned"
    assert payload["data_status"] == "complete"
    assert "valuation" in payload
    assert "positions" in payload

    # Now run without --stdout-only
    hw.main([dates[-1]])
    stdout2 = capsys.readouterr().out.strip()
    payload2 = json.loads(stdout2)

    # Check file was written
    out_path = tmp_path / "data/account_state" / f"{dates[-1]}_account_state.json"
    assert out_path.exists()
    file_content = json.loads(out_path.read_text(encoding="utf-8"))
    assert file_content == payload2

    # stdout from both runs should match (deterministic)
    assert payload == payload2


def test_eod_valuation_shares_times_close(tmp_path: Path, monkeypatch, capsys):
    dates = _make_dates(21)
    _write_cache(tmp_path, dates)
    _write_holdings(tmp_path, as_of=dates[-1])
    monkeypatch.setattr(hw, "CACHE", str(tmp_path / "data/cache"))
    monkeypatch.setattr(hw, "HOLDINGS", str(tmp_path / "data/holdings.json"))

    hw.main([dates[-1], "--stdout-only"])
    stdout = capsys.readouterr().out.strip()
    payload = json.loads(stdout)

    val = payload["valuation"]
    # 000001.SZ: 1000 × 12.5 = 12500; 000002.SZ: 500 × 15.0 = 7500
    assert val["market_value"] == 20000.0
    assert val["total_assets"] == 20000.0 + 50000.0
    assert val["status"] == "complete"
    assert val["valued_position_count"] == 2
    assert val["basis"] == "shares_x_eod_close"

    # Verify positions show close correctly
    codes = {r["ts_code"]: r for r in payload["positions"]}
    assert codes["000001.SZ"]["close"] == 12.5
    assert codes["000002.SZ"]["close"] == 15.0

    # Industry weights
    assert payload.get("industry_weights") is not None
    assert "银行" in payload["industry_weights"]
    assert "地产" in payload["industry_weights"]
    assert payload["industry_weights"]["银行"]["weight"] == pytest.approx(12500 / 70000, abs=1e-6)
    assert payload["industry_weights"]["地产"]["weight"] == pytest.approx(7500 / 70000, abs=1e-6)


def test_missing_close_is_incomplete(tmp_path: Path, monkeypatch, capsys):
    dates = _make_dates(21)
    # 只写入 000001.SZ 的行情
    daily_dir = tmp_path / "data/cache/daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    adj_dir = tmp_path / "data/cache/adj_factor"
    adj_dir.mkdir(parents=True, exist_ok=True)
    for d in dates:
        daily = pd.DataFrame({
            "ts_code": ["000001.SZ"],
            "close": [12.5],
            "low": [12.0],
            "pct_chg": [1.2],
        })
        daily.to_parquet(daily_dir / f"{d}.parquet", index=False)
        adj = pd.DataFrame({
            "ts_code": ["000001.SZ"],
            "adj_factor": [1.0],
        })
        adj.to_parquet(adj_dir / f"{d}.parquet", index=False)

    _write_holdings(tmp_path, as_of=dates[-1])
    monkeypatch.setattr(hw, "CACHE", str(tmp_path / "data/cache"))
    monkeypatch.setattr(hw, "HOLDINGS", str(tmp_path / "data/holdings.json"))

    hw.main([dates[-1], "--stdout-only"])
    stdout = capsys.readouterr().out.strip()
    payload = json.loads(stdout)

    val = payload["valuation"]
    # 000002.SZ 无行情 → incomplete
    assert val["status"] == "incomplete"
    assert payload["data_status"] == "incomplete"
    assert val["market_value"] is None
    assert val["total_assets"] is None
    assert val["valued_position_count"] == 1


def test_repeat_run_deterministic(tmp_path: Path, monkeypatch, capsys):
    dates = _make_dates(21)
    _write_cache(tmp_path, dates)
    _write_holdings(tmp_path, as_of=dates[-1])
    monkeypatch.setattr(hw, "CACHE", str(tmp_path / "data/cache"))
    monkeypatch.setattr(hw, "HOLDINGS", str(tmp_path / "data/holdings.json"))

    hw.main([dates[-1], "--stdout-only"])
    first = capsys.readouterr().out.strip()

    hw.main([dates[-1], "--stdout-only"])
    second = capsys.readouterr().out.strip()

    assert first == second


def test_does_not_modify_holdings(tmp_path: Path, monkeypatch):
    dates = _make_dates(21)
    _write_cache(tmp_path, dates)
    _write_holdings(tmp_path, as_of=dates[-1])
    monkeypatch.setattr(hw, "CACHE", str(tmp_path / "data/cache"))
    monkeypatch.setattr(hw, "HOLDINGS", str(tmp_path / "data/holdings.json"))

    original = (tmp_path / "data/holdings.json").read_text(encoding="utf-8")
    hw.main([dates[-1], "--stdout-only"])
    after = (tmp_path / "data/holdings.json").read_text(encoding="utf-8")
    assert original == after


def test_stdout_only_does_not_write_file(tmp_path: Path, monkeypatch, capsys):
    dates = _make_dates(21)
    _write_cache(tmp_path, dates)
    _write_holdings(tmp_path, as_of=dates[-1])
    monkeypatch.setattr(hw, "CACHE", str(tmp_path / "data/cache"))
    monkeypatch.setattr(hw, "HOLDINGS", str(tmp_path / "data/holdings.json"))

    out_dir = tmp_path / "data/account_state"

    hw.main([dates[-1], "--stdout-only"])
    stdout = capsys.readouterr().out.strip()
    assert len(stdout) > 0
    # 可能没有 account_state 目录,或者不存在文件
    if out_dir.exists():
        files = list(out_dir.glob("*.json"))
        assert not files


def test_output_filename_matches_as_of(tmp_path: Path, monkeypatch, capsys):
    dates = _make_dates(21, base="20260805")
    as_of = dates[-1]
    _write_cache(tmp_path, dates)
    _write_holdings(tmp_path, as_of=as_of)
    monkeypatch.setattr(hw, "CACHE", str(tmp_path / "data/cache"))
    monkeypatch.setattr(hw, "HOLDINGS", str(tmp_path / "data/holdings.json"))

    hw.main([as_of])
    capsys.readouterr()

    out_file = tmp_path / "data/account_state" / f"{as_of}_account_state.json"
    assert out_file.exists()


def test_no_positions(tmp_path: Path, monkeypatch, capsys):
    dates = _make_dates(21)
    _write_cache(tmp_path, dates)
    _write_holdings(tmp_path, as_of=dates[-1], positions=[])
    monkeypatch.setattr(hw, "CACHE", str(tmp_path / "data/cache"))
    monkeypatch.setattr(hw, "HOLDINGS", str(tmp_path / "data/holdings.json"))

    hw.main([dates[-1], "--stdout-only"])
    stdout = capsys.readouterr().out.strip()
    payload = json.loads(stdout)
    assert payload["valuation"]["position_count"] == 0
    assert payload["positions"] == []


@pytest.mark.parametrize(("account_as_of", "status"), [
    (None, "ACCOUNT_AS_OF_MISSING"),
    ("20260806", "ACCOUNT_AS_OF_STALE"),
    ("20260808", "ACCOUNT_AS_OF_FUTURE"),
    ("20260230", "ACCOUNT_AS_OF_INVALID"),
])
def test_account_freshness_fails_before_output(
        tmp_path: Path, monkeypatch, capsys, account_as_of, status):
    dates = _make_dates(21)
    _write_cache(tmp_path, dates)
    _write_holdings(tmp_path, as_of=account_as_of)
    monkeypatch.setattr(hw, "CACHE", str(tmp_path / "data/cache"))
    monkeypatch.setattr(hw, "HOLDINGS", str(tmp_path / "data/holdings.json"))

    with pytest.raises(AccountFreshnessError, match=status):
        hw.main([dates[-1]])
    assert capsys.readouterr().out == ""
    assert not list(Path(hw.OUT_DIR).glob("*"))


def test_incomplete_account_schema_fails_before_output(tmp_path: Path, monkeypatch, capsys):
    dates = _make_dates(21)
    _write_cache(tmp_path, dates)
    _write_holdings(tmp_path, as_of=dates[-1], cash=None)
    monkeypatch.setattr(hw, "CACHE", str(tmp_path / "data/cache"))
    monkeypatch.setattr(hw, "HOLDINGS", str(tmp_path / "data/holdings.json"))

    with pytest.raises(AccountSchemaError, match="missing=.*cash"):
        hw.main([dates[-1]])
    assert capsys.readouterr().out == ""
    assert not list(Path(hw.OUT_DIR).glob("*"))


def test_atomic_replace_failure_preserves_existing_snapshot(
        tmp_path: Path, monkeypatch, capsys):
    dates = _make_dates(21)
    _write_cache(tmp_path, dates)
    _write_holdings(tmp_path, as_of=dates[-1])
    monkeypatch.setattr(hw, "CACHE", str(tmp_path / "data/cache"))
    monkeypatch.setattr(hw, "HOLDINGS", str(tmp_path / "data/holdings.json"))
    out_dir = Path(hw.OUT_DIR)
    out_dir.mkdir(parents=True)
    out_path = out_dir / f"{dates[-1]}_account_state.json"
    old = '{"old": true}'
    out_path.write_text(old, encoding="utf-8")

    def fail_replace(*args, **kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(hw.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        hw.main([dates[-1]])
    assert capsys.readouterr().out == ""
    assert out_path.read_text(encoding="utf-8") == old
    assert not list(out_dir.glob(".tmp_account_state_*"))


def test_stdout_only_does_not_create_output_directory(tmp_path: Path, monkeypatch, capsys):
    dates = _make_dates(21)
    _write_cache(tmp_path, dates)
    _write_holdings(tmp_path, as_of=dates[-1])
    monkeypatch.setattr(hw, "CACHE", str(tmp_path / "data/cache"))
    monkeypatch.setattr(hw, "HOLDINGS", str(tmp_path / "data/holdings.json"))
    out_dir = Path(hw.OUT_DIR)

    hw.main([dates[-1], "--stdout-only"])
    assert capsys.readouterr().out
    assert not out_dir.exists()
