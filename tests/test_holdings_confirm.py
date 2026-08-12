"""holdings_confirm:人工确认 as_of 推进的安全性——只动一个字段,其余逐项不变。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.holdings_confirm import confirm_as_of


def _holdings(as_of: str = "20260805") -> dict:
    return {
        "as_of": as_of,
        "source": "manual",
        "note": "合成测试数据",
        "positions": [
            {"ts_code": "600001.SH", "name": "甲", "shares": 100, "cost": 10.0,
             "mv": 1000.0, "industry": "测试", "last": 10.0, "stop": 9.0},
            {"ts_code": "000002.SZ", "name": "乙", "shares": 200, "cost": 5.5,
             "mv": 1100.0, "industry": "测试", "last": 5.5},
        ],
        "cash": 8888.88,
        "conditional_orders": "legacy free text",
        "closed": [{"ts_code": "600999.SH", "reason": "系统规则(C2)"}],
    }


def _fixture(tmp_path: Path, *, as_of: str = "20260805",
             trade_days: tuple[str, ...] = ("20260805", "20260811")) -> Path:
    holdings_path = tmp_path / "holdings.json"
    holdings_path.write_text(json.dumps(_holdings(as_of), ensure_ascii=False),
                             encoding="utf-8")
    for day in trade_days:
        p = tmp_path / "cache" / "daily" / f"{day}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"ts_code": "600001.SH", "trade_date": day}]).to_parquet(p)
    return holdings_path


def test_confirm_advances_as_of_and_touches_nothing_else(tmp_path):
    path = _fixture(tmp_path)
    result = confirm_as_of("20260811", holdings_path=str(path),
                           cache_dir=str(tmp_path / "cache"))
    assert result == {"changed": True, "as_of": "20260811",
                      "previous_as_of": "20260805", "positions": 2}
    after = json.load(path.open(encoding="utf-8"))
    expected = _holdings("20260811")
    assert after == expected          # 除 as_of 外逐字段语义相等
    assert after["cash"] == 8888.88
    assert after["conditional_orders"] == "legacy free text"


def test_confirm_is_noop_when_already_current(tmp_path):
    path = _fixture(tmp_path, as_of="20260811")
    before = path.read_text(encoding="utf-8")
    result = confirm_as_of("20260811", holdings_path=str(path),
                           cache_dir=str(tmp_path / "cache"))
    assert result["changed"] is False
    assert path.read_text(encoding="utf-8") == before   # no-op 连格式都不动


def test_confirm_rejects_backwards_date(tmp_path):
    path = _fixture(tmp_path, as_of="20260811")
    with pytest.raises(SystemExit, match="倒退"):
        confirm_as_of("20260805", holdings_path=str(path),
                      cache_dir=str(tmp_path / "cache"))


def test_confirm_rejects_unknown_trade_day(tmp_path):
    path = _fixture(tmp_path)   # 分区只有 0805/0811
    before = path.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="交易日"):
        confirm_as_of("20260809", holdings_path=str(path),   # 周日,无分区
                      cache_dir=str(tmp_path / "cache"))
    assert path.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("bad", ["2026-08-11", "20261301", "近日", ""])
def test_confirm_rejects_malformed_date(tmp_path, bad):
    path = _fixture(tmp_path)
    with pytest.raises(SystemExit):
        confirm_as_of(bad, holdings_path=str(path),
                      cache_dir=str(tmp_path / "cache"))


def test_confirm_rejects_broken_holdings_without_write(tmp_path):
    path = _fixture(tmp_path)
    path.write_text('{"as_of": "20260805"}', encoding="utf-8")   # 缺 positions
    with pytest.raises(SystemExit, match="结构非法"):
        confirm_as_of("20260811", holdings_path=str(path),
                      cache_dir=str(tmp_path / "cache"))
    assert json.load(path.open(encoding="utf-8")) == {"as_of": "20260805"}


def test_write_failure_leaves_original_intact(tmp_path, monkeypatch):
    path = _fixture(tmp_path)
    before = path.read_text(encoding="utf-8")
    import scripts.holdings_confirm as hc

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(hc.os, "replace", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        confirm_as_of("20260811", holdings_path=str(path),
                      cache_dir=str(tmp_path / "cache"))
    assert path.read_text(encoding="utf-8") == before
    assert not list(tmp_path.glob(".tmp_holdings_*"))
