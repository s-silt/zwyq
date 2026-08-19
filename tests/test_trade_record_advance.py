"""trade_record --advance-as-of:成交日晚于账户 as_of 时,在同一原子写内前进式推进 as_of。

契约:前进式(不倒退)、须为交易日、与落账同一次 holdings 原子写;未加 flag 时保持旧行为
(date≠as_of 仍报错,要求先跑 holdings_confirm)。所有既有护栏不变。
"""
from __future__ import annotations

import json

import pytest

from scripts import trade_record as tr

DAYS = ["20260810", "20260811", "20260814", "20260817"]


def _cache(tmp_path, days=DAYS):
    daily = tmp_path / "cache" / "daily"
    daily.mkdir(parents=True)
    for d in days:
        (daily / f"{d}.parquet").write_bytes(b"")
    return str(tmp_path / "cache")


def _setup_sell(tmp_path, *, as_of="20260814"):
    holdings = {"as_of": as_of, "cash": 318.0, "closed": [], "positions": [
        {"ts_code": "600001.SH", "name": "合成电气", "industry": "电气设备",
         "shares": 1200, "cost": 24.0, "last": 26.5, "mv": 31800.0,
         "stop": None, "bucket": "long", "entry_date": "20260810",
         "bucket_note": "", "tag": "", "theme": "", "watch": False}]}
    hp = tmp_path / "holdings.json"
    jp = tmp_path / "journal.json"
    hp.write_text(json.dumps(holdings, ensure_ascii=False), encoding="utf-8")
    jp.write_text(json.dumps({"trades": []}), encoding="utf-8")
    return str(hp), str(jp), _cache(tmp_path)


def _sell(hp, jp, cache, **kw):
    args = dict(date="20260817", net=31700.0, exit_px=26.5, pnl_pct=4.2,
                reason_key="C2", holdings_path=hp, journal_path=jp, cache_dir=cache)
    args.update(kw)
    return tr.record_sell("600001.SH", **args)


def _read(path):
    return json.loads(open(path, encoding="utf-8").read())


def test_advance_moves_date_and_records_in_one_write(tmp_path):
    hp, jp, cache = _setup_sell(tmp_path, as_of="20260814")
    _sell(hp, jp, cache, date="20260817", advance_as_of=True)   # 20260817 > as_of 20260814
    holdings = _read(hp)
    assert holdings["as_of"] == "20260817"          # 前进式推进
    assert holdings["positions"] == []              # 同一次写内完成清仓
    assert holdings["cash"] == pytest.approx(318.0 + 31700.0)
    assert _read(jp)["trades"][-1]["exit_date"] == "20260817"


def test_without_flag_forward_date_still_errors(tmp_path):
    hp, jp, cache = _setup_sell(tmp_path, as_of="20260814")
    before = open(hp, encoding="utf-8").read()
    with pytest.raises(SystemExit, match="as_of"):
        _sell(hp, jp, cache, date="20260817", advance_as_of=False)
    assert open(hp, encoding="utf-8").read() == before   # 零副作用,旧行为不变


def test_backward_date_rejected_even_with_flag(tmp_path):
    hp, jp, cache = _setup_sell(tmp_path, as_of="20260817")
    before = open(hp, encoding="utf-8").read()
    with pytest.raises(SystemExit, match="不可倒退"):
        _sell(hp, jp, cache, date="20260814", advance_as_of=True)   # 倒退即便有 flag 也拒
    assert open(hp, encoding="utf-8").read() == before


def test_equal_date_with_flag_is_noop(tmp_path):
    hp, jp, cache = _setup_sell(tmp_path, as_of="20260817")
    _sell(hp, jp, cache, date="20260817", advance_as_of=True)
    holdings = _read(hp)
    assert holdings["as_of"] == "20260817" and holdings["positions"] == []


def test_buy_advance_as_of(tmp_path):
    holdings = {"as_of": "20260814", "cash": 100000.0, "closed": [], "positions": []}
    hp = tmp_path / "holdings.json"
    hp.write_text(json.dumps(holdings, ensure_ascii=False), encoding="utf-8")
    cache = _cache(tmp_path)
    holdscore = tmp_path / "holdscore"
    holdscore.mkdir(parents=True)
    (holdscore / "20260817_factor.json").write_text(json.dumps(
        [{"ts_code": "600002.SH", "name": "新票", "industry": "电气设备", "last": 9.9}]),
        encoding="utf-8")
    # gross=700×9.87=6909;net∈[6909, 7254.45];cash 充足
    tr.record_buy("600002.SH", date="20260817", shares=700, cost_px=9.87, net=6915.0,
                  bucket="long", advance_as_of=True, holdings_path=str(hp),
                  holdscore_dir=str(holdscore), cache_dir=cache)
    holdings = _read(str(hp))
    assert holdings["as_of"] == "20260817"
    assert [p["ts_code"] for p in holdings["positions"]] == ["600002.SH"]
