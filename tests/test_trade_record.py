"""trade_record:人工成交落账契约测试(合成值、临时目录,绝不碰真实账户)。

锁的语义(codex 复审后):pnl_pct 显式抄券商值不推导;entry_date 必填;
hold_days=交易日口径;方向性净额护栏;--date 必须=as_of 且为交易日;
journal 行恰合 trade_journal._FIELDS 契约;一切校验失败=文件零字节改动。
"""
from __future__ import annotations

import json

import pytest

from scripts import trade_record as tr


def _write_cache(tmp_path, days):
    daily = tmp_path / "cache" / "daily"
    daily.mkdir(parents=True)
    for d in days:
        (daily / f"{d}.parquet").write_bytes(b"")
    return str(tmp_path / "cache")


def _setup(tmp_path, *, entry_date="20260810"):
    holdings = {
        "as_of": "20260817", "cash": 318.0, "closed": [],
        "positions": [
            {"ts_code": "600001.SH", "name": "合成电气", "industry": "电气设备",
             "shares": 1200, "cost": 24.0, "last": 26.5, "mv": 31800.0,
             "stop": None, "bucket": "long", "entry_date": entry_date,
             "bucket_note": "", "tag": "", "theme": "", "watch": False},
        ],
    }
    hp = tmp_path / "holdings.json"
    jp = tmp_path / "journal.json"
    hp.write_text(json.dumps(holdings, ensure_ascii=False), encoding="utf-8")
    jp.write_text(json.dumps({"trades": []}), encoding="utf-8")
    cache = _write_cache(tmp_path, ["20260810", "20260811", "20260814", "20260817"])
    return str(hp), str(jp), cache


def _sell(hp, jp, cache, **kw):
    args = dict(date="20260817", net=31700.0, exit_px=26.5, pnl_pct=4.2,
                reason_key="C2", holdings_path=hp, journal_path=jp, cache_dir=cache)
    args.update(kw)
    return tr.record_sell("600001.SH", **args)


def test_sell_journal_contract_and_holdings(tmp_path):
    hp, jp, cache = _setup(tmp_path)
    result = _sell(hp, jp, cache)
    holdings = json.loads((tmp_path / "holdings.json").read_text(encoding="utf-8"))
    trade = json.loads((tmp_path / "journal.json").read_text(encoding="utf-8"))["trades"][-1]
    assert holdings["positions"] == []
    assert holdings["cash"] == pytest.approx(318.0 + 31700.0)
    assert trade["pnl_pct"] == 4.2                      # 券商值原样,不从价差推导
    assert trade["bucket"] == "长线"                     # short/long → 中文契约
    assert trade["entry_date"] == "20260810"
    assert trade["hold_days"] == 3                      # 交易日口径:0811,0814,0817
    assert trade["approx"] is False
    from scripts.trade_journal import _FIELDS
    assert set(trade) == set(_FIELDS)                   # 字段集恰合 journal 契约
    assert result["positions"] == 0


def test_sell_requires_entry_date_when_position_lacks_it(tmp_path):
    hp, jp, cache = _setup(tmp_path, entry_date=None)
    holdings = json.loads((tmp_path / "holdings.json").read_text(encoding="utf-8"))
    del holdings["positions"][0]["entry_date"]
    (tmp_path / "holdings.json").write_text(json.dumps(holdings), encoding="utf-8")
    before = (tmp_path / "holdings.json").read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="entry_date"):
        _sell(hp, jp, cache)
    assert (tmp_path / "holdings.json").read_text(encoding="utf-8") == before  # 零副作用
    _sell(hp, jp, cache, entry_date="20260811", approx=True)
    trade = json.loads((tmp_path / "journal.json").read_text(encoding="utf-8"))["trades"][-1]
    assert trade["entry_date"] == "20260811" and trade["approx"] is True


def test_sell_directional_net_guardrail(tmp_path):
    hp, jp, cache = _setup(tmp_path)
    with pytest.raises(SystemExit, match="应在"):
        _sell(hp, jp, cache, net=32000.0)               # 净回款 > gross:方向非法
    with pytest.raises(SystemExit, match="应在"):
        _sell(hp, jp, cache, net=29000.0)               # 低于 gross×0.95:抄错


def test_sell_date_gates(tmp_path):
    hp, jp, cache = _setup(tmp_path)
    with pytest.raises(SystemExit, match="不是本地缓存已知交易日"):
        _sell(hp, jp, cache, date="20260816")           # 周日
    with pytest.raises(SystemExit, match="as_of"):
        _sell(hp, jp, cache, date="20260814")           # 交易日但 ≠ as_of


def test_sell_duplicate_and_nan_source_rejected(tmp_path):
    hp, jp, cache = _setup(tmp_path)
    _sell(hp, jp, cache)
    holdings = json.loads((tmp_path / "holdings.json").read_text(encoding="utf-8"))
    holdings["positions"] = [{"ts_code": "600001.SH", "name": "合成电气", "shares": 1200,
                              "cost": 24.0, "bucket": "long", "entry_date": "20260810"}]
    (tmp_path / "holdings.json").write_text(json.dumps(holdings), encoding="utf-8")
    with pytest.raises(SystemExit, match="重复落账"):
        _sell(hp, jp, cache)
    (tmp_path / "holdings.json").write_text(
        '{"as_of": "20260817", "cash": NaN, "positions": []}', encoding="utf-8")
    with pytest.raises(SystemExit, match="NaN"):
        _sell(hp, jp, cache)


def test_sell_lock_blocks_concurrent_run(tmp_path):
    hp, jp, cache = _setup(tmp_path)
    (tmp_path / "holdings.lock").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="账本锁"):
        _sell(hp, jp, cache)


def test_shared_lock_blocks_holdings_confirm_and_journal_add(tmp_path, monkeypatch):
    """codex 二轮 P0:锁必须被全部写入方遵守——holdings_confirm 与 trade_journal --add
    在锁被持有时同样必须拒绝写入。"""
    hp, jp, cache = _setup(tmp_path)
    (tmp_path / "holdings.lock").write_text("", encoding="utf-8")
    from scripts.holdings_confirm import confirm_as_of
    with pytest.raises(SystemExit, match="账本锁"):
        confirm_as_of("20260817", holdings_path=hp, cache_dir=cache)
    from scripts import trade_journal as tj
    with pytest.raises(SystemExit, match="账本锁"):
        tj.main(["--add", "code=600009.SH,bucket=短线,entry_date=20260817,"
                          "entry_px=10.0,shares=100"], path=jp)


def test_hold_days_rejects_calendar_gap(tmp_path):
    """codex 二轮:daily 分区缺段会静默少算 hold_days——超 15 天缺口必须 fail-loud。"""
    hp, jp, _ = _setup(tmp_path)
    cache = _write_cache(tmp_path / "gappy", ["20260601", "20260817"])   # 77 天缺口
    holdings = json.loads((tmp_path / "holdings.json").read_text(encoding="utf-8"))
    holdings["positions"][0]["entry_date"] = "20260601"
    (tmp_path / "holdings.json").write_text(json.dumps(holdings), encoding="utf-8")
    with pytest.raises(SystemExit, match="缺口"):
        _sell(hp, jp, cache)


def test_sell_does_not_require_conditional_orders(tmp_path):
    hp, jp, cache = _setup(tmp_path)
    assert _sell(hp, jp, cache)["positions"] == 0


def test_sell_records_when_conditional_orders_unsynced(tmp_path):
    """条件单不是落账门禁:残留 active SELL 仍按成交事实落账,不改条件单。"""
    hp, jp, cache = _setup(tmp_path)
    holdings = json.loads(open(hp, encoding="utf-8").read())
    leftover = {
        "schema_version": 2,
        "orders": [{
            "order_id": "ord-001", "ts_code": "600001.SH", "side": "SELL",
            "condition": {"field": "close", "operator": "<="},
            "price": 20.9, "shares": 1200,
            "valid_from": "20260801", "valid_until": "20261231", "status": "active",
        }],
    }
    holdings["conditional_orders"] = leftover
    open(hp, "w", encoding="utf-8").write(json.dumps(holdings, ensure_ascii=False))
    assert _sell(hp, jp, cache)["positions"] == 0
    after = json.loads(open(hp, encoding="utf-8").read())
    assert after["conditional_orders"] == leftover


def test_sell_second_write_failure_rolls_back_bytes_and_retry_succeeds(
        tmp_path, monkeypatch):
    hp, jp, cache = _setup(tmp_path)
    from pathlib import Path
    h0, j0 = Path(hp).read_bytes(), Path(jp).read_bytes()
    real_replace = tr.os.replace

    def fail_holdings(src, dst):
        if str(dst).endswith("holdings.json"):
            raise OSError("disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(tr.os, "replace", fail_holdings)
    with pytest.raises(SystemExit, match="回滚到写入前字节"):
        _sell(hp, jp, cache)
    assert Path(hp).read_bytes() == h0
    assert Path(jp).read_bytes() == j0
    monkeypatch.setattr(tr.os, "replace", real_replace)
    result = _sell(hp, jp, cache)
    assert result["positions"] == 0
    trades = json.loads(Path(jp).read_text(encoding="utf-8"))["trades"]
    assert len(trades) == 1


def test_sell_second_write_and_rollback_failure_blocks_blind_retry(
        tmp_path, monkeypatch):
    hp, jp, cache = _setup(tmp_path)
    from pathlib import Path
    h0, j0 = Path(hp).read_bytes(), Path(jp).read_bytes()
    real_replace = tr.os.replace
    journal_replaces = {"n": 0}

    def fail_holdings_and_rollback(src, dst):
        dst_s = str(dst)
        if dst_s.endswith("holdings.json"):
            raise OSError("disk full")
        if dst_s.endswith("journal.json"):
            journal_replaces["n"] += 1
            if journal_replaces["n"] > 1:
                raise OSError("rollback disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(tr.os, "replace", fail_holdings_and_rollback)
    with pytest.raises(SystemExit, match="回滚失败"):
        _sell(hp, jp, cache)
    assert Path(hp).read_bytes() == h0
    assert Path(jp).read_bytes() != j0
    monkeypatch.setattr(tr.os, "replace", real_replace)
    with pytest.raises(SystemExit, match="重复落账"):
        _sell(hp, jp, cache)
    assert Path(hp).read_bytes() == h0
    assert len(json.loads(Path(jp).read_text(encoding="utf-8"))["trades"]) == 1


def test_sell_second_write_holdings_unread_blocks_blind_retry(tmp_path, monkeypatch):
    """except 内 holdings read_bytes 失败不得掩盖恢复提示,禁止盲目重试。"""
    hp, jp, cache = _setup(tmp_path)
    from pathlib import Path
    real_replace = tr.os.replace
    real_read_bytes = tr.Path.read_bytes
    after_holdings_write = {"on": False}

    def fail_holdings_replace(src, dst):
        if str(dst).endswith("holdings.json"):
            after_holdings_write["on"] = True
            raise OSError("disk full")
        return real_replace(src, dst)

    def fail_holdings_read(self):
        if after_holdings_write["on"] and self.name == "holdings.json":
            raise OSError("holdings unreadable")
        return real_read_bytes(self)

    monkeypatch.setattr(tr.os, "replace", fail_holdings_replace)
    monkeypatch.setattr(tr.Path, "read_bytes", fail_holdings_read)
    with pytest.raises(tr.TradeRecordError, match="holdings 状态未知") as ei:
        _sell(hp, jp, cache)
    err = ei.value
    msg = str(err)
    assert "不要盲目重跑" in msg
    assert "可重试" not in msg
    assert "恢复指引" in msg
    assert "disk full" in msg
    assert isinstance(err.__cause__, OSError)
    assert "holdings unreadable" in str(err.__cause__)
    assert err.__context__ is not None
    assert "disk full" in str(err.__context__)


def _buy_env(tmp_path):
    hp, jp, cache = _setup(tmp_path)
    holdings = json.loads((tmp_path / "holdings.json").read_text(encoding="utf-8"))
    holdings["cash"] = 20000.0
    (tmp_path / "holdings.json").write_text(json.dumps(holdings), encoding="utf-8")
    snap = tmp_path / "holdscore"
    snap.mkdir()
    (snap / "20260817_factor.json").write_text(json.dumps([
        {"ts_code": "600002.SH", "name": "合成百货", "industry": "百货", "last": 9.9}]),
        encoding="utf-8")
    return hp, str(snap), cache


def test_buy_writes_entry_date_and_cash(tmp_path):
    hp, snap, cache = _buy_env(tmp_path)
    tr.record_buy("600002.SH", date="20260817", shares=700, cost_px=9.87, net=6915.0,
                  holdings_path=hp, holdscore_dir=snap, cache_dir=cache)
    holdings = json.loads((tmp_path / "holdings.json").read_text(encoding="utf-8"))
    added = [p for p in holdings["positions"] if p["ts_code"] == "600002.SH"][0]
    assert added["entry_date"] == "20260817"            # 卖出侧免 --entry-date 的闭环
    assert added["industry"] == "百货" and added["mv"] == pytest.approx(700 * 9.9)
    assert holdings["cash"] == pytest.approx(20000.0 - 6915.0)


def test_buy_directional_guardrail_and_suspended_snapshot(tmp_path):
    hp, snap, cache = _buy_env(tmp_path)
    with pytest.raises(SystemExit, match="应在"):
        tr.record_buy("600002.SH", date="20260817", shares=700, cost_px=9.87, net=6800.0,
                      holdings_path=hp, holdscore_dir=snap, cache_dir=cache)  # 实扣 < gross
    (tmp_path / "holdscore" / "20260817_factor.json").write_text(json.dumps([
        {"ts_code": "600002.SH", "name": "合成百货", "industry": "百货", "last": None}]),
        encoding="utf-8")
    with pytest.raises(SystemExit, match="非正有限数"):
        tr.record_buy("600002.SH", date="20260817", shares=700, cost_px=9.87, net=6915.0,
                      holdings_path=hp, holdscore_dir=snap, cache_dir=cache)


def test_buy_rejects_overdraft_odd_lot_existing(tmp_path):
    hp, snap, cache = _buy_env(tmp_path)
    with pytest.raises(SystemExit, match="超过现金"):
        tr.record_buy("600002.SH", date="20260817", shares=2100, cost_px=9.87, net=20727.0,
                      holdings_path=hp, holdscore_dir=snap, cache_dir=cache)
    with pytest.raises(SystemExit, match="100 的倍数"):
        tr.record_buy("600002.SH", date="20260817", shares=730, cost_px=9.87, net=7205.0,
                      holdings_path=hp, holdscore_dir=snap, cache_dir=cache)
    with pytest.raises(SystemExit, match="已在持仓"):
        tr.record_buy("600001.SH", date="20260817", shares=100, cost_px=24.0, net=2400.0,
                      holdings_path=hp, holdscore_dir=snap, cache_dir=cache)


def test_validated_output_passes_buy_list_contract(tmp_path):
    """产物契约测试(codex P2):落账后的 holdings 必须能过 buy_list.validate_holdings。"""
    hp, snap, cache = _buy_env(tmp_path)
    tr.record_buy("600002.SH", date="20260817", shares=700, cost_px=9.87, net=6915.0,
                  holdings_path=hp, holdscore_dir=snap, cache_dir=cache)
    from scripts.buy_list import validate_holdings
    holdings = json.loads((tmp_path / "holdings.json").read_text(encoding="utf-8"))
    validate_holdings(holdings)   # 不抛=契约兼容
