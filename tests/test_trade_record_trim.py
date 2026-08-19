"""record_trim 部分减仓落账(+25% 减半锁利工具化):账务口径 + 护栏。

口径契约:journal 记**卖出部分**为一笔完整平仓行;holdings 该仓 shares 减少、
entry_date/cost/bucket/stop 不动;cash += net;卖光必须走 --sell(写 closed 段)。
"""
from __future__ import annotations

import json

import pytest

from scripts import trade_record as tr

DAYS = ["20260810", "20260811", "20260814", "20260817"]


def _setup(tmp_path, *, shares=1200, as_of="20260817"):
    holdings = {"as_of": as_of, "cash": 318.0, "closed": [], "positions": [
        {"ts_code": "600001.SH", "name": "合成电气", "industry": "电气设备",
         "shares": shares, "cost": 24.0, "last": 26.5, "mv": shares * 26.5,
         "stop": 20.9, "bucket": "long", "entry_date": "20260810",
         "bucket_note": "", "tag": "", "theme": "", "watch": False}]}
    hp = tmp_path / "holdings.json"
    jp = tmp_path / "journal.json"
    hp.write_text(json.dumps(holdings, ensure_ascii=False), encoding="utf-8")
    jp.write_text(json.dumps({"trades": []}), encoding="utf-8")
    daily = tmp_path / "cache" / "daily"
    daily.mkdir(parents=True)
    for d in DAYS:
        (daily / f"{d}.parquet").write_bytes(b"")
    return str(hp), str(jp), str(tmp_path / "cache")


def _trim(hp, jp, cache, **kw):
    args = dict(date="20260817", shares=600, net=15850.0, exit_px=26.5, pnl_pct=10.4,
                reason_key="MANUAL", holdings_path=hp, journal_path=jp, cache_dir=cache)
    args.update(kw)
    return tr.record_trim("600001.SH", **args)


def _read(p):
    return json.loads(open(p, encoding="utf-8").read())


def test_trim_halves_position_and_records_closed_part(tmp_path):
    hp, jp, cache = _setup(tmp_path)
    result = _trim(hp, jp, cache)                      # 1200 → 卖 600(减半锁利)
    pos = _read(hp)["positions"][0]
    trade = _read(jp)["trades"][-1]
    assert pos["shares"] == 600 and result["remaining"] == 600
    assert pos["mv"] == pytest.approx(600 * 26.5)      # mv 随剩余股数重算
    # 剩余仓的成本基础与纪律参数一律不动
    assert pos["cost"] == 24.0 and pos["entry_date"] == "20260810"
    assert pos["bucket"] == "long" and pos["stop"] == 20.9
    assert _read(hp)["cash"] == pytest.approx(318.0 + 15850.0)
    # journal 记卖出那部分为一笔完整平仓(统计口径正确)
    assert trade["shares"] == 600 and trade["exit_date"] == "20260817"
    assert trade["pnl_pct"] == 10.4 and trade["entry_px"] == 24.0
    assert trade["bucket"] == "长线" and trade["hold_days"] == 3
    from scripts.trade_journal import _FIELDS
    assert set(trade) == set(_FIELDS)
    # 部分减仓不写 closed 段(仓还在)
    assert _read(hp)["closed"] == []


def test_trim_full_position_rejected_use_sell(tmp_path):
    hp, jp, cache = _setup(tmp_path)
    before = open(hp, encoding="utf-8").read()
    with pytest.raises(SystemExit, match="整仓清盘请用"):
        _trim(hp, jp, cache, shares=1200, net=31700.0)
    assert open(hp, encoding="utf-8").read() == before          # 零副作用


def test_trim_more_than_held_rejected(tmp_path):
    hp, jp, cache = _setup(tmp_path)
    with pytest.raises(SystemExit, match="> 持仓"):
        _trim(hp, jp, cache, shares=1300, net=34000.0)


def test_trim_directional_net_guardrail(tmp_path):
    hp, jp, cache = _setup(tmp_path)
    with pytest.raises(SystemExit, match="应在"):
        _trim(hp, jp, cache, net=16500.0)      # 净回款 > gross(600×26.5=15900)
    with pytest.raises(SystemExit, match="应在"):
        _trim(hp, jp, cache, net=14000.0)      # 低于 gross×0.95


def test_trim_lot_and_date_gates(tmp_path):
    hp, jp, cache = _setup(tmp_path)
    with pytest.raises(SystemExit, match="均非 100 倍数"):
        _trim(hp, jp, cache, shares=650, net=17100.0)   # 650 卖 + 550 剩,两边都零散
    with pytest.raises(SystemExit, match="不是本地缓存已知交易日"):
        _trim(hp, jp, cache, date="20260816")           # 周日
    with pytest.raises(SystemExit, match="不可倒退"):
        _trim(hp, jp, cache, date="20260814")           # 早于 as_of=20260817


def test_trim_forward_date_without_flag_errors(tmp_path):
    # codex 测试盲区:日期前进但未加 --advance-as-of(与"倒退"是不同分支)
    hp, jp, cache = _setup(tmp_path, as_of="20260814")
    before = open(hp, encoding="utf-8").read()
    with pytest.raises(SystemExit, match="≠ 账户 as_of"):
        _trim(hp, jp, cache, date="20260817")
    assert open(hp, encoding="utf-8").read() == before


def test_trim_duplicate_same_day_rejected(tmp_path):
    # 1800 股:首次减 600 后仍剩 1200,第二次 600 仍属部分减仓 → 撞的必须是重复守卫
    hp, jp, cache = _setup(tmp_path, shares=1800)
    _trim(hp, jp, cache)
    with pytest.raises(SystemExit, match="重复落账"):
        _trim(hp, jp, cache)


def test_trim_rejects_when_mv_cannot_be_recomputed(tmp_path):
    """codex P1-1:last 非正有限数时 mv 无法与 shares 同步 → fail-loud,不留陈旧 mv。

    静默保留整仓旧 mv 会让 buy_list 的 account_value 虚高一个整仓、行业权重被稀释。
    """
    for i, bad in enumerate(("26.5", None, 0, -1.0, float("nan"))):
        sub = tmp_path / f"case{i}"
        sub.mkdir()
        hp, jp, cache = _setup(sub)
        h = _read(hp)
        h["positions"][0]["last"] = bad
        open(hp, "w", encoding="utf-8").write(json.dumps(h, ensure_ascii=False))
        before = open(hp, encoding="utf-8").read()
        # NaN 由更早的 _load_strict 非标准常量守卫拦下,其余由 last 前置校验拦下
        with pytest.raises(SystemExit, match="last|非标准 JSON 常量"):
            _trim(hp, jp, cache)
        assert open(hp, encoding="utf-8").read() == before
        assert _read(jp)["trades"] == []          # journal 侧亦零副作用


def test_trim_rejects_stale_sell_conditional_order(tmp_path):
    """codex P1-2:减仓后 active SELL 条件单股数 > 剩余持仓 = 账本自相矛盾。"""
    hp, jp, cache = _setup(tmp_path)
    h = _read(hp)
    h["conditional_orders"] = [
        {"ts_code": "600001.SH", "side": "SELL", "shares": 1200,
         "status": "active", "trigger": 20.9}]
    open(hp, "w", encoding="utf-8").write(json.dumps(h, ensure_ascii=False))
    before = open(hp, encoding="utf-8").read()
    with pytest.raises(SystemExit, match="条件单"):
        _trim(hp, jp, cache)                      # 卖 600 后剩 600 < 挂单 1200
    assert open(hp, encoding="utf-8").read() == before
    # 挂单股数 ≤ 剩余 → 放行
    h["conditional_orders"][0]["shares"] = 600
    open(hp, "w", encoding="utf-8").write(json.dumps(h, ensure_ascii=False))
    assert _trim(hp, jp, cache)["remaining"] == 600


def test_same_day_trim_then_sell_full_sequence(tmp_path):
    """codex P1-4 + 盲区8:同日"上午减半锁利 → 下午跌破止损清仓"必须全程走工具。"""
    hp, jp, cache = _setup(tmp_path)              # 1200 股 @24
    _trim(hp, jp, cache, shares=600, net=15850.0, exit_px=26.5, pnl_pct=10.4)
    tr.record_sell("600001.SH", date="20260817", net=12400.0, exit_px=20.9,
                   pnl_pct=-12.9, reason_key="STOP", holdings_path=hp,
                   journal_path=jp, cache_dir=cache)
    holdings, trades = _read(hp), _read(jp)["trades"]
    assert holdings["positions"] == []                          # 已清仓
    assert len(holdings["closed"]) == 1                         # closed 只由 sell 写
    assert holdings["cash"] == pytest.approx(318.0 + 15850.0 + 12400.0)
    assert [t["shares"] for t in trades] == [600, 600]          # 合计=原仓,无重复计入
    assert all(t["entry_date"] == "20260810" and t["entry_px"] == 24.0 for t in trades)
    # 按笔等权美化(+10.4/−12.9 → −1.25),shares 加权同为 −1.25(两腿等量);
    # 关键是加权字段存在且可用于不等量场景
    from scripts.trade_journal import stats
    s = stats(trades)
    assert s["n"] == 2 and s["n_w"] == 2
    assert s["expectancy_w"] == pytest.approx((10.4 * 600 + -12.9 * 600) / 1200)


def test_stats_weighted_exposes_partial_exit_distortion():
    """codex P1-3:小额锁利腿 + 大额亏损腿——按笔等权 vs shares 加权must不同。"""
    from scripts.trade_journal import stats
    trades = [
        {"code": "1.SH", "bucket": "长线", "exit_date": "20260817", "pnl_pct": 30.0,
         "shares": 100, "hold_days": 5},
        {"code": "1.SH", "bucket": "长线", "exit_date": "20260818", "pnl_pct": -10.0,
         "shares": 1100, "hold_days": 9},
    ]
    s = stats(trades)
    assert s["win_rate"] == 0.5 and s["expectancy"] == pytest.approx(10.0)   # 按笔:美化
    assert s["expectancy_w"] == pytest.approx((30 * 100 - 10 * 1100) / 1200)  # ≈ -6.67
    assert s["win_rate_w"] == pytest.approx(100 / 1200)


def test_trim_leaves_other_positions_untouched(tmp_path):
    """codex 盲区1:多持仓下重建 positions 不得改动/重排其他行。"""
    hp, jp, cache = _setup(tmp_path)
    h = _read(hp)
    other = {"ts_code": "000002.SZ", "name": "另一只", "industry": "地产", "shares": 500,
             "cost": 10.0, "last": 11.0, "mv": 5500.0, "stop": 8.7, "bucket": "long",
             "entry_date": "20260811", "bucket_note": "", "tag": "", "theme": "",
             "watch": False}
    h["positions"].append(other)
    open(hp, "w", encoding="utf-8").write(json.dumps(h, ensure_ascii=False))
    _trim(hp, jp, cache)
    positions = _read(hp)["positions"]
    assert [p["ts_code"] for p in positions] == ["600001.SH", "000002.SZ"]   # 顺序不变
    assert positions[1] == other                                            # 逐字段不变


def test_trim_odd_lot_allowed_when_remainder_round(tmp_path):
    """codex P2-2:送转后 1250 股卖 250(剩 1000 整百)是合法零股卖出。"""
    hp, jp, cache = _setup(tmp_path, shares=1250)
    result = _trim(hp, jp, cache, shares=250, net=6600.0, exit_px=26.5)
    assert result["remaining"] == 1000


def test_trim_same_day_intraday_t1_rejected(tmp_path):
    """codex P2-6:entry_date == exit_date 违反 T+1,必是日期抄错。"""
    hp, jp, cache = _setup(tmp_path)
    h = _read(hp)
    h["positions"][0]["entry_date"] = "20260817"
    open(hp, "w", encoding="utf-8").write(json.dumps(h, ensure_ascii=False))
    with pytest.raises(SystemExit, match="T\\+1"):
        _trim(hp, jp, cache)


def test_trim_advance_as_of(tmp_path):
    hp, jp, cache = _setup(tmp_path, as_of="20260814")
    _trim(hp, jp, cache, date="20260817", advance_as_of=True)
    assert _read(hp)["as_of"] == "20260817"
    assert _read(hp)["positions"][0]["shares"] == 600
