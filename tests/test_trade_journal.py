"""Tests for trade_journal —— 交易复盘账本(纯统计,不打分不荐股)。

口径约定(与 scripts/trade_journal.py docstring 一致):
- pnl_pct 以百分点存储(+3.8 表示 +3.8%),win_rate 为 0~1 小数;
- 只统计 exit_date 非空且 pnl_pct 非空的已平仓笔(持有中/换出未记录盈亏的不进统计);
- n<1 返回 {} —— 不伪造空样本统计;
- 无定义的量(无亏损笔时的 payoff 等)返回 NaN,不伪造为 0。
"""
import json
import math

import pytest

from scripts.trade_journal import (
    BUCKETS,
    JOURNAL_PATH,
    load_journal,
    main,
    parse_add,
    save_journal,
    stats,
)


def _t(**kw):
    """一笔已平仓交易的模板,kw 覆盖字段。"""
    base = {"code": "600000.SH", "name": "测试股", "bucket": "短线",
            "entry_date": "20260101", "entry_px": 10.0, "shares": 100,
            "exit_date": "20260105", "exit_px": 11.0, "pnl_pct": 10.0,
            "hold_days": 4, "reason": None, "approx": False}
    base.update(kw)
    return base


# ---------- stats:核心统计口径 ----------

def test_stats_mixed_win_loss():
    # 2 赢(+10, +20)1 亏(-5):胜率 2/3,avg_win=15,avg_loss=-5,payoff=3,期望=(10+20-5)/3
    trades = [_t(pnl_pct=10.0), _t(pnl_pct=20.0, hold_days=6), _t(pnl_pct=-5.0, hold_days=2)]
    s = stats(trades)
    assert s["n"] == 3
    assert abs(s["win_rate"] - 2 / 3) < 1e-9
    assert abs(s["avg_win_pct"] - 15.0) < 1e-9
    assert abs(s["avg_loss_pct"] + 5.0) < 1e-9
    assert abs(s["payoff"] - 3.0) < 1e-9
    assert abs(s["avg_hold_days"] - 4.0) < 1e-9          # (4+6+2)/3
    assert abs(s["expectancy"] - 25.0 / 3) < 1e-9
    # 期望恒等式:win_rate*avg_win + (1-win_rate)*avg_loss
    assert abs(s["expectancy"] - (s["win_rate"] * s["avg_win_pct"]
                                  + (1 - s["win_rate"]) * s["avg_loss_pct"])) < 1e-9


def test_stats_excludes_open_and_null_pnl():
    # 持有中(exit_date=None)与换出未记录盈亏(exit 有、pnl=None)都不进统计
    trades = [_t(pnl_pct=10.0),
              _t(exit_date=None, exit_px=None, pnl_pct=None, hold_days=None),
              _t(exit_date="20260106", exit_px=None, pnl_pct=None)]
    s = stats(trades)
    assert s["n"] == 1
    assert abs(s["win_rate"] - 1.0) < 1e-9


def test_stats_empty_returns_empty_dict():
    assert stats([]) == {}
    # 全是持有中 → 有效样本 0 → {} 不伪造
    assert stats([_t(exit_date=None, exit_px=None, pnl_pct=None)]) == {}


def test_stats_bucket_filter():
    trades = [_t(bucket="短线", pnl_pct=10.0), _t(bucket="长线", pnl_pct=-4.0)]
    s = stats(trades, bucket="短线")
    assert s["n"] == 1 and abs(s["win_rate"] - 1.0) < 1e-9
    s2 = stats(trades, bucket="长线")
    assert s2["n"] == 1 and abs(s2["win_rate"] - 0.0) < 1e-9
    assert stats(trades, bucket="制度前") == {}   # 该仓无样本 → {}


def test_stats_all_wins_undefined_loss_metrics_are_nan():
    # 无亏损笔:avg_loss/payoff 无定义 → NaN 不伪造;期望退化为均值
    s = stats([_t(pnl_pct=5.0), _t(pnl_pct=15.0)])
    assert math.isnan(s["avg_loss_pct"]) and math.isnan(s["payoff"])
    assert abs(s["expectancy"] - 10.0) < 1e-9


def test_stats_all_losses_undefined_win_metrics_are_nan():
    s = stats([_t(pnl_pct=-5.0), _t(pnl_pct=-15.0)])
    assert math.isnan(s["avg_win_pct"]) and math.isnan(s["payoff"])
    assert abs(s["win_rate"] - 0.0) < 1e-9
    assert abs(s["expectancy"] + 10.0) < 1e-9


def test_stats_zero_pnl_counts_as_loss():
    # 0% 不算赢(保守口径:不赢即输,扣掉摩擦成本后 0 名义盈亏实为小亏)
    s = stats([_t(pnl_pct=0.0), _t(pnl_pct=10.0)])
    assert abs(s["win_rate"] - 0.5) < 1e-9
    assert abs(s["avg_loss_pct"] - 0.0) < 1e-9
    assert math.isnan(s["payoff"])   # |avg_loss|=0 → 盈亏比无定义 → NaN


def test_stats_hold_days_all_missing_is_nan():
    s = stats([_t(hold_days=None)])
    assert math.isnan(s["avg_hold_days"])
    assert s["n"] == 1   # hold_days 缺不影响盈亏统计


# ---------- parse_add:CLI 追加一笔的解析(纯函数) ----------

def test_parse_add_full_fields_and_types():
    t = parse_add("code=601138.SH,name=工业富联,bucket=短线,entry_date=20260703,"
                  "entry_px=70.5,shares=200,exit_date=20260710,exit_px=77.0,"
                  "pnl_pct=9.2,hold_days=5,reason=移动止盈,approx=false")
    assert t["code"] == "601138.SH" and t["name"] == "工业富联"
    assert t["bucket"] == "短线"
    assert isinstance(t["entry_px"], float) and abs(t["entry_px"] - 70.5) < 1e-9
    assert isinstance(t["shares"], int) and t["shares"] == 200
    assert isinstance(t["hold_days"], int) and t["hold_days"] == 5
    assert t["approx"] is False


def test_parse_add_minimal_defaults():
    # 最小必填:code/bucket/entry_date/entry_px/shares;其余字段补 None(schema 齐全)
    t = parse_add("code=600000.SH,bucket=长线,entry_date=20260703,entry_px=10.0,shares=100")
    assert t["exit_date"] is None and t["exit_px"] is None and t["pnl_pct"] is None
    assert t["hold_days"] is None and t["reason"] is None
    assert t["approx"] is False   # 新增笔默认精确记录(approx 是历史种子专用标记)
    assert set(t) == {"code", "name", "bucket", "entry_date", "entry_px", "shares",
                      "exit_date", "exit_px", "pnl_pct", "hold_days", "reason", "approx"}


def test_parse_add_unknown_key_fails_loud():
    with pytest.raises(ValueError, match="未知字段"):
        parse_add("code=600000.SH,bucket=短线,entry_date=20260703,entry_px=1,shares=100,foo=1")


def test_parse_add_bad_bucket_fails_loud():
    with pytest.raises(ValueError, match="bucket"):
        parse_add("code=600000.SH,bucket=中线,entry_date=20260703,entry_px=1,shares=100")


def test_parse_add_missing_required_fails_loud():
    with pytest.raises(ValueError, match="缺必填"):
        parse_add("code=600000.SH,bucket=短线")


def test_parse_add_bad_date_fails_loud():
    with pytest.raises(ValueError, match="entry_date"):
        parse_add("code=600000.SH,bucket=短线,entry_date=2026-07-03,entry_px=1,shares=100")


def test_parse_add_bad_bool_fails_loud():
    with pytest.raises(ValueError, match="approx"):
        parse_add("code=600000.SH,bucket=短线,entry_date=20260703,entry_px=1,shares=100,approx=maybe")


# ---------- load/save:账本 IO,坏形状 fail-loud ----------

def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "journal.json"
    trades = [_t()]
    save_journal(trades, p)
    assert load_journal(p) == trades


def test_load_bad_shape_fails_loud(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([{"code": "X"}]), encoding="utf-8")   # 顶层不是 {trades: []}
    with pytest.raises(ValueError, match="trades"):
        load_journal(p)


# ---------- 种子数据:data/trade_journal.json 完整性 + 真实统计值 ----------

def test_seed_journal_schema_and_stats():
    # journal 是**活数据文件**(实盘交易持续追加),测试只钉两件不许腐的事:
    # ① 全量 schema 完整 + bucket 合法;② 种子历史(制度前·2026-07-02 之前的 6 笔,
    # 已定格的回忆口径)统计值不变。整体笔数只做"追加不减"下界,不写死
    # (此前 ==7 在 7/3 实盘三笔开仓入账后误报,钉活文件行数=测试腐坏)。
    trades = load_journal(JOURNAL_PATH)
    assert len(trades) >= 7                  # 追加不减(种子7笔是历史下界)
    keys = {"code", "name", "bucket", "entry_date", "entry_px", "shares",
            "exit_date", "exit_px", "pnl_pct", "hold_days", "reason", "approx"}
    for t in trades:
        assert set(t) == keys, t
        assert t["bucket"] in BUCKETS, t

    # 种子=制度前 & 入场早于 20260702(富联v2 及之后为实盘期,不属定格历史)
    seeds = [t for t in trades if t["bucket"] == "制度前" and t["entry_date"] < "20260702"]
    assert len(seeds) == 6
    assert all(t["approx"] is True for t in seeds)   # 历史回忆口径,全部标 approx

    # 种子已平仓且有盈亏的 5 笔:春秋+3.8 / 世纪华通-0.3 / 海康+4.3 / 韵升+6.65 / 富联v1-18.2
    # (甬金 exit 有但 pnl 未记录 → 不进统计)
    s = stats(seeds)
    assert s["n"] == 5
    assert abs(s["win_rate"] - 0.6) < 1e-9
    assert abs(s["avg_win_pct"] - (3.8 + 4.3 + 6.65) / 3) < 1e-9
    assert abs(s["avg_loss_pct"] - (-0.3 - 18.2) / 2) < 1e-9
    assert abs(s["payoff"] - ((3.8 + 4.3 + 6.65) / 3) / 9.25) < 1e-9
    assert abs(s["expectancy"] - (3.8 + 4.3 + 6.65 - 0.3 - 18.2) / 5) < 1e-9
    assert abs(s["avg_hold_days"] - 6.0) < 1e-9   # (8+1+7+9+5)/5,交易日口径见 journal reason 注


def test_buckets_constant_matches_regime():
    # 双仓制三档:短线/长线 + 制度生效(2026-07-03)前的历史仓
    assert set(BUCKETS) == {"短线", "长线", "制度前"}


# ---------- CLI main:--add 追加落盘,统计表可打印 ----------

def test_main_add_appends_and_persists(tmp_path, capsys):
    p = tmp_path / "journal.json"
    save_journal([_t()], p)
    main(["--add", "code=601138.SH,bucket=短线,entry_date=20260703,entry_px=70.5,shares=200"],
         path=p)
    trades = load_journal(p)
    assert len(trades) == 2
    assert trades[-1]["code"] == "601138.SH" and trades[-1]["exit_date"] is None
    out = capsys.readouterr().out
    assert "601138.SH" in out


def test_main_prints_stats_and_recent(tmp_path, capsys):
    p = tmp_path / "journal.json"
    save_journal([_t(bucket="制度前", pnl_pct=10.0), _t(bucket="短线", pnl_pct=-7.0)], p)
    main([], path=p)
    out = capsys.readouterr().out
    assert "短线" in out and "制度前" in out and "总体" in out
    main(["--bucket", "短线"], path=p)
    out2 = capsys.readouterr().out
    assert "短线" in out2 and "制度前" not in out2.split("最近")[0]   # 统计表只剩指定仓
