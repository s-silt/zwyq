"""P0① 退出侧卖出约束(吸纳终榜 2026-07-05,Claude×Codex 对抗后 P0 第1条)。

入场侧已剔"一字涨停买不进",退出侧此前仍按窗口内 ffill 假装崩盘股/停牌股卖得掉——
方向性**高估**多头收益。修正:计划退出日一字跌停(封死)或停牌 → 顺延到第一个可卖日
的开盘价;数据尽头仍不可卖(退市终局)→ 保持 ffill 最后成交价并 surface(不伪造)。
只认一字封死(开=高=低=收 且触及跌停价);盘中打开的跌停有成交机会,不算卖不出。
"""
import pandas as pd
import pytest

from scripts.factor_backtest import first_sellable_open, one_word_limit_down


def _day(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"ts_code": c, "trade_date": "20260102",
                          "open": o, "high": h, "low": lo, "close": cl}
                         for c, o, h, lo, cl in rows])


def _lim(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"ts_code": c, "trade_date": "20260102", "down_limit": d}
                         for c, d in rows])


# ---------- one_word_limit_down:一字跌停(卖不出)判定 ----------

def test_one_word_limit_down_detects_flat_at_limit():
    day = _day([("600001.SH", 9.0, 9.0, 9.0, 9.0)])
    lim = _lim([("600001.SH", 9.0)])
    assert one_word_limit_down(day, lim, ["600001.SH"]) == {"600001.SH"}


def test_one_word_limit_down_excludes_intraday_opened():
    # 盘中打开(high>low,有成交机会)→ 不算卖不出,即使收在跌停
    day = _day([("600002.SH", 9.5, 9.5, 9.0, 9.0)])
    lim = _lim([("600002.SH", 9.0)])
    assert one_word_limit_down(day, lim, ["600002.SH"]) == set()


def test_one_word_limit_down_excludes_flat_above_limit():
    # 全天一价但未及跌停(冷门股一笔成交)→ 可卖
    day = _day([("600003.SH", 9.5, 9.5, 9.5, 9.5)])
    lim = _lim([("600003.SH", 9.0)])
    assert one_word_limit_down(day, lim, ["600003.SH"]) == set()


def test_one_word_limit_down_missing_limit_treated_as_sellable():
    # 个券缺跌停价:daily 行里有真实成交价,无法证明封死 → 视为可卖(与买侧相反方向:
    # 买侧伪造"可买"会虚增收益故保守剔除;卖侧伪造"卖不掉"是无据地推迟真实成交价)
    day = _day([("600001.SH", 9.0, 9.0, 9.0, 9.0)])
    lim = _lim([("600999.SH", 8.0)])
    assert one_word_limit_down(day, lim, ["600001.SH"]) == set()


def test_one_word_limit_down_restricts_to_codes_and_fails_loud_on_empty():
    day = _day([("600001.SH", 9.0, 9.0, 9.0, 9.0), ("300999.SZ", 8.0, 8.0, 8.0, 8.0)])
    lim = _lim([("600001.SH", 9.0), ("300999.SZ", 8.0)])
    assert one_word_limit_down(day, lim, ["600001.SH"]) == {"600001.SH"}
    with pytest.raises(ValueError):
        one_word_limit_down(day.iloc[0:0], lim, ["600001.SH"])


# ---------- first_sellable_open:顺延到第一个可卖日 ----------

def test_first_sellable_immediate():
    opens = pd.Series([10.0, 9.5, 9.0])
    px, defer = first_sellable_open(opens, 1, lambda j: False)
    assert (px, defer) == (9.5, 0)


def test_first_sellable_skips_suspended_nan():
    # 停牌日 open=NaN → 顺延到复牌日开盘
    opens = pd.Series([10.0, float("nan"), float("nan"), 8.8])
    px, defer = first_sellable_open(opens, 1, lambda j: False)
    assert (px, defer) == (8.8, 2)


def test_first_sellable_skips_locked_days():
    # 连续一字跌停(locked)→ 顺延到首个非封死日
    opens = pd.Series([10.0, 9.0, 8.1, 7.3])
    px, defer = first_sellable_open(opens, 1, lambda j: j in (1, 2))
    assert (px, defer) == (7.3, 2)


def test_first_sellable_none_when_data_ends():
    # 数据尽头仍不可卖(退市终局)→ None,调用方保持 ffill 最后成交价并 surface
    opens = pd.Series([10.0, float("nan"), float("nan")])
    assert first_sellable_open(opens, 1, lambda j: False) is None


def test_defer_note_no_deferral_empty():
    from scripts.factor_backtest import defer_note
    assert defer_note(0, 0, 0) == ""


def test_defer_note_unresolved_only_no_division():
    # 顺延0只但未解>0(全退市终局的期)——曾除零崩掉 fwd=10 整跑
    from scripts.factor_backtest import defer_note
    s = defer_note(0, 0, 3)
    assert "未解3" in s and "均" not in s


def test_defer_note_with_average():
    from scripts.factor_backtest import defer_note
    assert "均2.5日" in defer_note(4, 10, 1)
