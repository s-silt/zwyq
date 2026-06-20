"""Tests for the factual-layer indicators (descriptive, not predictive).

These are the reusable, tested versions of the EMA / RSI / Bollinger numbers —
computed from real closes, never quoted from an unverified source.
"""

import pandas as pd
import pytest

from ashare_gauntlet.factsheet import (
    bollinger,
    build_factsheet,
    daily_tech_facts,
    ema,
    entry_rank,
    market_returns,
    north_flow_disclosure,
    north_turnover,
    rsi,
)


def test_ema_of_constant_series_is_constant():
    assert ema(pd.Series([5.0] * 12), span=3).iloc[-1] == pytest.approx(5.0)


def test_rsi_is_100_for_only_gains_and_0_for_only_losses():
    assert rsi(pd.Series(range(1, 40), dtype=float), n=14).iloc[-1] == pytest.approx(100.0)
    assert rsi(pd.Series(range(40, 1, -1), dtype=float), n=14).iloc[-1] == pytest.approx(0.0)


def test_bollinger_is_mean_plus_minus_k_sample_std_over_window():
    s = pd.Series([1.0, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6])
    window = s.iloc[-20:]
    lower, mid, upper = bollinger(s, n=20, k=2.0)

    assert mid == pytest.approx(window.mean())
    assert upper == pytest.approx(window.mean() + 2 * window.std())  # ddof=1
    assert lower == pytest.approx(window.mean() - 2 * window.std())


def test_build_factsheet_assembles_facts_for_one_stock_incl_northbound():
    days = [f"202506{d:02d}" for d in range(1, 26)]
    daily = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": d, "open": 10.0 + j * 0.1,
             "close": 10.0 + j * 0.1, "amount": 5e5 + j}
            for j, d in enumerate(days)
        ]
        + [{"ts_code": "600000.SH", "trade_date": d, "open": 9.0, "close": 9.0, "amount": 1e5} for d in days]
    )
    adj = pd.DataFrame([{"ts_code": c, "trade_date": d, "adj_factor": 1.0}
                        for c in ("000001.SZ", "600000.SH") for d in days])
    hk = pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": d, "ratio": str(1.0 + j * 0.1),
                        "exchange": "SZ"} for j, d in enumerate(days)])

    fs = build_factsheet("000001.SZ", daily, adj, hk)

    assert fs["ts_code"] == "000001.SZ"
    assert fs["as_of"] == "20250625"
    assert fs["close_raw"] == pytest.approx(12.4)  # 10 + 24*0.1
    assert "ema_short" in fs and "rsi" in fs and "boll" in fs
    assert fs["high_20d"] == pytest.approx(12.4)
    # northbound: latest ratio rising over the window (real data point, not invented)
    assert fs["north_ratio"] == pytest.approx(1.0 + 24 * 0.1)
    assert fs["north_ratio_chg_5"] == pytest.approx(0.5)


def _two_stock_market():
    days = [f"202506{d:02d}" for d in range(1, 26)]
    # WINNER rises steadily, LOSER falls steadily. 用真实沪深主板代码,使横截面
    # 分位 cohort(过滤到 600/000…)非空 —— 契约C2 后 pct 只对主板 cohort 计算。
    daily = pd.DataFrame(
        [{"ts_code": "000002.SZ", "trade_date": d, "open": 10 + j, "close": 10 + j, "amount": 1e6}
         for j, d in enumerate(days)]
        + [{"ts_code": "600519.SH", "trade_date": d, "open": 50 - j, "close": 50 - j, "amount": 1e6}
           for j, d in enumerate(days)]
    )
    adj = pd.DataFrame([{"ts_code": c, "trade_date": d, "adj_factor": 1.0}
                        for c in ("000002.SZ", "600519.SH") for d in days])
    return daily, adj


def test_market_returns_gives_each_stock_horizon_return():
    daily, adj = _two_stock_market()
    mr = market_returns(daily, adj, horizons=(5,))
    # WIN last=34, 5 sessions ago=29 -> +5/29; LOS last=26, 5 ago=31 -> -5/31
    assert mr[5]["000002.SZ"] == pytest.approx(34 / 29 - 1)
    assert mr[5]["600519.SH"] == pytest.approx(26 / 31 - 1)


def test_daily_tech_facts_labels_trend_and_cross_sectional_percentile():
    daily, adj = _two_stock_market()
    mr = market_returns(daily, adj, horizons=(5, 20))

    win = daily_tech_facts("000002.SZ", daily, adj, mr)  # 深主板,稳涨
    los = daily_tech_facts("600519.SH", daily, adj, mr)  # 沪主板,稳跌

    assert win["trend"] == "多头"  # price > EMA5 > EMA20 (steady rise)
    assert los["trend"] == "空头"
    # WIN beats LOS cross-sectionally, so its 5d percentile is the higher one.
    assert win["pct5"] > los["pct5"]
    assert set(["close", "rsi", "ret5_pct", "vol_ratio", "dist_60d_high_pct"]).issubset(win)


def test_north_flow_disclosure_states_the_standing_facts():
    # The honest standing limit must be self-carried by every report rather than
    # left to a manual footnote: daily northbound NET inflow was discontinued on
    # 2024-08-19 (per-stock holdings moved to quarterly), so the factual layer
    # must never quote a daily net figure that no longer officially exists.
    note = north_flow_disclosure()
    assert "2024-08-19" in note
    assert "停披露" in note
    assert "季度" in note
    assert "净" in note  # explicitly disclaims a net figure


def test_north_turnover_returns_turnover_in_yi_after_cutoff():
    # On/after 2024-08-19, moneyflow_hsgt's hgt/sgt/north_money carry TURNOVER
    # (成交额, 百万元). Convert to 亿元; north_money == hgt + sgt.
    mf = pd.DataFrame(
        [{"trade_date": "20260617", "hgt": "174913.28", "sgt": "210142.56",
          "north_money": "385055.84"}]
    )
    nt = north_turnover(mf, "20260617")
    assert nt["hgt_yi"] == pytest.approx(1749.13, abs=0.01)
    assert nt["sgt_yi"] == pytest.approx(2101.43, abs=0.01)
    assert nt["total_yi"] == pytest.approx(3850.56, abs=0.01)


def test_north_turnover_refuses_pre_cutoff_dates_to_avoid_semantic_drift():
    # LANDMINE: before 2024-08-19 the SAME columns are daily NET inflow (signed,
    # ~50x smaller), not turnover. Returning that as turnover would be a silent
    # ~50x fabrication, so the guard must refuse pre-cutoff dates outright.
    mf = pd.DataFrame(
        [{"trade_date": "20240815", "hgt": "8865.01", "sgt": "3340.83",
          "north_money": "12205.84"}]
    )
    with pytest.raises(ValueError):
        north_turnover(mf, "20240815")


def test_entry_rank_prefers_uptrend_penalizes_falling_knife_and_overbought():
    # Same relative strength (pct20=80); the entry discipline should still rank
    # an uptrend-pullback above a downtrend and above an overbought chase.
    up = {"trend": "多头", "rsi": 55.0, "close": 11.0, "ema_long": 10.8, "pct20": 80.0}
    down = {"trend": "空头", "rsi": 40.0, "close": 9.0, "ema_long": 10.0, "pct20": 80.0}
    hot = {"trend": "多头", "rsi": 78.0, "close": 13.0, "ema_long": 10.0, "pct20": 80.0}

    s_up, tag_up = entry_rank(up)
    s_down, tag_down = entry_rank(down)
    s_hot, tag_hot = entry_rank(hot)

    assert s_up > s_down  # uptrend beats falling knife
    assert s_up > s_hot   # disciplined entry beats chasing overbought
    assert "勿接" in tag_down
    assert "勿追" in tag_hot
    assert "回踩" in tag_up


# ---------------------------------------------------------------------------
# 契约C2:entry_rank 缺关键技术输入(pct20/close)时返回 None 分(不补 0/50/close 默认)
# ---------------------------------------------------------------------------
def test_entry_rank_returns_none_score_when_pct20_missing():
    # pct20 缺失(横截面分位算不出,如停牌/历史不足)→ 不得把 base 补 0 伪造一个分,
    # 返回 score=None,标签注明数据缺失
    score, tag = entry_rank({"trend": "多头", "rsi": 55.0, "close": 11.0, "ema_long": 10.8})
    assert score is None
    assert "缺" in tag


def test_entry_rank_returns_none_score_when_close_missing():
    # close 缺失 → 距 EMA 等判据无依据,返回 score=None,不补 close 默认
    score, tag = entry_rank({"trend": "多头", "rsi": 55.0, "ema_long": 10.8, "pct20": 80.0})
    assert score is None
    assert "缺" in tag


def test_entry_rank_present_inputs_still_numeric():
    # 关键输入俱在时仍返回数值分(守卫不得误伤正常路径)
    score, _ = entry_rank({"trend": "多头", "rsi": 55.0, "close": 11.0, "ema_long": 10.8, "pct20": 80.0})
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# 契约C2:daily_tech_facts 的 pct20/pct5 横截面分位 cohort 过滤到沪深主板
# (600/601/603/605/000/001/002/003;不含创业300/科创688/北交8xx)
# ---------------------------------------------------------------------------
def _cohort_market():
    days = [f"202506{d:02d}" for d in range(1, 26)]
    # 600519 是「主板 cohort」里最强的(3 只主板里跑赢另 2 只);cohort 里混入
    # 一只科创(688)与一只创业板(300),二者是全市场最强,若未过滤会污染主板股分位。
    rows = []
    for code, base, step in [
        ("600519.SH", 10.0, 1.0),   # 主板,被评估(主板内最强)
        ("000001.SZ", 30.0, -0.2),  # 主板,弱(cohort 内)
        ("601988.SH", 30.0, -0.3),  # 主板,更弱(cohort 内)
        ("688981.SH", 10.0, 5.0),   # 科创,极强(应被排除)
        ("300750.SZ", 10.0, 5.0),   # 创业板,极强(应被排除)
    ]:
        for j, d in enumerate(days):
            rows.append({"ts_code": code, "trade_date": d, "open": base + j * step,
                         "close": base + j * step, "amount": 1e6})
    daily = pd.DataFrame(rows)
    adj = pd.DataFrame([{"ts_code": c, "trade_date": d, "adj_factor": 1.0}
                        for c in daily["ts_code"].unique() for d in days])
    return daily, adj


def test_daily_tech_facts_percentile_cohort_is_main_board_only():
    # 主板 cohort = {600519 强, 000001 弱, 601988 更弱};600519 跑赢另 2 只主板,
    # cohort 内 2/3 在其下 → pct20 ≈ 66.7。若把极强的 688/300 算进 cohort
    # (5 只里 600519 排第 3),分位会跌到 ~40。验证 cohort 过滤生效。
    daily, adj = _cohort_market()
    mr = market_returns(daily, adj, horizons=(20,))
    fs = daily_tech_facts("600519.SH", daily, adj, mr)
    assert fs["pct20"] == pytest.approx(2 / 3 * 100, abs=0.1)
