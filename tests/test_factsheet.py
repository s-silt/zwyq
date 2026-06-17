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
    # WINNER rises steadily, LOSER falls steadily.
    daily = pd.DataFrame(
        [{"ts_code": "WIN.SZ", "trade_date": d, "open": 10 + j, "close": 10 + j, "amount": 1e6}
         for j, d in enumerate(days)]
        + [{"ts_code": "LOS.SH", "trade_date": d, "open": 50 - j, "close": 50 - j, "amount": 1e6}
           for j, d in enumerate(days)]
    )
    adj = pd.DataFrame([{"ts_code": c, "trade_date": d, "adj_factor": 1.0}
                        for c in ("WIN.SZ", "LOS.SH") for d in days])
    return daily, adj


def test_market_returns_gives_each_stock_horizon_return():
    daily, adj = _two_stock_market()
    mr = market_returns(daily, adj, horizons=(5,))
    # WIN last=34, 5 sessions ago=29 -> +5/29; LOS last=26, 5 ago=31 -> -5/31
    assert mr[5]["WIN.SZ"] == pytest.approx(34 / 29 - 1)
    assert mr[5]["LOS.SH"] == pytest.approx(26 / 31 - 1)


def test_daily_tech_facts_labels_trend_and_cross_sectional_percentile():
    daily, adj = _two_stock_market()
    mr = market_returns(daily, adj, horizons=(5, 20))

    win = daily_tech_facts("WIN.SZ", daily, adj, mr)
    los = daily_tech_facts("LOS.SH", daily, adj, mr)

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
