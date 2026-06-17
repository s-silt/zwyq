"""Tests for the factual-layer indicators (descriptive, not predictive).

These are the reusable, tested versions of the EMA / RSI / Bollinger numbers —
computed from real closes, never quoted from an unverified source.
"""

import pandas as pd
import pytest

from ashare_gauntlet.factsheet import bollinger, build_factsheet, ema, rsi


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
