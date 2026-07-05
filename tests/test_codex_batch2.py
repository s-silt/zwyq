"""Codex review 第二批(2026-07-05,范围 347cc6b..HEAD)的修复测试。

6 findings 对应 6 组测试:
1. fetch 缓存命中路径也校验 REQUIRED_MARKET_COLUMNS(退化 schema 缓存自愈/fail-loud)
2. factor_rank 入池门槛围绕 COMPOSITE_FACTORS(展示列不参与门槛)
3. costs round_trip 印花税按**卖出日**取段(跨税改期不再用信号日错扣)
4. backtest 真 Newey-West HAC t(Bartlett 核 + NW1994 自动带宽,替代 AR(1) 近似的错标)
5. factor_backtest 走 date_partition_files(在 test_partition.py 已测枚举语义,此处不重复)
6. costs 费率 NaN/inf fail-loud
"""
import math

import pandas as pd
import pytest

from ashare_gauntlet.backtest import newey_west_tstat
from ashare_gauntlet.costs import round_trip_cost_rate
from ashare_gauntlet.data.fetch import fetch_market_day
from scripts.factor_rank import composite_inputs_complete
from scripts.pick_track import cost_adjusted_excess


class _MarketPro:
    """按端点返回注册 DataFrame 的假 pro(与 test_fetch 同款);记录调用。"""

    def __init__(self, returns: dict[str, pd.DataFrame]) -> None:
        self._returns = returns
        self.calls: list[tuple[str, str]] = []

    def __getattr__(self, endpoint: str):
        def _method(trade_date: str, fields: str = "") -> pd.DataFrame:
            self.calls.append((endpoint, trade_date))
            return self._returns[endpoint]

        return _method


# ---------- 1. 缓存命中路径的退化 schema 自愈 ----------

def test_fetch_market_day_heals_degraded_cached_schema(tmp_path):
    # 历史退化 schema 缓存(缺 total_mv/pe_ttm/pb,实测 20160630)命中时不得直接返回——
    # 重拉自愈并覆盖缓存,否则显式 fields 重试永远不触发,下游 KeyError/错口径
    path = tmp_path / "daily_basic" / "20160630.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ts_code": ["600519.SH"], "trade_date": ["20160630"],
                  "pe": [30.0]}).to_parquet(path, index=False)
    full = pd.DataFrame({"ts_code": ["600519.SH"], "trade_date": ["20160630"],
                         "total_mv": [2e7], "pe_ttm": [28.0], "pb": [9.0],
                         "turnover_rate": [1.2]})
    pro = _MarketPro({"daily_basic": full})
    out = fetch_market_day(pro, "daily_basic", "20160630", tmp_path)
    assert {"total_mv", "pe_ttm", "pb", "turnover_rate"} <= set(out.columns)
    assert {"total_mv", "pe_ttm", "pb", "turnover_rate"} <= set(pd.read_parquet(path).columns)


def test_fetch_market_day_degraded_cache_and_source_fails_loud(tmp_path):
    # 缓存退化且源侧重拉仍退化 → fail-loud,绝不静默供给缺列数据
    path = tmp_path / "daily_basic" / "20160630.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    degraded = pd.DataFrame({"ts_code": ["600519.SH"], "trade_date": ["20160630"], "pe": [30.0]})
    degraded.to_parquet(path, index=False)
    pro = _MarketPro({"daily_basic": degraded})
    with pytest.raises(RuntimeError, match="必需列"):
        fetch_market_day(pro, "daily_basic", "20160630", tmp_path)


def test_fetch_market_day_healthy_cache_untouched(tmp_path):
    # 健康缓存命中不重拉(零 API 语义不回退)
    path = tmp_path / "daily_basic" / "20240102.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    full = pd.DataFrame({"ts_code": ["600519.SH"], "trade_date": ["20240102"],
                         "total_mv": [2e7], "pe_ttm": [28.0], "pb": [9.0],
                         "turnover_rate": [1.2]})
    full.to_parquet(path, index=False)
    pro = _MarketPro({"daily_basic": full})
    fetch_market_day(pro, "daily_basic", "20240102", tmp_path)
    assert pro.calls == []


# ---------- 2. 入池门槛围绕 COMPOSITE_FACTORS ----------

def test_composite_inputs_complete_gate():
    # 门槛原则(review 点名):入分因子原值全齐、展示列缺失不清退;因子集随
    # COMPOSITE_FACTORS 演进(当前 EP+BP+IVOL,见 test_factor_rank_composite)
    df = pd.DataFrame({
        "EP": [0.10, None], "BP": [0.50, 0.50], "IVOL": [0.02, 0.02],
        "ACC": [None, 0.02], "roe": [None, 9.9], "GPOA": [None, 0.30]})
    assert composite_inputs_complete(df).tolist() == [True, False]


# ---------- 3. 印花税按卖出日取段 ----------

def test_round_trip_stamp_duty_uses_sell_date():
    # 买 20230731(税率1‰)、卖 20230829(减半后0.5‰)→ 按卖出日
    rt = round_trip_cost_rate("20230731", 0.00025, 0.0015, sell_date="20230829")
    assert rt == pytest.approx(2 * 0.00025 + 2 * 0.0015 + 0.0005)


def test_round_trip_sell_date_defaults_to_trade_date():
    assert round_trip_cost_rate("20230731", 0.0, 0.0) == pytest.approx(0.001)


def test_round_trip_sell_before_buy_fails_loud():
    with pytest.raises(ValueError, match="sell_date"):
        round_trip_cost_rate("20230829", 0.0, 0.0, sell_date="20230731")


def test_cost_adjusted_excess_sell_date_regime():
    # pick_track 前向收益的计量终点跨过 2023-08-28 税改 → 印花税按终点日(卖出侧)取段
    pre = cost_adjusted_excess(0.10, 0.04, "20230827", 0.00025, 0.0015)
    post = cost_adjusted_excess(0.10, 0.04, "20230827", 0.00025, 0.0015, sell_date="20230901")
    assert post - pre == pytest.approx(0.001 - 0.0005)


# ---------- 4. 真 Newey-West HAC t ----------

def test_newey_west_t_below_iid_for_autocorrelated():
    # 正自相关 IC 序列 → HAC 方差 > iid 方差 → t 变小(与 AR(1) 近似同方向,但核不同)
    s = pd.Series([0.03] * 5 + [0.01] * 5)
    icir, t, lag = newey_west_tstat(s)
    iid_t = (s.mean() / s.std()) * math.sqrt(len(s))
    assert 0 < t < iid_t
    assert lag >= 1


def test_newey_west_automatic_lag_formula():
    # NW(1994) 自动带宽 q = floor(4·(N/100)^(2/9)):N=100 → 4(文献公式,非手拍常数)
    s = pd.Series(([0.01, -0.02, 0.03] * 34)[:100])
    _, _, lag = newey_west_tstat(s)
    assert lag == 4


def test_newey_west_finite_for_normal_series():
    s = pd.Series([0.05, 0.01, 0.03, 0.02, 0.04, 0.00, 0.03, 0.01])
    icir, t, lag = newey_west_tstat(s)
    assert math.isfinite(icir) and math.isfinite(t) and lag >= 0


def test_newey_west_insufficient_sample_returns_nan():
    icir, t, lag = newey_west_tstat(pd.Series([0.01, 0.02, 0.03]))
    assert math.isnan(icir) and math.isnan(t)


# ---------- 6. 费率 NaN/inf fail-loud ----------

@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_rates_nan_inf_fail_loud(bad):
    with pytest.raises(ValueError):
        round_trip_cost_rate("20240101", bad, 0.001)
    with pytest.raises(ValueError):
        round_trip_cost_rate("20240101", 0.001, bad)


def test_required_daily_basic_covers_panel_consumers(tmp_path):
    # P1 实跑踩雷:镜像对个别日(实测 20140108)默认返回仅 ts_code/pe_ttm/pb/total_mv 的
    # 退化 schema——恰好满足旧 required 三列,不触发显式重试;缺 trade_date/turnover_rate
    # 让 TURN 换手面板整跑崩。required 必须覆盖全部下游消费列,退化日才会自愈。
    from ashare_gauntlet.data.fetch import REQUIRED_MARKET_COLUMNS
    assert {"trade_date", "turnover_rate", "total_mv", "pe_ttm", "pb"} <= set(
        REQUIRED_MARKET_COLUMNS["daily_basic"])
    # 行为:退化缓存(缺 turnover_rate)命中 → 重拉自愈
    path = tmp_path / "daily_basic" / "20140108.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ts_code": ["600519.SH"], "total_mv": [2e7], "pe_ttm": [28.0],
                  "pb": [9.0]}).to_parquet(path, index=False)
    full = pd.DataFrame({"ts_code": ["600519.SH"], "trade_date": ["20140108"],
                         "total_mv": [2e7], "pe_ttm": [28.0], "pb": [9.0],
                         "turnover_rate": [1.2]})
    pro = _MarketPro({"daily_basic": full})
    out = fetch_market_day(pro, "daily_basic", "20140108", tmp_path)
    assert "turnover_rate" in out.columns
