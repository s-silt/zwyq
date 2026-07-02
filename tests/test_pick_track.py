"""Tests for pick_track —— 筛选器命中率闭环(D10 进出 diff + 前向收益),纯测量不打分。"""
import pandas as pd
import pytest

from scripts.pick_track import (
    EmptyIndexPullError,
    IncompleteIndexPullError,
    diff_picks,
    forward_returns,
    index_return,
    load_index_daily,
    regime_return,
)


def test_diff_picks_new_and_dropped():
    prev = ["A", "B", "C"]
    curr = ["B", "C", "D", "E"]
    d = diff_picks(prev, curr)
    assert d["new"] == ["D", "E"]
    assert d["dropped"] == ["A"]
    assert d["stay"] == ["B", "C"]


def test_diff_picks_no_prev_all_new():
    d = diff_picks([], ["A", "B"])
    assert d["new"] == ["A", "B"] and d["dropped"] == [] and d["stay"] == []


def _panel():
    # 两只股,4 个交易日,前复权价
    rows = []
    for i, dte in enumerate(["20260101", "20260102", "20260103", "20260106"]):
        rows.append({"ts_code": "A", "trade_date": dte, "adj_close": 10.0 + i})       # 10,11,12,13
        rows.append({"ts_code": "B", "trade_date": dte, "adj_close": 20.0 - i})       # 20,19,18,17
    return pd.DataFrame(rows)


def test_forward_returns_from_snapshot_date():
    px = _panel()
    r = forward_returns(["A", "B"], "20260101", px)
    # A: 13/10-1=+30%;B: 17/20-1=-15%(截至面板最新日)
    assert abs(r["A"] - 0.30) < 1e-9
    assert abs(r["B"] + 0.15) < 1e-9


def test_forward_returns_snapshot_date_not_traded_uses_next_available():
    px = _panel()
    r = forward_returns(["A"], "20260104", px)   # 周末快照 → 用其后首个交易日 20260106 起算
    assert abs(r["A"] - 0.0) < 1e-9              # 起点即最新日 → 0%


def test_forward_returns_missing_code_is_nan():
    px = _panel()
    r = forward_returns(["Z"], "20260101", px)
    assert r["Z"] != r["Z"]  # NaN


# ---------- 沪深300 基准:index_return(与 forward_returns 同口径的指数区间收益) ----------

def _idx(dates_closes):
    # 显式给列名:空表也带 schema(真实 tushare 空拉返回的就是带列的空 DataFrame)
    return pd.DataFrame([{"trade_date": d, "close": c} for d, c in dates_closes],
                        columns=["trade_date", "close"])


def test_index_return_from_snapshot_date():
    idx = _idx([("20260101", 4000.0), ("20260102", 4100.0), ("20260103", 4200.0), ("20260106", 4400.0)])
    # 4400/4000 - 1 = +10%
    assert abs(index_return(idx, "20260101") - 0.10) < 1e-9


def test_index_return_snapshot_on_non_trading_day_uses_next_trading_day():
    idx = _idx([("20260101", 4000.0), ("20260102", 4100.0), ("20260106", 4400.0)])
    # 快照落在 20260104(周末)→ 用其后首个交易日 20260106 起算 → 起点即最新日 = 0%
    assert abs(index_return(idx, "20260104") - 0.0) < 1e-9


def test_index_return_insufficient_data_is_nan():
    idx = _idx([("20260101", 4000.0), ("20260102", 4100.0)])
    r = index_return(idx, "20260107")  # 快照日在全部数据之后 → 数据不足,NaN 不伪造
    assert r != r
    r2 = index_return(_idx([]), "20260101")  # 空表同理
    assert r2 != r2


# ---------- regime 读数:最近 n 交易日指数涨跌 ----------

def test_regime_return_last_n_trading_days():
    # 5 个交易日,n=3:close[-1]/close[-1-3] − 1 = 4300/4000 − 1
    idx = _idx([("20260101", 3900.0), ("20260102", 4000.0), ("20260103", 4100.0),
                ("20260106", 4200.0), ("20260107", 4300.0)])
    assert abs(regime_return(idx, 3) - (4300.0 / 4000.0 - 1.0)) < 1e-9


def test_regime_return_insufficient_rows_is_nan():
    idx = _idx([("20260101", 4000.0), ("20260102", 4100.0)])
    r = regime_return(idx, 20)  # 行数 < n+1 → NaN 不伪造
    assert r != r


# ---------- index_daily 缓存:单文件、覆盖即用、不足整段重拉、空拉 fail-loud ----------

class _FakePro:
    """假 pro:按区间切片返回预置指数日线,并记录每次 API 调用。

    day_df 单独给单日拉取用 —— 模拟镜像实测行为:大区间拉取会稳定漏个别日
    (20260625 区间无),但单日拉取可靠(20260625 单日有)。
    """

    def __init__(self, df: pd.DataFrame, day_df: pd.DataFrame | None = None):
        self.df = df
        self.day_df = day_df
        self.calls: list[tuple[str, str, str]] = []

    def index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append((ts_code, start_date, end_date))
        src = self.day_df if (start_date == end_date and self.day_df is not None) else self.df
        m = (src["trade_date"] >= start_date) & (src["trade_date"] <= end_date)
        return src[m].copy()


def _fake_pro():
    return _FakePro(_idx([("20260101", 4000.0), ("20260102", 4100.0),
                          ("20260103", 4200.0), ("20260106", 4400.0)]))


def test_load_index_daily_pulls_once_and_caches(tmp_path):
    pro = _fake_pro()
    df = load_index_daily(pro, "000300.SH", "20260101", "20260106", cache_dir=tmp_path)
    assert len(pro.calls) == 1
    assert list(df["trade_date"]) == ["20260101", "20260102", "20260103", "20260106"]
    assert (tmp_path / "index_daily" / "000300.SH.parquet").exists()


def test_load_index_daily_covering_cache_skips_api(tmp_path):
    pro = _fake_pro()
    load_index_daily(pro, "000300.SH", "20260101", "20260106", cache_dir=tmp_path)
    # 第二次同区间(及更窄区间)命中缓存 → 不再调 API
    df = load_index_daily(pro, "000300.SH", "20260102", "20260103", cache_dir=tmp_path)
    assert len(pro.calls) == 1
    assert not df.empty


def test_load_index_daily_stale_cache_refetches_whole_range(tmp_path):
    pro = _fake_pro()
    load_index_daily(pro, "000300.SH", "20260101", "20260103", cache_dir=tmp_path)
    # end 前进到 20260106,缓存不覆盖 → 整段重拉一次并覆盖写
    df = load_index_daily(pro, "000300.SH", "20260101", "20260106", cache_dir=tmp_path)
    assert len(pro.calls) == 2
    assert pro.calls[-1] == ("000300.SH", "20260101", "20260106")
    assert list(df["trade_date"])[-1] == "20260106"
    # 覆盖写后缓存已更新:再取同区间不调 API
    load_index_daily(pro, "000300.SH", "20260101", "20260106", cache_dir=tmp_path)
    assert len(pro.calls) == 2


def test_load_index_daily_empty_pull_fails_loud(tmp_path):
    pro = _FakePro(_idx([]))
    with pytest.raises(EmptyIndexPullError):
        load_index_daily(pro, "000300.SH", "20260101", "20260106", cache_dir=tmp_path)
    # fail-loud 且不落盘:空拉不能写缓存毒化后续
    assert not (tmp_path / "index_daily" / "000300.SH.parquet").exists()


def test_load_index_daily_holey_pull_patched_by_single_day_refetch(tmp_path):
    # 镜像大区间拉取稳定漏个别日(实测 20260625:单日拉有、跨区间拉无)→
    # 对照交易日历发现缺日后逐日补拉(同源真数据,非伪造),补全后才落盘
    holey = _idx([("20260101", 4000.0), ("20260103", 4200.0), ("20260106", 4400.0)])
    full = _idx([("20260101", 4000.0), ("20260102", 4100.0),
                 ("20260103", 4200.0), ("20260106", 4400.0)])
    pro = _FakePro(holey, day_df=full)
    df = load_index_daily(pro, "000300.SH", "20260101", "20260106", cache_dir=tmp_path,
                          expected_days=["20260101", "20260102", "20260103", "20260106"])
    assert list(df["trade_date"]) == ["20260101", "20260102", "20260103", "20260106"]
    assert ("000300.SH", "20260102", "20260102") in pro.calls  # 缺日单日补拉
    # 补全后的完整序列已落盘:再取同区间零 API
    n = len(pro.calls)
    load_index_daily(pro, "000300.SH", "20260101", "20260106", cache_dir=tmp_path,
                     expected_days=["20260101", "20260102", "20260103", "20260106"])
    assert len(pro.calls) == n


def test_load_index_daily_unfillable_hole_fails_loud_and_not_cached(tmp_path):
    # 单日补拉也拿不到(源真缺)→ fail-loud 且拒绝落盘,带洞序列不能当真值
    holey = _idx([("20260101", 4000.0), ("20260103", 4200.0), ("20260106", 4400.0)])
    pro = _FakePro(holey)  # 单日拉取同样走带洞数据 → 补不上
    with pytest.raises(IncompleteIndexPullError, match="20260102"):
        load_index_daily(pro, "000300.SH", "20260101", "20260106", cache_dir=tmp_path,
                         expected_days=["20260101", "20260102", "20260103", "20260106"])
    assert not (tmp_path / "index_daily" / "000300.SH.parquet").exists()


def test_load_index_daily_holey_cache_self_heals_by_refetch(tmp_path):
    # 历史遗留的带洞缓存(完整性校验加入前写入的)→ 视为不覆盖,自动整段重拉修复
    holey = _idx([("20260101", 4000.0), ("20260103", 4200.0), ("20260106", 4400.0)])
    (tmp_path / "index_daily").mkdir(parents=True)
    holey.to_parquet(tmp_path / "index_daily" / "000300.SH.parquet", index=False)
    pro = _fake_pro()  # 源数据完整
    df = load_index_daily(pro, "000300.SH", "20260101", "20260106", cache_dir=tmp_path,
                          expected_days=["20260101", "20260102", "20260103", "20260106"])
    assert len(pro.calls) == 1  # 缓存有洞 → 重拉一次
    assert list(df["trade_date"]) == ["20260101", "20260102", "20260103", "20260106"]
