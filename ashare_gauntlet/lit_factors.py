"""A股有效因子(纯函数)—— 文献核查后在 A股 OOS 真有效、且本项目缓存可算的几类。

依据(见 presentation-layer-design memory):
- 盈利质量(现金含量):净现比(经营现金流/归母净利)、应计强度((净利−经营现金流)/总资产)。
  A股最有学术含金量的质量因子(PKU/Sloan/Dechow),给"现金流是分水岭"背书。
- 价值:EP(盈利收益率=1/PE),A股 EP 优于 BP(Liu-Stambaugh-Yuan 2019)。
- 短期反转:近20日前复权收益,A股以反转为主(涨多→下月倾向回调),故正值=回调风险。
- 低波动:近60日前复权日收益标准差,低波动异象在A股有效。

纯函数:不读文件不调接口,DataFrame 由调用方传入(IO 在 scripts 层)。数据缺失返回 None,不伪造默认值。
"""
import pandas as pd

ANNUAL = "20251231"


def latest_annual_end(*dfs: pd.DataFrame | None) -> str | None:
    """从传入的财报表动态取「最近年报期」= max(end_date 以 '1231' 结尾)。

    契约C3:'最近年报'不再硬编码 ``ANNUAL``。跨表(income/cashflow…)取并集里最大的
    年报期;取不到任何 …1231 → 返回 ``None``(由调用方标「年报缺失」,**不**套固定日期
    伪造一个年报期)。空表/None 安全跳过。
    """
    ends: set[str] = set()
    for df in dfs:
        if df is None or df.empty or "end_date" not in df.columns:
            continue
        col = df["end_date"].astype(str)
        ends.update(col[col.str.endswith("1231")].tolist())
    return max(ends) if ends else None


def _annual_value(df: pd.DataFrame | None, col: str, end: str) -> float | None:
    """取年报(end_date==end)某字段值;多行按 ann_date 取最新。缺失/非数返回 None。"""
    if df is None or col not in df.columns or "end_date" not in df.columns:
        return None
    rows = df[df["end_date"] == end]
    if rows.empty:
        return None
    if "ann_date" in rows.columns:
        rows = rows.sort_values("ann_date")
    v = rows.iloc[-1][col]
    return float(v) if pd.notna(v) else None


def net_cash_ratio(income: pd.DataFrame | None, cashflow: pd.DataFrame | None,
                   end: str = ANNUAL) -> float | None:
    """净现比 = 经营活动现金流净额 / 归母净利(年报口径)。净利≤0 时无意义返回 None。"""
    ni = _annual_value(income, "n_income_attr_p", end)
    ocf = _annual_value(cashflow, "n_cashflow_act", end)
    if ni is None or ocf is None or ni <= 0:
        return None
    return ocf / ni


def accrual_ratio(income: pd.DataFrame | None, cashflow: pd.DataFrame | None,
                  balancesheet: pd.DataFrame | None, end: str = ANNUAL) -> float | None:
    """应计强度 = (归母净利 − 经营现金流) / 总资产(年报口径)。越低/越负越干净。"""
    ni = _annual_value(income, "n_income_attr_p", end)
    ocf = _annual_value(cashflow, "n_cashflow_act", end)
    ta = _annual_value(balancesheet, "total_assets", end)
    if ni is None or ocf is None or ta is None or ta == 0:
        return None
    return (ni - ocf) / ta


def earnings_yield(pe_ttm: float | None) -> float | None:
    """EP 盈利收益率 = 1 / PE_TTM。PE≤0(亏损)或缺失返回 None。"""
    if pe_ttm is None or pe_ttm <= 0:
        return None
    return 1.0 / pe_ttm


def _qfq(daily_sub: pd.DataFrame, adj_sub: pd.DataFrame) -> pd.Series:
    """前复权收盘价序列(按 trade_date 升序)。以最新 adj_factor 为基准。"""
    m = daily_sub.merge(adj_sub, on="trade_date", how="inner").sort_values("trade_date")
    last_adj = m["adj_factor"].iloc[-1]
    return m["close"] * m["adj_factor"] / last_adj


def reversal(daily_sub: pd.DataFrame, adj_sub: pd.DataFrame, n: int = 20) -> float | None:
    """近 n 日前复权收益率(正=近期涨多=短期反转回调风险大)。历史不足返回 None。"""
    q = _qfq(daily_sub, adj_sub).reset_index(drop=True)
    if len(q) < n + 1:
        return None
    return q.iloc[-1] / q.iloc[-1 - n] - 1.0


def volatility(daily_sub: pd.DataFrame, adj_sub: pd.DataFrame, n: int = 60) -> float | None:
    """近 n 日前复权日收益标准差(低波动因子,越低越好)。历史不足返回 None。"""
    q = _qfq(daily_sub, adj_sub).reset_index(drop=True)
    rets = q.pct_change().dropna()
    if len(rets) < n:
        return None
    return float(rets.tail(n).std())
