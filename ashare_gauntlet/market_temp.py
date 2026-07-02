"""市场温度读数 —— 每日盯盘头部一行,指导仓位松紧;不选股、不预测。

只 surface 各读数,不做加权综合分(任何权重都会是 magic number,冷热判断留给人)。
四个读数全部定义性/相对自身 20 日基准(20 日 = 代码库既有窗口约定,见
pick_track.REGIME_WINDOW / pct20 / ret20):

① 涨停家数与炸板率:limit_list_d 的 limit 列(U=涨停收盘/Z=炸板/D=跌停,实测确认),
   炸板率 = Z/(U+Z) —— 封板成功率的补,是情绪退潮最早的定义性信号;
② 全市场成交额:今日 vs 前 20 日均值的比值(放量/缩量,只报比值不设阈值);
③ 北向成交额:净流入自 2024-08-19 制度性停披露(见 factsheet.py LANDMINE 注释),
   日频仅剩总成交额 —— 只 surface 成交额并标注口径,拒绝伪造净流入;
④ 沪深300 regime:复用 scripts.pick_track 的 20 日涨跌读数(组装在 scripts 层)。

全部纯函数;空表/坏数据 fail-loud 不伪造(分析优先级:数据源纯净 > 结果效率)。
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd

from ashare_gauntlet.factsheet import NORTH_FLOW_SEMANTICS_CUTOFF, north_turnover

# limit_list_d 的 limit 列编码(tushare 文档 + 实测):U=涨停收盘 Z=炸板 D=跌停
LIMIT_STATUS_UP = "U"
LIMIT_STATUS_BROKEN = "Z"
LIMIT_STATUS_DOWN = "D"
_KNOWN_LIMIT_STATUSES = frozenset({LIMIT_STATUS_UP, LIMIT_STATUS_BROKEN, LIMIT_STATUS_DOWN})

# 成交额基准窗口 = 20 交易日 ≈ 一个自然月,复用代码库既有约定(REGIME_WINDOW/pct20/ret20)
BASELINE_WINDOW = 20

# 北向成交额累计窗口 = 5 交易日 = 一个自然周(日历常数,非调出来的参数)
NORTH_SUM_DAYS = 5


class EmptyLimitListError(RuntimeError):
    """涨跌停名单空表 —— A 股全市场 ~5000 只,真实交易日不可能 0 只触板;
    空表只能是拉取失败/日期未发布,当真值计数会伪造"今日零涨停",fail-loud。"""


class UnknownLimitStatusError(RuntimeError):
    """limit 列出现 U/Z/D 之外的编码 —— 镜像 schema 漂移;静默丢行会低估计数,fail-loud。"""


class InsufficientAmountHistoryError(RuntimeError):
    """成交额序列不足 window+1 个值或窗口内含 NaN —— 基准会错移/被污染,fail-loud
    (window+1 约定同 regime_return 的 n+1:前 window 日做基准 + 今日)。"""


class EmptyNorthTurnoverError(RuntimeError):
    """北向成交额无有效数据(空表或 north_money 空值)—— 空值不是 0,fail-loud 不伪造。"""


class InsufficientNorthHistoryError(RuntimeError):
    """北向有数日不足累计窗口 —— 20 交易日拉取窗口内 HK 最长连休也留 ≥5 个有数日,
    不足只能是拉取失败;缩窗累计会伪造口径,fail-loud。"""


def limit_counts(limit_list: pd.DataFrame) -> dict[str, float]:
    """涨停/炸板/跌停家数 + 炸板率(= Z/(U+Z),分母 0 → NaN 不伪造成 0)。

    输入:limit_list_d 单日全市场名单,须含 ``limit`` 列(U/Z/D)。
    空表/未知编码 → fail-loud(见各异常 docstring)。
    """
    if limit_list.empty:
        raise EmptyLimitListError(
            "limit_list_d 返回 0 行 —— 真实交易日不可能零触板,疑为未发布/拉取失败;"
            "拒绝把空表当'今日零涨停'")
    statuses = limit_list["limit"].astype(str)
    unknown = sorted(set(statuses) - _KNOWN_LIMIT_STATUSES)
    if unknown:
        raise UnknownLimitStatusError(
            f"limit 列出现未知编码 {unknown}(已知 U=涨停/Z=炸板/D=跌停)—— "
            f"疑为 schema 漂移,拒绝静默丢行")
    up = int((statuses == LIMIT_STATUS_UP).sum())
    broken = int((statuses == LIMIT_STATUS_BROKEN).sum())
    down = int((statuses == LIMIT_STATUS_DOWN).sum())
    rate = broken / (up + broken) if (up + broken) > 0 else math.nan
    return {"up": up, "broken": broken, "down": down, "broken_rate": rate}


def amount_ratio(amounts: Sequence[float], window: int = BASELINE_WINDOW) -> dict[str, float]:
    """今日全市场成交额 vs 前 ``window`` 日均值的比值(>1 放量,<1 缩量;不设阈值)。

    输入:按交易日升序的日总成交额序列(单位由调用方统一,比值本身无量纲)。
    基准不含今日 —— 今日放量若计入自身基准会稀释读数;需要 window+1 个值
    (同 regime_return 的 n+1 约定),不足或窗口内含 NaN → fail-loud。
    """
    vals = [float(a) for a in amounts]
    if len(vals) < window + 1:
        raise InsufficientAmountHistoryError(
            f"成交额序列仅 {len(vals)} 个值,需 {window + 1}(前 {window} 日基准 + 今日)—— "
            f"缓存不足,拒绝缩窗伪造基准")
    tail = vals[-(window + 1):]
    if any(v != v for v in tail):
        raise InsufficientAmountHistoryError(
            f"成交额窗口内含 NaN(最近 {window + 1} 日)—— 坏缓存值会污染基准,不藏不补")
    today = tail[-1]
    baseline = sum(tail[:-1]) / window
    return {"today": today, "baseline": baseline, "ratio": today / baseline, "window": window}


def north_turnover_recent(mf: pd.DataFrame, days: int = NORTH_SUM_DAYS) -> dict[str, float | str]:
    """北向(沪股通+深股通)最近一有数日**总成交额**(亿)+ 最近 ``days`` 个有数日累计。

    口径:成交额,非净流入 —— 净流入 2024-08-19 制度性停披露(factsheet LANDMINE),
    窗口内混入换义前日期 → ValueError 拒绝(~50x 量级差,误读即灾难)。
    HK 休市日整行缺失是合法的(按有数日取,不补零);有行但 north_money 空值是坏数据
    → fail-loud。单位换算复用 factsheet.north_turnover(百万元→亿元)。
    """
    if mf.empty:
        raise EmptyNorthTurnoverError("moneyflow_hsgt 返回 0 行 —— 拉取失败/窗口无数据,不伪造")
    bad = mf[mf["north_money"].isna()]
    if not bad.empty:
        raise EmptyNorthTurnoverError(
            f"north_money 存在空值行(trade_date={sorted(str(d) for d in bad['trade_date'])})—— "
            f"HK 休市应整行缺失而非 NaN 行,疑为坏数据,不藏不补")
    rows = mf.sort_values("trade_date")
    if len(rows) < days:
        raise InsufficientNorthHistoryError(
            f"北向有数日仅 {len(rows)} 个,累计窗口需 {days} —— 拉取窗口应保证充足,"
            f"拒绝缩窗伪造口径")
    tail = rows.iloc[-days:]
    dates = [str(d) for d in tail["trade_date"]]
    if dates[0] < NORTH_FLOW_SEMANTICS_CUTOFF:
        raise ValueError(
            f"累计窗口最早日 {dates[0]} 在 {NORTH_FLOW_SEMANTICS_CUTOFF} 前 —— "
            f"north_money 换义前是净流入(~50x 量级差),拒绝混算")
    latest = north_turnover(mf, dates[-1])  # 复用:亿元换算 + cutoff 守卫
    sum_yi = float(tail["north_money"].astype(float).sum()) / 100.0  # 百万元 → 亿元
    return {"latest_date": dates[-1], "latest_yi": latest["total_yi"],
            "sum_yi": sum_yi, "days": days}


def _pct(x: float, signed: bool = False, digits: int = 0) -> str:
    """百分比渲染;NaN → 'n/a' 不伪造成 0。"""
    if x != x:
        return "n/a"
    return f"{x * 100:{'+' if signed else ''}.{digits}f}%"


def summary_line(as_of: str, lim: dict[str, float], amt: dict[str, float],
                 north: dict[str, float | str], regime_pct: float, regime_window: int) -> str:
    """一行市场温度摘要:只并列 surface 四个读数,不加权、不给冷热结论。"""
    return (
        f"[市场温度 {as_of}] "
        f"涨停{lim['up']} 炸板{lim['broken']}(炸板率{_pct(lim['broken_rate'])}) 跌停{lim['down']}"
        f" | 成交{amt['today']:.0f}亿={amt['window']}日均×{amt['ratio']:.2f}"
        f" | 北向成交{north['latest_yi']:.0f}亿/近{north['days']}日{north['sum_yi']:.0f}亿"
        f"(净流入已停披露)"
        f" | 沪深300近{regime_window}日{_pct(regime_pct, signed=True, digits=1)}"
    )
