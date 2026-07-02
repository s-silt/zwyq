"""执行层纯函数:右侧确认(entry_readiness)+ A股整手仓位公式(position_size)。

双仓制的执行侧(制度出处 memory trading-constraints:短线仓右侧入场=缩量企稳+收复
5日线,单笔风险=账户风险预算的固定比例)。本模块只做**定义性比较**与**恒等式换算**,
零可调常数:

- 5日 = A股最短通用均线约定(MA5,图表软件通用窗口;也与代码库既有"近5日"窗口约定
  一致,见 factor_model.touched_limit_up);
- "收复5日线" = 最新前复权收盘 > MA5(严格不等:平走不算收复);
- "缩量" = 最新成交量 < **不含最新一根**的前5日均量(被比较项不进基准,否则今天的
  天量会拉高自己的对照基准、放量被自我稀释;严格不等:平量不算缩量);
- 一手 = 100 股,交易所交易规则(监管常数,非本库发明);
- 仓位 = floor(风险预算 / 单股风险 / 100) × 100,纯恒等式:风险预算与止损距离给定后
  股数被唯一确定,向下取整手只会降低而不会突破风险预算。

risk_pct(风险比例)与止损距离是**用户制度参数**,不在本模块写死——由 CLI 层
(scripts/entry_check.py)按仓别注入并注明出处。

fail-loud:数据不足 / 参数非法一律 raise,不静默给出看似合理的判定或仓位。
"""
from __future__ import annotations

import math

import pandas as pd

# MA5 窗口:A股最短通用均线约定(5日线),与库内"近5日"窗口约定一致(touched_limit_up)
MA5_WINDOW = 5
# 最少K线根数 = 前5日量能基准(不含最新) + 最新一根 = 5 + 1(推导值,非独立常数)
MIN_BARS = MA5_WINDOW + 1
# A股一手 = 100 股(交易所交易规则,监管常数)
LOT_SIZE = 100

# 右侧判定标签(唯一事实源,CLI 与测试都从这里取,避免 emoji 拼写漂移)
LABEL_RIGHT = "右侧✓"          # 两判据全真
LABEL_STABILIZING = "企稳中"   # 仅一真
LABEL_LEFT = "左侧⚠"           # 全假


def entry_readiness(adj_close: pd.Series, vol: pd.Series) -> dict:
    """右侧确认判据(缩量企稳 + 收复5日线),入参按时间**升序**、最新在末。

    - above_ma5:最新前复权收盘 > MA5(含最新一根,图表约定);
    - shrinking_vol:最新成交量 < 前5日均量(**不含最新**,今天的量不进自己的基准);
    - label:两真=右侧✓ / 一真=企稳中 / 全假=左侧⚠。

    NaN 不算有效根数(dropna 后计数);任一序列有效根数 < MIN_BARS → fail-loud,
    拒绝在残缺数据上给出看似有据的判定。返回判定所用数字证据,CLI 直接打印不重算。
    """
    c = adj_close.dropna().astype(float)
    v = vol.dropna().astype(float)
    if len(c) < MIN_BARS or len(v) < MIN_BARS:
        raise ValueError(
            f"entry_readiness: 有效K线不足 {MIN_BARS} 根(close={len(c)}, vol={len(v)})"
            f"—— 缩量基准需要不含最新的前{MA5_WINDOW}日,数据不足拒绝判定"
        )
    close = float(c.iloc[-1])
    ma5 = float(c.iloc[-MA5_WINDOW:].mean())          # 含最新一根:图表 MA5 定义
    latest_vol = float(v.iloc[-1])
    vol_ma5 = float(v.iloc[-MIN_BARS:-1].mean())       # 不含最新:前5日量能基准
    above_ma5 = close > ma5
    shrinking_vol = latest_vol < vol_ma5
    if above_ma5 and shrinking_vol:
        label = LABEL_RIGHT
    elif above_ma5 or shrinking_vol:
        label = LABEL_STABILIZING
    else:
        label = LABEL_LEFT
    return {
        "above_ma5": above_ma5,
        "shrinking_vol": shrinking_vol,
        "label": label,
        "close": close,
        "ma5": ma5,
        "vol": latest_vol,
        "vol_ma5": vol_ma5,
    }


def position_size(account_value: float, risk_pct: float,
                  entry_price: float, stop_price: float) -> dict:
    """风险预算 → A股整手仓位:shares = floor(account×risk_pct/(entry−stop)/100)×100。

    恒等式换算,无自由参数:风险预算(account_value×risk_pct)除以单股风险
    (entry−stop)得理论股数,向下取整手(一手=100股,交易所规则)——只会降低、
    永不突破风险预算。预算不足一手 → 如实返回 0 手,不硬凑一手突破预算。

    fail-loud:任一参数非正、或 stop_price ≥ entry_price(单股风险≤0,公式失义,
    往往是止损方向写反)→ raise。risk_pct 是用户制度参数(如短线=账户1%),
    由调用方传入,本函数不设默认。
    """
    if account_value <= 0 or risk_pct <= 0 or entry_price <= 0 or stop_price <= 0:
        raise ValueError(
            f"position_size: 参数须全为正(account={account_value}, risk_pct={risk_pct}, "
            f"entry={entry_price}, stop={stop_price})"
        )
    if stop_price >= entry_price:
        raise ValueError(
            f"position_size: 止损价 {stop_price} ≥ 入场价 {entry_price} —— 单股风险≤0,"
            f"公式失义(止损应在入场价下方)"
        )
    per_share_risk = entry_price - stop_price
    lots = math.floor(account_value * risk_pct / per_share_risk / LOT_SIZE)
    shares = lots * LOT_SIZE
    return {
        "shares": shares,
        "lots": lots,
        "cost": shares * entry_price,
        "max_loss": shares * per_share_risk,
    }
