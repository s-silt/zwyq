"""入场特征与准入门禁(spec §6)——特征只做研究评测,过八条门禁前不产生 BUY。

实验注册(spec §6.1 防挑选偏差:先验先行、受测规则登记于此、阈值带邻域受测):
- R_READY  entry_readiness(收复5日线+缩量):用户既有短线信号,科远实盘证伪后冻结,
  本实验为其正式过审。先验=回调后企稳的行为金融文献族;登记 2026-07-19。
- R_DMA20  价格相对 MA20 回踩:触发带"回落带内买入"哲学的检验。先验=用户触发带
  既有约定(MA20 回踩带);阈值 0,邻域 {-2%, 0, +2%}。
- R_RET5   近5日收益为负(回调日入场):先验=自家 12 年追涨族全反向证据(反转向)。
  阈值 0,邻域 {-2%, 0, +2%}。
- R_GAP    T+1 开盘跳空 ≤3%(不追高开):执行层沿用的手拍规则正式受审。先验=
  momentum-screen-limitup(涨停/大幅高开=已涨完);阈值 3%,邻域 {2%, 3%, 4%}。
市场状态切片仅用于报告,不作择时开关(spec §6.1)。
"""
from __future__ import annotations

import math

import pandas as pd


def dma20(close_hist: pd.DataFrame) -> pd.Series:
    """末日收盘相对 20 日均线(前复权面板,行=日升序):last/MA20 − 1;不足 20 根 NaN。"""
    if len(close_hist) < 20:
        return pd.Series(float("nan"), index=close_hist.columns)
    ma = close_hist.tail(20).mean()
    return close_hist.iloc[-1] / ma - 1.0


def ret_n(close_hist: pd.DataFrame, n: int = 5) -> pd.Series:
    """近 n 日收益:last/close[-1-n] − 1;历史不足 NaN。"""
    if len(close_hist) < n + 1:
        return pd.Series(float("nan"), index=close_hist.columns)
    return close_hist.iloc[-1] / close_hist.iloc[-1 - n] - 1.0


def gap_pct(open_next: pd.Series, close_t: pd.Series) -> pd.Series:
    """T+1 开盘跳空幅度:open(t+1)/close(t) − 1(对齐索引;任一侧缺=NaN)。"""
    return open_next / close_t - 1.0


# —— 八条准入门禁(spec §6.3;正式阈值实验后固化 methodology,不预写死) ——
_GATES = ("PRIOR_REGISTERED", "NET_POSITIVE_VS_BASE", "LOYO_STABLE", "REGIME_CONSISTENT",
          "SIGNIFICANT_AT_PRESPECIFIED_HORIZON", "COVERAGE_FLOOR", "MDD_NOT_WORSE",
          "NEIGHBORHOOD_STABLE")
COVERAGE_FLOOR = 0.20   # spec §6.3:候选覆盖率不低于基础 D10 的 20%


def gate_verdict(stats: dict) -> dict:
    """入场规则准入判定。stats 键:

    prior_registered(bool)、net_diff(成本后相对基础组,%/期)、loyo_signs(逐年
    diff 符号列表,+1/-1/0)、up_diff/dn_diff(涨跌市切片 diff)、
    sig_t(预先指定持有期的 NW t)、coverage(规则组/基础组覆盖率)、
    mdd_rule/mdd_base(≤0)、neighborhood_diffs(邻域阈值的 net_diff 列表)。
    返回 {passed, failed:[gate...]}——全过才可入生产(缺键 KeyError fail-loud)。
    """
    failed: list[str] = []
    if not stats["prior_registered"]:
        failed.append("PRIOR_REGISTERED")
    if not (math.isfinite(stats["net_diff"]) and stats["net_diff"] > 0):
        failed.append("NET_POSITIVE_VS_BASE")
    signs = [s for s in stats["loyo_signs"] if s != 0]
    if not signs or any(s != signs[0] for s in signs):
        failed.append("LOYO_STABLE")
    up, dn = stats["up_diff"], stats["dn_diff"]
    if not (math.isfinite(up) and math.isfinite(dn)):
        failed.append("REGIME_CONSISTENT")     # 切片缺数据=不可判,不得视为满足(Codex P1)
    elif up * dn < 0 and min(abs(up), abs(dn)) > 1e-12:
        # 涨跌市方向相反且都非零=状态反转,需可解释;机械判定先挡下(spec:不发生
        # 无法解释的方向反转)
        failed.append("REGIME_CONSISTENT")
    if not (math.isfinite(stats["sig_t"]) and abs(stats["sig_t"]) > 2.0 and stats["net_diff"] > 0):
        failed.append("SIGNIFICANT_AT_PRESPECIFIED_HORIZON")
    if not (math.isfinite(stats["coverage"]) and stats["coverage"] >= COVERAGE_FLOOR):
        failed.append("COVERAGE_FLOOR")
    if not (math.isfinite(stats["mdd_rule"]) and math.isfinite(stats["mdd_base"])):
        failed.append("MDD_NOT_WORSE")         # 回撤不可算=不可判,不得视为满足(Codex P1)
    elif stats["mdd_rule"] < stats["mdd_base"] - 1e-12:
        failed.append("MDD_NOT_WORSE")
    nb = [d for d in stats["neighborhood_diffs"] if math.isfinite(d)]
    if not nb or any(d <= 0 for d in nb):
        failed.append("NEIGHBORHOOD_STABLE")
    return {"passed": not failed, "failed": failed, "gates": list(_GATES)}
