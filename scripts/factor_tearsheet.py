"""因子 tear-sheet(吸纳终榜 P1 之 6+7)—— 年度 LOYO + 市场状态切片 + 准入判定。

读 factor_backtest 的逐期明细(data/holdscore/factor_ic_backtest.json),对每个因子:
- **LOYO**(leave-one-year-out):逐年剔除后重算 NW t——暴露"只靠单一年份撑起"的因子
  (Qlib workflow 实验纪律 / GKX 验证集思想的月频轻量版,不拟合任何权重);
- **市场状态切片**:按当期宇宙等权前向收益(mkt_fwd)正/负分涨市/跌市,各报 IC 均值
  ——暴露"只在单边行情成立"的因子;
- **准入判定**(admission_verdict,仅适用入 composite 的因子,不适用治理/风控标签):
  NW t>3(Harvey-Liu-Zhu 2016:同一份数据反复挖因子,发现率膨胀,单次检验的 t>2
  不够)+ 真实净>0(τ×round_trip 折扣后)+ LOYO 各折同号且不塌 + 涨/跌市 IC 同号。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.factor_tearsheet
       (先跑 scripts.factor_backtest [--candidates] 产出明细)
"""
from __future__ import annotations

import json

import pandas as pd

from ashare_gauntlet.backtest import newey_west_tstat

DETAIL = "data/holdscore/factor_ic_backtest.json"
T_ADMIT = 3.0     # Harvey-Liu-Zhu (2016) 多重检验准入线(文献常数,非手拍)
T_FOLD = 2.0      # 折内仍显著的常规单次检验线(同一文献体系的基准线)


def loyo_tstats(res: pd.DataFrame, ic_col: str) -> dict[str, float]:
    """逐年剔除(leave-one-year-out)后的 NW t。折 = 被剔除的年份。

    读法:某折 t 远低于其余折 → 该年是因子的单一引擎(全样本显著性被一年撑起);
    各折同号且都过 T_FOLD → 时间稳健。
    """
    y = res["date"].astype(str).str[:4]
    out: dict[str, float] = {}
    for year in sorted(y.unique()):
        ic = res.loc[y != year, ic_col].dropna()
        _, t, _ = newey_west_tstat(ic)
        out[year] = t
    return out


def state_split(res: pd.DataFrame, ic_col: str) -> tuple[float, float]:
    """(涨市 IC 均值, 跌市 IC 均值)。状态 = 当期宇宙等权前向收益 mkt_fwd 的符号
    (定义性切分,无阈值;正/负号是"多头环境/空头环境"的最朴素代理)。"""
    up = res.loc[res["mkt_fwd"] > 0, ic_col].dropna()
    down = res.loc[res["mkt_fwd"] <= 0, ic_col].dropna()
    return float(up.mean()), float(down.mean())


def admission_verdict(full_t: float, real_net: float, loyo: dict[str, float],
                      up_ic: float, down_ic: float) -> tuple[bool, list[str]]:
    """入 composite 准入判定:四门全过 → (True, []);否则 (False, 未过理由)。

    仅适用**入分因子**;治理雷/流动性警示/人工复核标签不适用此线(对抗轮划定:
    风控信号的价值不在预测收益,用收益 t 值门槛会误杀)。
    """
    reasons: list[str] = []
    if not abs(full_t) > T_ADMIT:
        reasons.append(f"NW t>3 未过(t={full_t:+.2f};Harvey-Liu-Zhu 多重检验线)")
    if not real_net > 0:
        reasons.append(f"真实净>0 未过({real_net * 100:+.2f}%/期)")
    ts = list(loyo.values())
    same_sign = all(t > 0 for t in ts) or all(t < 0 for t in ts)
    if not (same_sign and all(abs(t) > T_FOLD for t in ts)):
        reasons.append("LOYO 未过(存在变号折或 |t|≤2 折=单一年份引擎/时间不稳)")
    if not (up_ic * down_ic > 0):
        reasons.append(f"市场状态未过(涨市 {up_ic:+.3f} vs 跌市 {down_ic:+.3f} 变号)")
    return not reasons, reasons


def main() -> None:
    res = pd.DataFrame(json.load(open(DETAIL, encoding="utf-8")))
    if "mkt_fwd" not in res.columns:
        raise SystemExit("明细缺 mkt_fwd 列——先用新版 factor_backtest 重跑产出")
    facs = sorted(c[3:] for c in res.columns if c.startswith("IC_"))
    print(f"=== 因子 tear-sheet(N={len(res)},{res['date'].min()}→{res['date'].max()};"
          f"准入线=NW t>3+真实净>0+LOYO 同号过2+涨跌市同号)===")
    for fac in facs:
        ic = res["IC_" + fac].dropna()
        if ic.empty:
            print(f"\n◆ {fac}: 无有效 IC(数据窗不足),跳过")
            continue
        _, full_t, _ = newey_west_tstat(ic)
        to = res.get("TO_" + fac)
        real_net = float((res["SPR_" + fac] - to * res["cost_rt"]).dropna().mean()) if to is not None else float("nan")
        loyo = loyo_tstats(res, "IC_" + fac)
        up_ic, down_ic = state_split(res, "IC_" + fac)
        ok, reasons = admission_verdict(full_t, real_net, loyo, up_ic, down_ic)
        worst_year = min(loyo, key=lambda k: abs(loyo[k]))
        print(f"\n◆ {fac}: IC {ic.mean():+.3f} | NW t {full_t:+.2f} | 真实净 {real_net * 100:+.2f}%/期"
              f" | 涨市 {up_ic:+.3f}/跌市 {down_ic:+.3f}"
              f" | LOYO 最弱折 {worst_year}(t {loyo[worst_year]:+.2f})")
        print("  LOYO: " + " ".join(f"{y}:{t:+.1f}" for y, t in loyo.items()))
        print(f"  准入: {'✅ 过四门(可提入分讨论)' if ok else '❌ ' + ';'.join(reasons)}")
    print("\n(准入仅适用入分因子;风控/治理标签不适用此线。终判永远人工。)")


if __name__ == "__main__":
    main()
