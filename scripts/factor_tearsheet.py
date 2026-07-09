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


def tradable_real_net(res: pd.DataFrame, fac: str) -> float:
    """**可交易方向**的真实净:sign(IC 均值)×SPR_t − τ_t×cost_t 的均值。

    方向盲区(P1 实跑发现):反转向因子(IVOL/MAX/TURN 等,IC<0)的 SPR 恒为负——
    那是"做多高投机腿"的方向,没人会这么交易;可交易方向是做多低因子腿(规避彩票票),
    净值 = |SPR| − 成本。按原始方向算会把整个反转族在"真实净>0"门机械误杀。
    """
    ic_mean = res["IC_" + fac].dropna().mean()
    sign = 1.0 if ic_mean >= 0 else -1.0
    net = sign * res["SPR_" + fac] - res["TO_" + fac] * res["cost_rt"]
    return float(net.dropna().mean())


def long_leg_net(res: pd.DataFrame, fac: str) -> float:
    """可交易向**多头腿**成本后超额:sign(IC)选腿(QHI/QLO)与对应腿换手(TOHI/TOLO)。

    两处口径修正(review 第三批 P1):①多头腿成本必须用**对应腿**的 τ——低腿换手 10%/
    高腿 90% 时用两腿平均 50% 扣会错估纯多头可拿收益;②该值升格为准入第五门
    (MOM 教训:过四门但多头腿 -0.28%,'腿分解为准入必查'写了要执行)。
    旧明细无逐腿 τ 列时回退 TO_(均值,标注保守偏差方向不定)。
    """
    ic_mean = res["IC_" + fac].dropna().mean()
    sign = 1.0 if ic_mean >= 0 else -1.0
    leg = res[("QHI_" if sign > 0 else "QLO_") + fac]
    to_col = ("TOHI_" if sign > 0 else "TOLO_") + fac
    tau = res[to_col] if to_col in res.columns else res["TO_" + fac]
    return float((leg - res["mkt_fwd"] - tau * res["cost_rt"]).dropna().mean())


def admission_verdict(full_t: float, real_net: float, loyo: dict[str, float],
                      up_ic: float, down_ic: float, leg_net: float) -> tuple[bool, list[str]]:
    """入 composite 准入判定:**五门**全过 → (True, []);否则 (False, 未过理由)。

    第五门=可交易向多头腿成本后>0(纯多头产品的命门,MOM 教训后升格为门)。
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
    if not leg_net > 0:
        reasons.append(f"多头腿成本后>0 未过({leg_net * 100:+.2f}%/期,纯多头拿不到)")
    return not reasons, reasons


def main() -> None:
    res = pd.DataFrame(json.load(open(DETAIL, encoding="utf-8")))
    if "mkt_fwd" not in res.columns:
        raise SystemExit("明细缺 mkt_fwd 列——先用新版 factor_backtest 重跑产出")
    facs = sorted(c[3:] for c in res.columns if c.startswith("IC_"))
    print(f"=== 因子 tear-sheet(N={len(res)},{res['date'].min()}→{res['date'].max()};"
          f"准入五门=NW t>3+真实净>0+LOYO 同号过2+涨跌市同号+多头腿净>0)===")
    for fac in facs:
        ic = res["IC_" + fac].dropna()
        if ic.empty:
            print(f"\n◆ {fac}: 无有效 IC(数据窗不足),跳过")
            continue
        _, full_t, _ = newey_west_tstat(ic)
        real_net = tradable_real_net(res, fac) if "TO_" + fac in res.columns else float("nan")
        loyo = loyo_tstats(res, "IC_" + fac)
        up_ic, down_ic = state_split(res, "IC_" + fac)
        has_leg = "QLO_" + fac in res.columns
        leg_net = long_leg_net(res, fac) if has_leg else float("nan")
        ok, reasons = admission_verdict(full_t, real_net, loyo, up_ic, down_ic, leg_net)
        worst_year = min(loyo, key=lambda k: abs(loyo[k]))
        print(f"\n◆ {fac}: IC {ic.mean():+.3f} | NW t {full_t:+.2f} | 可交易向真实净 {real_net * 100:+.2f}%/期"
              f" | 涨市 {up_ic:+.3f}/跌市 {down_ic:+.3f}"
              f" | LOYO 最弱折 {worst_year}(t {loyo[worst_year]:+.2f})")
        # 腿分解(纯多头命门):多头腿 − 宇宙等权 − **对应腿** τ×成本(第五门输入)
        if has_leg:
            sign = 1.0 if ic.mean() >= 0 else -1.0
            leg = res["QHI_" + fac] if sign > 0 else res["QLO_" + fac]
            leg_ex = (leg - res["mkt_fwd"]).dropna().mean()
            print(f"  多头腿 vs 宇宙: 超额 {leg_ex * 100:+.2f}%/期,成本后 {leg_net * 100:+.2f}%/期"
                  f"(spread 的多头腿贡献占比 {leg_ex / abs(res['SPR_' + fac].dropna().mean()) * 100:.0f}%)")
        print("  LOYO: " + " ".join(f"{y}:{t:+.1f}" for y, t in loyo.items()))
        print(f"  准入: {'✅ 过五门(可提入分讨论)' if ok else '❌ ' + ';'.join(reasons)}")
    print("\n(准入仅适用入分因子;风控/治理标签不适用此线。终判永远人工。)")


if __name__ == "__main__":
    main()
