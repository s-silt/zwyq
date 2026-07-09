"""P1 因子 tear-sheet(吸纳终榜 6+7):年度 leave-one-year-out + 市场状态切片 + 准入判定。

防"同一份 149 期数据反复挖因子":新因子入 composite 的准入线 = NW t>3(Harvey-Liu-Zhu
多重检验)+ 真实净>0 + LOYO 各折方向一致 + 涨/跌市同号。只适用入分因子,不适用治理/
风控标签(对抗轮划定的适用范围)。
"""
import numpy as np
import pandas as pd

from scripts.factor_tearsheet import admission_verdict, loyo_tstats, state_split

RNG = np.random.default_rng(11)


def _res(n_years: int = 5) -> pd.DataFrame:
    dates, ic, mkt = [], [], []
    for y in range(2020, 2020 + n_years):
        for m in range(1, 13):
            dates.append(f"{y}{m:02d}28")
            ic.append(0.04 + RNG.normal(0, 0.01))
            mkt.append(RNG.normal(0, 0.03))
    return pd.DataFrame({"date": dates, "IC_X": ic, "mkt_fwd": mkt})


def test_loyo_tstats_one_fold_per_year():
    out = loyo_tstats(_res(), "IC_X")
    assert set(out) == {"2020", "2021", "2022", "2023", "2024"}
    assert all(t > 0 for t in out.values())          # 稳定正因子:去掉任一年仍显著为正


def test_loyo_tstats_detects_single_year_driver():
    # 因子只在 2022 一年有效(其余噪声)→ 去掉 2022 的折 t 崩、其余折 t 高:折间号向/量级不一致
    res = _res()
    res["IC_X"] = RNG.normal(0, 0.01, len(res))
    res.loc[res["date"].str.startswith("2022"), "IC_X"] = 0.15
    out = loyo_tstats(res, "IC_X")
    assert out["2022"] < 2.0                          # 去掉引擎年,显著性塌
    assert max(out.values()) > 2.0                    # 其余折被单年撑着


def test_state_split_up_down():
    res = _res()
    res["mkt_fwd"] = [0.05, -0.05] * (len(res) // 2)
    res.loc[res["mkt_fwd"] > 0, "IC_X"] = 0.06        # 涨市有效
    res.loc[res["mkt_fwd"] <= 0, "IC_X"] = -0.02      # 跌市反向
    up, down = state_split(res, "IC_X")
    assert up > 0 > down


def test_admission_verdict_all_gates():
    ok, reasons = admission_verdict(full_t=3.5, real_net=0.002,
                                    loyo={"2020": 3.1, "2021": 3.4}, up_ic=0.03, down_ic=0.02,
                                    leg_net=0.001)
    assert ok and reasons == []
    # 任一门不过 → 拒绝并给出理由(第五门=多头腿净,见 test_codex_batch3)
    bad, reasons = admission_verdict(full_t=2.5, real_net=0.002,
                                     loyo={"2020": 3.1, "2021": 3.4}, up_ic=0.03, down_ic=0.02,
                                     leg_net=0.001)
    assert not bad and any("t>3" in r for r in reasons)
    bad, reasons = admission_verdict(full_t=3.5, real_net=-0.001,
                                     loyo={"2020": 3.1, "2021": 3.4}, up_ic=0.03, down_ic=0.02,
                                     leg_net=0.001)
    assert not bad and any("真实净" in r for r in reasons)
    bad, reasons = admission_verdict(full_t=3.5, real_net=0.002,
                                     loyo={"2020": 3.1, "2021": -0.4}, up_ic=0.03, down_ic=0.02,
                                     leg_net=0.001)
    assert not bad and any("LOYO" in r for r in reasons)
    bad, reasons = admission_verdict(full_t=3.5, real_net=0.002,
                                     loyo={"2020": 3.1, "2021": 3.4}, up_ic=0.03, down_ic=-0.01,
                                     leg_net=0.001)
    assert not bad and any("状态" in r for r in reasons)


def test_tradable_real_net_flips_reversal_direction():
    # 方向盲区(P1 实跑发现):IVOL 类反转因子 SPR<0(=低IVOL腿赢),按原始方向算
    # 真实净恒为负、被"真实净>0"门机械否决;可交易方向 = sign(IC均值)×SPR − τ×成本
    from scripts.factor_tearsheet import tradable_real_net
    res = pd.DataFrame({
        "IC_Y": [-0.05, -0.06, -0.04],          # 稳定负 IC(反转向)
        "SPR_Y": [-0.015, -0.014, -0.016],      # 高因子腿输 1.5%/期
        "TO_Y": [0.5, 0.5, 0.5],
        "cost_rt": [0.004, 0.004, 0.004],
    })
    net = tradable_real_net(res, "Y")
    assert abs(net - (0.015 - 0.5 * 0.004)) < 1e-9   # 翻向后 +1.3%/期
    # 正向因子不受影响
    res2 = res.copy()
    res2[["IC_Y", "SPR_Y"]] = -res2[["IC_Y", "SPR_Y"]]
    assert abs(tradable_real_net(res2, "Y") - (0.015 - 0.002)) < 1e-9


def test_quantile_leg_means_monotone():
    # 腿分解:纯多头产品的命门——spread 若全靠空头腿(垃圾崩),多头腿跑不赢宇宙=纸面alpha
    from scripts.factor_backtest import quantile_leg_means
    f = pd.Series({f"c{i}": float(i) for i in range(50)})
    r = pd.Series({f"c{i}": float(i) / 100 for i in range(50)})
    lo, hi = quantile_leg_means(f, r, 5)
    assert abs(lo - 0.045) < 1e-9 and abs(hi - 0.445) < 1e-9   # 底组均值 (0..9)/100,顶组 (40..49)/100


def test_quantile_leg_means_small_sample_nan():
    import math
    from scripts.factor_backtest import quantile_leg_means
    f = pd.Series({f"c{i}": float(i) for i in range(10)})
    lo, hi = quantile_leg_means(f, f / 100, 5)
    assert math.isnan(lo) and math.isnan(hi)
