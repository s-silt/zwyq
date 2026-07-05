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
                                    loyo={"2020": 3.1, "2021": 3.4}, up_ic=0.03, down_ic=0.02)
    assert ok and reasons == []
    # 任一门不过 → 拒绝并给出理由
    bad, reasons = admission_verdict(full_t=2.5, real_net=0.002,
                                     loyo={"2020": 3.1, "2021": 3.4}, up_ic=0.03, down_ic=0.02)
    assert not bad and any("t>3" in r for r in reasons)
    bad, reasons = admission_verdict(full_t=3.5, real_net=-0.001,
                                     loyo={"2020": 3.1, "2021": 3.4}, up_ic=0.03, down_ic=0.02)
    assert not bad and any("真实净" in r for r in reasons)
    bad, reasons = admission_verdict(full_t=3.5, real_net=0.002,
                                     loyo={"2020": 3.1, "2021": -0.4}, up_ic=0.03, down_ic=0.02)
    assert not bad and any("LOYO" in r for r in reasons)
    bad, reasons = admission_verdict(full_t=3.5, real_net=0.002,
                                     loyo={"2020": 3.1, "2021": 3.4}, up_ic=0.03, down_ic=-0.01)
    assert not bad and any("状态" in r for r in reasons)
