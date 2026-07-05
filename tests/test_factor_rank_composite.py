"""composite 因子集测试 —— 钉住"哪些因子入分"这一模型定义本身。

演进(证据链见 memory factor-backtest-a-share):
- 2026-07-05:5→3(EP+BP+ACC),剔 ROE/GP(12年噪声+互相冗余0.63);
- 2026-07-06 上午:3→2(EP+BP),P0 三修后 ACC 现形(t 2.36/IC 0.008/真实净≈0);
- 2026-07-06 晚:2→3(**EP+BP+IVOL负向**),P1 交易行为族过完整门禁:IVOL t-14.7、
  13折 LOYO 无一变号、涨跌市同号、**多头腿成本后+0.34%/期**(纯多头拿得到),
  与 EP/BP 相关仅 -0.2/-0.3(正交信息);同族 MAX/NLIMIT/TURN 多头腿≈0 → 风险标签层。
新因子入 composite 须过准入纪律(NW t>3+成本后>0+LOYO+状态)**加腿分解**(多头腿
成本后>0)——MOM 过四门但多头腿-0.28% 的教训。
"""
import pandas as pd

from scripts.factor_rank import COMPOSITE_FACTORS, composite_inputs_complete, composite_score


def test_composite_factors_are_the_validated_three():
    # 入分=EP+BP+IVOL(f_IVOL 已是"低波=高分"方向的百分位);其余全展示列
    assert set(COMPOSITE_FACTORS) == {"f_EP", "f_BP", "f_IVOL"}


def test_composite_score_invariant_to_display_factors():
    # 行为测试:扰动展示列不改变 score
    base = pd.DataFrame({
        "f_EP": [0.9, 0.1], "f_BP": [0.6, 0.4], "f_IVOL": [0.3, 0.7],
        "f_ACC": [0.99, 0.01], "f_ROE": [0.99, 0.01], "f_GPOA": [0.99, 0.01], "f_MOM": [0.99, 0.01],
    })
    perturbed = base.copy()
    display = ["f_ACC", "f_ROE", "f_GPOA", "f_MOM"]
    perturbed[display] = 1.0 - perturbed[display]
    pd.testing.assert_series_equal(composite_score(base), composite_score(perturbed))
    assert abs(composite_score(base).iloc[0] - (0.9 + 0.6 + 0.3) / 3) < 1e-12


def test_composite_score_skips_missing_factor():
    df = pd.DataFrame({"f_EP": [0.8], "f_BP": [None], "f_IVOL": [0.4]})
    assert abs(composite_score(df).iloc[0] - 0.6) < 1e-12


def test_composite_inputs_complete_gate_three_factor():
    # 入池门槛=EP/BP/IVOL 原值全齐(新上市<21日无 IVOL → 如实清退);展示列缺失不清退
    df = pd.DataFrame({
        "EP": [0.10, 0.10, None], "BP": [0.50, 0.50, 0.50], "IVOL": [0.02, None, 0.02],
        "ACC": [None, 0.02, 0.02], "roe": [None, 9.9, 9.9], "GPOA": [None, 0.3, 0.3]})
    assert composite_inputs_complete(df).tolist() == [True, False, False]


def test_spec_crowd_flags_union_of_family_top_decile():
    # 🎰投机拥挤标签:IVOL/MAX/NLIMIT 任一处于当日横截面 top decile(to_decile==10,
    # 复用十分位既有约定,零新常数;union 为定义性聚合)。原值口径(非中性化)——
    # 标签回答"这票现在是不是彩票票",人类可读性优先
    from scripts.factor_rank import spec_crowd_flags
    n = 20
    ivol = pd.Series([0.01] * n, index=[f"c{i}" for i in range(n)])
    mx = pd.Series([0.02] * n, index=ivol.index)
    nl = pd.Series([0.0] * n, index=ivol.index)
    ivol.iloc[0] = 0.9        # c0 仅 IVOL 顶
    mx.iloc[1] = 0.5          # c1 仅 MAX 顶
    nl.iloc[2] = 8.0          # c2 仅 NLIMIT 顶
    flags = spec_crowd_flags(ivol, mx, nl)
    assert bool(flags["c0"]) and bool(flags["c1"]) and bool(flags["c2"])
    assert not bool(flags["c5"])
