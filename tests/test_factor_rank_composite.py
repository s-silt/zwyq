"""composite 因子集测试 —— 钉住"哪些因子入分"这一模型定义本身。

演进(证据链见 memory factor-backtest-a-share):
- 2026-07-05:5→3(EP+BP+ACC),剔 ROE/GP(12年噪声+互相冗余0.63);
- 2026-07-06:3→2(**EP+BP**),P0 三修(退出侧卖出约束/真实换手/退市股财务回填)后
  ACC 现形——退市高应计公司补入样本 t 3.27→2.36、IC 0.008、真实净≈0,按自家标准
  (✓需 t>2 且 |IC|>0.02)仅"~弱";与 ROE/GP 同待遇降为展示列。质地关(现金流方向/
  净现比)独立把守盈利含金量,不依赖 ACC 入分。
此测试防止未来"顺手加回去"式回归;新因子入 composite 须过准入纪律(NW t>3+成本后>0
+方向稳定+年度切片,见吸纳终榜 P1)。
"""
import pandas as pd

from scripts.factor_rank import COMPOSITE_FACTORS, composite_inputs_complete, composite_score


def test_composite_factors_are_the_validated_two():
    # 入分因子=P0 三修后仍站住的两个;ACC/ROE/GP/MOM 全部展示列
    assert set(COMPOSITE_FACTORS) == {"f_EP", "f_BP"}


def test_composite_score_invariant_to_display_factors():
    # 行为测试:扰动 f_ACC/f_ROE/f_GPOA/f_MOM 不改变 score(展示列不泄漏进合成)
    base = pd.DataFrame({
        "f_EP": [0.9, 0.1], "f_BP": [0.6, 0.4],
        "f_ACC": [0.99, 0.01], "f_ROE": [0.99, 0.01], "f_GPOA": [0.99, 0.01], "f_MOM": [0.99, 0.01],
    })
    perturbed = base.copy()
    display = ["f_ACC", "f_ROE", "f_GPOA", "f_MOM"]
    perturbed[display] = 1.0 - perturbed[display]
    pd.testing.assert_series_equal(composite_score(base), composite_score(perturbed))
    # 等权口径不变:score = mean(EP,BP)
    assert abs(composite_score(base).iloc[0] - (0.9 + 0.6) / 2) < 1e-12


def test_composite_score_skips_missing_factor():
    # 缺某因子 → 可得因子均值(沿用 composite 的 skipna 语义,不当 0 罚)
    df = pd.DataFrame({"f_EP": [0.8], "f_BP": [None], "f_ACC": [0.1]})
    assert abs(composite_score(df).iloc[0] - 0.8) < 1e-12


def test_composite_inputs_complete_gate_two_factor():
    # 入池门槛=EP/BP 原值全齐;ACC 已是展示列,缺失不清退(与 roe/GPOA 同待遇)
    df = pd.DataFrame({
        "EP": [0.10, 0.10, None], "BP": [0.50, None, 0.50],
        "ACC": [None, 0.02, 0.02], "roe": [None, 9.9, 9.9], "GPOA": [None, 0.3, 0.3]})
    assert composite_inputs_complete(df).tolist() == [True, False, False]
