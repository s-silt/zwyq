"""composite 因子集测试 —— 钉住"哪些因子入分"这一模型定义本身。

2026-07 P1 终审(N=149, 2014-2026, 成本后+剔壳稳健,见 memory factor-backtest-a-share):
EP/BP 可交易有效、ACC 统计显著且与一切正交;ROE/GP 12 年噪声(t<1)+ 互相冗余(相关 0.63)
+ ROE 半被 EP 吸收(0.49)→ 降出 composite 只留展示。此测试防止未来"顺手加回去"式回归。
"""
import pandas as pd

from scripts.factor_rank import COMPOSITE_FACTORS, composite_score


def test_composite_factors_are_the_validated_three():
    # 入分因子=本地 12.5 年实证过的三个;ROE/GP/MOM 不入分(展示列)
    assert set(COMPOSITE_FACTORS) == {"f_EP", "f_BP", "f_ACC"}


def test_composite_score_invariant_to_display_factors():
    # 行为测试:扰动 f_ROE/f_GPOA/f_MOM 不改变 score(展示列不泄漏进合成)
    base = pd.DataFrame({
        "f_EP": [0.9, 0.1], "f_BP": [0.6, 0.4], "f_ACC": [0.3, 0.7],
        "f_ROE": [0.99, 0.01], "f_GPOA": [0.99, 0.01], "f_MOM": [0.99, 0.01],
    })
    perturbed = base.copy()
    perturbed[["f_ROE", "f_GPOA", "f_MOM"]] = 1.0 - perturbed[["f_ROE", "f_GPOA", "f_MOM"]]
    pd.testing.assert_series_equal(composite_score(base), composite_score(perturbed))
    # 等权口径不变:score = mean(EP,BP,ACC)
    assert abs(composite_score(base).iloc[0] - (0.9 + 0.6 + 0.3) / 3) < 1e-12


def test_composite_score_skips_missing_factor():
    # 缺某因子 → 可得因子均值(沿用 composite 的 skipna 语义,不当 0 罚)
    df = pd.DataFrame({"f_EP": [0.8], "f_BP": [None], "f_ACC": [0.4],
                       "f_ROE": [0.5], "f_GPOA": [0.5]})
    assert abs(composite_score(df).iloc[0] - 0.6) < 1e-12
